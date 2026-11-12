// This file is part of text-to-audiobook
//
// text-to-audiobook is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// text-to-audiobook is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Provisional audiobook while stage 2 is running.
//!
//! After every newly spoken chunk, `final/audiobook_preview.mp3` is built
//! again from all chunks done so far, on a thread of its own, while the TTS
//! goes on with the next chunk. One build runs at a time; chunks finished
//! during a build go into the next one, so only the newest state is built
//! and the previews never hold up the synthesis.
//!
//! When the synthesis stops early (Stop, error), the preview is brought up
//! to date with every chunk done. When it completes, the running build is
//! dropped: the final assembly follows and deletes the preview.

use std::path::{Path, PathBuf};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::Instant;

use crate::audio::assembler::{assemble_preview, chunk_id, part_path};
use crate::cancel::{StopFlag, forced};
use crate::config::Settings;
use crate::segment::Segment;

/// File name of the provisional audiobook in `<output>/<project>/final/`.
pub const PREVIEW_FILE: &str = "audiobook_preview.mp3";

#[derive(Default)]
struct State {
    /// Newest WAV list that is not built yet.
    latest: Option<Vec<PathBuf>>,
    /// No more lists come: build the newest one, then end.
    closed: bool,
    /// Drop everything and abort the running build.
    cancelled: bool,
    /// The preview thread is building a list.
    building: bool,
}

/// Hand-over from the synthesis to the preview thread: a slot for the
/// newest WAV list, which replaces one not built yet.
#[derive(Default)]
pub struct PreviewQueue {
    state: Mutex<State>,
    wake: Condvar,
}

impl PreviewQueue {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Every WAV so far, after a newly spoken chunk.
    pub fn push(&self, wavs: Vec<PathBuf>) {
        let mut s = self.lock();
        if !s.closed && !s.cancelled {
            s.latest = Some(wavs);
            self.wake.notify_all();
        }
    }

    /// No more lists; the newest one is still built.
    pub fn close(&self) {
        self.lock().closed = true;
        self.wake.notify_all();
    }

    /// Drop the waiting list and abort the running build.
    pub fn cancel(&self) {
        let mut s = self.lock();
        s.cancelled = true;
        s.latest = None;
        self.wake.notify_all();
    }

    /// [`Self::cancel`] unless already closed.
    fn cancel_if_open(&self) {
        if !self.lock().closed {
            self.cancel();
        }
    }

    /// Wait for the next list to build; None: done.
    pub fn next(&self) -> Option<Vec<PathBuf>> {
        let mut s = self.lock();
        s.building = false;
        loop {
            if s.cancelled {
                return None;
            }
            if let Some(wavs) = s.latest.take() {
                s.building = true;
                return Some(wavs);
            }
            if s.closed {
                return None;
            }
            s = self.wake.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The running build is no longer wanted: cancelled, or closed with a
    /// newer list waiting (no new chunk can supersede that one).
    pub fn superseded(&self) -> bool {
        let s = self.lock();
        s.cancelled || (s.closed && s.latest.is_some())
    }

    /// A list is waiting or being built.
    pub fn busy(&self) -> bool {
        let s = self.lock();
        s.building || s.latest.is_some()
    }
}

/// Cancels the queue when dropped unless it was closed, so the preview
/// thread also ends when the synthesis panics.
pub struct EndOnDrop<'a>(pub &'a PreviewQueue);

impl Drop for EndOnDrop<'_> {
    fn drop(&mut self) {
        self.0.cancel_if_open();
    }
}

/// `ends_paragraph` per chunk of `wavs` (sub-parts of one chunk count
/// once), as [`crate::audio::assembler::silence_plan`] expects it.
pub fn ends_paragraph_of(wavs: &[PathBuf], segments: &[Segment]) -> Vec<bool> {
    let mut ids: Vec<usize> = wavs.iter().filter_map(|w| chunk_id(w)).collect();
    ids.dedup();
    ids.iter()
        .map(|i| segments.get(*i).is_some_and(|s| s.ends_paragraph))
        .collect()
}

/// `h:mm:ss` or `m:ss`.
fn clock(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

/// Build previews into `mp3_path` from what `queue` hands over until it is
/// closed or cancelled. The first preview is announced through `report`,
/// later ones go to the debug log. Returns the description of the last
/// preview written.
pub fn run(
    queue: &PreviewQueue,
    settings: &Settings,
    segments: &[Segment],
    mp3_path: &Path,
    stop: Option<&StopFlag>,
    report: &(dyn Fn(&str) + Sync),
) -> Option<String> {
    let mut last = None;
    let mut warned = false;
    while let Some(wavs) = queue.next() {
        let ends = ends_paragraph_of(&wavs, segments);
        let abort = || queue.superseded() || forced(stop);
        let started = Instant::now();
        match assemble_preview(&wavs, &ends, mp3_path, settings, &abort) {
            Ok(Some(secs)) => {
                let msg = format!(
                    "provisional audiobook ({} of {} chunks, {}): {}",
                    ends.len(),
                    segments.len(),
                    clock(secs),
                    mp3_path.display()
                );
                if last.is_none() {
                    report(&format!("{msg} (updated after every chunk)"));
                } else {
                    log::debug!("{msg} (built in {:.1} s)", started.elapsed().as_secs_f64());
                }
                last = Some(msg);
            }
            Ok(None) => log::debug!("Provisional audiobook dropped, a newer one follows"),
            Err(e) => {
                let msg = format!("Could not update the provisional audiobook: {e:#}");
                if warned {
                    log::debug!("{msg}");
                } else {
                    log::warn!("{msg}");
                    warned = true;
                }
            }
        }
    }
    last
}

/// Delete the provisional audiobook of `final_dir` (after the final
/// assembly).
pub fn remove(final_dir: &Path) {
    let preview = final_dir.join(PREVIEW_FILE);
    for p in [part_path(&preview), preview] {
        if p.exists()
            && let Err(e) = std::fs::remove_file(&p)
        {
            log::warn!("Could not delete {}: {e}", p.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queue_builds_only_the_newest_list() {
        let q = PreviewQueue::default();
        q.push(vec![PathBuf::from("chunk_000000.wav")]);
        q.push(vec![
            PathBuf::from("chunk_000000.wav"),
            PathBuf::from("chunk_000001.wav"),
        ]);
        assert_eq!(q.next().unwrap().len(), 2);
        assert!(!q.superseded());
        assert!(q.busy());
        // Closed with a list waiting: the running build gives way to it.
        q.push(vec![PathBuf::from("chunk_000002.wav")]);
        q.close();
        assert!(q.superseded());
        assert_eq!(q.next().unwrap().len(), 1);
        assert!(!q.superseded());
        assert!(q.next().is_none());
        assert!(!q.busy());
        // Nothing is taken after closing.
        q.push(vec![PathBuf::from("x.wav")]);
        assert!(q.next().is_none());
    }

    #[test]
    fn cancel_drops_the_waiting_list() {
        let q = PreviewQueue::default();
        q.push(vec![PathBuf::from("chunk_000000.wav")]);
        q.cancel();
        assert!(q.superseded());
        assert!(q.next().is_none());
        // The drop guard leaves a closed queue alone.
        let q = PreviewQueue::default();
        q.push(vec![PathBuf::from("chunk_000000.wav")]);
        q.close();
        drop(EndOnDrop(&q));
        assert!(q.next().is_some());
    }

    #[test]
    fn next_waits_for_the_synthesis() {
        let q = PreviewQueue::default();
        std::thread::scope(|s| {
            let worker = s.spawn(|| {
                let mut built = Vec::new();
                while let Some(w) = q.next() {
                    built.push(w.len());
                }
                built
            });
            std::thread::sleep(std::time::Duration::from_millis(20));
            q.push(vec![PathBuf::from("chunk_000000.wav")]);
            std::thread::sleep(std::time::Duration::from_millis(20));
            q.close();
            assert_eq!(worker.join().unwrap(), [1]);
        });
    }

    #[test]
    fn paragraph_ends_per_chunk() {
        let seg = |ends| Segment {
            ends_paragraph: ends,
            ..Segment::with_text("x")
        };
        let segments = [seg(true), seg(false), seg(true)];
        let wavs: Vec<PathBuf> = [
            "chunk_000000.wav",
            "chunk_000001_0.wav",
            "chunk_000001_1.wav",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        assert_eq!(ends_paragraph_of(&wavs, &segments), [true, false]);
        assert_eq!(clock(59.6), "1:00");
        assert_eq!(clock(3725.0), "1:02:05");
    }
}
