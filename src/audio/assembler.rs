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

//! Join chunk WAVs with crossfades and sentence/paragraph pauses into the
//! final WAV and MP3.
//!
//! Mixing follows pydub's `AudioSegment.append` as used by the Python
//! version: linear fades between -120 dB and 0 dB, chunks are joined with
//! `CROSSFADE_MS`, pauses are appended with pydub's default crossfade of
//! 100 ms. The result is streamed to disk, so long books need little memory.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow};
use regex::Regex;

use super::{read_wav, resample, to_i16};

/// pydub's default crossfade for `append()` without an explicit value;
/// the Python version appended every pause that way.
pub const SILENCE_CROSSFADE_MS: u32 = 100;

/// -120 dB, the floor of pydub's fades.
const FADE_FLOOR: f64 = 1e-6;

/// Keep this many seconds in memory before streaming to disk.
const FLUSH_SECONDS: usize = 30;

/// `chunk_000012` (single part) or `chunk_000012_3` (sub-part 3 of chunk 12).
static CHUNK_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^chunk_(\d+)(?:_(\d+))?$").unwrap());

/// Chunk number of a WAV named `chunk_NNN.wav` or `chunk_NNN_K.wav`.
pub fn chunk_id(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_str()?;
    CHUNK_NAME.captures(stem)?.get(1)?.as_str().parse().ok()
}

/// Pause (ms, or None) to insert after each WAV.
///
/// One segment can be synthesized as several sub-part WAVs. Consecutive
/// WAVs of the same chunk form one group, and group k belongs to
/// `ends_paragraph[k]`. Between sub-parts the sentence pause is used; after
/// the last part the paragraph pause if the segment ends a paragraph. WAVs
/// whose names do not follow the chunk pattern are one group each.
pub fn silence_plan(
    wavs: &[PathBuf],
    ends_paragraph: &[bool],
    period_ms: u32,
    paragraph_ms: u32,
) -> Vec<Option<u32>> {
    #[derive(PartialEq)]
    enum Key {
        Chunk(usize),
        Pos(usize),
    }
    let keys: Vec<Key> = wavs
        .iter()
        .enumerate()
        .map(|(pos, w)| chunk_id(w).map(Key::Chunk).unwrap_or(Key::Pos(pos)))
        .collect();
    let mut plan = Vec::with_capacity(wavs.len());
    let mut group: isize = -1;
    for i in 0..keys.len() {
        if i == 0 || keys[i] != keys[i - 1] {
            group += 1;
        }
        let Some(&ends) = ends_paragraph.get(group as usize) else {
            plan.push(None);
            continue;
        };
        let last_of_group = i == keys.len() - 1 || keys[i + 1] != keys[i];
        plan.push(Some(if last_of_group && ends {
            paragraph_ms
        } else {
            period_ms
        }));
    }
    plan
}

/// Streaming WAV + MP3 writer with pydub-style crossfades.
struct Mixer {
    sample_rate: u32,
    pending: Vec<f32>,
    keep: usize,
    started: bool,
    total_frames: usize,
    wav: hound::WavWriter<BufWriter<File>>,
    mp3: mp3lame_encoder::Encoder,
    mp3_file: BufWriter<File>,
    mp3_buf: Vec<u8>,
}

impl Mixer {
    fn new(wav_path: &Path, mp3_path: &Path, sample_rate: u32, keep: usize) -> Result<Self> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let wav = hound::WavWriter::create(wav_path, spec)
            .with_context(|| format!("creating {}", wav_path.display()))?;
        let mut builder =
            mp3lame_encoder::Builder::new().ok_or_else(|| anyhow!("LAME init failed"))?;
        // MPEG-2 (sample rates below 32 kHz) tops out at 160 kbit/s.
        let bitrate = if sample_rate >= 32000 {
            mp3lame_encoder::Bitrate::Kbps192
        } else {
            mp3lame_encoder::Bitrate::Kbps160
        };
        builder
            .set_num_channels(1)
            .map_err(|e| anyhow!("LAME: {e:?}"))?;
        builder
            .set_sample_rate(sample_rate)
            .map_err(|e| anyhow!("LAME: {e:?}"))?;
        builder
            .set_brate(bitrate)
            .map_err(|e| anyhow!("LAME: {e:?}"))?;
        builder
            .set_quality(mp3lame_encoder::Quality::NearBest)
            .map_err(|e| anyhow!("LAME: {e:?}"))?;
        let mp3 = builder.build().map_err(|e| anyhow!("LAME: {e:?}"))?;
        let mp3_file = BufWriter::new(
            File::create(mp3_path).with_context(|| format!("creating {}", mp3_path.display()))?,
        );
        Ok(Self {
            sample_rate,
            pending: Vec::new(),
            keep,
            started: false,
            total_frames: 0,
            wav,
            mp3,
            mp3_file,
            mp3_buf: Vec::new(),
        })
    }

    fn ms_to_frames(&self, ms: u32) -> usize {
        (ms as u64 * self.sample_rate as u64 / 1000) as usize
    }

    /// Append `audio`, overlapping the last `crossfade` frames: the tail
    /// fades out and the head of `audio` fades in (both linear in amplitude).
    fn append(&mut self, audio: &[f32], crossfade: usize) -> Result<()> {
        if !self.started {
            self.started = true;
            self.pending.extend_from_slice(audio);
            self.total_frames += audio.len();
        } else {
            let cf = crossfade.min(self.pending.len()).min(audio.len());
            let start = self.pending.len() - cf;
            for (i, (tail, head)) in self.pending[start..].iter_mut().zip(audio).enumerate() {
                let t = i as f64 / cf as f64;
                let fade_out = 1.0 + (FADE_FLOOR - 1.0) * t;
                let fade_in = FADE_FLOOR + (1.0 - FADE_FLOOR) * t;
                let mixed = *tail as f64 * fade_out + *head as f64 * fade_in;
                *tail = mixed.clamp(-1.0, 1.0) as f32;
            }
            self.pending.extend_from_slice(&audio[cf..]);
            self.total_frames += audio.len() - cf;
        }
        if self.pending.len() > FLUSH_SECONDS * self.sample_rate as usize + self.keep {
            let n = self.pending.len() - self.keep;
            self.write_out(n)?;
        }
        Ok(())
    }

    fn write_out(&mut self, n: usize) -> Result<()> {
        let pcm: Vec<i16> = self.pending[..n].iter().map(|s| to_i16(*s)).collect();
        {
            let mut w16 = self.wav.get_i16_writer(pcm.len() as u32);
            for s in &pcm {
                w16.write_sample(*s);
            }
            w16.flush()?;
        }
        self.mp3_buf.clear();
        self.mp3_buf
            .reserve(mp3lame_encoder::max_required_buffer_size(pcm.len()));
        self.mp3
            .encode_to_vec(mp3lame_encoder::MonoPcm(&pcm), &mut self.mp3_buf)
            .map_err(|e| anyhow!("MP3 encoding failed: {e:?}"))?;
        self.mp3_file.write_all(&self.mp3_buf)?;
        self.pending.drain(..n);
        Ok(())
    }

    fn finish(mut self) -> Result<usize> {
        let n = self.pending.len();
        self.write_out(n)?;
        self.wav.finalize()?;
        self.mp3_buf.clear();
        self.mp3_buf.reserve(8192);
        self.mp3
            .flush_to_vec::<mp3lame_encoder::FlushNoGap>(&mut self.mp3_buf)
            .map_err(|e| anyhow!("MP3 encoding failed: {e:?}"))?;
        self.mp3_file.write_all(&self.mp3_buf)?;
        self.mp3_file.flush()?;
        Ok(self.total_frames)
    }
}

/// Sample rate of a WAV without decoding it.
fn wav_rate(path: &Path) -> Result<u32> {
    match hound::WavReader::open(path) {
        Ok(r) => Ok(r.spec().sample_rate),
        Err(_) => Ok(read_wav(path)?.sample_rate),
    }
}

/// Join `chunk_wavs` into `<out>.wav` and `<out>.mp3`; returns the MP3 path.
///
/// `ends_paragraph[k]` belongs to the k-th segment (see [`silence_plan`]).
/// With an empty `ends_paragraph` no pauses are inserted.
pub fn assemble(
    chunk_wavs: &[PathBuf],
    ends_paragraph: &[bool],
    out_path: &Path,
    settings: &crate::config::Settings,
) -> Result<PathBuf> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    log::info!("Assembling {} audio chunks", chunk_wavs.len());
    let mp3_out = out_path.with_extension("mp3");
    let wav_out = out_path.with_extension("wav");
    if chunk_wavs.is_empty() {
        log::warn!("No chunks to combine");
        return Ok(out_path.to_path_buf());
    }

    let result = (|| -> Result<PathBuf> {
        let mut rate = 0u32;
        for w in chunk_wavs {
            rate = rate.max(wav_rate(w)?);
        }
        let plan = silence_plan(
            chunk_wavs,
            ends_paragraph,
            settings.silence_after_period_ms,
            settings.silence_after_paragraph_ms,
        );
        let crossfade_ms = settings.crossfade_ms;
        let keep = (crossfade_ms.max(SILENCE_CROSSFADE_MS) as usize * rate as usize) / 1000 + 1;
        let mut mixer = Mixer::new(&wav_out, &mp3_out, rate, keep)?;
        let chunk_xf = mixer.ms_to_frames(crossfade_ms);
        let silence_xf = mixer.ms_to_frames(SILENCE_CROSSFADE_MS);

        for (i, wav_path) in chunk_wavs.iter().enumerate() {
            log::debug!("Loading chunk {i}: {}", wav_path.display());
            let audio = read_wav(wav_path)?;
            let mut samples = audio.to_mono();
            if audio.sample_rate != rate {
                samples = resample(&samples, audio.sample_rate, rate);
            }
            mixer.append(&samples, chunk_xf)?;
            if let Some(ms) = plan[i] {
                let silence = vec![0f32; mixer.ms_to_frames(ms)];
                mixer.append(&silence, silence_xf)?;
            }
        }
        let frames = mixer.finish()?;
        log::info!(
            "Exported WAV: {} ({:.1} s)",
            wav_out.display(),
            frames as f64 / rate as f64
        );
        log::info!("Exported MP3: {}", mp3_out.display());
        Ok(mp3_out.clone())
    })();
    if let Err(e) = &result {
        log::error!("Assembly failed: {e:#}");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::write_mono_wav;
    use crate::config::Settings;

    #[test]
    fn silence_plan_handles_sub_parts() {
        let wavs: Vec<PathBuf> = [
            "chunk_000000.wav",
            "chunk_000001_0.wav",
            "chunk_000001_1.wav",
            "chunk_000001_2.wav",
            "chunk_000002.wav",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        let plan = silence_plan(&wavs, &[true, true, false], 500, 1200);
        assert_eq!(
            plan,
            [Some(1200), Some(500), Some(500), Some(1200), Some(500)]
        );
    }

    #[test]
    fn silence_plan_without_meta_adds_nothing() {
        let wavs = vec![PathBuf::from("a.wav"), PathBuf::from("b.wav")];
        assert_eq!(silence_plan(&wavs, &[], 500, 1200), [None, None]);
        // Non-chunk names are one group each.
        assert_eq!(
            silence_plan(&wavs, &[false, true], 500, 1200),
            [Some(500), Some(1200)]
        );
    }

    #[test]
    fn chunk_ids() {
        assert_eq!(chunk_id(Path::new("x/chunk_000012.wav")), Some(12));
        assert_eq!(chunk_id(Path::new("chunk_000012_3.wav")), Some(12));
        assert_eq!(chunk_id(Path::new("anchor.wav")), None);
    }

    #[test]
    fn assemble_writes_wav_and_mp3_with_pydub_timing() {
        let dir = tempfile::tempdir().unwrap();
        let sr = 24000;
        let tone: Vec<f32> = (0..sr).map(|i| 0.5 * (i as f32 * 0.07).sin()).collect();
        let wavs: Vec<PathBuf> = (0..2)
            .map(|i| {
                let p = dir.path().join(format!("chunk_{i:06}.wav"));
                write_mono_wav(&p, &tone, sr as u32).unwrap();
                p
            })
            .collect();
        let out = dir.path().join("final/audiobook.mp3");
        let settings = Settings::default();
        let mp3 = assemble(&wavs, &[false, true], &out, &settings).unwrap();
        assert_eq!(mp3, out);
        assert!(std::fs::metadata(&mp3).unwrap().len() > 1000);
        let wav = crate::audio::read_wav(&out.with_extension("wav")).unwrap();
        // 1000 + (500 - 100) + (1000 - 80) + (1200 - 100) ms
        let expected_ms = 1000 + 400 + 920 + 1100;
        assert_eq!(wav.samples.len(), expected_ms * sr / 1000);
        assert!(
            wav.samples[wav.samples.len() - 100..]
                .iter()
                .all(|s| *s == 0.0)
        );
    }

    #[test]
    fn assemble_resamples_mismatched_rates() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("chunk_000000.wav");
        let b = dir.path().join("chunk_000001.wav");
        write_mono_wav(&a, &vec![0.1; 16000], 16000).unwrap();
        write_mono_wav(&b, &vec![0.1; 24000], 24000).unwrap();
        let out = dir.path().join("out.mp3");
        let settings = Settings {
            crossfade_ms: 0,
            ..Settings::default()
        };
        assemble(&[a, b], &[], &out, &settings).unwrap();
        let wav = crate::audio::read_wav(&out.with_extension("wav")).unwrap();
        assert_eq!(wav.sample_rate, 24000);
        assert_eq!(wav.samples.len(), 48000);
    }

    #[test]
    fn assemble_without_chunks_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.mp3");
        assert_eq!(assemble(&[], &[], &out, &Settings::default()).unwrap(), out);
        assert!(!out.exists());
    }
}
