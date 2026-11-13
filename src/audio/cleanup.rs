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

//! Stage 3: audit chunk WAVs for TTS failures.
//!
//! Per chunk:
//! 1. Speech ratio: share of 30 ms frames that are voiced speech.
//! 2. Volume: mean RMS of voiced frames, share of clipped samples.
//! 3. Voice match: cosine similarity of the MFCC fingerprint with the
//!    voice reference.
//!
//! Failing chunks are deleted (with all sub-parts) so stage 2 regenerates
//! them; passing chunks have long noisy stretches cut out in place.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use super::{mfcc, read_wav, tags, write_mono_wav};
use crate::natsort::natural_cmp;

const FRAME_MS: usize = 30;
/// Spectral flatness above this counts as noise (speech ~0.1-0.5, white noise ~1.0).
const FLATNESS_THRESH: f64 = 0.75;
/// Frames quieter than this (int16 scale) are silence, not noise.
const MIN_RMS: f64 = 150.0;
/// Samples above this (int16 scale) count as clipped.
const CLIP_THRESH: f64 = 32000.0;

#[derive(Debug, Clone)]
pub struct Thresholds {
    pub max_noise_ms: usize,
    pub min_speech_ratio: f64,
    pub min_avg_rms: f64,
    pub max_clip_ratio: f64,
    pub min_voice_similarity: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            max_noise_ms: 1500,
            min_speech_ratio: 0.40,
            min_avg_rms: 300.0,
            max_clip_ratio: 0.02,
            min_voice_similarity: 0.55,
        }
    }
}

/// Cleanup result of one project.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProjectCleanup {
    pub edited: usize,
    /// (file name, reason) of every deleted WAV.
    pub deleted: Vec<(String, String)>,
}

/// Frame analysis of one chunk.
#[derive(Debug, Clone)]
pub struct Analysis {
    pub n_frames: usize,
    pub frame_size: usize,
    pub frame_is_noise: Vec<bool>,
    pub speech_ratio: f64,
    pub avg_rms: f64,
    pub clip_ratio: f64,
}

/// Analyse int16-scaled mono samples in 30 ms frames.
pub fn analyze(samples: &[f64], sample_rate: u32) -> Analysis {
    let frame_size = (sample_rate as usize * FRAME_MS / 1000).max(1);
    let n_frames = samples.len() / frame_size;
    // numpy.hanning (symmetric)
    let window: Vec<f64> = if frame_size == 1 {
        vec![1.0]
    } else {
        (0..frame_size)
            .map(|n| {
                0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (frame_size - 1) as f64).cos()
            })
            .collect()
    };
    let fft = FftPlanner::<f64>::new().plan_fft_forward(frame_size);
    let n_bins = frame_size / 2 + 1;
    let mut buf = vec![Complex::new(0.0, 0.0); frame_size];
    let mut frame_is_noise = Vec::with_capacity(n_frames);
    let mut voiced_rms = Vec::new();
    for fi in 0..n_frames {
        let frame = &samples[fi * frame_size..(fi + 1) * frame_size];
        let rms = (frame.iter().map(|x| x * x).sum::<f64>() / frame_size as f64).sqrt();
        if rms < MIN_RMS {
            frame_is_noise.push(false); // silence — leave it alone
            continue;
        }
        for (n, x) in frame.iter().enumerate() {
            buf[n] = Complex::new(x * window[n], 0.0);
        }
        fft.process(&mut buf);
        let mags: Vec<f64> = buf[..n_bins].iter().map(|c| c.norm() + 1e-10).collect();
        let mean_log = mags.iter().map(|m| m.ln()).sum::<f64>() / n_bins as f64;
        let mean = mags.iter().sum::<f64>() / n_bins as f64;
        let flatness = mean_log.exp() / mean;
        let is_noise = flatness > FLATNESS_THRESH;
        frame_is_noise.push(is_noise);
        if !is_noise {
            voiced_rms.push(rms);
        }
    }
    let speech_ratio = if n_frames > 0 {
        voiced_rms.len() as f64 / n_frames as f64
    } else {
        0.0
    };
    let avg_rms = if voiced_rms.is_empty() {
        0.0
    } else {
        voiced_rms.iter().sum::<f64>() / voiced_rms.len() as f64
    };
    let clip_ratio = if samples.is_empty() {
        0.0
    } else {
        samples.iter().filter(|s| s.abs() > CLIP_THRESH).count() as f64 / samples.len() as f64
    };
    Analysis {
        n_frames,
        frame_size,
        frame_is_noise,
        speech_ratio,
        avg_rms,
        clip_ratio,
    }
}

/// Python-style percentage (`f"{x:.0%}"`).
fn pct(x: f64, decimals: usize) -> String {
    format!("{:.*}%", decimals, x * 100.0)
}

/// Reason why a chunk fails the audit, if it does.
pub fn fail_reason(
    a: &Analysis,
    th: &Thresholds,
    voice_sim: impl FnOnce() -> Option<f64>,
) -> Option<String> {
    if a.n_frames < 5 {
        return Some(format!("too short ({} frames)", a.n_frames));
    }
    if a.speech_ratio < th.min_speech_ratio {
        return Some(format!(
            "low speech ratio {} (<{})",
            pct(a.speech_ratio, 0),
            pct(th.min_speech_ratio, 0)
        ));
    }
    if a.avg_rms < th.min_avg_rms {
        return Some(format!(
            "too quiet (avg RMS {:.0} < {:.0})",
            a.avg_rms, th.min_avg_rms
        ));
    }
    if a.clip_ratio > th.max_clip_ratio {
        return Some(format!(
            "clipping {} (>{})",
            pct(a.clip_ratio, 1),
            pct(th.max_clip_ratio, 1)
        ));
    }
    if let Some(sim) = voice_sim()
        && sim < th.min_voice_similarity
    {
        return Some(format!(
            "voice mismatch (sim {sim:.2} < {:.2})",
            th.min_voice_similarity
        ));
    }
    None
}

/// Sample ranges of noise runs of at least `min_frames` frames.
pub fn noisy_regions(a: &Analysis, min_frames: usize) -> Vec<(usize, usize)> {
    let mut regions = Vec::new();
    let f = &a.frame_is_noise;
    let mut i = 0;
    while i < f.len() {
        if f[i] {
            let mut j = i + 1;
            while j < f.len() && f[j] {
                j += 1;
            }
            if j - i >= min_frames {
                regions.push((i * a.frame_size, j * a.frame_size));
            }
            i = j;
        } else {
            i += 1;
        }
    }
    regions
}

/// Mono samples in int16 scale (stereo is averaged and truncated like numpy's
/// `mean(axis=1).astype(int16)`).
fn load_int16_scale(path: &Path) -> anyhow::Result<(Vec<f64>, u32)> {
    let audio = read_wav(path)?;
    let ch = audio.channels.max(1) as usize;
    let samples: Vec<f64> = if ch == 1 {
        audio
            .samples
            .iter()
            .map(|s| (*s as f64 * 32768.0).round())
            .collect()
    } else {
        audio
            .samples
            .chunks_exact(ch)
            .map(|fr| (fr.iter().map(|s| *s as f64 * 32768.0).sum::<f64>() / ch as f64).trunc())
            .collect()
    };
    Ok((samples, audio.sample_rate))
}

static CHUNK_BASE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(chunk_\d+)").unwrap());

/// The WAV and all of its sibling parts (`chunk_N.wav`, `chunk_N_K.wav`).
fn siblings(chunks_dir: &Path, wav: &Path) -> Vec<PathBuf> {
    let stem = wav.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    let base = CHUNK_BASE
        .captures(stem)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| stem.to_string());
    let part = Regex::new(&format!(r"^{}(_\d+)?\.wav$", regex::escape(&base))).unwrap();
    let mut out: Vec<PathBuf> = std::fs::read_dir(chunks_dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| part.is_match(n))
                })
                .collect()
        })
        .unwrap_or_default();
    out.sort();
    out
}

/// `chunk_*.wav` in `dir`, in natural name order (`chunk_N_2` before
/// `chunk_N_10`), which is reading order.
pub fn chunk_wavs(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|it| {
            it.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.is_file()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("chunk_") && n.ends_with(".wav"))
                })
                .collect()
        })
        .unwrap_or_default();
    let name = |p: &PathBuf| {
        p.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    };
    v.sort_by(|a, b| natural_cmp(&name(a), &name(b)));
    v
}

/// Audit (and clean or delete) the chunk WAVs of the given projects below
/// `output_dir` (all project folders when `project_names` is None).
pub fn cleanup_noisy_chunks(
    output_dir: &Path,
    project_names: Option<&[String]>,
    progress: &dyn Fn(&str),
    voice_ref_path: &Path,
    th: &Thresholds,
) -> IndexMap<String, ProjectCleanup> {
    cleanup_noisy_chunks_with(
        output_dir,
        project_names,
        progress,
        voice_ref_path,
        th,
        &|_, _| {},
    )
}

/// [`cleanup_noisy_chunks`], calling `audited(done, total)` after every
/// WAV file of a project.
pub fn cleanup_noisy_chunks_with(
    output_dir: &Path,
    project_names: Option<&[String]>,
    progress: &dyn Fn(&str),
    voice_ref_path: &Path,
    th: &Thresholds,
    audited: &dyn Fn(usize, usize),
) -> IndexMap<String, ProjectCleanup> {
    let mut results = IndexMap::new();

    let mut ref_fp: Option<Vec<f64>> = None;
    if voice_ref_path.exists() {
        match read_wav(voice_ref_path) {
            Ok(a) => {
                ref_fp = mfcc::fingerprint(&a.to_mono(), a.sample_rate);
                if ref_fp.is_none() {
                    progress(
                        "Voice reference fingerprint unavailable (ref too short); skipping voice-match check",
                    );
                }
            }
            Err(e) => log::warn!(
                "Failed to load voice reference {}: {e:#}",
                voice_ref_path.display()
            ),
        }
    } else {
        progress(&format!(
            "No voice reference at {}; skipping voice-match check",
            voice_ref_path.display()
        ));
    }

    let dirs: Vec<PathBuf> = match project_names {
        Some(names) => names
            .iter()
            .map(|n| output_dir.join(n))
            .filter(|d| d.is_dir())
            .collect(),
        None => {
            let mut v: Vec<PathBuf> = std::fs::read_dir(output_dir)
                .map(|it| {
                    it.filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.is_dir())
                        .collect()
                })
                .unwrap_or_default();
            v.sort();
            v
        }
    };
    if dirs.is_empty() {
        progress("No project output directories found for cleanup");
        return results;
    }

    let min_noisy_frames = (th.max_noise_ms / FRAME_MS).max(1);
    for project_dir in dirs {
        let project = project_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();
        let chunks_dir = project_dir.join("chunks");
        if !chunks_dir.exists() {
            progress(&format!("[{project}] no chunks directory, skipping"));
            continue;
        }
        let wavs = chunk_wavs(&chunks_dir);
        if wavs.is_empty() {
            progress(&format!("[{project}] no chunk WAV files found"));
            continue;
        }
        progress(&format!(
            "[{project}] auditing {} chunks (discriminator + volume + voice match)...",
            wavs.len()
        ));
        let mut r = ProjectCleanup::default();
        for (n, wav) in wavs.iter().enumerate() {
            if n > 0 {
                audited(n, wavs.len());
            }
            if !wav.exists() {
                continue; // removed together with a failing sibling
            }
            let name = wav
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let outcome = (|| -> anyhow::Result<()> {
                let (samples, sr) = load_int16_scale(wav)?;
                let a = analyze(&samples, sr);
                let reason = fail_reason(&a, th, || {
                    let rf = ref_fp.as_ref()?;
                    let s32: Vec<f32> = samples.iter().map(|s| *s as f32).collect();
                    let fp = mfcc::fingerprint(&s32, sr)?;
                    Some(mfcc::cosine(rf, &fp))
                });
                if let Some(reason) = reason {
                    // A failed sub-part means the whole chunk is regenerated.
                    let mut removed = Vec::new();
                    for sib in siblings(&chunks_dir, wav) {
                        match std::fs::remove_file(&sib) {
                            Ok(()) => removed.push(
                                sib.file_name()
                                    .map(|n| n.to_string_lossy().to_string())
                                    .unwrap_or_default(),
                            ),
                            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                            Err(e) => log::error!("Failed to delete {}: {e}", sib.display()),
                        }
                    }
                    if removed.is_empty() {
                        return Ok(());
                    }
                    let note = if removed.len() > 1 {
                        format!(" (+{} sibling part(s))", removed.len() - 1)
                    } else {
                        String::new()
                    };
                    for n in removed {
                        r.deleted.push((n, reason.clone()));
                    }
                    progress(&format!("[{project}] {name}: DELETED — {reason}{note}"));
                    return Ok(());
                }

                // Passed: cut out long noisy stretches in place.
                let regions = noisy_regions(&a, min_noisy_frames);
                if regions.is_empty() {
                    return Ok(());
                }
                let mut kept: Vec<f32> = Vec::with_capacity(samples.len());
                let mut prev = 0;
                for (start, end) in &regions {
                    if prev < *start {
                        kept.extend(samples[prev..*start].iter().map(|s| (*s / 32768.0) as f32));
                    }
                    prev = *end;
                }
                if prev < samples.len() {
                    kept.extend(samples[prev..].iter().map(|s| (*s / 32768.0) as f32));
                }
                let tag = tags::read_tag(wav);
                write_mono_wav(wav, &kept, sr)?;
                if let Some(tag) = tag {
                    tags::write_tag(wav, &tag);
                }
                progress(&format!(
                    "[{project}] {name}: removed {} noise region(s), {:.1}s -> {:.1}s",
                    regions.len(),
                    samples.len() as f64 / sr as f64,
                    kept.len() as f64 / sr as f64
                ));
                r.edited += 1;
                Ok(())
            })();
            if let Err(e) = outcome {
                log::error!("Failed to process {name}: {e:#}");
            }
        }
        audited(wavs.len(), wavs.len());
        progress(&format!(
            "[{project}] cleanup done: {} cleaned, {} deleted",
            r.edited,
            r.deleted.len()
        ));
        results.insert(project, r);
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speechlike(sr: u32, secs: f32) -> Vec<f32> {
        // Harmonic tone with a slow envelope: low spectral flatness, loud.
        let n = (sr as f32 * secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let env = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * 3.0 * t).sin();
                env * (0..5)
                    .map(|h| 0.08 * (2.0 * std::f32::consts::PI * 150.0 * (h + 1) as f32 * t).sin())
                    .sum::<f32>()
            })
            .collect()
    }

    fn noise(n: usize, amp: f32) -> Vec<f32> {
        let mut x: u32 = 12345;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    #[test]
    fn chunk_wavs_keep_reading_order_past_ten_parts() {
        let dir = tempfile::tempdir().unwrap();
        let names = [
            "chunk_000000.wav",
            "chunk_000001_0.wav",
            "chunk_000001_1.wav",
            "chunk_000001_2.wav",
            "chunk_000001_9.wav",
            "chunk_000001_10.wav",
            "chunk_000001_11.wav",
            "chunk_000002.wav",
        ];
        for n in names.iter().rev() {
            std::fs::write(dir.path().join(n), b"x").unwrap();
        }
        let got: Vec<String> = chunk_wavs(dir.path())
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(got, names);
    }

    #[test]
    fn audit_deletes_silent_chunk_with_siblings_and_cleans_noise() {
        let dir = tempfile::tempdir().unwrap();
        let chunks = dir.path().join("out/proj/chunks");
        std::fs::create_dir_all(&chunks).unwrap();
        let sr = 24000;
        // Chunk 0: speech. Chunk 1 (two parts): part 0 silent -> both deleted.
        write_mono_wav(&chunks.join("chunk_000000.wav"), &speechlike(sr, 3.0), sr).unwrap();
        write_mono_wav(&chunks.join("chunk_000001_0.wav"), &vec![0.0; 48000], sr).unwrap();
        write_mono_wav(&chunks.join("chunk_000001_1.wav"), &speechlike(sr, 1.0), sr).unwrap();
        // Chunk 2: speech with 2 s of loud white noise in the middle.
        let mut s = speechlike(sr, 3.0);
        s.extend(noise(48000, 0.5));
        s.extend(speechlike(sr, 3.0));
        write_mono_wav(&chunks.join("chunk_000002.wav"), &s, sr).unwrap();

        let msgs = std::sync::Mutex::new(Vec::new());
        let progress = |m: &str| msgs.lock().unwrap().push(m.to_string());
        let audited = std::sync::Mutex::new(Vec::new());
        let r = cleanup_noisy_chunks_with(
            &dir.path().join("out"),
            Some(&["proj".to_string()]),
            &progress,
            &dir.path().join("missing_ref.wav"),
            &Thresholds::default(),
            &|done, total| audited.lock().unwrap().push((done, total)),
        );
        // Every file counts, also the sibling deleted before its turn.
        assert_eq!(*audited.lock().unwrap(), [(1, 4), (2, 4), (3, 4), (4, 4)]);
        let p = &r["proj"];
        assert_eq!(p.deleted.len(), 2, "{:?}", msgs.lock().unwrap());
        assert!(p.deleted[0].1.starts_with("low speech ratio 0%"));
        assert!(!chunks.join("chunk_000001_1.wav").exists());
        assert!(chunks.join("chunk_000000.wav").exists());
        assert_eq!(p.edited, 1);
        let cleaned = read_wav(&chunks.join("chunk_000002.wav")).unwrap();
        assert!(cleaned.duration_secs() < 7.0);
        assert!(cleaned.duration_secs() > 5.5);
    }

    #[test]
    fn fail_reason_formatting() {
        let a = Analysis {
            n_frames: 10,
            frame_size: 720,
            frame_is_noise: vec![],
            speech_ratio: 0.9,
            avg_rms: 1000.0,
            clip_ratio: 0.0321,
        };
        assert_eq!(
            fail_reason(&a, &Thresholds::default(), || None).unwrap(),
            "clipping 3.2% (>2.0%)"
        );
        let a = Analysis {
            clip_ratio: 0.0,
            ..a
        };
        assert_eq!(
            fail_reason(&a, &Thresholds::default(), || Some(0.3)).unwrap(),
            "voice mismatch (sim 0.30 < 0.55)"
        );
        assert!(fail_reason(&a, &Thresholds::default(), || Some(0.9)).is_none());
    }
}
