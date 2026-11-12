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
//!
//! With `LOUDNESS_TARGET_LUFS` below 0 the chunks are measured first and the
//! whole file gets one gain to that loudness; a limiter catches the peaks
//! (see [`super::loudness`]).
//!
//! [`assemble_preview`] builds the provisional MP3 written while stage 2 is
//! still running: the same mix from chunks that are not denoised yet.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow};
use regex::Regex;

use super::filter::denoise_audio;
use super::loudness::{
    Limiter, LoudnessMeter, PEAK_CEILING_DBFS, db_to_gain, normalization_gain_db,
};
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

/// Receives the mixed audio block by block.
type Sink<'a> = dyn FnMut(&[f32]) -> Result<()> + 'a;

/// pydub-style crossfading. Finished samples go to the sink; the last
/// `keep` frames stay back for the next crossfade.
struct Crossfader {
    pending: Vec<f32>,
    keep: usize,
    flush_at: usize,
    started: bool,
    total_frames: usize,
}

impl Crossfader {
    fn new(sample_rate: u32, keep: usize) -> Self {
        Self {
            pending: Vec::new(),
            keep,
            flush_at: FLUSH_SECONDS * sample_rate as usize + keep,
            started: false,
            total_frames: 0,
        }
    }

    /// Append `audio`, overlapping the last `crossfade` frames: the tail
    /// fades out and the head of `audio` fades in (both linear in amplitude).
    fn append(&mut self, audio: &[f32], crossfade: usize, sink: &mut Sink) -> Result<()> {
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
        if self.pending.len() > self.flush_at {
            let n = self.pending.len() - self.keep;
            sink(&self.pending[..n])?;
            self.pending.drain(..n);
        }
        Ok(())
    }

    /// Hand over the rest; returns the number of frames mixed.
    fn finish(self, sink: &mut Sink) -> Result<usize> {
        sink(&self.pending)?;
        Ok(self.total_frames)
    }
}

/// Returned when the abort check of [`assemble_preview`] said stop.
#[derive(Debug, thiserror::Error)]
#[error("assembly aborted")]
pub struct Aborted;

/// How [`mix`] reads the chunks.
#[derive(Clone, Copy, Default)]
struct Source<'a> {
    /// Filter every chunk like [`super::filter::denoise`] while reading
    /// (chunks that are not denoised on disk yet).
    denoise: bool,
    /// Asked before every chunk; true gives up with [`Aborted`].
    abort: Option<&'a dyn Fn() -> bool>,
}

/// Mix the chunks with crossfades and the pauses of `plan` at `rate`.
fn mix(
    chunk_wavs: &[PathBuf],
    plan: &[Option<u32>],
    rate: u32,
    crossfade_ms: u32,
    source: Source,
    sink: &mut Sink,
) -> Result<usize> {
    let frames = |ms: u32| (ms as u64 * rate as u64 / 1000) as usize;
    let keep = frames(crossfade_ms.max(SILENCE_CROSSFADE_MS)) + 1;
    let chunk_xf = frames(crossfade_ms);
    let silence_xf = frames(SILENCE_CROSSFADE_MS);
    let mut mixer = Crossfader::new(rate, keep);
    for (i, wav_path) in chunk_wavs.iter().enumerate() {
        if source.abort.is_some_and(|abort| abort()) {
            return Err(Aborted.into());
        }
        log::trace!("Loading chunk {i}: {}", wav_path.display());
        let mut audio = read_wav(wav_path)?;
        if source.denoise {
            denoise_audio(&mut audio)?;
        }
        let mut samples = audio.to_mono();
        if audio.sample_rate != rate {
            samples = resample(&samples, audio.sample_rate, rate);
        }
        mixer.append(&samples, chunk_xf, sink)?;
        if let Some(ms) = plan[i] {
            mixer.append(&vec![0f32; frames(ms)], silence_xf, sink)?;
        }
    }
    mixer.finish(sink)
}

/// Streaming MP3 (+ WAV) writer, with optional gain and limiter.
struct Writer {
    /// The WAV mirror (None: MP3 only).
    wav: Option<hound::WavWriter<BufWriter<File>>>,
    mp3: mp3lame_encoder::Encoder,
    mp3_file: BufWriter<File>,
    mp3_buf: Vec<u8>,
    /// Loudness gain and the limiter behind it (None: written as mixed).
    level: Option<(f32, Limiter)>,
    log_level: log::Level,
}

impl Writer {
    fn new(
        wav_path: Option<&Path>,
        mp3_path: &Path,
        sample_rate: u32,
        gain_db: Option<f64>,
        log_level: log::Level,
    ) -> Result<Self> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let wav = match wav_path {
            Some(p) => Some(
                hound::WavWriter::create(p, spec)
                    .with_context(|| format!("creating {}", p.display()))?,
            ),
            None => None,
        };
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
            wav,
            mp3,
            mp3_file,
            mp3_buf: Vec::new(),
            level: gain_db.map(|db| {
                (
                    db_to_gain(db) as f32,
                    Limiter::new(PEAK_CEILING_DBFS, sample_rate),
                )
            }),
            log_level,
        })
    }

    fn write(&mut self, samples: &[f32]) -> Result<()> {
        match &mut self.level {
            Some((gain, limiter)) => {
                let louder: Vec<f32> = samples.iter().map(|s| s * *gain).collect();
                let mut limited = Vec::with_capacity(louder.len());
                limiter.process(&louder, &mut limited);
                self.encode(&limited)
            }
            None => self.encode(samples),
        }
    }

    fn encode(&mut self, samples: &[f32]) -> Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        let pcm: Vec<i16> = samples.iter().map(|s| to_i16(*s)).collect();
        if let Some(wav) = &mut self.wav {
            let mut w16 = wav.get_i16_writer(pcm.len() as u32);
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
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        if let Some((_, mut limiter)) = self.level.take() {
            let mut tail = Vec::new();
            limiter.flush(&mut tail);
            self.encode(&tail)?;
            let reduction = limiter.max_reduction_db();
            if reduction < 0.0 {
                log::log!(
                    self.log_level,
                    "Limiter: peaks lowered by up to {:.1} dB",
                    -reduction
                );
            }
        }
        if let Some(wav) = self.wav.take() {
            wav.finalize()?;
        }
        self.mp3_buf.clear();
        self.mp3_buf.reserve(8192);
        self.mp3
            .flush_to_vec::<mp3lame_encoder::FlushNoGap>(&mut self.mp3_buf)
            .map_err(|e| anyhow!("MP3 encoding failed: {e:?}"))?;
        self.mp3_file.write_all(&self.mp3_buf)?;
        self.mp3_file.flush()?;
        Ok(())
    }
}

/// Sample rate of a WAV without decoding it.
fn wav_rate(path: &Path) -> Result<u32> {
    match hound::WavReader::open(path) {
        Ok(r) => Ok(r.spec().sample_rate),
        Err(_) => Ok(read_wav(path)?.sample_rate),
    }
}

/// Gain in dB for `LOUDNESS_TARGET_LUFS`, measured on a first mixing pass
/// (None: off, or no speech to measure).
fn loudness_gain_db(
    chunk_wavs: &[PathBuf],
    plan: &[Option<u32>],
    rate: u32,
    settings: &crate::config::Settings,
    source: Source,
    log_level: log::Level,
) -> Result<Option<f64>> {
    let target = settings.loudness_target_lufs;
    if target >= 0 {
        return Ok(None);
    }
    let mut meter = LoudnessMeter::new(rate);
    mix(
        chunk_wavs,
        plan,
        rate,
        settings.crossfade_ms,
        source,
        &mut |s| {
            meter.process(s);
            Ok(())
        },
    )?;
    let Some(measured) = meter.integrated_lufs() else {
        log::log!(log_level, "Loudness: no speech found, level left unchanged");
        return Ok(None);
    };
    let gain = normalization_gain_db(measured, target as f64);
    log::log!(
        log_level,
        "Loudness: {measured:.1} LUFS measured, target {target} LUFS, gain {gain:+.1} dB"
    );
    Ok(Some(gain))
}

/// Mix `chunk_wavs` into `mp3_out` (and `wav_out`); returns the length in
/// seconds.
fn render(
    chunk_wavs: &[PathBuf],
    ends_paragraph: &[bool],
    wav_out: Option<&Path>,
    mp3_out: &Path,
    settings: &crate::config::Settings,
    source: Source,
    log_level: log::Level,
) -> Result<f64> {
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
    let gain_db = loudness_gain_db(chunk_wavs, &plan, rate, settings, source, log_level)?;
    let mut writer = Writer::new(wav_out, mp3_out, rate, gain_db, log_level)?;
    let frames = mix(
        chunk_wavs,
        &plan,
        rate,
        settings.crossfade_ms,
        source,
        &mut |s| writer.write(s),
    )?;
    writer.finish()?;
    Ok(frames as f64 / rate as f64)
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
        let secs = render(
            chunk_wavs,
            ends_paragraph,
            Some(&wav_out),
            &mp3_out,
            settings,
            Source::default(),
            log::Level::Info,
        )?;
        log::info!("Exported WAV: {} ({secs:.1} s)", wav_out.display());
        log::info!("Exported MP3: {}", mp3_out.display());
        Ok(mp3_out.clone())
    })();
    if let Err(e) = &result {
        log::error!("Assembly failed: {e:#}");
    }
    result
}

/// Name under which [`assemble_preview`] writes before it replaces `mp3`.
pub fn part_path(mp3: &Path) -> PathBuf {
    let mut name = mp3.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    mp3.with_file_name(name)
}

/// Provisional MP3 of the chunks done so far, without WAV mirror: the same
/// crossfades, pauses and loudness as [`assemble`], and the chunks are
/// denoised in memory as stage 2 later does on disk, so it sounds like the
/// finished book. It is written to [`part_path`] and then renamed to
/// `mp3_path`, so a player never sees a half-written file.
///
/// `abort` is asked before every chunk; true stops and leaves `mp3_path`
/// as it was. Returns the length in seconds, or None when aborted or
/// without chunks.
pub fn assemble_preview(
    chunk_wavs: &[PathBuf],
    ends_paragraph: &[bool],
    mp3_path: &Path,
    settings: &crate::config::Settings,
    abort: &dyn Fn() -> bool,
) -> Result<Option<f64>> {
    if chunk_wavs.is_empty() {
        return Ok(None);
    }
    if let Some(parent) = mp3_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part = part_path(mp3_path);
    let source = Source {
        denoise: true,
        abort: Some(abort),
    };
    let result = render(
        chunk_wavs,
        ends_paragraph,
        None,
        &part,
        settings,
        source,
        log::Level::Debug,
    )
    .and_then(|secs| {
        std::fs::rename(&part, mp3_path)
            .with_context(|| format!("replacing {}", mp3_path.display()))?;
        Ok(secs)
    });
    match result {
        Ok(secs) => Ok(Some(secs)),
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            if e.is::<Aborted>() { Ok(None) } else { Err(e) }
        }
    }
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

    fn tone_chunks(dir: &Path, amp: f32, n: usize) -> Vec<PathBuf> {
        let sr = 24000;
        let tone: Vec<f32> = (0..sr * 3)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 997.0 * i as f32 / sr as f32).sin())
            .collect();
        (0..n)
            .map(|i| {
                let p = dir.join(format!("chunk_{i:06}.wav"));
                write_mono_wav(&p, &tone, sr as u32).unwrap();
                p
            })
            .collect()
    }

    fn lufs(path: &Path) -> f64 {
        let wav = crate::audio::read_wav(path).unwrap();
        let mut m = crate::audio::loudness::LoudnessMeter::new(wav.sample_rate);
        m.process(&wav.samples);
        m.integrated_lufs().unwrap()
    }

    #[test]
    fn assemble_brings_quiet_speech_to_the_target_loudness() {
        let dir = tempfile::tempdir().unwrap();
        // About -29 LUFS.
        let wavs = tone_chunks(dir.path(), 0.05, 3);
        let out = dir.path().join("out.mp3");
        let settings = Settings::default();
        assemble(&wavs, &[false, true, true], &out, &settings).unwrap();
        let l = lufs(&out.with_extension("wav"));
        assert!(
            (l - settings.loudness_target_lufs as f64).abs() < 0.05,
            "{l} LUFS"
        );

        // Too loud for the target: peaks stay below the ceiling.
        let wavs = tone_chunks(dir.path(), 0.99, 2);
        let settings = Settings {
            loudness_target_lufs: -2,
            ..Settings::default()
        };
        assemble(&wavs, &[], &out, &settings).unwrap();
        let wav = crate::audio::read_wav(&out.with_extension("wav")).unwrap();
        let peak = wav.samples.iter().fold(0f32, |m, s| m.max(s.abs()));
        let ceiling = crate::audio::loudness::db_to_gain(PEAK_CEILING_DBFS) as f32;
        assert!(peak <= ceiling + 1e-4, "peak {peak}");
    }

    #[test]
    fn assemble_keeps_the_level_with_loudness_off() {
        let dir = tempfile::tempdir().unwrap();
        let wavs = tone_chunks(dir.path(), 0.05, 1);
        let out = dir.path().join("out.mp3");
        let settings = Settings {
            loudness_target_lufs: 0,
            ..Settings::default()
        };
        assemble(&wavs, &[], &out, &settings).unwrap();
        let input = crate::audio::read_wav(&wavs[0]).unwrap();
        let output = crate::audio::read_wav(&out.with_extension("wav")).unwrap();
        assert_eq!(output.samples, input.samples);
    }

    #[test]
    fn preview_is_mp3_only_and_an_abort_keeps_the_old_file() {
        let dir = tempfile::tempdir().unwrap();
        let wavs = tone_chunks(dir.path(), 0.05, 2);
        let out = dir.path().join("final/preview.mp3");
        let settings = Settings::default();
        let secs = assemble_preview(&wavs, &[true, true], &out, &settings, &|| false)
            .unwrap()
            .unwrap();
        assert!((secs - (3.0 + 1.1 + 2.92 + 1.1)).abs() < 1e-3, "{secs}");
        assert!(out.exists());
        assert!(!out.with_extension("wav").exists());
        assert!(!part_path(&out).exists());
        let before = std::fs::read(&out).unwrap();
        // Aborted: nothing replaced, no leftovers.
        let calls = std::cell::Cell::new(0);
        let abort = || {
            calls.set(calls.get() + 1);
            calls.get() > 1
        };
        let one = tone_chunks(dir.path(), 0.5, 1);
        assert!(
            assemble_preview(&one, &[true], &out, &settings, &abort)
                .unwrap()
                .is_none()
        );
        assert_eq!(std::fs::read(&out).unwrap(), before);
        assert!(!part_path(&out).exists());
    }

    #[test]
    fn assemble_without_chunks_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out.mp3");
        assert_eq!(assemble(&[], &[], &out, &Settings::default()).unwrap(), out);
        assert!(!out.exists());
    }
}
