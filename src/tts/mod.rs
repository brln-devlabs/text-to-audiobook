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

//! Stage 2: speech synthesis with Qwen3-TTS voice cloning.
//!
//! Voice anchoring: chunk 0 is spoken with the voice reference
//! (`cache/voice_reference.wav`, generated on first use); every later chunk
//! clones the voice of chunk 0, which keeps the narrator consistent.
//! Segments over the token limit are split into sub-parts
//! `chunk_NNN_0.wav`, `chunk_NNN_1.wav`, ...

#[cfg(feature = "tts")]
pub mod qwen;

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Result, bail};
use regex::Regex;

use crate::audio::{tags, write_mono_wav};
use crate::cancel::{ForceStopped, StopFlag, forced, stopped};
use crate::chunker::word_count;
use crate::config::{Paths, Settings};
use crate::segment::Segment;

pub const MAX_SUB_PARTS: usize = 10;
pub const DEFAULT_LANGUAGE: &str = "english";

/// Sampling for chunks (Python: temperature=0.6, top_p=0.9).
pub const CLONE_TEMPERATURE: f64 = 0.6;
pub const CLONE_TOP_P: f64 = 0.9;
/// Sampling for the generated voice reference.
pub const REFERENCE_TEMPERATURE: f64 = 0.3;
pub const REFERENCE_TOP_P: f64 = 0.9;

/// Short fixed texts for generating the voice reference WAV.
pub fn voice_ref_text(language: &str) -> &'static str {
    match language {
        "german" => {
            "Dies ist eine kurze Sprachprobe. Sie legt fest, wie die Stimme \
             des Erzählers im ganzen Hörbuch klingen wird."
        }
        _ => {
            "This is a short voice sample. It defines how the narrator \
             will sound throughout the entire audiobook."
        }
    }
}

/// Language for Qwen3-TTS: `TTS_LANGUAGE`, or the segment's own detected
/// language when `TTS_LANGUAGE` is "auto".
pub fn resolve_language(settings: &Settings, segment_language: Option<&str>) -> String {
    let fixed = settings.tts_language_normalized();
    if fixed != "auto" {
        return fixed;
    }
    segment_language
        .filter(|l| !l.is_empty())
        .unwrap_or(DEFAULT_LANGUAGE)
        .to_string()
}

/// Mono audio as produced by a TTS backend.
#[derive(Debug, Clone)]
pub struct Wave {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Qwen3-TTS operations needed by the engine (mockable in tests).
pub trait TtsBackend: Send {
    /// Speak `text` in the voice of `ref_audio` (x-vector only cloning).
    fn generate_voice_clone(
        &mut self,
        text: &str,
        language: &str,
        ref_audio: &Path,
        temperature: f64,
        top_p: f64,
        seed: u64,
    ) -> Result<Wave>;

    /// Speak `text` without a reference (random voice); used once to create
    /// the voice reference.
    fn generate(
        &mut self,
        text: &str,
        language: &str,
        temperature: f64,
        top_p: f64,
        seed: u64,
    ) -> Result<Wave>;

    /// Number of text tokens, or None when no tokenizer is available.
    fn count_tokens(&self, text: &str) -> Option<usize>;
}

static PARAGRAPH_BREAK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\s*\n").unwrap());

/// Split at whitespace runs that follow `.!?` (Python `(?<=[.!?])\s+`).
fn split_after_sentence_end(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut iter = text.char_indices().peekable();
    let mut prev: Option<char> = None;
    while let Some((i, c)) = iter.next() {
        if c.is_whitespace() && matches!(prev, Some('.' | '!' | '?')) {
            let mut end = i + c.len_utf8();
            while let Some(&(j, d)) = iter.peek() {
                if d.is_whitespace() {
                    end = j + d.len_utf8();
                    iter.next();
                } else {
                    break;
                }
            }
            out.push(text[start..i].to_string());
            start = end;
            prev = None;
            continue;
        }
        prev = Some(c);
    }
    out.push(text[start..].to_string());
    out
}

pub struct TtsEngine {
    pub model_name: String,
    backend: Option<Box<dyn TtsBackend>>,
    pub anchor_path: Option<PathBuf>,
    voice_ref_path: Option<PathBuf>,
    settings: Settings,
    paths: Paths,
}

impl TtsEngine {
    pub fn new(settings: &Settings, paths: &Paths) -> Self {
        Self {
            model_name: settings.tts_model_name.clone(),
            backend: None,
            anchor_path: None,
            voice_ref_path: None,
            settings: settings.clone(),
            paths: paths.clone(),
        }
    }

    /// Engine with an already loaded backend (tests, other engines).
    pub fn with_backend(backend: Box<dyn TtsBackend>, settings: &Settings, paths: &Paths) -> Self {
        let mut e = Self::new(settings, paths);
        e.backend = Some(backend);
        e.voice_ref_path = Some(paths.voice_ref_path());
        e
    }

    pub fn is_loaded(&self) -> bool {
        self.backend.is_some()
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Load the TTS model and make sure the voice reference exists
    /// (generating it with the loaded model when missing).
    pub fn load(&mut self) -> Result<()> {
        if self.backend.is_none() && self.settings.mock_models {
            log::info!("Mock mode: TTS model not loaded, writing silence");
            return Ok(());
        }
        #[cfg(feature = "tts")]
        {
            if self.backend.is_none() {
                let b = qwen::QwenBackend::load(&self.model_name, &self.settings.tts_dtype)?;
                self.backend = Some(Box::new(b));
            }
        }
        #[cfg(not(feature = "tts"))]
        {
            if self.backend.is_none() {
                log::warn!("Qwen3-TTS support not compiled in (feature \"tts\"), using mock mode");
                return Ok(());
            }
        }
        let ref_path = self.paths.voice_ref_path();
        if !ref_path.exists() {
            self.generate_voice_reference(&ref_path)?;
        }
        if ref_path.exists() {
            log::info!(
                "Voice reference ready: {}",
                ref_path.file_name().unwrap_or_default().to_string_lossy()
            );
            self.voice_ref_path = Some(ref_path);
        }
        Ok(())
    }

    pub fn unload(&mut self) {
        if self.backend.take().is_some() {
            log::info!("Unloading TTS model");
        }
        self.voice_ref_path = None;
    }

    /// Generate the voice reference WAV with the loaded model.
    pub fn generate_voice_reference(&mut self, ref_path: &Path) -> Result<()> {
        let Some(backend) = self.backend.as_mut() else {
            bail!("TTS model not loaded");
        };
        log::info!("Generating voice reference with {} ...", self.model_name);
        let language = resolve_language(&self.settings, Some("english"));
        let wave = backend.generate(
            voice_ref_text(&language),
            &language,
            REFERENCE_TEMPERATURE,
            REFERENCE_TOP_P,
            self.settings.voice_seed,
        )?;
        if wave.samples.is_empty() {
            bail!("Base model returned no audio for reference generation");
        }
        write_mono_wav(ref_path, &wave.samples, wave.sample_rate)?;
        log::info!("Voice reference saved: {}", ref_path.display());
        Ok(())
    }

    /// Delete and regenerate the voice reference (GUI "Regen Voice Ref").
    pub fn regenerate_voice_reference(settings: &Settings, paths: &Paths) -> Result<PathBuf> {
        let ref_path = paths.voice_ref_path();
        if ref_path.exists() {
            std::fs::remove_file(&ref_path)?;
        }
        let mut engine = TtsEngine::new(settings, paths);
        engine.load()?;
        if !engine.is_loaded() {
            bail!("TTS model not available");
        }
        engine.unload();
        Ok(ref_path)
    }

    fn write_silence(&self, out_path: &Path) -> Result<PathBuf> {
        let sr = self.settings.sample_rate;
        write_mono_wav(out_path, &vec![0.0; sr as usize * 2], sr)?;
        Ok(out_path.to_path_buf())
    }

    // ------------------------------------------------------------------
    // Text splitting
    // ------------------------------------------------------------------

    /// Split text over the token limit (`TTS_MAX_TOKENS`, or
    /// `TTS_MAX_WORDS_FALLBACK` words without tokenizer) into parts packed
    /// to 85% of the limit, on paragraph and then sentence boundaries.
    pub fn split_if_too_long(&self, text: &str) -> Vec<String> {
        let tokenizer = self
            .backend
            .as_ref()
            .is_some_and(|b| b.count_tokens("probe").is_some());
        let (limit, unit) = if tokenizer {
            (self.settings.tts_max_tokens, "tokens")
        } else {
            (self.settings.tts_max_words_fallback, "words")
        };
        // 15% headroom: Qwen3-TTS gets unstable near its hard ceiling.
        let target = (limit as f64 * 0.85) as usize;
        let measure = |s: &str| -> usize {
            if tokenizer {
                self.backend
                    .as_ref()
                    .and_then(|b| b.count_tokens(s))
                    .unwrap_or_else(|| word_count(s))
            } else {
                word_count(s)
            }
        };

        let total = measure(text);
        if total <= limit {
            return vec![text.to_string()];
        }

        // Atoms: paragraphs, or sentences of paragraphs still over target.
        let mut atoms: Vec<(String, usize)> = Vec::new();
        for para in PARAGRAPH_BREAK.split(text) {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }
            let count = measure(para);
            if count <= target {
                atoms.push((para.to_string(), count));
                continue;
            }
            for sent in split_after_sentence_end(para) {
                let sent = sent.trim();
                if !sent.is_empty() {
                    atoms.push((sent.to_string(), measure(sent)));
                }
            }
        }

        let mut parts: Vec<String> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let mut current_count = 0usize;
        for (atom, count) in atoms {
            if !current.is_empty() && (current_count + count > limit || current_count >= target) {
                parts.push(current.join(" "));
                current = vec![atom];
                current_count = count;
            } else {
                current.push(atom);
                current_count += count;
            }
        }
        if !current.is_empty() {
            parts.push(current.join(" "));
        }

        let oversize = parts.iter().filter(|p| measure(p) > limit).count();
        if oversize > 0 {
            log::warn!(
                "Split produced {oversize} sub-part(s) still over the {limit}-{unit} limit \
                 (atomic sentences too long); TTS may truncate or hallucinate"
            );
        }
        log::debug!(
            "Split long segment ({total} {unit}, target {target}, limit {limit}) into {} sub-parts",
            parts.len()
        );
        parts
    }

    // ------------------------------------------------------------------
    // Synthesis
    // ------------------------------------------------------------------

    fn synthesize_single(
        &mut self,
        text: &str,
        ref_audio: &Path,
        out_path: &Path,
        language: &str,
    ) -> Result<()> {
        let seed = self.settings.voice_seed;
        let backend = self.backend.as_mut().expect("backend checked by caller");
        let wave = backend.generate_voice_clone(
            text,
            if language.is_empty() {
                DEFAULT_LANGUAGE
            } else {
                language
            },
            ref_audio,
            CLONE_TEMPERATURE,
            CLONE_TOP_P,
            seed,
        )?;
        if wave.samples.is_empty() {
            bail!("generate_voice_clone returned no audio");
        }
        write_mono_wav(out_path, &wave.samples, wave.sample_rate)
    }

    /// Synthesize `text` to `out_path` (or to `out_path`'s `_N` sub-parts),
    /// cloning the voice of `ref_audio` (the voice reference when None).
    pub fn synthesize(
        &mut self,
        text: &str,
        ref_audio: Option<&Path>,
        out_path: &Path,
        language: &str,
        stop: Option<&StopFlag>,
    ) -> Result<Vec<PathBuf>> {
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if self.backend.is_none() {
            return Ok(vec![self.write_silence(out_path)?]);
        }
        let ref_audio: PathBuf = match ref_audio {
            Some(p) => p.to_path_buf(),
            None => match &self.voice_ref_path {
                Some(p) => p.clone(),
                None => bail!("no voice reference available"),
            },
        };
        let result = (|| -> Result<Vec<PathBuf>> {
            let parts = self.split_if_too_long(text);
            if parts.len() == 1 {
                self.synthesize_single(&parts[0], &ref_audio, out_path, language)?;
                return Ok(vec![out_path.to_path_buf()]);
            }
            if parts.len() > MAX_SUB_PARTS {
                bail!(
                    "Segment split produced {} sub-parts; max supported is {MAX_SUB_PARTS}",
                    parts.len()
                );
            }
            // Multi-part chunks: every part gets a _N suffix (0-indexed); the
            // unsuffixed name is reserved for single-part chunks.
            let stem = out_path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let ext = out_path
                .extension()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let dir = out_path.parent().map(Path::to_path_buf).unwrap_or_default();
            let mut paths = Vec::new();
            for (idx, part) in parts.iter().enumerate() {
                let part_path = dir.join(format!("{stem}_{idx}.{ext}"));
                if part_path.exists() {
                    log::info!(
                        "Part {}/{} already exists, reusing: {}",
                        idx + 1,
                        parts.len(),
                        part_path.file_name().unwrap_or_default().to_string_lossy()
                    );
                    paths.push(part_path);
                    continue;
                }
                if forced(stop) {
                    return Err(ForceStopped.into());
                }
                self.synthesize_single(part, &ref_audio, &part_path, language)?;
                log::warn!(
                    "Text split into {} parts; saved as {}",
                    parts.len(),
                    part_path.file_name().unwrap_or_default().to_string_lossy()
                );
                paths.push(part_path);
            }
            Ok(paths)
        })();
        if let Err(e) = &result
            && !e.is::<ForceStopped>()
        {
            log::error!("TTS synthesis failed: {e:#}");
        }
        result
    }

    /// First chunk: clone the voice reference; the result becomes the anchor.
    pub fn synthesize_first(
        &mut self,
        text: &str,
        out_path: &Path,
        language: &str,
    ) -> Result<Vec<PathBuf>> {
        let paths = self.synthesize(text, None, out_path, language, None)?;
        if self.backend.is_some() {
            self.anchor_path = paths.first().cloned();
        }
        Ok(paths)
    }

    /// Later chunks: clone the voice of the anchor WAV.
    pub fn synthesize_with_anchor(
        &mut self,
        text: &str,
        anchor: &Path,
        out_path: &Path,
        language: &str,
    ) -> Result<Vec<PathBuf>> {
        self.synthesize(text, Some(anchor), out_path, language, None)
    }

    /// Rename legacy 1-indexed multi-part names (`_1`, `_2`, ... without a
    /// base file) to the 0-indexed scheme. Idempotent.
    pub fn migrate_legacy_chunk_names(output_dir: &Path, total_chunks: usize) {
        let mut renamed = 0;
        for i in 0..total_chunks {
            let stem = format!("chunk_{i:06}");
            if output_dir.join(format!("{stem}.wav")).exists()
                || output_dir.join(format!("{stem}_0.wav")).exists()
            {
                continue;
            }
            let legacy: Vec<(usize, PathBuf)> = (1..=MAX_SUB_PARTS)
                .map(|k| (k, output_dir.join(format!("{stem}_{k}.wav"))))
                .filter(|(_, p)| p.exists())
                .collect();
            if legacy.is_empty() {
                continue;
            }
            for (k, src) in legacy {
                let dst = output_dir.join(format!("{stem}_{}.wav", k - 1));
                if dst.exists() {
                    let _ = std::fs::remove_file(&dst);
                }
                if let Err(e) = std::fs::rename(&src, &dst) {
                    log::warn!("Could not rename {}: {e}", src.display());
                }
            }
            renamed += 1;
        }
        if renamed > 0 {
            log::info!("Migrated {renamed} chunk(s) to 0-indexed multi-part naming");
        }
    }

    /// Existing WAVs of chunk `i`: `chunk_NNN.wav`, or `_0`, `_1`, ...
    pub fn existing_parts(output_dir: &Path, i: usize) -> Vec<PathBuf> {
        let base = output_dir.join(format!("chunk_{i:06}.wav"));
        if base.exists() {
            return vec![base];
        }
        let mut parts = Vec::new();
        for k in 0..MAX_SUB_PARTS {
            let p = output_dir.join(format!("chunk_{i:06}_{k}.wav"));
            if !p.exists() {
                break;
            }
            parts.push(p);
        }
        parts
    }

    /// Synthesize all segments into `output_dir`, skipping chunks that
    /// already exist. Returns every WAV (including sub-parts) in order.
    pub fn process_all(
        &mut self,
        segments: &[Segment],
        output_dir: &Path,
        stop: Option<&StopFlag>,
        chunk_done: Option<&(dyn Fn() + Sync)>,
    ) -> Result<Vec<PathBuf>> {
        std::fs::create_dir_all(output_dir)?;
        Self::migrate_legacy_chunk_names(output_dir, segments.len());
        let mut wav_paths = Vec::new();

        for (i, seg) in segments.iter().enumerate() {
            if stopped(stop) {
                log::info!(
                    "Stop requested after {i}/{} chunks generated",
                    segments.len()
                );
                break;
            }
            let out_path = output_dir.join(format!("chunk_{i:06}.wav"));

            let existing = Self::existing_parts(output_dir, i);
            if !existing.is_empty() {
                log::debug!(
                    "Chunk {i} already exists ({} part(s)), skipping",
                    existing.len()
                );
                for w in &existing {
                    tags::tag_wav(w, seg);
                }
                if i == 0 {
                    self.anchor_path = existing.first().cloned();
                }
                wav_paths.extend(existing);
                if let Some(cb) = chunk_done {
                    cb();
                }
                continue;
            }

            let language = resolve_language(&self.settings, seg.language.as_deref());
            let ref_audio = if i == 0 {
                None
            } else {
                self.anchor_path.clone()
            };
            let result = self.synthesize(
                &seg.normalized_text,
                ref_audio.as_deref(),
                &out_path,
                &language,
                stop,
            );
            let paths = match result {
                Ok(p) => p,
                Err(e) if e.is::<ForceStopped>() => {
                    if out_path.exists() {
                        let _ = std::fs::remove_file(&out_path);
                    }
                    log::warn!("Force-stopped at chunk {i}. Partial file removed.");
                    return Err(e);
                }
                Err(e) => return Err(e),
            };
            if i == 0 {
                self.anchor_path = paths.first().cloned();
            }
            for w in &paths {
                tags::tag_wav(w, seg);
            }
            wav_paths.extend(paths);
            if let Some(cb) = chunk_done {
                cb();
            }
            if (i + 1) % 20 == 0 {
                log::info!("Checkpoint at chunk {}", i + 1);
            }
        }
        Ok(wav_paths)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Records calls and returns one second of silence.
    #[derive(Clone, Default)]
    pub struct FakeTts {
        pub clone_calls: Arc<Mutex<Vec<(String, String, PathBuf)>>>,
        pub generate_calls: Arc<Mutex<Vec<(String, String)>>>,
        pub tokens: bool,
        pub on_clone: Option<Arc<dyn Fn(usize) + Send + Sync>>,
    }

    impl TtsBackend for FakeTts {
        fn generate_voice_clone(
            &mut self,
            text: &str,
            language: &str,
            ref_audio: &Path,
            _t: f64,
            _p: f64,
            _s: u64,
        ) -> Result<Wave> {
            let n = {
                let mut c = self.clone_calls.lock().unwrap();
                c.push((text.into(), language.into(), ref_audio.into()));
                c.len()
            };
            if let Some(cb) = &self.on_clone {
                cb(n);
            }
            Ok(Wave {
                samples: vec![0.0; 24000],
                sample_rate: 24000,
            })
        }

        fn generate(
            &mut self,
            text: &str,
            language: &str,
            _t: f64,
            _p: f64,
            _s: u64,
        ) -> Result<Wave> {
            self.generate_calls
                .lock()
                .unwrap()
                .push((text.into(), language.into()));
            Ok(Wave {
                samples: vec![0.0; 24000],
                sample_rate: 24000,
            })
        }

        fn count_tokens(&self, text: &str) -> Option<usize> {
            self.tokens.then(|| word_count(text))
        }
    }

    fn setup(settings: &Settings) -> (tempfile::TempDir, FakeTts, TtsEngine) {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        paths.ensure_dirs().unwrap();
        let fake = FakeTts::default();
        let mut engine = TtsEngine::with_backend(Box::new(fake.clone()), settings, &paths);
        engine.load().unwrap();
        (dir, fake, engine)
    }

    fn segs(texts: &[(&str, Option<&str>)]) -> Vec<Segment> {
        texts
            .iter()
            .map(|(t, l)| Segment {
                language: l.map(String::from),
                ..Segment::with_text(*t)
            })
            .collect()
    }

    #[test]
    fn load_generates_missing_voice_reference() {
        let (dir, fake, _engine) = setup(&Settings::default());
        assert!(dir.path().join("cache/voice_reference.wav").exists());
        let calls = fake.generate_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "german");
        assert!(calls[0].0.starts_with("Dies ist eine kurze Sprachprobe"));
    }

    #[test]
    fn process_all_anchors_on_first_chunk() {
        let (dir, fake, mut engine) = setup(&Settings::default());
        let out = dir.path().join("out");
        let wavs = engine
            .process_all(
                &segs(&[("First.", None), ("Second.", None), ("Third.", None)]),
                &out,
                None,
                None,
            )
            .unwrap();
        assert_eq!(wavs.len(), 3);
        assert_eq!(engine.anchor_path.as_ref(), Some(&wavs[0]));
        let calls = fake.clone_calls.lock().unwrap();
        assert_eq!(calls.len(), 3);
        assert!(calls[0].2.ends_with("voice_reference.wav"));
        assert_eq!(calls[1].2, wavs[0]);
        assert_eq!(calls[2].2, wavs[0]);
    }

    #[test]
    fn process_all_resumes_skipping_existing() {
        let (dir, fake, mut engine) = setup(&Settings::default());
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        for i in 0..2 {
            write_mono_wav(&out.join(format!("chunk_{i:06}.wav")), &[0.0; 10], 24000).unwrap();
        }
        let wavs = engine
            .process_all(
                &segs(&[("First.", None), ("Second.", None), ("Third.", None)]),
                &out,
                None,
                None,
            )
            .unwrap();
        assert_eq!(wavs.len(), 3);
        assert_eq!(fake.clone_calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn stop_flag_breaks_loop() {
        let (dir, mut fake, _) = setup(&Settings::default());
        let stop = StopFlag::new();
        let s2 = stop.clone();
        fake.on_clone = Some(Arc::new(move |n| {
            if n == 1 {
                s2.set();
            }
        }));
        let paths = Paths::new(dir.path());
        let mut engine =
            TtsEngine::with_backend(Box::new(fake.clone()), &Settings::default(), &paths);
        let texts: Vec<String> = (0..5).map(|i| format!("chunk {i}")).collect();
        let s: Vec<Segment> = texts
            .iter()
            .map(|t| Segment::with_text(t.as_str()))
            .collect();
        let wavs = engine
            .process_all(&s, &dir.path().join("out"), Some(&stop), None)
            .unwrap();
        assert_eq!(wavs.len(), 1);
        assert_eq!(fake.clone_calls.lock().unwrap().len(), 1);
    }

    #[test]
    fn mock_mode_writes_silence() {
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let mut engine = TtsEngine::new(&Settings::default(), &paths);
        let out = dir.path().join("silence.wav");
        let r = engine.synthesize_first("text", &out, "german").unwrap();
        assert_eq!(r, vec![out.clone()]);
        let a = crate::audio::read_wav(&out).unwrap();
        assert_eq!(a.samples.len(), 48000);
    }

    #[test]
    fn per_segment_language_with_auto() {
        let s = Settings {
            tts_language: "auto".into(),
            ..Settings::default()
        };
        let (dir, fake, mut engine) = setup(&s);
        engine
            .process_all(
                &segs(&[
                    ("Hello.", Some("english")),
                    ("Hola.", Some("spanish")),
                    ("Bonjour.", Some("french")),
                    ("x", None),
                ]),
                &dir.path().join("out"),
                None,
                None,
            )
            .unwrap();
        let langs: Vec<String> = fake
            .clone_calls
            .lock()
            .unwrap()
            .iter()
            .map(|c| c.1.clone())
            .collect();
        assert_eq!(langs, ["english", "spanish", "french", DEFAULT_LANGUAGE]);
    }

    #[test]
    fn fixed_language_overrides_segments() {
        let (dir, fake, mut engine) = setup(&Settings::default());
        engine
            .process_all(
                &segs(&[
                    ("Erster Satz.", Some("english")),
                    ("Zweiter Satz.", Some("japanese")),
                    ("Dritter Satz.", None),
                ]),
                &dir.path().join("out"),
                None,
                None,
            )
            .unwrap();
        let langs: Vec<String> = fake
            .clone_calls
            .lock()
            .unwrap()
            .iter()
            .map(|c| c.1.clone())
            .collect();
        assert_eq!(langs, ["german", "german", "german"]);
    }

    #[test]
    fn long_segments_become_sub_parts() {
        let s = Settings {
            tts_max_words_fallback: 10,
            ..Settings::default()
        };
        let (dir, fake, mut engine) = setup(&s);
        let text =
            "Eins zwei drei vier. Fünf sechs sieben acht. Neun zehn elf zwölf. Dreizehn vierzehn.";
        assert_eq!(engine.split_if_too_long(text).len(), 2);
        let out = dir.path().join("out");
        let wavs = engine
            .process_all(&segs(&[(text, None)]), &out, None, None)
            .unwrap();
        let names: Vec<String> = wavs
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, ["chunk_000000_0.wav", "chunk_000000_1.wav"]);
        assert_eq!(fake.clone_calls.lock().unwrap().len(), 2);
        assert_eq!(engine.anchor_path.as_ref(), Some(&wavs[0]));
        // Second run reuses both parts.
        let again = engine
            .process_all(&segs(&[(text, None)]), &out, None, None)
            .unwrap();
        assert_eq!(again, wavs);
        assert_eq!(fake.clone_calls.lock().unwrap().len(), 2);
    }

    #[test]
    fn split_uses_tokens_when_available() {
        let s = Settings {
            tts_max_tokens: 4,
            tts_max_words_fallback: 1000,
            ..Settings::default()
        };
        let dir = tempfile::tempdir().unwrap();
        let paths = Paths::new(dir.path());
        let fake = FakeTts {
            tokens: true,
            ..Default::default()
        };
        let engine = TtsEngine::with_backend(Box::new(fake), &s, &paths);
        assert_eq!(
            engine.split_if_too_long("A b c. D e f. G h."),
            ["A b c.", "D e f.", "G h."]
        );
    }

    #[test]
    fn legacy_names_are_migrated() {
        let dir = tempfile::tempdir().unwrap();
        for k in 1..=2 {
            std::fs::write(dir.path().join(format!("chunk_000000_{k}.wav")), b"x").unwrap();
        }
        TtsEngine::migrate_legacy_chunk_names(dir.path(), 1);
        assert!(dir.path().join("chunk_000000_0.wav").exists());
        assert!(dir.path().join("chunk_000000_1.wav").exists());
        assert!(!dir.path().join("chunk_000000_2.wav").exists());
        assert_eq!(TtsEngine::existing_parts(dir.path(), 0).len(), 2);
    }

    #[test]
    fn sentence_end_split() {
        assert_eq!(
            split_after_sentence_end("A. b!  C? d"),
            ["A.", "b!", "C?", "d"]
        );
    }
}
