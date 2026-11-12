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

//! Where the narrator voice comes from (`VOICE_SOURCE`).
//!
//! Both Qwen3-TTS Base models (0.6B and 1.7B) only *clone* a voice: they
//! need a short recording, `cache/voice_reference.wav`. That recording is
//! - a file chosen by the user ("file"),
//! - spoken by the separate VoiceDesign model from the narrator description
//!   ("design"; the model exists only as 1.7B and is loaded just for this),
//! - or spoken by the Base model itself without a reference ("random": a
//!   voice that depends on `VOICE_SEED`).
//!
//! `cache/voice_reference.json` records what the current recording was made
//! from, so it is rebuilt only when the choice changes.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::audio::{read_wav, write_mono_wav};
use crate::config::{Paths, Settings};

pub const META_FILE: &str = "voice_reference.json";

/// Longest part of a chosen recording that is kept.
pub const MAX_REFERENCE_SECONDS: f32 = 20.0;
/// Shorter recordings give an unstable voice.
pub const MIN_REFERENCE_SECONDS: f32 = 3.0;

/// What a voice reference was made from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceMeta {
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Size and modification time of the chosen file: a re-recorded file
    /// under the same name is imported again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_stamp: Option<(u64, u64)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// The source in the settings, normalized ("random" when unknown).
pub fn source(settings: &Settings) -> &'static str {
    match settings.voice_source.trim().to_lowercase().as_str() {
        "file" => "file",
        "design" => "design",
        _ => "random",
    }
}

fn file_stamp(path: &Path) -> Option<(u64, u64)> {
    let m = std::fs::metadata(path).ok()?;
    let mtime = m
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_secs();
    Some((m.len(), mtime))
}

/// The recording the settings ask for.
pub fn wanted(settings: &Settings, language: &str) -> VoiceMeta {
    let mut meta = VoiceMeta {
        source: source(settings).into(),
        file: None,
        file_stamp: None,
        description: None,
        model: None,
        seed: None,
        language: None,
    };
    match meta.source.as_str() {
        "file" => {
            let f = settings.voice_reference_file.trim();
            meta.file = Some(f.to_string());
            meta.file_stamp = file_stamp(Path::new(f));
        }
        "design" => {
            meta.description = Some(settings.narrator_prompt.trim().to_string());
            meta.model = Some(settings.voice_design_model.trim().to_string());
            meta.seed = Some(settings.voice_seed);
            meta.language = Some(language.to_string());
        }
        _ => {
            meta.seed = Some(settings.voice_seed);
            meta.language = Some(language.to_string());
        }
    }
    meta
}

pub fn meta_path(paths: &Paths) -> PathBuf {
    paths.cache_dir.join(META_FILE)
}

pub fn read_meta(paths: &Paths) -> Option<VoiceMeta> {
    let text = std::fs::read_to_string(meta_path(paths)).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn write_meta(paths: &Paths, meta: &VoiceMeta) -> Result<()> {
    std::fs::create_dir_all(&paths.cache_dir)?;
    std::fs::write(meta_path(paths), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

/// True if `cache/voice_reference.wav` has to be (re)made for `want`.
///
/// A reference without record that the user put there by hand (as the docs
/// used to recommend) is kept while the source is "random", the default.
pub fn needs_update(paths: &Paths, want: &VoiceMeta) -> bool {
    if !paths.voice_ref_path().exists() {
        return true;
    }
    match read_meta(paths) {
        Some(have) => &have != want,
        None => want.source != "random",
    }
}

/// Copy a chosen recording into the voice reference: mono, at most
/// [`MAX_REFERENCE_SECONDS`]. Returns the kept length in seconds.
pub fn import_file(src: &Path, dst: &Path) -> Result<f32> {
    if src.as_os_str().is_empty() {
        bail!("no voice recording chosen (VOICE_REFERENCE_FILE is empty)");
    }
    let audio = read_wav(src)
        .with_context(|| format!("reading the voice recording {} (WAV needed)", src.display()))?;
    let mut samples = audio.to_mono();
    let sr = audio.sample_rate.max(1);
    let max = (MAX_REFERENCE_SECONDS * sr as f32) as usize;
    if samples.len() > max {
        samples.truncate(max);
    }
    let seconds = samples.len() as f32 / sr as f32;
    if seconds < MIN_REFERENCE_SECONDS {
        bail!(
            "the voice recording {} is only {seconds:.1} s long; use 10–15 s of speech",
            src.display()
        );
    }
    if let Some(dir) = dst.parent() {
        std::fs::create_dir_all(dir)?;
    }
    write_mono_wav(dst, &samples, sr)?;
    Ok(seconds)
}

/// Short description of the current voice reference for the GUI.
pub fn describe(paths: &Paths) -> String {
    let path = paths.voice_ref_path();
    let Ok(audio) = read_wav(&path) else {
        return "none yet (made when audio is generated)".into();
    };
    let secs =
        audio.samples.len() as f32 / audio.channels.max(1) as f32 / audio.sample_rate.max(1) as f32;
    let from = match read_meta(paths) {
        Some(m) => match m.source.as_str() {
            "file" => format!(
                "from {}",
                Path::new(m.file.as_deref().unwrap_or(""))
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default()
            ),
            "design" => "from the description".into(),
            _ => format!("random voice, seed {}", m.seed.unwrap_or_default()),
        },
        None => "placed by hand".into(),
    };
    format!("{secs:.1} s, {from}")
}

/// Identity of the current voice reference (FNV-1a of the file), stored with
/// every chunk so a new voice makes the chunks be synthesized again.
pub fn voice_id(path: &Path) -> String {
    match std::fs::read(path) {
        Ok(bytes) => {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            format!("{h:016x}")
        }
        Err(_) => "none".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(dir: &Path) -> Paths {
        let p = Paths::new(dir);
        std::fs::create_dir_all(&p.cache_dir).unwrap();
        p
    }

    fn tone(path: &Path, seconds: f32, sr: u32) {
        let n = (seconds * sr as f32) as usize;
        let s: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        write_mono_wav(path, &s, sr).unwrap();
    }

    #[test]
    fn hand_placed_reference_is_kept_for_random_only() {
        let dir = tempfile::tempdir().unwrap();
        let p = paths(dir.path());
        tone(&p.voice_ref_path(), 5.0, 24000);
        let mut s = Settings::default();
        assert!(!needs_update(&p, &wanted(&s, "german")));
        s.voice_source = "design".into();
        assert!(needs_update(&p, &wanted(&s, "german")));
    }

    #[test]
    fn changes_of_the_choice_are_noticed() {
        let dir = tempfile::tempdir().unwrap();
        let p = paths(dir.path());
        tone(&p.voice_ref_path(), 5.0, 24000);
        let mut s = Settings {
            voice_source: "design".into(),
            ..Settings::default()
        };
        write_meta(&p, &wanted(&s, "german")).unwrap();
        assert!(!needs_update(&p, &wanted(&s, "german")));
        s.narrator_prompt = "Eine tiefe, ruhige Männerstimme.".into();
        assert!(needs_update(&p, &wanted(&s, "german")));
        s.voice_source = "random".into();
        write_meta(&p, &wanted(&s, "german")).unwrap();
        s.voice_seed += 1;
        assert!(needs_update(&p, &wanted(&s, "german")));
    }

    #[test]
    fn import_makes_mono_and_cuts_long_recordings() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("aufnahme.wav");
        let dst = dir.path().join("cache/voice_reference.wav");
        tone(&src, 30.0, 16000);
        let secs = import_file(&src, &dst).unwrap();
        assert!((secs - MAX_REFERENCE_SECONDS).abs() < 0.01);
        let a = read_wav(&dst).unwrap();
        assert_eq!(a.channels, 1);
        assert_eq!(a.sample_rate, 16000);
        // Too short and missing files are refused.
        tone(&src, 1.0, 16000);
        assert!(import_file(&src, &dst).is_err());
        assert!(import_file(Path::new(""), &dst).is_err());
    }

    #[test]
    fn voice_id_follows_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("v.wav");
        assert_eq!(voice_id(&f), "none");
        tone(&f, 4.0, 24000);
        let a = voice_id(&f);
        tone(&f, 4.5, 24000);
        assert_ne!(voice_id(&f), a);
    }
}
