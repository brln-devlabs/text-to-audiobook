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

//! Model downloads from Hugging Face.
//!
//! - The Llama GGUF goes to `models/` (same place as the Python version).
//! - TTS models use the shared Hugging Face cache (`HF_HOME`, default
//!   `~/.cache/huggingface`), so models downloaded by the Python version are
//!   reused. `HF_TOKEN` is sent when set.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::config::{LLM_MODEL_NAME, LLM_MODEL_REPO};

fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
}

fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|e| !e.trim().is_empty())
        .unwrap_or_else(|| "https://huggingface.co".into())
        .trim_end_matches('/')
        .to_string()
}

fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new().try_proxy_from_env(true).build()
}

/// Download `repo`/`filename` (branch main) to `dest`, via `dest.part`.
pub fn download_file(repo: &str, filename: &str, dest: &Path) -> Result<PathBuf> {
    let url = format!("{}/{repo}/resolve/main/{filename}", hf_endpoint());
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let token = hf_token();
    if token.is_none() {
        log::warn!("HF_TOKEN not set in environment; download may fail for private models");
    }
    log::info!("Downloading {repo}/{filename} ...");
    let mut req = agent().get(&url);
    if let Some(t) = &token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = req.call().with_context(|| format!("GET {url}"))?;
    let total: Option<u64> = resp.header("Content-Length").and_then(|v| v.parse().ok());
    let part = dest.with_extension(format!(
        "{}.part",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("")
    ));
    let mut file =
        std::fs::File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut done: u64 = 0;
    let mut next_report = 10u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        done += n as u64;
        if let Some(total) = total.filter(|t| *t > 0) {
            let pct = done * 100 / total;
            if pct >= next_report {
                log::info!("  {filename}: {pct}% of {} MB", total / (1024 * 1024));
                next_report = (pct / 10 + 1) * 10;
            }
        }
    }
    file.flush()?;
    drop(file);
    if let Some(total) = total
        && done != total
    {
        bail!("download of {filename} incomplete ({done} of {total} bytes)");
    }
    std::fs::rename(&part, dest)?;
    log::info!("Model downloaded to {}", dest.display());
    Ok(dest.to_path_buf())
}

/// Path of the Llama GGUF, downloading it to `model_path` if missing.
pub fn ensure_llm_model(model_path: &Path) -> Result<PathBuf> {
    if model_path.exists() {
        return Ok(model_path.to_path_buf());
    }
    log::info!("Downloading Llama model from HuggingFace...");
    download_file(LLM_MODEL_REPO, LLM_MODEL_NAME, model_path)
}

/// State of a model download, for progress displays.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DownloadProgress {
    /// Model id.
    pub model: String,
    /// File being downloaded.
    pub file: String,
    /// Bytes downloaded so far (files that were already there do not count).
    pub done: u64,
    /// Bytes of all files to download; 0 while unknown.
    pub total: u64,
}

impl DownloadProgress {
    /// Fraction done, 0.0 ..= 1.0 (0 while the size is unknown).
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0) as f32
        }
    }
}

/// Callback for [`download_tts_model`].
pub type ProgressFn<'a> = &'a (dyn Fn(&DownloadProgress) + Sync);

const MB: u64 = 1024 * 1024;

/// "1.2 GB" / "350 MB".
pub fn human_size(bytes: u64) -> String {
    if bytes >= 1024 * MB {
        format!("{:.1} GB", bytes as f64 / (1024 * MB) as f64)
    } else {
        format!("{} MB", bytes.div_ceil(MB))
    }
}

/// Directory with config.json, tokenizer and safetensors of a TTS model.
///
/// `model` is either a local directory or a Hugging Face model id; for ids
/// a complete snapshot in the cache is used as is, otherwise the snapshot is
/// downloaded into the cache.
pub fn resolve_tts_model_dir(model: &str) -> Result<PathBuf> {
    if let Some(dir) = available_tts_model_dir(model) {
        log::debug!("Using TTS model at {}", dir.display());
        return Ok(dir);
    }
    download_tts_model(model, None)
}

/// True if `model` is a local model directory or complete in the cache, so
/// loading it needs no download.
pub fn tts_model_available(model: &str) -> bool {
    available_tts_model_dir(model).is_some()
}

fn available_tts_model_dir(model: &str) -> Option<PathBuf> {
    let local = Path::new(model);
    if local.join("config.json").is_file() {
        return Some(local.to_path_buf());
    }
    let cache = hf_hub::Cache::from_env();
    let config = cache.model(model.to_string()).get("config.json")?;
    let dir = config.parent()?;
    snapshot_looks_complete(dir).then(|| dir.to_path_buf())
}

/// Where Hugging Face models are stored (`HF_HOME`, default
/// `~/.cache/huggingface`).
pub fn model_cache_dir() -> PathBuf {
    hf_hub::Cache::from_env().path().clone()
}

/// Download the files of `model` that are not in the cache yet and return
/// the snapshot directory. An interrupted download continues where it
/// stopped: finished files are kept, a partial file is resumed.
pub fn download_tts_model(model: &str, on_progress: Option<ProgressFn>) -> Result<PathBuf> {
    log::info!("Downloading TTS model {model} into the Hugging Face cache ...");
    let cache = hf_hub::Cache::from_env();
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .with_token(hf_token().or_else(|| cache.token()))
        .with_progress(false)
        .with_retries(5)
        .build()
        .context("creating Hugging Face client")?;
    let repo = api.model(model.to_string());
    let files = list_model_files(&repo)
        .with_context(|| format!("listing files of {model} (set HF_TOKEN if needed)"))?;
    let cached = cache.model(model.to_string());
    let todo: Vec<&(String, u64)> = files
        .iter()
        .filter(|(name, _)| cached.get(name).is_none())
        .collect();
    let total: u64 = todo.iter().map(|(_, size)| size).sum();
    if !todo.is_empty() {
        log::info!("  {} file(s), {}", todo.len(), human_size(total));
    }
    let mut report = Reporter {
        state: DownloadProgress {
            model: model.to_string(),
            file: String::new(),
            done: 0,
            total,
        },
        base: 0,
        current: 0,
        on_progress,
        next_log: 10,
        last_call: None,
    };
    let n = todo.len();
    for (i, (name, size)) in todo.iter().enumerate() {
        log::info!("  [{}/{n}] {name} ({})", i + 1, human_size(*size));
        report.state.file = name.clone();
        report.current = 0;
        report.emit(true);
        repo.download_with_progress(name, &mut report)
            .with_context(|| format!("downloading {model}/{name}"))?;
        report.base += size;
        report.current = 0;
    }
    report.emit(true);
    let config = cached
        .get("config.json")
        .with_context(|| format!("{model} has no config.json"))?;
    Ok(config
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(".")))
}

/// Files of a model repo that are needed to run it, with their sizes
/// (README, .gitattributes and the like are skipped).
fn list_model_files(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<(String, u64)>> {
    let info: serde_json::Value = repo
        .info_request()
        .query("blobs", "true")
        .call()?
        .into_json()?;
    let siblings = info
        .get("siblings")
        .and_then(|s| s.as_array())
        .context("unexpected answer from Hugging Face (no file list)")?;
    Ok(siblings
        .iter()
        .filter_map(|s| {
            let name = s.get("rfilename")?.as_str()?;
            let size = s.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
            needed_file(name).then(|| (name.to_string(), size))
        })
        .collect())
}

fn needed_file(name: &str) -> bool {
    !(name.starts_with('.') || name.ends_with(".md"))
}

/// Turns hf-hub's per-file progress into [`DownloadProgress`] for the whole
/// model and logs every 10 %.
struct Reporter<'a> {
    state: DownloadProgress,
    /// Bytes of the files finished before the current one.
    base: u64,
    /// Bytes of the current file.
    current: u64,
    on_progress: Option<ProgressFn<'a>>,
    next_log: u64,
    last_call: Option<std::time::Instant>,
}

impl Reporter<'_> {
    fn emit(&mut self, force: bool) {
        let done = self.base + self.current;
        self.state.done = if self.state.total > 0 {
            done.min(self.state.total)
        } else {
            done
        };
        if let Some(pct) = (self.state.done * 100).checked_div(self.state.total)
            && pct >= self.next_log
            && pct < 100
        {
            log::info!(
                "  {}: {pct}% of {}",
                self.state.model,
                human_size(self.state.total)
            );
            self.next_log = (pct / 10 + 1) * 10;
        }
        let Some(cb) = self.on_progress else {
            return;
        };
        // A few updates per second are plenty for a progress bar.
        let now = std::time::Instant::now();
        if force
            || self
                .last_call
                .is_none_or(|t| now.duration_since(t).as_millis() >= 200)
        {
            self.last_call = Some(now);
            cb(&self.state);
        }
    }
}

impl hf_hub::api::Progress for &mut Reporter<'_> {
    fn init(&mut self, _size: usize, _filename: &str) {
        // Called again before each (resumed) attempt, followed by update()
        // with the bytes already on disk.
        self.current = 0;
    }

    fn update(&mut self, size: usize) {
        self.current += size as u64;
        self.emit(false);
    }

    fn finish(&mut self) {
        self.emit(true);
    }
}

/// True if `dir` has a config, weights and the speech tokenizer.
fn snapshot_looks_complete(dir: &Path) -> bool {
    let has_weights = dir.join("model.safetensors").is_file()
        || dir.join("model.safetensors.index.json").is_file();
    let has_tokenizer = dir.join("tokenizer.json").is_file()
        || (dir.join("vocab.json").is_file() && dir.join("merges.txt").is_file());
    let has_speech_tokenizer = dir.join("speech_tokenizer").join("config.json").is_file();
    has_weights && has_tokenizer && has_speech_tokenizer
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    #[test]
    fn local_model_dir_is_used_directly() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        let p = resolve_tts_model_dir(dir.path().to_str().unwrap()).unwrap();
        assert_eq!(p, dir.path());
    }

    #[test]
    fn existing_llm_model_is_not_downloaded() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m.gguf");
        std::fs::write(&p, b"x").unwrap();
        assert_eq!(ensure_llm_model(&p).unwrap(), p);
    }

    #[test]
    fn local_model_dir_is_available() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!tts_model_available(dir.path().to_str().unwrap()));
        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        assert!(tts_model_available(dir.path().to_str().unwrap()));
    }

    #[test]
    fn sizes_and_fraction() {
        assert_eq!(human_size(350 * MB), "350 MB");
        assert_eq!(human_size(1), "1 MB");
        assert_eq!(human_size(2560 * MB), "2.5 GB");
        let mut p = DownloadProgress {
            total: 200,
            done: 50,
            ..Default::default()
        };
        assert_eq!(p.fraction(), 0.25);
        p.total = 0;
        assert_eq!(p.fraction(), 0.0);
    }

    #[test]
    fn progress_spans_all_files() {
        use hf_hub::api::Progress;
        let seen = Mutex::new(Vec::new());
        let cb = |p: &DownloadProgress| seen.lock().unwrap().push(p.done);
        let mut r = Reporter {
            state: DownloadProgress {
                total: 300,
                ..Default::default()
            },
            base: 0,
            current: 0,
            on_progress: Some(&cb),
            next_log: 10,
            last_call: None,
        };
        let mut h = &mut r;
        h.init(100, "a");
        h.update(100);
        h.finish();
        r.base += 100;
        // second file: interrupted after 50 bytes, resumed from there
        let mut h = &mut r;
        h.init(200, "b");
        h.update(50);
        h.init(200, "b");
        h.update(50);
        h.update(150);
        h.finish();
        assert_eq!(r.state.done, 300);
        let seen = seen.into_inner().unwrap();
        assert_eq!(seen.first(), Some(&100));
        assert_eq!(seen.last(), Some(&300));
        assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{seen:?}");
    }

    #[test]
    fn readme_and_dotfiles_are_skipped() {
        assert!(needed_file("model.safetensors"));
        assert!(needed_file("speech_tokenizer/config.json"));
        assert!(!needed_file("README.md"));
        assert!(!needed_file(".gitattributes"));
    }

    #[test]
    fn snapshot_completeness() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!snapshot_looks_complete(dir.path()));
        std::fs::write(dir.path().join("model.safetensors"), b"").unwrap();
        std::fs::write(dir.path().join("tokenizer.json"), b"").unwrap();
        std::fs::create_dir(dir.path().join("speech_tokenizer")).unwrap();
        std::fs::write(dir.path().join("speech_tokenizer/config.json"), b"{}").unwrap();
        assert!(snapshot_looks_complete(dir.path()));
    }
}
