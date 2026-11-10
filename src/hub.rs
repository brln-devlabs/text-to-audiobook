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

/// Directory with config.json, tokenizer and safetensors of a TTS model.
///
/// `model` is either a local directory or a Hugging Face model id; for ids
/// a complete snapshot in the cache is used as is, otherwise the snapshot is
/// downloaded into the cache.
pub fn resolve_tts_model_dir(model: &str) -> Result<PathBuf> {
    let local = Path::new(model);
    if local.join("config.json").is_file() {
        return Ok(local.to_path_buf());
    }

    let cache = hf_hub::Cache::from_env();
    if let Some(config) = cache.model(model.to_string()).get("config.json")
        && let Some(dir) = config.parent()
        && snapshot_looks_complete(dir)
    {
        log::debug!("Using cached TTS model at {}", dir.display());
        return Ok(dir.to_path_buf());
    }

    log::info!("Downloading TTS model {model} into the Hugging Face cache ...");
    let api = hf_hub::api::sync::ApiBuilder::from_env()
        .with_token(hf_token().or_else(|| cache.token()))
        .with_progress(false)
        .build()
        .context("creating Hugging Face client")?;
    let repo = api.model(model.to_string());
    let info = repo
        .info()
        .with_context(|| format!("listing files of {model} (set HF_TOKEN if needed)"))?;
    let mut config_path: Option<PathBuf> = None;
    let n = info.siblings.len();
    for (i, s) in info.siblings.iter().enumerate() {
        let name = &s.rfilename;
        if name.starts_with('.') || name.ends_with(".md") {
            continue;
        }
        log::info!("  [{}/{n}] {name}", i + 1);
        let p = repo
            .get(name)
            .with_context(|| format!("downloading {model}/{name}"))?;
        if name == "config.json" {
            config_path = Some(p);
        }
    }
    let config = config_path.with_context(|| format!("{model} has no config.json"))?;
    Ok(config
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(".")))
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
