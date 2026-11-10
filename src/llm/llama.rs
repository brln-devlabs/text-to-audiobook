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

//! llama.cpp backend (GGUF models) for [`super::LlmBackend`].
//!
//! Plain text completion with the same defaults as llama-cpp-python's
//! `Llama.__call__`: BOS added, special tokens parsed, top-k 40, top-p 0.95,
//! min-p 0.05, temperature 0.8, random seed, 4096 tokens of context.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::LazyLock;

use anyhow::{Context, Result, anyhow, bail};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;

use super::LlmBackend;

const N_CTX: u32 = 4096;
const N_BATCH: usize = 512;

/// llama.cpp may only be initialised once per process.
static BACKEND: LazyLock<Result<LlamaBackend, String>> = LazyLock::new(|| {
    // Keep llama.cpp quiet (verbose=False in the Python version).
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
    LlamaBackend::init().map_err(|e| e.to_string())
});

fn backend() -> Result<&'static LlamaBackend> {
    BACKEND
        .as_ref()
        .map_err(|e| anyhow!("llama.cpp backend init failed: {e}"))
}

pub struct LlamaCppBackend {
    model: LlamaModel,
    threads: i32,
}

impl LlamaCppBackend {
    /// Load a GGUF model. `gpu_layers`: 0 = CPU only, -1 = all layers.
    pub fn load(path: &Path, gpu_layers: i32) -> Result<Self> {
        let backend = backend()?;
        let n_gpu = if gpu_layers < 0 {
            1000
        } else {
            gpu_layers as u32
        };
        let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu);
        let model = LlamaModel::load_from_file(backend, path, &params)
            .with_context(|| format!("loading {}", path.display()))?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .clamp(1, 16);
        Ok(Self { model, threads })
    }
}

fn random_seed() -> u32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    (nanos ^ (nanos >> 32)) as u32
}

/// Byte offset of the earliest stop string in `text`, if any.
fn first_stop(text: &str, stops: &[&str]) -> Option<usize> {
    stops
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s))
        .min()
}

impl LlmBackend for LlamaCppBackend {
    fn complete(&mut self, prompt: &str, max_tokens: usize, stops: &[&str]) -> Result<String> {
        let backend = backend()?;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(N_CTX))
            .with_n_batch(N_BATCH as u32)
            .with_n_threads(self.threads)
            .with_n_threads_batch(self.threads);
        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .context("creating llama.cpp context")?;
        let vocab = self.model.vocab();

        let tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        if tokens.is_empty() {
            bail!("empty prompt");
        }
        if tokens.len() >= N_CTX as usize {
            bail!(
                "Requested tokens ({}) exceed context window of {N_CTX}",
                tokens.len()
            );
        }
        let max_tokens = max_tokens.min(N_CTX as usize - tokens.len());

        // Evaluate the prompt in batches; logits only for its last token.
        let mut batch = LlamaBatch::new(N_BATCH, 1);
        let last = tokens.len() - 1;
        for (start, piece) in tokens
            .chunks(N_BATCH)
            .enumerate()
            .map(|(i, c)| (i * N_BATCH, c))
        {
            batch.clear();
            for (k, tok) in piece.iter().enumerate() {
                let pos = start + k;
                batch.add(*tok, pos as i32, &[0], pos == last)?;
            }
            ctx.decode(&mut batch)
                .context("llama.cpp decode (prompt)")?;
        }

        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::top_k(40),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::min_p(0.05, 1),
            LlamaSampler::temp(0.8),
            LlamaSampler::dist(random_seed()),
        ]);

        let mut out: Vec<u8> = Vec::new();
        let mut logits_idx = batch.n_tokens() - 1;
        for n_cur in (tokens.len() as i32..).take(max_tokens) {
            let token = sampler.sample(&ctx, logits_idx);
            if vocab.is_eog(token) {
                break;
            }
            out.extend(vocab.token_to_piece(token, false, None));
            let text = String::from_utf8_lossy(&out);
            if let Some(pos) = first_stop(&text, stops) {
                return Ok(text[..pos].to_string());
            }
            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            ctx.decode(&mut batch).context("llama.cpp decode")?;
            logits_idx = 0;
        }
        let text = String::from_utf8_lossy(&out).to_string();
        Ok(match first_stop(&text, stops) {
            Some(pos) => text[..pos].to_string(),
            None => text,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::first_stop;

    #[test]
    fn stops_cut_at_earliest_match() {
        assert_eq!(
            first_stop("abc\n\nJSON: x</s>", &["</s>", "\n\nJSON:"]),
            Some(3)
        );
        assert_eq!(first_stop("abc", &["</s>"]), None);
    }
}
