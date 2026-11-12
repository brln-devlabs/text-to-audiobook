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

//! Qwen3-TTS backend on Candle (crate `qwen_tts`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context, Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use qwen_tts::model::Model;
use qwen_tts::model::loader::{LoaderConfig, ModelLoader};
use qwen_tts::nn::generation::Output;
use qwen_tts::nn::generation_options::GenerationOptions;

use super::{TtsBackend, Wave};
use crate::audio::{read_wav, resample};

/// Sequences longer than this are decoded in chunks (less VRAM).
const CHUNK_THRESHOLD: usize = 512;
const CHUNK_SIZE: usize = 256;
const LEFT_CONTEXT: usize = 32;

pub struct QwenBackend {
    model: Model,
    device: Device,
    /// Speaker embeddings per reference file (path, mtime).
    speakers: HashMap<PathBuf, (Option<SystemTime>, Tensor)>,
}

fn pick_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        match Device::cuda_if_available(0) {
            Ok(d) if d.is_cuda() => return d,
            Ok(_) => {}
            Err(e) => log::warn!("CUDA not usable: {e}"),
        }
    }
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(d) => return d,
            Err(e) => log::warn!("Metal not usable: {e}"),
        }
    }
    Device::Cpu
}

/// Precision from `TTS_DTYPE`; bf16 falls back to f16 on GPUs without it.
/// The CPU always runs in f32.
fn resolve_dtype(device: &Device, name: &str) -> DType {
    if device.is_cpu() {
        return DType::F32;
    }
    match name.trim().to_lowercase().as_str() {
        "float32" => DType::F32,
        "float16" => DType::F16,
        "bfloat16" => {
            let probe = Tensor::ones((2, 2), DType::BF16, device).and_then(|t| t.matmul(&t));
            match probe {
                Ok(_) => DType::BF16,
                Err(_) => {
                    log::warn!("GPU has no bfloat16 support, falling back to float16");
                    DType::F16
                }
            }
        }
        other => {
            log::warn!("Unknown TTS_DTYPE '{other}', using bfloat16");
            DType::BF16
        }
    }
}

impl QwenBackend {
    /// Load a Base model (voice cloning).
    pub fn load(model_name: &str, dtype_name: &str) -> Result<Self> {
        let b = Self::load_any(model_name, dtype_name)?;
        if !b.model.is_base_model() {
            bail!("{model_name} is not a Base model; voice cloning needs a *-Base checkpoint");
        }
        Ok(b)
    }

    /// Load a VoiceDesign model (voice from a description).
    pub fn load_voice_design(model_name: &str, dtype_name: &str) -> Result<Self> {
        let b = Self::load_any(model_name, dtype_name)?;
        if !b.model.is_voice_design_model() {
            bail!(
                "{model_name} is not a VoiceDesign model; a voice from a description needs \
                 Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
            );
        }
        Ok(b)
    }

    /// Load `model_name` (Hugging Face id or local directory).
    fn load_any(model_name: &str, dtype_name: &str) -> Result<Self> {
        let dir = crate::hub::resolve_tts_model_dir(model_name)?;
        let device = pick_device();
        let dtype = resolve_dtype(&device, dtype_name);
        log::info!(
            "Loading TTS model: {model_name} ({}, {dtype:?})",
            if device.is_cpu() { "CPU" } else { "GPU" }
        );
        let loader = ModelLoader::from_local_dir(&dir)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("reading model config in {}", dir.display()))?;
        let cfg = LoaderConfig {
            dtype,
            ..Default::default()
        };
        let model = loader
            .load_tts_model(&device, &cfg)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("loading {model_name}"))?;
        if !model.has_text_processor() {
            bail!("{model_name}: no text tokenizer found in {}", dir.display());
        }
        Ok(Self {
            model,
            device,
            speakers: HashMap::new(),
        })
    }

    fn speaker_embedding(&mut self, ref_audio: &Path) -> Result<Tensor> {
        let mtime = std::fs::metadata(ref_audio).and_then(|m| m.modified()).ok();
        if let Some((t, emb)) = self.speakers.get(ref_audio)
            && *t == mtime
        {
            return Ok(emb.clone());
        }
        let audio = read_wav(ref_audio)?;
        let target = self.model.speaker_encoder_sample_rate();
        let mut samples = audio.to_mono();
        if audio.sample_rate != target {
            samples = resample(&samples, audio.sample_rate, target);
        }
        let n = samples.len();
        let t = Tensor::from_vec(samples, (n,), &self.device)?;
        let emb = self.model.extract_speaker_embedding(&t)?;
        self.speakers
            .insert(ref_audio.to_path_buf(), (mtime, emb.clone()));
        Ok(emb)
    }

    fn run(
        &self,
        text: &str,
        options: GenerationOptions,
        temperature: f64,
        top_p: f64,
        seed: u64,
    ) -> Result<Wave> {
        qwen_tts::nn::mt_rng::set_seed(seed);
        let ids = self.model.tokenize_text(text).map_err(|e| anyhow!(e))?;
        let n = ids.len();
        let text_ids = Tensor::from_vec(ids, (1, n), &self.device)?;
        let defaults = self.model.generate_defaults();
        let sampling = defaults.merge(
            None,
            None,
            Some(top_p),
            Some(temperature),
            None,
            None,
            None,
            None,
            None,
        );
        let max_tokens = defaults.effective_max_tokens(None);
        let output = self
            .model
            .model()
            .generate(&text_ids, &options, max_tokens, &sampling)?;
        let audio = self.decode(&output)?;
        let samples: Vec<f32> = audio.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        Ok(Wave {
            samples,
            sample_rate: self.model.sample_rate() as u32,
        })
    }

    /// Codes -> waveform, without the EOS step, chunked for long outputs.
    fn decode(&self, output: &Output) -> Result<Tensor> {
        let steps = output.codes.dim(1)?;
        if steps == 0 {
            bail!("Generation produced no audio (0 steps)");
        }
        let len = output
            .effective_lengths
            .as_ref()
            .and_then(|l| l.first().copied())
            .unwrap_or(steps)
            .min(steps);
        if len == 0 {
            bail!("Generation produced no audio (0 steps)");
        }
        let codes = output.codes.i((0..1, 0..len, ..))?;
        let tokenizer = self
            .model
            .audio_tokenizer()
            .ok_or_else(|| anyhow!("speech tokenizer not loaded (speech_tokenizer/ missing?)"))?;
        Ok(if len > CHUNK_THRESHOLD {
            tokenizer.chunked_decode(&codes, CHUNK_SIZE, LEFT_CONTEXT)?
        } else {
            tokenizer.decode(&codes)?
        })
    }
}

impl TtsBackend for QwenBackend {
    fn generate_voice_clone(
        &mut self,
        text: &str,
        language: &str,
        ref_audio: &Path,
        temperature: f64,
        top_p: f64,
        seed: u64,
    ) -> Result<Wave> {
        let emb = self.speaker_embedding(ref_audio)?;
        let emb = if emb.dims().len() == 1 {
            emb.unsqueeze(0)?
        } else {
            emb
        };
        let options = GenerationOptions {
            language: language.to_lowercase(),
            speaker_embed: Some(emb),
            x_vector_only_mode: true,
            non_streaming_mode: false,
            ..Default::default()
        };
        self.run(text, options, temperature, top_p, seed)
    }

    fn generate(
        &mut self,
        text: &str,
        language: &str,
        temperature: f64,
        top_p: f64,
        seed: u64,
    ) -> Result<Wave> {
        let options = GenerationOptions {
            language: language.to_lowercase(),
            ..Default::default()
        };
        self.run(text, options, temperature, top_p, seed)
    }

    fn count_tokens(&self, text: &str) -> Option<usize> {
        self.model.text_processor().map(|p| p.tokenize(text).len())
    }

    fn generate_voice_design(
        &mut self,
        text: &str,
        language: &str,
        description: &str,
        temperature: f64,
        top_p: f64,
        seed: u64,
    ) -> Result<Wave> {
        let processor = self
            .model
            .text_processor()
            .ok_or_else(|| anyhow!("no text tokenizer loaded"))?;
        let ids = processor.tokenize_instruct(description);
        let n = ids.len();
        let instruct_ids = Tensor::from_vec(ids, (1, n), &self.device)?;
        // As qwen_tts' generate_voice_design_full: the description is the
        // instruction, non-streaming prompt layout.
        let options = GenerationOptions {
            language: language.to_lowercase(),
            instruct: Some(description.to_string()),
            instruct_ids: Some(instruct_ids),
            non_streaming_mode: true,
            ..Default::default()
        };
        self.run(text, options, temperature, top_p, seed)
    }
}
