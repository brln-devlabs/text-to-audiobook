# License Information

This project incorporates several open-source libraries. Below is a list of the direct dependencies and their licenses. The complete dependency tree with exact versions is pinned in [`Cargo.lock`](../Cargo.lock).

## Project License

text to audiobook is provided under the [AGPL License](../LICENSE).

---

## Third-Party Dependencies

### Machine Learning

| Crate | Version | License | Purpose |
|---|---|---|---|
| [candle-core](https://github.com/huggingface/candle) | 0.9 | MIT OR Apache-2.0 | tensor library (CPU, CUDA, Metal) |
| [qwen_tts](https://github.com/danielclough/qwen3-tts-rs) | 0.1 | MIT OR Apache-2.0 | Qwen3-TTS implementation for Candle |
| [llama-cpp-2](https://github.com/utilityai/llama-cpp-rs) | 0.1 | MIT OR Apache-2.0 | Rust bindings for llama.cpp |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (bundled by llama-cpp-sys-2) | — | MIT | GGUF inference for the Llama helper model |
| [tokenizers](https://github.com/huggingface/tokenizers) (via qwen_tts) | 0.21 | Apache-2.0 | text tokenizer of Qwen3-TTS |
| [hf-hub](https://github.com/huggingface/hf-hub) | 0.4 | Apache-2.0 | Hugging Face downloads and cache |

### Audio

| Crate | Version | License | Purpose |
|---|---|---|---|
| [hound](https://github.com/ruuda/hound) | 3.5 | Apache-2.0 | WAV reading and writing |
| [mp3lame-encoder](https://github.com/DoumanAsh/mp3lame-encoder) / mp3lame-sys | 0.2 | LGPL-3.0 | MP3 export (bundles the LAME encoder, LGPL) |
| [rustfft](https://github.com/ejmahler/RustFFT) | 6 | MIT OR Apache-2.0 | FFT for the quality audit and MFCCs |
| [id3](https://codeberg.org/polyfloyd/rust-id3) | 1 | MIT | ID3 tags in chunk WAVs |

### Application

| Crate | Version | License | Purpose |
|---|---|---|---|
| [eframe / egui](https://github.com/emilk/egui) | 0.37 | MIT OR Apache-2.0 | desktop GUI |
| [rfd](https://github.com/PolyMeilex/rfd) | 0.17 | MIT | file dialogs |
| [rusqlite](https://github.com/rusqlite/rusqlite) (bundled SQLite) | 0.37 | MIT (SQLite: public domain) | SQLite storage backend |
| [clap](https://github.com/clap-rs/clap) | 4 | MIT OR Apache-2.0 | command line |
| [serde](https://github.com/serde-rs/serde), [serde_json](https://github.com/serde-rs/json) | 1 | MIT OR Apache-2.0 | settings and cache files |
| [regex](https://github.com/rust-lang/regex) | 1 | MIT OR Apache-2.0 | text cleaning |
| [ureq](https://github.com/algesten/ureq) | 2 | MIT OR Apache-2.0 | model download |
| [chrono](https://github.com/chronotope/chrono) | 0.4 | MIT OR Apache-2.0 | log timestamps, tag year |
| [indexmap](https://github.com/indexmap-rs/indexmap) | 2 | Apache-2.0 OR MIT | ordered maps |
| [log](https://github.com/rust-lang/log), [anyhow](https://github.com/dtolnay/anyhow), [thiserror](https://github.com/dtolnay/thiserror) | — | MIT OR Apache-2.0 | logging and errors |

### Testing

| Crate | Version | License | Purpose |
|---|---|---|---|
| [tempfile](https://github.com/Stebalien/tempfile) | 3 | MIT OR Apache-2.0 | temporary directories in tests |

---

## Models

| Model | License |
|---|---|
| [Qwen3-TTS](https://huggingface.co/Qwen) (0.6B / 1.7B Base) | Apache-2.0 |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (GGUF by bartowski) | Llama 3.2 Community License Agreement |

Models are downloaded at runtime and are not part of this repository or its binaries.

---

## License Summary

- Nearly all crates use permissive licenses (MIT, Apache-2.0, BSD, Zlib, ISC, Unicode-3.0).
- The MP3 encoder (LAME, via mp3lame-sys/mp3lame-encoder) is LGPL. It is linked into the AGPL-licensed program, whose complete source is available, which satisfies the LGPL.

## Obtaining Full License Texts

1. Visit the project's repository (links above)
2. Inspect the crate sources in `~/.cargo/registry/src/` after a build
3. Generate a full report, e.g. with [`cargo-about`](https://github.com/EmbarkStudios/cargo-about) or [`cargo-license`](https://github.com/onur/cargo-license)

---

## Compliance Notes

- Modifications must retain copyright and license notices
- No warranty is provided for any third-party library
- Check the model licenses before distributing generated audio commercially

For questions regarding licensing, please contact: admin@seccodesmith.pl

---

*Last Updated: 2026-11-10*
