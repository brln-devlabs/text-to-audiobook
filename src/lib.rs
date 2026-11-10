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

//! Turn Markdown texts into audiobooks.
//!
//! Stage 1 cleans and chunks the text and lets a small Llama model tag
//! emotions, language and chapters ([`llm`]). Stage 2 speaks every segment
//! with Qwen3-TTS voice cloning ([`tts`]). Stage 3 audits the chunks and
//! stage 4 joins them into WAV/MP3 files ([`audio`], [`pipeline`]).

pub mod audio;
pub mod cancel;
pub mod chunker;
pub mod config;
#[cfg(feature = "gui")]
pub mod gui;
pub mod hub;
pub mod llm;
pub mod logging;
pub mod markdown_cleaner;
pub mod natsort;
pub mod pipeline;
pub mod pyval;
pub mod segment;
pub mod storage;
pub mod tts;
