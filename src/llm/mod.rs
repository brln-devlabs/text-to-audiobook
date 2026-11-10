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

//! Stage 1 helper LLM (Llama 3.2 1B via llama.cpp): emotion spans, language,
//! chapter/story titles, speaking speed and file ordering.
//!
//! Every call asks for JSON only, extracts the first balanced JSON value
//! from the answer and validates it; failed calls are retried up to
//! `LLM_MAX_RETRIES` times. Without a loaded model ("mock mode") every
//! method returns a neutral default, which keeps the pipeline usable for
//! tests and machines without the model.

#[cfg(feature = "llm")]
pub mod llama;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};

use crate::cancel::{StopFlag, stopped};
use crate::chunker::{Chunk, split_to_chunks, word_count};
use crate::config::Settings;
use crate::natsort::natsorted;
use crate::pyval;
use crate::segment::{ChunkMeta, EmotionSpan, Segment};
use crate::storage::create_storage;

/// Text completion as provided by llama.cpp (`Llama.__call__` in Python).
pub trait LlmBackend: Send {
    /// Complete `prompt` with at most `max_tokens` new tokens. The answer is
    /// cut before the first occurrence of any string in `stops`.
    fn complete(&mut self, prompt: &str, max_tokens: usize, stops: &[&str]) -> Result<String>;
}

/// Callback for human-readable progress messages.
pub type ProgressCb<'a> = Option<&'a (dyn Fn(&str) + Sync)>;

fn report(cb: ProgressCb, msg: &str) {
    log::info!("{msg}");
    if let Some(cb) = cb {
        cb(msg);
    }
}

pub const EMOTIONS: &[&str] = &[
    // Basic / core emotions
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    // Calm / low-arousal positive
    "calm",
    "content",
    "relieved",
    "serene",
    "peaceful",
    "tranquil",
    "satisfied",
    "relaxed",
    // High-arousal positive
    "ecstatic",
    "elated",
    "euphoric",
    "joyful",
    "excited",
    "thrilled",
    "exhilarated",
    "enthusiastic",
    "playful",
    "amused",
    "cheerful",
    "delighted",
    "gleeful",
    "jubilant",
    // Hope / aspiration
    "hopeful",
    "inspired",
    "optimistic",
    "motivated",
    "determined",
    "confident",
    "empowered",
    "eager",
    "anticipating",
    // Affection / connection
    "affectionate",
    "loving",
    "tender",
    "compassionate",
    "empathetic",
    "sympathetic",
    "warm",
    "nurturing",
    "protective",
    "trusting",
    "admiring",
    "grateful",
    "appreciative",
    // Pride / accomplishment
    "proud",
    "accomplished",
    "triumphant",
    "victorious",
    "validated",
    // Sadness spectrum
    "melancholic",
    "sorrowful",
    "grief_stricken",
    "heartbroken",
    "disappointed",
    "dejected",
    "despondent",
    "hopeless",
    "despairing",
    "miserable",
    "gloomy",
    "wistful",
    "nostalgic",
    "homesick",
    "regretful",
    // Loneliness / isolation
    "lonely",
    "isolated",
    "abandoned",
    "rejected",
    "excluded",
    "alienated",
    // Anger spectrum
    "frustrated",
    "irritated",
    "annoyed",
    "agitated",
    "resentful",
    "bitter",
    "indignant",
    "outraged",
    "furious",
    "enraged",
    "hostile",
    "contemptuous",
    "spiteful",
    "vengeful",
    // Fear / anxiety spectrum
    "tense",
    "anxious",
    "worried",
    "nervous",
    "uneasy",
    "apprehensive",
    "panicked",
    "terrified",
    "horrified",
    "dread",
    "paranoid",
    "insecure",
    "vulnerable",
    "intimidated",
    // Shame / guilt family
    "guilty",
    "ashamed",
    "embarrassed",
    "humiliated",
    "remorseful",
    "self_conscious",
    "awkward",
    "mortified",
    // Envy / jealousy
    "envious",
    "jealous",
    "covetous",
    "possessive",
    // Disgust spectrum
    "repulsed",
    "revolted",
    "appalled",
    "horrified",
    "contempt",
    // Surprise spectrum
    "astonished",
    "amazed",
    "shocked",
    "stunned",
    "bewildered",
    "dumbfounded",
    "startled",
    // Confusion / cognitive
    "confused",
    "perplexed",
    "puzzled",
    "uncertain",
    "doubtful",
    "skeptical",
    "suspicious",
    "curious",
    "intrigued",
    "fascinated",
    "contemplative",
    "pensive",
    "reflective",
    "focused",
    "absorbed",
    // Low-energy / disengaged
    "bored",
    "apathetic",
    "indifferent",
    "numb",
    "detached",
    "dissociated",
    "exhausted",
    "weary",
    "fatigued",
    "drained",
    "listless",
    "lethargic",
    "sluggish",
    // Overwhelmed
    "overwhelmed",
    "stressed",
    "burdened",
    "conflicted",
    "torn",
    "ambivalent",
    "restless",
    // Mischievous / complex social
    "mischievous",
    "smug",
    "sarcastic",
    "defiant",
    "rebellious",
    "stubborn",
    "impatient",
    "flirtatious",
    "sentimental",
    "bittersweet",
    // Awe / transcendent
    "awestruck",
    "reverent",
    "humbled",
    "moved",
    "touched",
    "vulnerable_open",
];

/// Languages Qwen3-TTS accepts.
pub const SUPPORTED_LANGUAGES: &[&str] = &[
    "english",
    "chinese",
    "spanish",
    "french",
    "german",
    "italian",
    "portuguese",
    "russian",
    "japanese",
    "korean",
];
pub const DEFAULT_LANGUAGE: &str = "english";

pub const SPEAKING_SPEEDS: &[&str] = &["slow", "normal", "fast"];
pub const DEFAULT_SPEAKING_SPEED: &str = "normal";

const TEXT_STOPS: &[&str] = &["\n\nText:", "\n\nJSON:", "</s>", "<|eot_id|>"];

/// Chapter information found by [`LlmNormalizer::detect_chapter_info`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChapterInfo {
    pub chapter_number: Option<i64>,
    pub chapter_title: Option<String>,
}

/// File name plus a short content preview, input of [`LlmNormalizer::sort_files`].
#[derive(Debug, Clone)]
pub struct FilePreview {
    pub name: String,
    pub preview: String,
}

/// Result of the legacy single-shot [`LlmNormalizer::normalize`].
#[derive(Debug, Clone, PartialEq)]
pub struct Normalized {
    pub normalized_text: String,
    pub emotion: String,
    pub detected_emotions: Vec<String>,
    pub language: String,
}

/// Extract the first balanced JSON value delimited by `opener`/`closer`
/// from messy LLM output, skipping characters inside string literals.
pub fn extract_json(text: &str, opener: char, closer: char) -> Option<Value> {
    let text = text.trim();
    let mut search_from = 0;
    while let Some(rel) = text[search_from..].find(opener) {
        let start = search_from + rel;
        let mut depth = 0i64;
        let mut in_string = false;
        let mut escape = false;
        for (off, ch) in text[start..].char_indices() {
            if in_string {
                if escape {
                    escape = false;
                } else if ch == '\\' {
                    escape = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            if ch == '"' {
                in_string = true;
            } else if ch == opener {
                depth += 1;
            } else if ch == closer {
                depth -= 1;
                if depth == 0 {
                    let end = start + off + ch.len_utf8();
                    if let Ok(v) = serde_json::from_str(&text[start..end]) {
                        return Some(v);
                    }
                    break;
                }
            }
        }
        search_from = start + opener.len_utf8();
    }
    None
}

/// Languages plausible for the Unicode scripts in `text`. Filters LLM
/// hallucinations such as labelling pure ASCII text as Japanese.
pub fn script_candidates(text: &str) -> BTreeSet<&'static str> {
    let (mut latin, mut cyrillic, mut kana, mut hangul, mut han) = (0, 0, 0, 0, 0);
    for ch in text.chars() {
        let cp = ch as u32;
        match cp {
            0x0041..=0x024F => latin += 1,
            0x0400..=0x04FF => cyrillic += 1,
            0x3040..=0x30FF => kana += 1,
            0xAC00..=0xD7AF => hangul += 1,
            0x4E00..=0x9FFF => han += 1,
            _ => {}
        }
    }
    let set = |langs: &[&'static str]| langs.iter().copied().collect::<BTreeSet<_>>();
    if hangul > 5 {
        return set(&["korean"]);
    }
    if kana > 5 {
        return set(&["japanese"]);
    }
    if han > 20 && kana == 0 && hangul == 0 {
        return set(&["chinese"]);
    }
    if cyrillic > 20 {
        return set(&["russian"]);
    }
    if latin > 20 {
        return set(&[
            "english",
            "spanish",
            "french",
            "german",
            "italian",
            "portuguese",
        ]);
    }
    set(SUPPORTED_LANGUAGES)
}

/// Readable chapter title from a source file name:
/// "Chapter_01.md" -> "Chapter 01", "01_prologue" -> "01 Prologue".
pub fn format_chapter_title(source_file: &str) -> String {
    let stem = Path::new(source_file)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let name = stem.replace(['_', '-'], " ");
    let words: Vec<String> = name
        .split_whitespace()
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(first) => {
                    first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase()
                }
                None => String::new(),
            }
        })
        .collect();
    let title = words.join(" ");
    if title.is_empty() {
        "Chapter".into()
    } else {
        title
    }
}

/// Split at whitespace runs preceded by `.!?` and followed by `A-Z`, `"` or
/// `'` (Python: `re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)`).
fn split_emotion_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut prev: Option<char> = None;
    let mut iter = text.char_indices().peekable();
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
            let next = text[end..].chars().next();
            if matches!(next, Some(n) if n.is_ascii_uppercase() || n == '"' || n == '\'') {
                out.push(text[start..i].to_string());
                start = end;
            }
            prev = text[..end].chars().next_back();
            continue;
        }
        prev = Some(c);
    }
    out.push(text[start..].to_string());
    out.into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

pub struct LlmNormalizer {
    pub model_path: PathBuf,
    llm: Option<Box<dyn LlmBackend>>,
    /// Retries per LLM call (`LLM_MAX_RETRIES`).
    pub max_retries: u32,
    /// Layers offloaded to the GPU (`LLM_GPU_LAYERS`).
    pub gpu_layers: i32,
    /// Never load a model (see [`Settings::mock_models`]).
    pub mock: bool,
}

impl LlmNormalizer {
    pub fn new(model_path: impl Into<PathBuf>, settings: &Settings) -> Self {
        Self {
            model_path: model_path.into(),
            llm: None,
            max_retries: settings.llm_max_retries,
            gpu_layers: settings.llm_gpu_layers,
            mock: settings.mock_models,
        }
    }

    /// Normalizer that talks to an existing backend (tests, other engines).
    pub fn with_backend(backend: Box<dyn LlmBackend>, settings: &Settings) -> Self {
        Self {
            model_path: PathBuf::new(),
            llm: Some(backend),
            max_retries: settings.llm_max_retries,
            gpu_layers: settings.llm_gpu_layers,
            mock: false,
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.llm.is_some()
    }

    /// Load the GGUF model (downloading it on first use). Failures are logged
    /// and leave the normalizer in mock mode, like the Python version.
    pub fn load(&mut self) {
        if self.mock {
            log::info!("Mock mode: LLM not loaded, using neutral defaults");
            return;
        }
        #[cfg(feature = "llm")]
        {
            let result = crate::hub::ensure_llm_model(&self.model_path).and_then(|path| {
                log::info!("Loading LLM from {}", path.display());
                llama::LlamaCppBackend::load(&path, self.gpu_layers)
            });
            match result {
                Ok(b) => self.llm = Some(Box::new(b)),
                Err(e) => {
                    log::warn!("Failed to load LLM: {e:#}. Using mock mode.");
                    self.llm = None;
                }
            }
        }
        #[cfg(not(feature = "llm"))]
        {
            log::warn!("llama.cpp support not compiled in (feature \"llm\"), using mock mode");
        }
    }

    pub fn unload(&mut self) {
        if self.llm.take().is_some() {
            log::info!("Unloading LLM");
        }
    }

    fn call_llm(&mut self, prompt: &str, max_tokens: usize, stops: &[&str]) -> Result<String> {
        let llm = self.llm.as_mut().ok_or_else(|| anyhow!("LLM not loaded"))?;
        Ok(llm.complete(prompt, max_tokens, stops)?.trim().to_string())
    }

    /// Run `attempt` up to `max_retries` times; logs every failure.
    fn retry<T>(
        &mut self,
        what: &str,
        mut attempt: impl FnMut(&mut Self) -> Result<T>,
    ) -> Result<T> {
        let max = self.max_retries.max(1);
        let mut last_error = anyhow!("no attempts made");
        for n in 1..=max {
            match attempt(self) {
                Ok(v) => return Ok(v),
                Err(e) => {
                    log::warn!("{what} attempt {n}/{max} failed: {e:#}");
                    last_error = e;
                }
            }
        }
        Err(last_error)
    }

    /// Ask for a JSON object; error if the answer has none.
    fn ask_object(&mut self, prompt: &str, max_tokens: usize, stops: &[&str]) -> Result<Value> {
        let output = self.call_llm(prompt, max_tokens, stops)?;
        match extract_json(&output, '{', '}') {
            Some(v @ Value::Object(_)) => Ok(v),
            _ => bail!("non-dict response: {output:?}"),
        }
    }

    // ------------------------------------------------------------------
    // Detection helpers
    // ------------------------------------------------------------------

    /// Recommend "slow", "normal" or "fast" for narrating `text`.
    pub fn detect_speaking_speed(&mut self, text: &str) -> String {
        if self.llm.is_none() || text.trim().is_empty() {
            return DEFAULT_SPEAKING_SPEED.into();
        }
        let sample = pyval::prefix(text, 500);
        let system_prompt = format!(
            "Analyze the text and recommend a speaking speed for narration.\n\
             Options: {}\n\
             Consider: sentence length, punctuation density, emotional indicators, complexity.\n\
             Output ONLY a JSON object with one key: speed (lowercase).\n\
             If unsure, use default: {DEFAULT_SPEAKING_SPEED}\n\
             Example: {{\"speed\": \"normal\"}}",
            SPEAKING_SPEEDS.join(", ")
        );
        let prompt = format!("{system_prompt}\n\nText:\n{sample}\n\nJSON:");
        let result = self.retry("detect_speaking_speed", |s| {
            let data = s.ask_object(&prompt, 64, TEXT_STOPS)?;
            let speed = data.get("speed").map(pyval::str).unwrap_or_default();
            let speed = speed.trim().to_lowercase();
            Ok(if SPEAKING_SPEEDS.contains(&speed.as_str()) {
                speed
            } else {
                DEFAULT_SPEAKING_SPEED.to_string()
            })
        });
        result.unwrap_or_else(|e| {
            log::warn!(
                "detect_speaking_speed failed after {} attempts: {e:#}; using default",
                self.max_retries
            );
            DEFAULT_SPEAKING_SPEED.into()
        })
    }

    /// Chapter number and title from patterns like "CHAPTER 2. The Beginning".
    pub fn detect_chapter_info(&mut self, text: &str) -> ChapterInfo {
        if self.llm.is_none() || text.trim().is_empty() {
            return ChapterInfo::default();
        }
        let sample = pyval::prefix(text, 1000);
        let system_prompt = "Extract chapter information from the text.\n\
             Look for patterns like: 'CHAPTER 2. The Beginning', 'Chapter II - Title', etc.\n\
             Output ONLY a JSON object with two keys:\n  \
             - chapter_number: extracted integer chapter number, or null if not found\n  \
             - chapter_title: extracted chapter title string, or null if not found\n\
             Example: {\"chapter_number\": 2, \"chapter_title\": \"The Beginning\"}";
        let prompt = format!("{system_prompt}\n\nText:\n{sample}\n\nJSON:");
        let result = self.retry("detect_chapter_info", |s| {
            let data = s.ask_object(&prompt, 128, TEXT_STOPS)?;
            let chapter_number = pyval::get(&data, "chapter_number").and_then(pyval::int);
            let chapter_title = pyval::get(&data, "chapter_title")
                .map(|v| pyval::str(v).trim().to_string())
                .filter(|t| !t.is_empty());
            Ok(ChapterInfo {
                chapter_number,
                chapter_title,
            })
        });
        result.unwrap_or_else(|e| {
            log::warn!(
                "detect_chapter_info failed after {} attempts: {e:#}",
                self.max_retries
            );
            ChapterInfo::default()
        })
    }

    /// Story or book title inferred from the text, if any.
    pub fn detect_story_title(&mut self, text: &str) -> Option<String> {
        if self.llm.is_none() || text.trim().is_empty() {
            return None;
        }
        let sample = pyval::prefix(text, 1000);
        let system_prompt = "Infer the story, book, or novel title from the provided text.\n\
             Look for: title pages, chapter headers, author references, or contextual clues.\n\
             Output ONLY a JSON object with one key:\n  \
             - title: the inferred title string, or null if not determinable\n\
             Example: {\"title\": \"The Fellowship of the Ring\"}";
        let prompt = format!("{system_prompt}\n\nText:\n{sample}\n\nJSON:");
        let result = self.retry("detect_story_title", |s| {
            let data = s.ask_object(&prompt, 128, TEXT_STOPS)?;
            Ok(pyval::get(&data, "title")
                .map(|v| pyval::str(v).trim().to_string())
                .filter(|t| !t.is_empty()))
        });
        result.unwrap_or_else(|e| {
            log::warn!(
                "detect_story_title failed after {} attempts: {e:#}",
                self.max_retries
            );
            None
        })
    }

    /// Language of `text`, combining the LLM with a Unicode-script check
    /// that rejects answers contradicted by the characters in the text.
    pub fn detect_language(&mut self, text: &str) -> Result<String> {
        if self.llm.is_none() || text.trim().is_empty() {
            return Ok(DEFAULT_LANGUAGE.into());
        }
        let sample = pyval::prefix(text, 1000);
        let plausible = script_candidates(sample);
        // Fast path: the script alone decides.
        if plausible.len() == 1 {
            return Ok(plausible.iter().next().unwrap().to_string());
        }
        let options = plausible.iter().copied().collect::<Vec<_>>().join(", ");
        let system_prompt = format!(
            "Identify the language of the text. Choose ONE from: {options}.\n\
             Output ONLY a JSON object with one key: language (lowercase).\n\
             If your confidence is below 80%, use the default: {DEFAULT_LANGUAGE}.\n\
             Example: {{\"language\": \"english\"}}"
        );
        let prompt = format!("{system_prompt}\n\nText:\n{sample}\n\nJSON:");
        let result = self.retry("detect_language", |s| {
            let data = s.ask_object(&prompt, 64, TEXT_STOPS)?;
            let lang = data
                .get("language")
                .map(pyval::str)
                .unwrap_or_default()
                .trim()
                .to_lowercase();
            if !SUPPORTED_LANGUAGES.contains(&lang.as_str()) {
                return Ok(DEFAULT_LANGUAGE.to_string());
            }
            if !plausible.contains(lang.as_str()) {
                bail!(
                    "LLM said {lang:?} but Unicode scripts in text only support {:?} — rejecting hallucination",
                    plausible
                );
            }
            Ok(lang)
        });
        result.map_err(|e| {
            anyhow!(
                "FATAL ERROR: detect_language failed after {} attempts: {e:#}",
                self.max_retries
            )
        })
    }

    /// Ask the LLM for the narrative reading order of chapter files. Falls
    /// back to natural sort when the model is missing or the answer is bad.
    pub fn sort_files(&mut self, files: &[FilePreview]) -> Vec<String> {
        let names_in: Vec<String> = files.iter().map(|f| f.name.clone()).collect();
        if self.llm.is_none() || files.len() <= 1 {
            return natsorted(&names_in);
        }
        let listing = files
            .iter()
            .map(|f| {
                format!(
                    "- {}: \"{}\"",
                    f.name,
                    pyval::prefix(&f.preview, 200).trim()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let system_prompt = "You are sorting book chapter files into correct reading order.\n\
             Rules:\n  \
             1. Files are shown in their default order. Only reorder when the names or      \
             content clearly indicate a different narrative sequence.\n  \
             2. Recognise ordinal words: 'first'<'second'<'third';      \
             'one'<'two'<'three'<'four'<'five'<'six'<'seven'<'eight'<'nine'<'ten' etc.\n  \
             3. Recognise structural keywords: 'prologue'/'preface'/'introduction' come      \
             BEFORE chapters; 'epilogue'/'afterword'/'appendix' come AFTER chapters.\n  \
             4. Output ONLY a JSON array of ALL file names in reading order. No commentary.\n\
             Example:\n  \
             Input:\n    \
             - Chapter_Two.md: \"The morning was cold...\"\n    \
             - Prologue.md: \"Long before the war began...\"\n    \
             - Chapter_One.md: \"She stepped off the train...\"\n  \
             Output: [\"Prologue.md\", \"Chapter_One.md\", \"Chapter_Two.md\"]";
        let prompt = format!("{system_prompt}\n\nInput:\n{listing}\n\nOutput:");
        let result = (|| -> Result<Vec<String>> {
            let output = self.call_llm(
                &prompt,
                2048,
                &["\n\nInput:", "\n\nOutput:", "</s>", "<|eot_id|>"],
            )?;
            let Some(Value::Array(ordered)) = extract_json(&output, '[', ']') else {
                bail!("LLM did not return a JSON array");
            };
            let mut valid: Vec<String> = Vec::new();
            for n in ordered.iter().filter_map(Value::as_str) {
                if names_in.iter().any(|x| x == n) && !valid.iter().any(|x| x == n) {
                    valid.push(n.to_string());
                }
            }
            let missing: Vec<&String> = names_in.iter().filter(|n| !valid.contains(n)).collect();
            if !missing.is_empty() {
                log::warn!(
                    "LLM sort missed {} files; appending in natsort order",
                    missing.len()
                );
                valid.extend(natsorted(missing));
            }
            if valid.is_empty() {
                bail!("no valid file names returned");
            }
            log::info!("LLM ordered {} files", valid.len());
            Ok(valid)
        })();
        result.unwrap_or_else(|e| {
            log::warn!("File sort failed: {e:#}; falling back to natsort");
            natsorted(&names_in)
        })
    }

    // ------------------------------------------------------------------
    // Step A: emotion analysis
    // ------------------------------------------------------------------

    /// Split `text` into spans of one emotion each.
    ///
    /// The text is pre-split into numbered sentences and the LLM only
    /// returns integer index ranges, so it cannot alter the prose.
    pub fn analyze_emotions(&mut self, text: &str) -> Result<Vec<EmotionSpan>> {
        let neutral = || {
            vec![EmotionSpan {
                text: text.to_string(),
                emotion: "neutral".into(),
            }]
        };
        if self.llm.is_none() {
            return Ok(neutral());
        }
        let sentences = split_emotion_sentences(text);
        let n_sent = sentences.len();
        if n_sent <= 1 {
            return Ok(neutral());
        }

        let numbered = sentences
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{}. {s}", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let system_prompt = format!(
            "You tag an audiobook passage with emotion spans by SENTENCE NUMBER.\n\
             Allowed emotions (pick exactly one): {emotions}.\n\
             The Input is a numbered list of sentences (1..N). Group consecutive \
             sentences that share the same emotion into spans.\n\
             Output a JSON array of objects with these EXACT keys:\n  \
             \"Start\"   — INTEGER sentence number where the span begins (1-based).\n  \
             \"Stop\"    — INTEGER sentence number where the span ends (inclusive).\n  \
             \"Emotion\" — one allowed emotion value.\n\
             HARD RULES:\n  \
             1. Start and Stop are integers in [1, {n_sent}], with Start <= Stop.\n  \
             2. Spans MUST be in order: each span's Start = previous span's Stop + 1.\n  \
             3. The first span's Start = 1; the last span's Stop = {n_sent}. NO gaps, NO overlaps.\n  \
             4. If unsure, output ONE object: {{\"Start\": 1, \"Stop\": {n_sent}, \"Emotion\": \"neutral\"}}.\n  \
             5. Output ONLY the JSON array — no prose, no markdown, no comments.\n\
             Schema example (numbers are placeholders):\n  \
             [{{\"Start\": 1, \"Stop\": 3, \"Emotion\": \"neutral\"}}, \
             {{\"Start\": 4, \"Stop\": 5, \"Emotion\": \"tense\"}}]",
            emotions = EMOTIONS.join(", ")
        );
        let prompt = format!("{system_prompt}\n\nInput:\n{numbered}\n\nJSON array:");

        let max = self.max_retries.max(1);
        let mut last_error = anyhow!("no attempts made");
        let mut last_raw = String::new();
        for attempt in 1..=max {
            let result = (|| -> Result<Vec<EmotionSpan>> {
                let output = self.call_llm(
                    &prompt,
                    1024,
                    &["\n\nInput:", "\n\nJSON array:", "</s>", "<|eot_id|>"],
                )?;
                last_raw = output.clone();
                let raw_head = pyval::prefix(&last_raw, 200).to_string();
                let segments = match extract_json(&output, '[', ']') {
                    Some(Value::Array(a)) if !a.is_empty() => a,
                    _ => bail!("no segments parsed | raw[:200]={raw_head:?}"),
                };
                // More spans than sentences means the LLM hallucinated.
                if segments.len() > n_sent {
                    bail!(
                        "too many spans: LLM returned {} but only {n_sent} sentence(s) available — likely hallucinated",
                        segments.len()
                    );
                }

                let mut spans: Vec<(usize, usize, String)> = Vec::new();
                let mut rejected: Vec<String> = Vec::new();
                for seg in &segments {
                    if !seg.is_object() {
                        rejected.push(format!("not-a-dict: {seg}"));
                        continue;
                    }
                    let raw_start = pyval::get(seg, "Start").or_else(|| pyval::get(seg, "start"));
                    let raw_stop = pyval::get(seg, "Stop").or_else(|| pyval::get(seg, "stop"));
                    let (Some(raw_start), Some(raw_stop)) = (raw_start, raw_stop) else {
                        rejected.push(format!("missing-range: {seg}"));
                        continue;
                    };
                    let (Some(start), Some(stop)) = (pyval::int(raw_start), pyval::int(raw_stop))
                    else {
                        rejected.push(format!("non-int-range: {seg}"));
                        continue;
                    };
                    let emotion_val = [seg.get("Emotion"), seg.get("emotion")]
                        .into_iter()
                        .flatten()
                        .find(|v| pyval::truthy(v));
                    let mut emotion = emotion_val
                        .map(pyval::str)
                        .unwrap_or_else(|| "neutral".into())
                        .trim()
                        .to_lowercase();
                    if !EMOTIONS.contains(&emotion.as_str()) {
                        emotion = "neutral".into();
                    }
                    if !(1 <= start && start <= stop && stop <= n_sent as i64) {
                        rejected.push(format!("out-of-range[1..{n_sent}]: ({start}, {stop})"));
                        continue;
                    }
                    spans.push((start as usize, stop as usize, emotion));
                }
                if spans.is_empty() {
                    bail!(
                        "no valid spans | rejected: {:?} | raw[:200]={raw_head:?}",
                        &rejected[..rejected.len().min(3)]
                    );
                }

                // Order & gap analysis (the "dead spaces" check)
                spans.sort_by_key(|s| s.0);
                let mut gaps = Vec::new();
                let mut overlaps = Vec::new();
                let mut expected = 1usize;
                for (st, sp, _) in &spans {
                    if *st > expected {
                        gaps.push((expected, st - 1));
                    } else if *st < expected {
                        overlaps.push((*st, expected - 1));
                    }
                    expected = expected.max(sp + 1);
                }
                if expected <= n_sent {
                    gaps.push((expected, n_sent));
                }
                if !overlaps.is_empty() {
                    bail!(
                        "overlapping spans at sentences {:?}",
                        &overlaps[..overlaps.len().min(3)]
                    );
                }
                if !gaps.is_empty() {
                    log::debug!(
                        "  filling {} dead-space gap(s) with neutral: {:?}",
                        gaps.len(),
                        &gaps[..gaps.len().min(5)]
                    );
                    for (gs, ge) in gaps {
                        spans.push((gs, ge, "neutral".into()));
                    }
                    spans.sort_by_key(|s| s.0);
                }

                let result: Vec<EmotionSpan> = spans
                    .into_iter()
                    .filter_map(|(st, sp, emotion)| {
                        let seg_text = sentences[st - 1..sp].join(" ").trim().to_string();
                        (!seg_text.is_empty()).then_some(EmotionSpan {
                            text: seg_text,
                            emotion,
                        })
                    })
                    .collect();
                if result.is_empty() {
                    bail!("empty result after assembly");
                }
                Ok(result)
            })();
            match result {
                Ok(r) => return Ok(r),
                Err(e) => {
                    log::warn!("analyze_emotions attempt {attempt}/{max} failed: {e:#}");
                    if !last_raw.is_empty() {
                        log::debug!(
                            "  raw LLM output[:300]: {:?}",
                            pyval::prefix(&last_raw, 300)
                        );
                    }
                    last_error = e;
                }
            }
        }
        bail!(
            "FATAL ERROR: analyze_emotions failed after {max} attempts: {last_error:#} | last_raw[:200]={:?}",
            pyval::prefix(&last_raw, 200)
        )
    }

    /// Use LLM segments only as emotion anchors and cut the final segments
    /// from the source text, so no LLM paraphrasing reaches the TTS.
    pub fn reconstruct_from_anchors(source: &str, anchors: &[EmotionSpan]) -> Vec<EmotionSpan> {
        let neutral = || {
            vec![EmotionSpan {
                text: source.to_string(),
                emotion: "neutral".into(),
            }]
        };
        let mut positions: Vec<(usize, String)> = Vec::new();
        let mut cursor = 0usize;
        for a in anchors {
            let mut head = pyval::prefix(&a.text, 60);
            let mut idx = source
                .get(cursor..)
                .and_then(|s| s.find(head))
                .map(|i| i + cursor);
            if idx.is_none() {
                head = pyval::prefix(&a.text, 30);
                idx = if head.is_empty() {
                    None
                } else {
                    source
                        .get(cursor..)
                        .and_then(|s| s.find(head))
                        .map(|i| i + cursor)
                };
            }
            let Some(idx) = idx else { continue };
            positions.push((idx, a.emotion.clone()));
            cursor = idx + head.len().max(1);
            while cursor < source.len() && !source.is_char_boundary(cursor) {
                cursor += 1;
            }
        }
        if positions.is_empty() {
            log::warn!("Could not anchor any LLM segment to source; using single neutral segment");
            return neutral();
        }
        let mut result = Vec::new();
        let head_text = source[..positions[0].0].trim();
        if !head_text.is_empty() {
            result.push(EmotionSpan {
                text: head_text.to_string(),
                emotion: "neutral".into(),
            });
        }
        for (k, (start, emotion)) in positions.iter().enumerate() {
            let end = positions.get(k + 1).map(|p| p.0).unwrap_or(source.len());
            let slice = source[*start..end].trim();
            if !slice.is_empty() {
                result.push(EmotionSpan {
                    text: slice.to_string(),
                    emotion: emotion.clone(),
                });
            }
        }
        let joined_len = result.iter().map(|s| s.text.chars().count()).sum::<usize>()
            + result.len().saturating_sub(1);
        let source_len = source.trim().chars().count();
        if joined_len < (0.9 * source_len as f64) as usize {
            log::warn!(
                "Reconstruction covered only {joined_len}/{source_len} chars; falling back to single neutral segment"
            );
            return neutral();
        }
        result
    }

    // ------------------------------------------------------------------
    // Steps C/D: location-specific edits
    // ------------------------------------------------------------------

    /// Check that every `{find, occurrence, replace}` exists in `source` and
    /// resolve it to a byte range. Errors on missing finds or overlaps.
    pub fn validate_replacements(
        source: &str,
        replacements: &[Value],
    ) -> Result<Vec<(usize, usize, String)>> {
        let mut resolved = Vec::new();
        for r in replacements {
            if !r.is_object() {
                bail!("replacement is not a dict: {r}");
            }
            let find = match r.get("find") {
                Some(Value::String(s)) if !s.is_empty() => s.as_str(),
                _ => bail!("missing/empty 'find' in {r}"),
            };
            let replace = match r.get("replace") {
                Some(Value::String(s)) => s.clone(),
                _ => bail!("missing 'replace' in {r}"),
            };
            let occurrence = match r.get("occurrence") {
                None => 1,
                Some(v) => pyval::int(v).ok_or_else(|| anyhow!("invalid 'occurrence' in {r}"))?,
            };
            if occurrence < 1 {
                bail!("occurrence must be >= 1 in {r}");
            }
            let mut start: Option<usize> = None;
            let mut cursor = 0usize;
            for _ in 0..occurrence {
                start = source
                    .get(cursor..)
                    .and_then(|s| s.find(find))
                    .map(|i| i + cursor);
                match start {
                    Some(s) => {
                        cursor = s + 1;
                        while cursor < source.len() && !source.is_char_boundary(cursor) {
                            cursor += 1;
                        }
                    }
                    None => break,
                }
            }
            let Some(start) = start else {
                bail!("'find'={find:?} occurrence #{occurrence} not present in source");
            };
            resolved.push((start, start + find.len(), replace));
        }
        resolved.sort_by_key(|r| r.0);
        for w in resolved.windows(2) {
            if w[0].1 > w[1].0 {
                bail!("overlapping replacements");
            }
        }
        Ok(resolved)
    }

    pub fn apply_replacements(source: &str, resolved: &[(usize, usize, String)]) -> String {
        let mut out = String::with_capacity(source.len());
        let mut cursor = 0;
        for (start, end, replace) in resolved {
            out.push_str(&source[cursor..*start]);
            out.push_str(replace);
            cursor = *end;
        }
        out.push_str(&source[cursor..]);
        out
    }

    fn replacement_call(
        &mut self,
        what: &str,
        text: &str,
        prompt: &str,
        stops: &[&str],
    ) -> Result<String> {
        let result = self.retry(what, |s| {
            let output = s.call_llm(prompt, 1024, stops)?;
            let data = extract_json(&output, '{', '}');
            let Some(reps) = data.as_ref().and_then(|d| d.get("replacements")) else {
                bail!("no 'replacements' key in response");
            };
            let Value::Array(reps) = reps else {
                bail!("'replacements' is not a list");
            };
            let resolved = Self::validate_replacements(text, reps)?;
            Ok(Self::apply_replacements(text, &resolved))
        });
        result.map_err(|e| {
            anyhow!(
                "FATAL ERROR: {what} failed after {} attempts: {e:#}",
                self.max_retries
            )
        })
    }

    /// Expand numbers, dates and abbreviations via location-specific edits.
    pub fn normalize_text(&mut self, text: &str) -> Result<String> {
        if self.llm.is_none() {
            return Ok(text.to_string());
        }
        let system_prompt = "You normalize text for a TTS narrator. Do NOT rewrite the passage. \
             Return ONLY the substrings that must change (abbreviations, numbers, \
             dates, times, symbols) so they read naturally aloud.\n\
             For each edit output: 'find' = the EXACT substring copied verbatim \
             from the input, 'occurrence' = which appearance (1 = first), \
             'replace' = the spoken form.\n\
             Rules: do not paraphrase; do not merge or reorder words; if nothing \
             needs changing return an empty list.\n\
             Output ONLY a JSON object with one key: replacements (a list).\n\
             Example:\n  \
             Input: \"Dr. Smith arrived at 3:45 PM in 1995. Dr. Smith smiled.\"\n  \
             Output: {\"replacements\": [\
             {\"find\": \"Dr.\", \"occurrence\": 1, \"replace\": \"Doctor\"}, \
             {\"find\": \"3:45 PM\", \"occurrence\": 1, \"replace\": \"three forty-five PM\"}, \
             {\"find\": \"1995\", \"occurrence\": 1, \"replace\": \"nineteen ninety-five\"}, \
             {\"find\": \"Dr.\", \"occurrence\": 2, \"replace\": \"Doctor\"}\
             ]}";
        let prompt = format!("{system_prompt}\n\nText:\n{text}\n\nJSON:");
        self.replacement_call("normalize_text", text, &prompt, TEXT_STOPS)
    }

    /// Resolve pronouns and elided references via location-specific edits.
    pub fn fill_context(&mut self, text: &str, prev_text: &str, next_text: &str) -> Result<String> {
        if self.llm.is_none() {
            return Ok(text.to_string());
        }
        let system_prompt = "Resolve pronouns and elided references in the SEGMENT using PREVIOUS \
             and NEXT context. Do NOT rewrite the segment. Return ONLY the \
             substrings inside SEGMENT that must change (e.g. an ambiguous pronoun \
             like 'she' replaced with the actual name from context).\n\
             For each edit output: 'find' = the EXACT substring copied verbatim \
             from SEGMENT, 'occurrence' = which appearance in SEGMENT (1 = first), \
             'replace' = the resolved form.\n\
             Rules: do not paraphrase, shorten, or reorder; if SEGMENT needs no \
             edits return an empty list.\n\
             Output ONLY a JSON object with one key: replacements (a list).\n\
             Example:\n  \
             PREVIOUS: \"Sarah opened the door.\"\n  \
             SEGMENT: \"Then she walked inside.\"\n  \
             NEXT: \"The room was dark.\"\n  \
             Output: {\"replacements\": [{\"find\": \"she\", \"occurrence\": 1, \"replace\": \"Sarah\"}]}";
        let none = |s: &str| {
            if s.is_empty() {
                "(none)".to_string()
            } else {
                s.to_string()
            }
        };
        let prompt = format!(
            "{system_prompt}\n\nPREVIOUS:\n{}\n\nSEGMENT:\n{text}\n\nNEXT:\n{}\n\nJSON:",
            none(prev_text),
            none(next_text)
        );
        self.replacement_call(
            "fill_context",
            text,
            &prompt,
            &[
                "\n\nPREVIOUS:",
                "\n\nSEGMENT:",
                "\n\nNEXT:",
                "\n\nJSON:",
                "</s>",
                "<|eot_id|>",
            ],
        )
    }

    /// Legacy single-shot normalization (text normalization + language).
    pub fn normalize(&mut self, chunk_text: &str) -> Normalized {
        if self.llm.is_none() {
            return Normalized {
                normalized_text: chunk_text.to_string(),
                emotion: "neutral".into(),
                detected_emotions: vec!["neutral".into()],
                language: DEFAULT_LANGUAGE.into(),
            };
        }
        let normalized = self.normalize_text(chunk_text).unwrap_or_else(|_| {
            log::warn!("LLM normalization failed, returning text unchanged");
            chunk_text.to_string()
        });
        let language = self.detect_language(chunk_text).unwrap_or_else(|_| {
            log::warn!("LLM language detection failed, using default");
            DEFAULT_LANGUAGE.into()
        });
        Normalized {
            normalized_text: normalized,
            emotion: "neutral".into(),
            detected_emotions: vec!["neutral".into()],
            language,
        }
    }

    // ------------------------------------------------------------------
    // Orchestrator
    // ------------------------------------------------------------------

    /// Stage 1 for one project: emotion analysis (A), re-chunking of
    /// over-long spans (B) and the final TTS segment records, each step
    /// cached in `cache_dir` so interrupted runs resume.
    ///
    /// Steps C/D (normalization and context fill) are skipped: the 1B model
    /// copies the few-shot examples instead of processing the input, so the
    /// verbatim text goes to the TTS.
    pub fn process_all(
        &mut self,
        chunks: &[Chunk],
        cache_dir: &Path,
        settings: &Settings,
        progress: ProgressCb,
        stop: Option<&StopFlag>,
    ) -> Result<Vec<Segment>> {
        let storage = create_storage(&settings.storage_backend, cache_dir)?;
        storage.init_schema()?;

        // Project language: fixed in the settings, or the majority vote over
        // the first ten chunks (single detections of the 1B model are noisy).
        let mut project_language: Option<String> = None;
        let fixed_language = settings.tts_language_normalized();
        if !chunks.is_empty() && fixed_language != "auto" {
            report(
                progress,
                &format!("[A] project language: {fixed_language} (fixed in settings)"),
            );
            project_language = Some(fixed_language);
        } else if !chunks.is_empty() {
            let n_probe = chunks.len().min(10);
            let mut votes: Vec<String> = Vec::new();
            for probe in &chunks[..n_probe] {
                match self.detect_language(&probe.text) {
                    Ok(l) => votes.push(l),
                    Err(e) => log::debug!("language probe failed on a chunk: {e:#}"),
                }
            }
            if votes.is_empty() {
                log::warn!(
                    "All {n_probe} project-language probes failed; falling back to per-chunk detection"
                );
            } else {
                let mut tally: Vec<(String, usize)> = Vec::new();
                for v in &votes {
                    match tally.iter_mut().find(|(l, _)| l == v) {
                        Some(entry) => entry.1 += 1,
                        None => tally.push((v.clone(), 1)),
                    }
                }
                // Most common; ties go to the language seen first.
                let (lang, top) = tally
                    .iter()
                    .fold(None::<&(String, usize)>, |best, item| match best {
                        Some(b) if b.1 >= item.1 => Some(b),
                        _ => Some(item),
                    })
                    .cloned()
                    .unwrap();
                let tally_str = tally
                    .iter()
                    .map(|(l, n)| format!("'{l}': {n}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                report(
                    progress,
                    &format!(
                        "[A] project language: {lang} ({top}/{} votes; tally={{{tally_str}}})",
                        votes.len()
                    ),
                );
                project_language = Some(lang);
            }
        }

        // --- Step A: per-chunk emotion analysis -------------------------
        struct Analyzed {
            text: String,
            emotion: String,
            ends_paragraph: bool,
            source_chunk: usize,
            source_file: String,
            language: String,
        }
        let mut analyzed: Vec<Analyzed> = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            if stopped(stop) {
                report(
                    progress,
                    &format!("Stop requested after {i}/{} chunks analyzed", chunks.len()),
                );
                break;
            }
            let cached = storage
                .load_chunk_meta(i)
                .and_then(|v| serde_json::from_value::<ChunkMeta>(v).ok());
            let (segs, ends_paragraph, language) = match cached {
                Some(meta) => (
                    meta.segments,
                    meta.ends_paragraph.unwrap_or(chunk.ends_paragraph),
                    meta.language
                        .filter(|l| !l.is_empty())
                        .unwrap_or_else(|| DEFAULT_LANGUAGE.into()),
                ),
                None => {
                    report(
                        progress,
                        &format!("[A] analyzing chunk {}/{}", i + 1, chunks.len()),
                    );
                    let segs = self.analyze_emotions(&chunk.text).unwrap_or_else(|_| {
                        log::warn!(
                            "LLM emotion analysis failed for chunk {i}, using neutral fallback"
                        );
                        vec![EmotionSpan {
                            text: chunk.text.clone(),
                            emotion: "neutral".into(),
                        }]
                    });
                    let language = match &project_language {
                        Some(l) => l.clone(),
                        None => {
                            let l = self.detect_language(&chunk.text).unwrap_or_else(|_| {
                                log::warn!(
                                    "LLM language detection failed for chunk {i}, using default"
                                );
                                DEFAULT_LANGUAGE.into()
                            });
                            report(progress, &format!("[A] chunk {} language: {l}", i + 1));
                            l
                        }
                    };
                    let meta = ChunkMeta {
                        segments: segs.clone(),
                        ends_paragraph: Some(chunk.ends_paragraph),
                        language: Some(language.clone()),
                    };
                    storage.save_chunk_meta(i, &serde_json::to_value(&meta)?);
                    // Legacy file in the cache root (kept for compatibility).
                    let legacy = json!({
                        "normalized_text": chunk.text,
                        "emotion": segs.first().map(|s| s.emotion.as_str()).unwrap_or("neutral"),
                        "detected_emotions": if segs.is_empty() {
                            vec!["neutral".to_string()]
                        } else {
                            segs.iter().map(|s| s.emotion.clone()).collect()
                        },
                        "language": language,
                    });
                    let legacy_path = cache_dir.join(format!("chunk_{i:03}.json"));
                    if let Err(e) = std::fs::write(&legacy_path, serde_json::to_string(&legacy)?) {
                        log::warn!("Could not write {}: {e}", legacy_path.display());
                    }
                    (segs, chunk.ends_paragraph, language)
                }
            };
            let n = segs.len();
            for (j, seg) in segs.into_iter().enumerate() {
                analyzed.push(Analyzed {
                    text: seg.text,
                    emotion: seg.emotion,
                    ends_paragraph: if j == n - 1 { ends_paragraph } else { false },
                    source_chunk: i,
                    source_file: chunk.source_file.clone(),
                    language: language.clone(),
                });
            }
        }

        // --- Step B: re-chunk over-budget segments ----------------------
        let max_words = settings.max_words_per_chunk;
        let mut rechunked: Vec<Analyzed> = Vec::new();
        let analyzed_count = analyzed.len();
        for seg in analyzed {
            if word_count(&seg.text) <= max_words {
                rechunked.push(seg);
                continue;
            }
            let sub = split_to_chunks(&seg.text, max_words, &seg.source_file);
            let n = sub.len();
            for (k, s) in sub.into_iter().enumerate() {
                rechunked.push(Analyzed {
                    text: s.text,
                    emotion: seg.emotion.clone(),
                    ends_paragraph: if k == n - 1 {
                        seg.ends_paragraph
                    } else {
                        false
                    },
                    source_chunk: seg.source_chunk,
                    source_file: seg.source_file.clone(),
                    language: seg.language.clone(),
                });
            }
        }
        report(
            progress,
            &format!(
                "After A+B: {analyzed_count} emotion segments -> {} TTS-sized segments",
                rechunked.len()
            ),
        );

        // --- Final records -----------------------------------------------
        let mut final_segments: Vec<Segment> = Vec::new();
        let mut current_source: Option<String> = None;
        let mut current_chapter_number: i64 = 1;
        let mut current_chapter_title = "Chapter 1".to_string();
        let mut story_title: Option<String> = None;
        let mut story_title_detected = false;

        for (i, seg) in rechunked.iter().enumerate() {
            if let Some(record) = storage.load_segment_meta(i).and_then(Segment::from_value) {
                final_segments.push(record);
                current_source = Some(seg.source_file.clone());
                continue;
            }

            let mut text = seg.text.clone();
            let source_file = seg.source_file.clone();
            let new_source = current_source.as_deref() != Some(source_file.as_str());

            if !story_title_detected {
                story_title = Some(
                    self.detect_story_title(&seg.text)
                        .unwrap_or_else(|| "Audiobook".into()),
                );
                story_title_detected = true;
                if story_title.as_deref() != Some("Audiobook") {
                    report(
                        progress,
                        &format!(
                            "  detected story title: '{}'",
                            story_title.as_deref().unwrap_or("")
                        ),
                    );
                }
            }

            if new_source {
                let info = self.detect_chapter_info(&seg.text);
                if let Some(num) = info.chapter_number {
                    current_chapter_number = num;
                    report(
                        progress,
                        &format!("  detected chapter number: {current_chapter_number}"),
                    );
                }
                match info.chapter_title {
                    Some(t) => {
                        current_chapter_title = t;
                        report(
                            progress,
                            &format!("  detected chapter title: '{current_chapter_title}'"),
                        );
                    }
                    None => current_chapter_title = format!("Chapter {current_chapter_number}"),
                }
            }

            let chapter_title_added = settings.vocalize_chapter_names && new_source;
            if chapter_title_added {
                let chapter_title = format_chapter_title(&source_file);
                text = format!("{chapter_title}. {text}");
                report(
                    progress,
                    &format!(
                        "  added chapter title: '{chapter_title}' to segment {}",
                        i + 1
                    ),
                );
            }
            current_source = Some(source_file.clone());

            let speaking_speed = self.detect_speaking_speed(&seg.text);

            let record = Segment {
                normalized_text: text,
                emotion: seg.emotion.clone(),
                detected_emotions: vec![seg.emotion.clone()],
                language: Some(seg.language.clone()),
                speaking_speed: Some(speaking_speed),
                ends_paragraph: seg.ends_paragraph,
                source_chunk: seg.source_chunk,
                source_file,
                chapter_title_added,
                chunk_number: i,
                chapter_number: current_chapter_number,
                chapter_title: Some(current_chapter_title.clone()),
                story_title: story_title.clone(),
                extra: Default::default(),
            };
            storage.save_segment_meta(i, &record.to_value());
            final_segments.push(record);
        }

        report(
            progress,
            &format!(
                "Stage 1 complete: {} input chunks -> {} TTS segments",
                chunks.len(),
                final_segments.len()
            ),
        );
        Ok(final_segments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Backend answering from a fixed script; records the prompts it got.
    pub struct ScriptedLlm {
        pub answers: Vec<String>,
        pub prompts: Arc<Mutex<Vec<String>>>,
        pub fallback: String,
    }

    impl ScriptedLlm {
        pub fn new(answers: &[&str]) -> Self {
            Self {
                answers: answers.iter().rev().map(|s| s.to_string()).collect(),
                prompts: Arc::default(),
                fallback: "{}".into(),
            }
        }
    }

    impl LlmBackend for ScriptedLlm {
        fn complete(&mut self, prompt: &str, _max: usize, _stops: &[&str]) -> Result<String> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            Ok(self.answers.pop().unwrap_or_else(|| self.fallback.clone()))
        }
    }

    fn settings() -> Settings {
        Settings::default()
    }

    fn with(answers: &[&str]) -> LlmNormalizer {
        LlmNormalizer::with_backend(Box::new(ScriptedLlm::new(answers)), &settings())
    }

    #[test]
    fn extract_json_skips_noise_and_strings() {
        let v = extract_json("Sure! {\"a\": \"}{\", \"b\": [1]} trailing", '{', '}').unwrap();
        assert_eq!(v["a"], "}{");
        let v = extract_json("x [1, 2] [3]", '[', ']').unwrap();
        assert_eq!(v, json!([1, 2]));
        assert!(extract_json("{broken", '{', '}').is_none());
        let v = extract_json("{bad} {\"ok\": 1}", '{', '}').unwrap();
        assert_eq!(v["ok"], 1);
    }

    #[test]
    fn script_candidates_by_script() {
        assert_eq!(
            script_candidates("안녕하세요 반갑습니다 여러분"),
            BTreeSet::from(["korean"])
        );
        assert_eq!(
            script_candidates("Привет, как дела у тебя сегодня утром?"),
            BTreeSet::from(["russian"])
        );
        assert!(script_candidates("This is plain latin text, long enough.").contains("german"));
        assert_eq!(script_candidates("hi").len(), SUPPORTED_LANGUAGES.len());
    }

    #[test]
    fn chapter_titles_from_file_names() {
        assert_eq!(format_chapter_title("Chapter_01.md"), "Chapter 01");
        assert_eq!(format_chapter_title("intro.txt"), "Intro");
        assert_eq!(format_chapter_title("the-dark_TOWER"), "The Dark Tower");
        assert_eq!(format_chapter_title("___"), "Chapter");
    }

    #[test]
    fn emotion_sentence_split_matches_python_regex() {
        assert_eq!(
            split_emotion_sentences("One. Two!  \"Three?\" four. Five"),
            ["One.", "Two!", "\"Three?\" four.", "Five"]
        );
        assert_eq!(split_emotion_sentences("Über. Ärger."), ["Über. Ärger."]);
    }

    #[test]
    fn mock_mode_defaults() {
        let mut n = LlmNormalizer::new("fake_model.gguf", &settings());
        assert!(!n.is_loaded());
        let r = n.normalize("Sample text");
        assert_eq!(r.normalized_text, "Sample text");
        assert_eq!(r.emotion, "neutral");
        assert_eq!(n.detect_language("hola mundo").unwrap(), DEFAULT_LANGUAGE);
        assert_eq!(n.detect_speaking_speed("x"), "normal");
        assert_eq!(n.detect_story_title("x"), None);
        assert_eq!(n.sort_files(&[]), Vec::<String>::new());
        n.unload();
        assert!(!n.is_loaded());
    }

    #[test]
    fn detect_language_accepts_supported_answer() {
        let mut n = with(&[r#"{"language": "spanish"}"#]);
        assert_eq!(n.detect_language("Hola mundo").unwrap(), "spanish");
    }

    #[test]
    fn detect_language_falls_back_on_unknown_answer() {
        let mut n = with(&[r#"{"language": "klingon"}"#]);
        assert_eq!(n.detect_language("nuqneH").unwrap(), DEFAULT_LANGUAGE);
    }

    #[test]
    fn detect_language_rejects_script_contradiction() {
        let text = "This is an English sentence with plenty of latin letters.";
        let mut n = with(&[
            r#"{"language": "japanese"}"#,
            "garbage",
            r#"{"language": "german"}"#,
        ]);
        assert_eq!(n.detect_language(text).unwrap(), "german");
        let mut n = with(&[r#"{"language": "japanese"}"#; 3]);
        assert!(n.detect_language(text).is_err());
    }

    #[test]
    fn analyze_emotions_fills_gaps_and_validates() {
        let text = "One is here. Two is here. Three is here. Four is here.";
        let mut n = with(&[
            r#"[{"Start": 1, "Stop": 1, "Emotion": "Happy"}, {"start": 3, "stop": "4", "emotion": "bogus"}]"#,
        ]);
        let spans = n.analyze_emotions(text).unwrap();
        assert_eq!(
            spans,
            vec![
                EmotionSpan {
                    text: "One is here.".into(),
                    emotion: "happy".into()
                },
                EmotionSpan {
                    text: "Two is here.".into(),
                    emotion: "neutral".into()
                },
                EmotionSpan {
                    text: "Three is here. Four is here.".into(),
                    emotion: "neutral".into()
                },
            ]
        );
        // Overlaps are retried, then fail.
        let mut n = with(
            &[r#"[{"Start": 1, "Stop": 3, "Emotion": "sad"}, {"Start": 2, "Stop": 4, "Emotion": "sad"}]"#;
                3],
        );
        assert!(n.analyze_emotions(text).is_err());
        // Single sentence: no LLM call needed.
        let mut n = with(&[]);
        assert_eq!(
            n.analyze_emotions("Just one.").unwrap()[0].emotion,
            "neutral"
        );
    }

    #[test]
    fn replacements_are_located_and_applied() {
        let src = "Dr. Smith met Dr. Jones at 3 PM.";
        let reps = vec![
            json!({"find": "Dr.", "occurrence": 2, "replace": "Doctor"}),
            json!({"find": "3 PM", "replace": "three PM"}),
        ];
        let resolved = LlmNormalizer::validate_replacements(src, &reps).unwrap();
        assert_eq!(
            LlmNormalizer::apply_replacements(src, &resolved),
            "Dr. Smith met Doctor Jones at three PM."
        );
        assert!(
            LlmNormalizer::validate_replacements(
                src,
                &[json!({"find": "Dr.", "occurrence": 3, "replace": ""})]
            )
            .is_err()
        );
        assert!(
            LlmNormalizer::validate_replacements(
                src,
                &[
                    json!({"find": "Dr. Smith", "replace": "a"}),
                    json!({"find": "Smith", "replace": "b"})
                ]
            )
            .is_err()
        );
        let mut n =
            with(&[r#"{"replacements": [{"find": "she", "occurrence": 1, "replace": "Sarah"}]}"#]);
        assert_eq!(
            n.fill_context("Then she left.", "Sarah came.", "").unwrap(),
            "Then Sarah left."
        );
    }

    #[test]
    fn sort_files_dedupes_and_appends_missing() {
        let files: Vec<FilePreview> = ["b.md", "a.md", "c10.md", "c2.md"]
            .iter()
            .map(|n| FilePreview {
                name: n.to_string(),
                preview: "x".into(),
            })
            .collect();
        let mut n = with(&[r#"Output: ["c2.md", "c2.md", "zzz.md", "a.md"]"#]);
        assert_eq!(n.sort_files(&files), ["c2.md", "a.md", "b.md", "c10.md"]);
        let mut n = with(&["no json"]);
        assert_eq!(n.sort_files(&files), ["a.md", "b.md", "c2.md", "c10.md"]);
    }

    #[test]
    fn reconstruct_from_anchors_covers_source() {
        let src = "Intro words. The storm came suddenly. Everyone ran home.";
        let anchors = vec![
            EmotionSpan {
                text: "The storm came".into(),
                emotion: "tense".into(),
            },
            EmotionSpan {
                text: "Everyone ran".into(),
                emotion: "fearful".into(),
            },
        ];
        let r = LlmNormalizer::reconstruct_from_anchors(src, &anchors);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].emotion, "neutral");
        assert_eq!(r[1].text, "The storm came suddenly.");
        assert_eq!(r[2].emotion, "fearful");
    }

    fn chunks() -> Vec<Chunk> {
        vec![
            Chunk::new("First text.", false),
            Chunk::new("Second text.", true),
        ]
    }

    #[test]
    fn process_all_writes_cache_and_records() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let mut n = LlmNormalizer::new("fake.gguf", &settings());
        let segs = n
            .process_all(&chunks(), &cache, &settings(), None, None)
            .unwrap();
        assert_eq!(segs.len(), 2);
        assert!(cache.join("chunk_000.json").exists());
        assert!(cache.join("chunk_001.json").exists());
        assert!(cache.join("analyze/chunk_000.json").exists());
        assert!(cache.join("segments/seg_0001.json").exists());
        assert!(segs.iter().all(|s| !s.normalized_text.is_empty()));
        assert!(segs[1].ends_paragraph);
        assert_eq!(segs[0].language.as_deref(), Some("german"));
        assert_eq!(segs[0].story_title.as_deref(), Some("Audiobook"));
    }

    #[test]
    fn process_all_resumes_from_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = dir.path().join("cache");
        let mut n = LlmNormalizer::new("fake.gguf", &settings());
        n.process_all(&chunks(), &cache, &settings(), None, None)
            .unwrap();
        let t1 = std::fs::metadata(cache.join("chunk_000.json"))
            .unwrap()
            .modified()
            .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        n.process_all(&chunks(), &cache, &settings(), None, None)
            .unwrap();
        let t2 = std::fs::metadata(cache.join("chunk_000.json"))
            .unwrap()
            .modified()
            .unwrap();
        assert_eq!(t1, t2);
    }

    #[test]
    fn process_all_auto_language_votes_and_chapter_titles() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = settings();
        s.tts_language = "auto".into();
        s.vocalize_chapter_names = true;
        s.storage_backend = "sqlite".into();
        let mut llm = ScriptedLlm::new(&[]);
        llm.fallback =
            r#"{"language": "french", "title": "Le Livre", "chapter_number": 4, "speed": "slow"}"#
                .into();
        let mut n = LlmNormalizer::with_backend(Box::new(llm), &s);
        let mut cs = chunks();
        cs[1].source_file = "chapter_two".into();
        let text = "Ceci est une phrase française assez longue pour le test.";
        cs[0].text = text.into();
        cs[1].text = text.into();
        let segs = n.process_all(&cs, dir.path(), &s, None, None).unwrap();
        assert_eq!(segs[0].language.as_deref(), Some("french"));
        assert_eq!(segs[0].story_title.as_deref(), Some("Le Livre"));
        assert_eq!(segs[0].chapter_number, 4);
        assert!(segs[0].normalized_text.starts_with("Default. "));
        assert!(segs[1].normalized_text.starts_with("Chapter Two. "));
        assert!(segs[1].chapter_title_added);
        assert_eq!(segs[1].speaking_speed.as_deref(), Some("slow"));
    }

    #[test]
    fn process_all_honours_stop_flag() {
        let dir = tempfile::tempdir().unwrap();
        let stop = StopFlag::new();
        stop.set();
        let mut n = LlmNormalizer::new("fake.gguf", &settings());
        let segs = n
            .process_all(&chunks(), dir.path(), &settings(), None, Some(&stop))
            .unwrap();
        assert!(segs.is_empty());
    }
}
