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

//! Records exchanged between the stages and stored in the cache. The JSON
//! layout is the same as in the Python version, so existing caches load.

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

/// Treat JSON `null` like a missing field.
fn null_default<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

fn default_emotion() -> String {
    "neutral".into()
}

fn default_source() -> String {
    "default".into()
}

fn default_chapter_number() -> i64 {
    1
}

/// One emotion span produced by step A.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmotionSpan {
    #[serde(default, deserialize_with = "null_default")]
    pub text: String,
    #[serde(default = "default_emotion", deserialize_with = "null_default")]
    pub emotion: String,
}

/// Stage-1 step A cache entry (`analyze/chunk_NNN.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkMeta {
    #[serde(default, deserialize_with = "null_default")]
    pub segments: Vec<EmotionSpan>,
    #[serde(default)]
    pub ends_paragraph: Option<bool>,
    #[serde(default)]
    pub language: Option<String>,
}

/// A TTS-ready segment (final output of stage 1, `segments/seg_GGGG.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Segment {
    #[serde(default, deserialize_with = "null_default")]
    pub normalized_text: String,
    #[serde(default = "default_emotion", deserialize_with = "null_default")]
    pub emotion: String,
    #[serde(default, deserialize_with = "null_default")]
    pub detected_emotions: Vec<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaking_speed: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    pub ends_paragraph: bool,
    #[serde(default, deserialize_with = "null_default")]
    pub source_chunk: usize,
    #[serde(default = "default_source", deserialize_with = "null_default")]
    pub source_file: String,
    #[serde(default, deserialize_with = "null_default")]
    pub chapter_title_added: bool,
    #[serde(default, deserialize_with = "null_default")]
    pub chunk_number: usize,
    #[serde(default = "default_chapter_number", deserialize_with = "null_default")]
    pub chapter_number: i64,
    #[serde(default)]
    pub chapter_title: Option<String>,
    #[serde(default)]
    pub story_title: Option<String>,
    /// Unknown fields (e.g. `segment_id` from the SQLite listing) are kept.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl Default for Segment {
    fn default() -> Self {
        Self {
            normalized_text: String::new(),
            emotion: default_emotion(),
            detected_emotions: Vec::new(),
            language: None,
            speaking_speed: None,
            ends_paragraph: false,
            source_chunk: 0,
            source_file: default_source(),
            chapter_title_added: false,
            chunk_number: 0,
            chapter_number: 1,
            chapter_title: None,
            story_title: None,
            extra: Map::new(),
        }
    }
}

impl Segment {
    /// A segment with just text (handy for tests and the legacy API).
    pub fn with_text(text: impl Into<String>) -> Self {
        Self {
            normalized_text: text.into(),
            ..Default::default()
        }
    }

    pub fn from_value(v: Value) -> Option<Segment> {
        serde_json::from_value(v).ok()
    }

    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_python_record_with_nulls() {
        let v: Value = serde_json::from_str(
            r#"{"normalized_text": "Hi.", "emotion": null, "language": null,
                "chapter_number": 3, "story_title": "Book", "segment_id": 7}"#,
        )
        .unwrap();
        let s = Segment::from_value(v).unwrap();
        assert_eq!(s.normalized_text, "Hi.");
        assert_eq!(s.emotion, "");
        assert_eq!(s.language, None);
        assert_eq!(s.chapter_number, 3);
        assert_eq!(s.source_file, "default");
        assert_eq!(s.extra["segment_id"], 7);
    }

    #[test]
    fn writes_python_field_order() {
        let s = Segment::with_text("x");
        let v = s.to_value();
        let keys: Vec<&String> = v.as_object().unwrap().keys().collect();
        assert_eq!(keys[0], "normalized_text");
        assert_eq!(keys[1], "emotion");
        assert_eq!(keys.last().unwrap().as_str(), "story_title");
    }
}
