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

//! Split cleaned text into paragraph-aligned chunks of bounded word count.

use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Chunk {
    pub text: String,
    pub ends_paragraph: bool,
    pub source_file: String,
}

impl Chunk {
    pub fn new(text: impl Into<String>, ends_paragraph: bool) -> Self {
        Self {
            text: text.into(),
            ends_paragraph,
            source_file: "default".into(),
        }
    }
}

/// Paragraphs are separated by one or more blank lines.
static PARAGRAPH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\s*\n").unwrap());

/// Closing quotes/brackets that may follow the end punctuation of a sentence.
const SENTENCE_CLOSERS: &[char] = &['"', '\'', '“', '”', '»', '«', '’', ')', ']'];
const SENTENCE_END: &[char] = &['.', '!', '?', '…'];

/// True if `s` ends like a sentence: end punctuation, optionally followed by
/// closing quotes/brackets (`[.!?…]["'“”»«’)\]]*$`).
fn ends_sentence(s: &str) -> bool {
    let trimmed = s.trim_end_matches(SENTENCE_CLOSERS);
    trimmed.ends_with(SENTENCE_END)
}

/// True if `c` can start a sentence: ASCII capital, German umlaut capital,
/// digit, or an opening quote/bracket (`[A-ZÄÖÜ0-9"'„“»«‚(\[]`).
fn starts_sentence(c: char) -> bool {
    c.is_ascii_uppercase()
        || c.is_ascii_digit()
        || matches!(
            c,
            'Ä' | 'Ö' | 'Ü' | '"' | '\'' | '„' | '“' | '»' | '«' | '‚' | '(' | '['
        )
}

/// Words with a full stop that do not end a sentence.
const ABBREVIATIONS: &[&str] = &[
    "al.", "bzw.", "ca.", "Dr.", "Prof.", "Nr.", "vgl.", "Vgl.", "evtl.", "ggf.", "inkl.", "zzgl.",
    "Abb.", "Tab.", "Kap.", "Abschn.", "Bd.", "Hrsg.", "Jh.", "Mio.", "Mrd.", "Str.", "St.",
    "geb.", "gest.", "sog.", "u.", "Mr.", "Mrs.", "Ms.", "vs.", "Fig.", "Eq.", "Sec.", "pp.",
    "Vol.", "No.", "Jan.", "Feb.", "Aug.", "Sept.", "Okt.", "Nov.", "Dez.",
];

/// True if the last word of `before` is an abbreviation, an initial or an
/// ordinal number, so the full stop does not end the sentence: "z. B.",
/// "Lee et al. (2021)", "Christopher A. Choquette", "am 3. Mai".
pub(crate) fn ends_with_abbreviation(before: &str) -> bool {
    let word = before
        .rsplit(|c: char| c.is_whitespace() || c == '(' || c == '[')
        .next()
        .unwrap_or("");
    if ABBREVIATIONS.contains(&word) {
        return true;
    }
    let Some(stem) = word.strip_suffix('.') else {
        return false;
    };
    // Initials and letter abbreviations: "A.", "z.", "z.B.", "U.S."
    if !stem.is_empty()
        && stem
            .split('.')
            .all(|p| p.chars().count() == 1 && p.chars().all(char::is_alphabetic))
    {
        return true;
    }
    // "3." / "15." before a word is an ordinal or a date.
    !stem.is_empty() && stem.len() <= 2 && stem.chars().all(|c| c.is_ascii_digit())
}

/// Split one paragraph at sentence boundaries. Covers German umlauts and
/// German/French quotes as well as ASCII.
pub fn split_sentences(paragraph: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    let mut iter = paragraph.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        if !c.is_whitespace() {
            continue;
        }
        // Whitespace run [i, end)
        let mut end = i + c.len_utf8();
        while let Some(&(j, d)) = iter.peek() {
            if d.is_whitespace() {
                end = j + d.len_utf8();
                iter.next();
            } else {
                break;
            }
        }
        let before = &paragraph[start..i];
        let after = paragraph[end..].chars().next();
        if let Some(a) = after
            && ends_sentence(before)
            && starts_sentence(a)
            && !ends_with_abbreviation(before)
        {
            sentences.push(before.trim().to_string());
            start = end;
        }
    }
    let tail = paragraph[start..].trim();
    if !tail.is_empty() {
        sentences.push(tail.to_string());
    }
    sentences.retain(|s| !s.is_empty());
    sentences
}

pub fn word_count(s: &str) -> usize {
    s.split_whitespace().count()
}

/// Split `text` into chunks of at most `max_words` words.
///
/// Chunks never span a paragraph break, so every paragraph ends a chunk and
/// the last chunk of each paragraph is flagged `ends_paragraph`. The
/// assembler uses that flag to insert the longer paragraph pause. Within a
/// paragraph, sentences are packed greedily; a single sentence longer than
/// `max_words` is kept whole (splitting mid-sentence damages prosody).
pub fn split_to_chunks(text: &str, max_words: usize, source_file: &str) -> Vec<Chunk> {
    let mut result = Vec::new();

    for paragraph in PARAGRAPH.split(text) {
        // Line breaks inside a paragraph are just wrapping.
        let paragraph = paragraph.split_whitespace().collect::<Vec<_>>().join(" ");
        if paragraph.is_empty() {
            continue;
        }

        let mut packed: Vec<String> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let mut current_words = 0usize;
        for sentence in split_sentences(&paragraph) {
            let words = word_count(&sentence);
            if !current.is_empty() && current_words + words > max_words {
                packed.push(current.join(" "));
                current = vec![sentence];
                current_words = words;
            } else {
                current.push(sentence);
                current_words += words;
            }
        }
        if !current.is_empty() {
            packed.push(current.join(" "));
        }

        let n = packed.len();
        for (i, chunk_text) in packed.into_iter().enumerate() {
            result.push(Chunk {
                text: chunk_text,
                ends_paragraph: i == n - 1,
                source_file: source_file.to_string(),
            });
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(chunks: &[Chunk]) -> Vec<&str> {
        chunks.iter().map(|c| c.text.as_str()).collect()
    }

    #[test]
    fn chunks_under_max_words() {
        let chunks = split_to_chunks(
            "First sentence. Second sentence. Third sentence.",
            10,
            "default",
        );
        for c in &chunks {
            assert!(word_count(&c.text) < 11);
        }
    }

    #[test]
    fn chunks_end_with_sentence_boundary() {
        let chunks = split_to_chunks(
            "This is sentence one. This is sentence two. This is sentence three.",
            499,
            "default",
        );
        for c in &chunks {
            assert!(c.text.trim_end().ends_with(['.', '!', '?']));
        }
    }

    #[test]
    fn single_short_text() {
        let chunks = split_to_chunks("Short text.", 499, "default");
        assert_eq!(texts(&chunks), ["Short text."]);
    }

    #[test]
    fn empty_text() {
        assert!(split_to_chunks("", 499, "default").is_empty());
    }

    #[test]
    fn very_long_sentence_is_kept_whole() {
        let long = "This is a very long sentence that contains many words ".repeat(30) + ".";
        let chunks = split_to_chunks(&long, 50, "default");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunks_never_span_paragraphs() {
        let text = "Erster Absatz. Noch ein Satz.\n\nZweiter Absatz.\n\nDritter Absatz. Ende.";
        let chunks = split_to_chunks(text, 499, "default");
        assert_eq!(
            texts(&chunks),
            [
                "Erster Absatz. Noch ein Satz.",
                "Zweiter Absatz.",
                "Dritter Absatz. Ende."
            ]
        );
        assert!(chunks.iter().all(|c| c.ends_paragraph));
    }

    #[test]
    fn long_paragraph_only_last_chunk_ends_paragraph() {
        let paragraph = (0..20)
            .map(|i| format!("Satz Nummer {i} hat fünf Wörter."))
            .collect::<Vec<_>>()
            .join(" ");
        let chunks = split_to_chunks(&(paragraph + "\n\nKurz."), 20, "default");
        assert!(chunks.len() > 2);
        let (short, rest) = chunks.split_last().unwrap();
        let (last_of_para, first_para) = rest.split_last().unwrap();
        assert!(first_para.iter().all(|c| !c.ends_paragraph));
        assert!(last_of_para.ends_paragraph);
        assert_eq!(short.text, "Kurz.");
        assert!(short.ends_paragraph);
        assert!(chunks.iter().all(|c| word_count(&c.text) <= 20));
    }

    #[test]
    fn german_sentence_starts_are_split() {
        let text = "Erster Satz. Über alles. „Zitat“, sagte er. Ärger folgt.";
        let chunks = split_to_chunks(text, 3, "default");
        assert_eq!(
            texts(&chunks),
            [
                "Erster Satz.",
                "Über alles.",
                "„Zitat“, sagte er.",
                "Ärger folgt."
            ]
        );
    }

    #[test]
    fn line_breaks_inside_paragraph_are_joined() {
        let chunks = split_to_chunks("Eine Zeile\nund die nächste.", 499, "default");
        assert_eq!(texts(&chunks), ["Eine Zeile und die nächste."]);
    }

    #[test]
    fn closing_quotes_after_punctuation() {
        let s = split_sentences("Er rief: „Halt!“ Dann ging er. (Leise.) Ende");
        assert_eq!(s, ["Er rief: „Halt!“", "Dann ging er.", "(Leise.)", "Ende"]);
    }

    #[test]
    fn source_file_is_kept() {
        let chunks = split_to_chunks("One. Two.\n\nThree.", 499, "story_ch1");
        assert!(chunks.iter().all(|c| c.source_file == "story_ch1"));
        assert_eq!(Chunk::new("x", false).source_file, "default");
    }

    #[test]
    fn abbreviations_and_initials_do_not_split() {
        let s = split_sentences(
            "Sie unterscheidet nicht (z. B. Preisgabe) und (z.B. Zitate). Lee et al. (2021) zeigen es. Christopher A. Choquette-Choo schrieb am 15. September. Danach kam U. Müller.",
        );
        assert_eq!(
            s,
            vec![
                "Sie unterscheidet nicht (z. B. Preisgabe) und (z.B. Zitate).",
                "Lee et al. (2021) zeigen es.",
                "Christopher A. Choquette-Choo schrieb am 15. September.",
                "Danach kam U. Müller.",
            ]
        );
        // A year or a word still ends the sentence.
        assert_eq!(split_sentences("Das war 1990. Danach nicht.").len(), 2);
    }
}
