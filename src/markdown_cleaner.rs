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

//! Strip Markdown syntax so only readable prose reaches the TTS.

use std::sync::LazyLock;

use regex::{Captures, Regex};

static HEADER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*#+\s").unwrap());
static IMAGE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"!\[([^\]]*)\]\([^\)]*\)").unwrap());
static LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\([^\)]+\)").unwrap());
static FOOTNOTE_REF: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[\^[^\]]+\]:?").unwrap());
static BOLD_STAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*\*([^\*]+)\*\*").unwrap());
static ITALIC_STAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*([^\*]+)\*").unwrap());
static BOLD_UNDERSCORE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"__([^_]+)__").unwrap());
static ITALIC_UNDERSCORE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(^|[^\p{L}\p{N}])_([^_]+)_($|[^\p{L}\p{N}])").unwrap());
static INLINE_CODE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"`([^`]+)`").unwrap());
/// A URL, with "unter"/"at" in front ("verfügbar unter https://…"). Trailing
/// punctuation stays: it usually ends the sentence.
static URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?:\s+(?:unter|at)\s+)?(?:https?://|www\.)[^\s]*[^\s.,;:!?)\]»«“”"']"#).unwrap()
});
/// HTML tags only; "a < b und c > d" is not a tag.
static HTML_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>").unwrap());
static MANY_NEWLINES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());
/// Backslash escape: `\*`, `\_`, `\[` … and stray ones in exported files (`\²`).
static ESCAPE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\\([^\p{L}\p{Nd}\s])").unwrap());
static LIST_MARKER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?:[-*+•–]|\d{1,3}[.)])\s+").unwrap());
static BULLET: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*[-*+•–]\s+").unwrap());
static BLOCKQUOTE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*(?:>\s?)+").unwrap());
static RULE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*(?:(?:-\s*){3,}|(?:\*\s*){3,}|(?:_\s*){3,})$").unwrap());
static TABLE_SEPARATOR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$").unwrap());
/// Footnote marks: superscript digits after a word or punctuation
/// ("Carlini¹", "2020).²", "Lee¹ʼ³") and in front of a footnote ("¹ Google").
/// After a digit they are exponents ("10²") and stay for the normalizer.
static FOOTNOTE_MARK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"([^\d\s])[¹²³⁰⁴⁵⁶⁷⁸⁹]+(?:[ʼ’',][¹²³⁰⁴⁵⁶⁷⁸⁹]+)*|(^|\s)[¹²³⁰⁴⁵⁶⁷⁸⁹*†‡]+(\s)|[†‡]")
        .unwrap()
});
static LONE_STAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*+").unwrap());
/// "bleu_score" -> "bleu score"; a leading "_low" just loses the underscore.
static WORD_UNDERSCORE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([\p{L}\p{N}])_+([\p{L}\p{N}])").unwrap());
static STRAY_UNDERSCORE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_+").unwrap());
/// Brackets emptied by a removed URL: "API (https://…)" -> "API ()".
static EMPTY_BRACKETS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+\(\s*\)|\s+\[\s*\]").unwrap());
static SPACE_BEFORE_PUNCT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+([.,;:!?])").unwrap());
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]{2,}").unwrap());

/// Escaped characters are hidden in a private-use plane while the emphasis
/// rules run, so `\*` or `\_` never start emphasis.
const ESCAPE_OFFSET: u32 = 0xF0000;

fn hide_escapes(line: &str) -> String {
    ESCAPE
        .replace_all(line, |c: &Captures| {
            let ch = c[1].chars().next().unwrap();
            match char::from_u32(ESCAPE_OFFSET + ch as u32) {
                Some(h) if (ch as u32) < 0xFFFE => h.to_string(),
                _ => ch.to_string(),
            }
        })
        .into_owned()
}

fn restore_escapes(line: &str) -> String {
    line.chars()
        .map(|c| match c as u32 {
            v if (ESCAPE_OFFSET..ESCAPE_OFFSET + 0xFFFE).contains(&v) => {
                char::from_u32(v - ESCAPE_OFFSET).unwrap_or(c)
            }
            _ => c,
        })
        .collect()
}

fn is_table_row(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.len() > 1
}

/// "| a | b |  |" -> "a, b." (empty cells dropped).
fn table_row(line: &str) -> String {
    let cells: Vec<&str> = line
        .trim()
        .trim_matches('|')
        .split('|')
        .map(str::trim)
        .filter(|c| !c.is_empty())
        .collect();
    if cells.is_empty() {
        return String::new();
    }
    let row = cells.join(", ");
    if row.ends_with(['.', '!', '?', ':']) {
        row
    } else {
        format!("{row}.")
    }
}

/// Inline Markdown of one line.
fn clean_inline(line: &str) -> String {
    let line = hide_escapes(line);
    let line = IMAGE.replace_all(&line, "$1");
    let line = LINK.replace_all(&line, "$1");
    let line = FOOTNOTE_REF.replace_all(&line, "");
    let line = BOLD_STAR.replace_all(&line, "$1");
    let line = ITALIC_STAR.replace_all(&line, "$1");
    let line = BOLD_UNDERSCORE.replace_all(&line, "$1");
    let line = ITALIC_UNDERSCORE.replace_all(&line, "$1$2$3");
    let line = INLINE_CODE.replace_all(&line, "$1");
    let line = URL.replace_all(&line, "");
    let line = HTML_TAG.replace_all(&line, "");
    let line = restore_escapes(&line);
    let line = FOOTNOTE_MARK.replace_all(&line, "$1$2$3");
    let line = LONE_STAR.replace_all(&line, "");
    let line = WORD_UNDERSCORE.replace_all(&line, "$1 $2");
    let line = WORD_UNDERSCORE.replace_all(&line, "$1 $2");
    let line = STRAY_UNDERSCORE.replace_all(&line, "");
    let line = EMPTY_BRACKETS.replace_all(&line, "");
    let line = SPACE_BEFORE_PUNCT.replace_all(&line, "$1");
    SPACES.replace_all(&line, " ").into_owned()
}

/// Remove headings, code fences, emphasis, links, URLs, HTML tags, list
/// markers, table pipes and footnote marks. Paragraphs (blank-line
/// separated) are preserved; every list item becomes its own paragraph and
/// every table row its own sentence.
pub fn clean(md: &str) -> String {
    let mut cleaned: Vec<String> = Vec::new();
    let mut in_code_fence = false;
    let mut in_table = false;

    let blank = |cleaned: &mut Vec<String>| {
        if cleaned.last().is_some_and(|l| !l.is_empty()) {
            cleaned.push(String::new());
        }
    };

    for line in md.split('\n') {
        if line.trim().starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence || HEADER.is_match(line) {
            continue;
        }
        if RULE.is_match(line) || TABLE_SEPARATOR.is_match(line) {
            continue;
        }
        let line = BLOCKQUOTE.replace(line, "");

        if is_table_row(&line) {
            if !in_table {
                blank(&mut cleaned);
                in_table = true;
            }
            let row = table_row(&clean_inline(&line));
            if !row.is_empty() {
                cleaned.push(row);
            }
            continue;
        }
        if in_table {
            in_table = false;
            blank(&mut cleaned);
        }

        // A list item starts a paragraph of its own (references, bullet
        // lists); bullets are dropped, "1." numbers stay.
        let item = LIST_MARKER.is_match(&line);
        let line = if item {
            blank(&mut cleaned);
            BULLET.replace(&line, "").into_owned()
        } else {
            line.into_owned()
        };

        let line = clean_inline(&line);
        if !line.trim().is_empty() {
            cleaned.push(line.trim().to_string());
        } else {
            blank(&mut cleaned);
        }
    }

    let text = cleaned.join("\n");
    MANY_NEWLINES.replace_all(&text, "\n\n").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::clean;

    #[test]
    fn strips_headers() {
        let r = clean("# Title\nSome text");
        assert!(!r.contains("Title"));
        assert!(r.contains("Some text"));
    }

    #[test]
    fn strips_bold_and_italic() {
        let r = clean("This is **bold** and *italic* and __b__ and _i_ text");
        assert_eq!(r, "This is bold and italic and b and i text");
    }

    #[test]
    fn strips_links_and_urls() {
        let r = clean("Check out [this link](https://example.com) or https://x.org/a now");
        assert!(r.contains("this link"));
        assert!(!r.contains("https://"));
    }

    #[test]
    fn preserves_paragraphs() {
        assert_eq!(
            clean("First paragraph.\n\n\n\nSecond paragraph."),
            "First paragraph.\n\nSecond paragraph."
        );
    }

    #[test]
    fn strips_code_fence_and_inline_code() {
        let r = clean("Text\n```python\ncode\n```\nMore text with `function()` here");
        assert!(!r.contains("code\n"));
        assert!(r.contains("Text"));
        assert!(r.contains("More text with function() here"));
        assert!(!r.contains('`'));
    }

    #[test]
    fn strips_html_tags() {
        assert_eq!(clean("A <b>bold</b> word<br/>"), "A bold word");
    }

    #[test]
    fn backslash_escapes_are_literal() {
        assert_eq!(clean(r"Tramèr\², Nasr\¹"), "Tramèr, Nasr");
        assert_eq!(
            clean(r"die große Modelle \[memorisieren\], so"),
            "die große Modelle [memorisieren], so"
        );
        // \_ must not open emphasis: F\_fp(D\_n) used to lose its letters.
        assert_eq!(
            clean(r"Sei F\_fp(D\_n) der Filter"),
            "Sei F fp(D n) der Filter"
        );
        assert_eq!(clean(r"\* Die übrigen Autoren"), "Die übrigen Autoren");
        assert_eq!(
            clean(r"Unterstrich („\_low“, „\_high“)"),
            "Unterstrich („low“, „high“)"
        );
        assert_eq!(
            clean("nltk.translate.bleu_score"),
            "nltk.translate.bleu score"
        );
        assert_eq!(clean(r"MLSum-es\* (Spanisch)"), "MLSum-es (Spanisch)");
    }

    #[test]
    fn footnote_marks_go_exponents_stay() {
        assert_eq!(
            clean("Daphne Ippolito¹, Katherine Lee¹ʼ³, Nicholas Carlini¹"),
            "Daphne Ippolito, Katherine Lee, Nicholas Carlini"
        );
        assert_eq!(
            clean("¹ Google Research · ² ETH Zürich"),
            "Google Research · ETH Zürich"
        );
        assert_eq!(
            clean("Crawls stammt (Gao et al., 2020).¹"),
            "Crawls stammt (Gao et al., 2020)."
        );
        assert_eq!(clean("Fläche von 10² Metern"), "Fläche von 10² Metern");
        assert_eq!(
            clean("Text[^1] weiter\n\n[^1]: Fußnote."),
            "Text weiter\n\nFußnote."
        );
    }

    #[test]
    fn urls_keep_sentence_end_and_drop_unter() {
        assert_eq!(
            clean("Quelle ist der Common-Crawl-Datensatz unter https://commoncrawl.org."),
            "Quelle ist der Common-Crawl-Datensatz."
        );
        assert_eq!(clean("Siehe (https://x.org/a), dann"), "Siehe, dann");
        assert_eq!(
            clean("die Transformers-API (https://huggingface.co). Code"),
            "die Transformers-API. Code"
        );
    }

    #[test]
    fn comparisons_are_not_html() {
        assert_eq!(
            clean("BLEU > 0,75 und (< 5) Änderungen"),
            "BLEU > 0,75 und (< 5) Änderungen"
        );
    }

    #[test]
    fn list_items_become_paragraphs() {
        let r = clean(
            "Literatur:\n- Abadi, M. (2016). Deep learning.\n- Anil, R. (2021). Large.\n* drittens\n1. Erstens",
        );
        assert_eq!(
            r,
            "Literatur:\n\nAbadi, M. (2016). Deep learning.\n\nAnil, R. (2021). Large.\n\ndrittens\n\n1. Erstens"
        );
    }

    #[test]
    fn tables_are_read_row_by_row() {
        let r = clean(
            "Vorher.\n\n| Datensatz | BLEU | |\n|---|:---:|---|\n| MLSum-es\\* | 58,7 | |\nNachher.",
        );
        assert_eq!(
            r,
            "Vorher.\n\nDatensatz, BLEU.\nMLSum-es, 58,7.\n\nNachher."
        );
    }

    #[test]
    fn rules_and_quotes() {
        assert_eq!(clean("A\n\n---\n\n> zitiert\n\nB"), "A\n\nzitiert\n\nB");
        assert_eq!(clean("![Bild eins](a.png) Text"), "Bild eins Text");
    }
}
