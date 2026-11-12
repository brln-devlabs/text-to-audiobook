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
/// A "_" on its own ("„_“ vor den Variablen") is spoken.
static LONE_UNDERSCORE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(^|[\s„“"'(«»‚‘])_+($|[\s„“"')«»‚‘.,;:])"#).unwrap());
/// "verlinkt unter URL1 und URL2." -> "verlinkt und ." -> "verlinkt."
static DANGLING_AND: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+(?:und|oder|and|or)\s*([.,;:)])").unwrap());
static STRAY_UNDERSCORE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_+").unwrap());
/// Brackets emptied by a removed URL: "API (https://…)" -> "API ()".
static EMPTY_BRACKETS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+\(\s*\)|\s+\[\s*\]").unwrap());
static SPACE_BEFORE_PUNCT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\s+([.,;:!?])").unwrap());
static SPACES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]{2,}").unwrap());
/// HTML entities left in exported Markdown: "&amp;", "&nbsp;", "&#8211;".
static ENTITY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"&(?:#(\d{1,7})|#[xX]([0-9a-fA-F]{1,6})|([a-zA-Z]{2,8}));").unwrap()
});

/// Decode HTML entities, so "&amp;" reaches the normalizer as "&" (and is
/// spoken "und") and "&nbsp;" is a space. Unknown names stay as written.
fn decode_entities(line: &str) -> String {
    ENTITY
        .replace_all(line, |c: &Captures| {
            let code = match (c.get(1), c.get(2)) {
                (Some(d), _) => d.as_str().parse::<u32>().ok(),
                (_, Some(h)) => u32::from_str_radix(h.as_str(), 16).ok(),
                _ => None,
            };
            if let Some(ch) = code.and_then(char::from_u32).filter(|ch| !ch.is_control()) {
                return ch.to_string();
            }
            let decoded = match c.get(3).map(|m| m.as_str()) {
                Some("amp") => "&",
                Some("lt") => "<",
                Some("gt") => ">",
                Some("quot") => "\"",
                Some("apos") => "'",
                Some("nbsp") => " ",
                Some("ndash") => "–",
                Some("mdash") => "—",
                Some("hellip") => "…",
                Some("shy") => "",
                _ => return c[0].to_string(),
            };
            decoded.to_string()
        })
        .into_owned()
}

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
    let had_url = URL.is_match(&line);
    let line = URL.replace_all(&line, "");
    let line = if had_url {
        DANGLING_AND
            .replace_all(&SPACE_BEFORE_PUNCT.replace_all(&line, "$1"), "$1")
            .into_owned()
    } else {
        line.into_owned()
    };
    let line = HTML_TAG.replace_all(&line, "");
    let line = decode_entities(&line);
    let line = restore_escapes(&line);
    let line = FOOTNOTE_MARK.replace_all(&line, "$1$2$3");
    let line = LONE_STAR.replace_all(&line, "");
    let line = WORD_UNDERSCORE.replace_all(&line, "$1 $2");
    let line = WORD_UNDERSCORE.replace_all(&line, "$1 $2");
    let line = LONE_UNDERSCORE.replace_all(&line, "${1}Unterstrich$2");
    let line = STRAY_UNDERSCORE.replace_all(&line, "");
    let line = EMPTY_BRACKETS.replace_all(&line, "");
    let line = SPACE_BEFORE_PUNCT.replace_all(&line, "$1");
    SPACES.replace_all(&line, " ").into_owned()
}

/// Text of a heading line ("## Titel ##" -> "Titel"), formatting removed;
/// None for other lines and for empty headings.
pub fn heading_text(line: &str) -> Option<String> {
    if !HEADER.is_match(line) {
        return None;
    }
    let rest = line.trim().trim_start_matches('#');
    let title = clean(rest.trim().trim_end_matches('#').trim());
    (!title.is_empty()).then_some(title)
}

/// A heading spoken as a paragraph of its own ends like a sentence, so the
/// voice drops at its end: "Kernaussage" -> "Kernaussage.".
fn spoken_heading(title: &str) -> String {
    let core = title.trim_end_matches(['"', '\'', '“', '”', '»', '«', '’', ')', ']']);
    if core.ends_with(['.', '!', '?', ':', '…']) {
        title.to_string()
    } else {
        format!("{title}.")
    }
}

/// Remove headings, code fences, emphasis, links, URLs, HTML tags, list
/// markers, table pipes and footnote marks. Paragraphs (blank-line
/// separated) are preserved; every list item becomes its own paragraph and
/// every table row its own sentence.
pub fn clean(md: &str) -> String {
    clean_with(md, false)
}

/// Like [`clean`]; with `speak_headings` every heading after the first one
/// stays as a paragraph of its own (`SPEAK_HEADINGS`). The first heading of
/// a file is its chapter name, spoken by `VOCALIZE_CHAPTER_NAMES`, and is
/// always dropped.
pub fn clean_with(md: &str, speak_headings: bool) -> String {
    let mut cleaned: Vec<String> = Vec::new();
    let mut in_code_fence = false;
    let mut in_table = false;
    let mut seen_heading = false;

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
        if in_code_fence {
            continue;
        }
        if HEADER.is_match(line) {
            if let Some(title) = heading_text(line) {
                if speak_headings && seen_heading {
                    in_table = false;
                    blank(&mut cleaned);
                    cleaned.push(spoken_heading(&title));
                    cleaned.push(String::new());
                }
                seen_heading = true;
            }
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
            clean(r"ein vorangestelltes „\_“ vor den Variablen"),
            "ein vorangestelltes „Unterstrich“ vor den Variablen"
        );
        assert_eq!(
            clean("Hot 100 ausgewählt, verlinkt unter https://a.org/x und https://b.org/y."),
            "Hot 100 ausgewählt, verlinkt."
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
    fn headings_after_the_first_are_paragraphs() {
        let md = "# Zusammenfassung: Titel\n\n**Ippolito, Carlini**, INLG 2023\n\n\
                  ## Kernaussage\n\nViele Arbeiten.\n### 1. GitHub Copilots Filter ##\n\
                  - Erster Punkt\n\n## Fazit: kurz?\n\n## *\n\nEnde.";
        assert_eq!(
            super::clean_with(md, true),
            "Ippolito, Carlini, INLG 2023\n\nKernaussage.\n\nViele Arbeiten.\n\n\
             1. GitHub Copilots Filter.\n\nErster Punkt\n\nFazit: kurz?\n\nEnde."
        );
        // Without SPEAK_HEADINGS (and in previews) all headings go.
        assert_eq!(
            clean(md),
            "Ippolito, Carlini, INLG 2023\n\nViele Arbeiten.\n\nErster Punkt\n\nEnde."
        );
        // Headings in code blocks are code.
        assert_eq!(
            super::clean_with("# T\n\nA\n```\n# Kommentar\n```\n## B", true),
            "A\n\nB."
        );
    }

    #[test]
    fn heading_text_and_sentence_end() {
        assert_eq!(super::heading_text("## Titel ##").as_deref(), Some("Titel"));
        assert_eq!(super::heading_text("#Titel"), None);
        assert_eq!(super::heading_text("##  "), None);
        assert_eq!(super::spoken_heading("Teil (2023)"), "Teil (2023).");
        assert_eq!(super::spoken_heading("Was nun?“"), "Was nun?“");
    }

    #[test]
    fn html_entities_are_decoded() {
        assert_eq!(
            clean("Q&amp;A, 5&nbsp;GB, 1990&#8211;1995, &#x201E;so&#x201C;, &unknown; &#0;"),
            "Q&A, 5 GB, 1990–1995, „so“, &unknown; &#0;"
        );
    }

    #[test]
    fn rules_and_quotes() {
        assert_eq!(clean("A\n\n---\n\n> zitiert\n\nB"), "A\n\nzitiert\n\nB");
        assert_eq!(clean("![Bild eins](a.png) Text"), "Bild eins Text");
    }
}
