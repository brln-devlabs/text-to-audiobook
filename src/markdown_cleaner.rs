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

use regex::Regex;

static HEADER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\s*#+\s").unwrap());
static LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\([^\)]+\)").unwrap());
static BOLD_STAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*\*([^\*]+)\*\*").unwrap());
static ITALIC_STAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\*([^\*]+)\*").unwrap());
static BOLD_UNDERSCORE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"__([^_]+)__").unwrap());
static ITALIC_UNDERSCORE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_([^_]+)_").unwrap());
static INLINE_CODE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"`([^`]+)`").unwrap());
static URL: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"https?://[^\s]+").unwrap());
static HTML_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"<[^>]+>").unwrap());
static MANY_NEWLINES: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n{3,}").unwrap());

/// Remove headings, code fences, emphasis, links, URLs and HTML tags.
/// Paragraphs (blank-line separated) are preserved.
pub fn clean(md: &str) -> String {
    let mut cleaned: Vec<String> = Vec::new();
    let mut in_code_fence = false;

    for line in md.split('\n') {
        if line.trim().starts_with("```") {
            in_code_fence = !in_code_fence;
            continue;
        }
        if in_code_fence || HEADER.is_match(line) {
            continue;
        }

        let line = LINK.replace_all(line, "$1");
        let line = BOLD_STAR.replace_all(&line, "$1");
        let line = ITALIC_STAR.replace_all(&line, "$1");
        let line = BOLD_UNDERSCORE.replace_all(&line, "$1");
        let line = ITALIC_UNDERSCORE.replace_all(&line, "$1");
        let line = INLINE_CODE.replace_all(&line, "$1");
        let line = URL.replace_all(&line, "");
        let line = HTML_TAG.replace_all(&line, "");

        if !line.trim().is_empty() {
            cleaned.push(line.into_owned());
        } else if cleaned.last().is_some_and(|l| !l.is_empty()) {
            cleaned.push(String::new());
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
}
