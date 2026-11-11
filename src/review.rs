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

//! Guided review of the spots the number rules cannot decide
//! (`REVIEW_NUMBERS`): ordinal numbers without a clear article or
//! preposition in front and Roman numerals outside the safe patterns.
//!
//! Every spot comes with suggested readings. The choice is stored per
//! project in `cache/<project>/review.json` and applied to the text right
//! before the number normalization.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::config::Settings;
use crate::normalize_de::{
    MONTHS, ROMAN_KEYWORDS, cardinal, ordinal, ordinal_ending, prev_word_raw, prev_words,
    roman_value,
};
use crate::segment::Segment;
use crate::tts::resolve_language;

pub const REVIEW_FILE: &str = "review.json";

/// One suggested reading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    /// Text that replaces the spot.
    pub replacement: String,
    /// Short explanation shown next to the button.
    pub hint: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpotKind {
    Ordinal,
    Roman,
}

/// A place in a segment the rules cannot decide.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spot {
    /// Byte range in the segment text.
    pub start: usize,
    pub end: usize,
    pub original: String,
    pub kind: SpotKind,
    pub choices: Vec<Choice>,
    /// Equal keys = same situation (spot, word before, word after); used to
    /// apply one decision to all of them.
    pub key: String,
}

fn re(p: &str) -> Regex {
    Regex::new(p).expect("valid regex")
}

/// "3." followed by a word.
static ORDINAL_SPOT: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d{1,3})\.(\s+)(\p{L}+)"));
/// Roman numeral (I, V, X, L, C), optionally with a full stop.
static ROMAN_SPOT: LazyLock<Regex> = LazyLock::new(|| re(r"\b([IVXLC]{1,8})\b(\.?)"));
static NEXT_WORD: LazyLock<Regex> = LazyLock::new(|| re(r"^\s*(\p{L}+)"));

/// Roman-looking words that are usually something else.
const NOT_ROMAN: &[&str] = &[
    "CV", "XL", "XXL", "CC", "CL", "LI", "VC", "IC", "CI", "LC", "XC",
];

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn starts_upper(s: &str) -> bool {
    s.chars().next().is_some_and(char::is_uppercase)
}

fn is_month(w: &str) -> bool {
    MONTHS.contains(&w) || w == "Jänner"
}

fn choice(replacement: String, hint: &str) -> Choice {
    Choice {
        replacement,
        hint: hint.to_string(),
    }
}

fn ordinal_choices(n: u64, next_upper: bool, prev_is_word: bool) -> Vec<Choice> {
    let forms = vec![
        choice(ordinal(n, "e"), "die/der/das …-e"),
        choice(ordinal(n, "en"), "den/dem/des …-en"),
        choice(ordinal(n, "er"), "ein …-er (maskulin)"),
        choice(ordinal(n, "es"), "ein …-es (neutral)"),
        choice(
            capitalize(&ordinal(n, "e")),
            "als Substantiv („wurde Dritte“)",
        ),
        choice(
            capitalize(&ordinal(n, "er")),
            "als Substantiv („wurde Dritter“)",
        ),
    ];
    let cardinal_choice = choice(
        format!("{}.", cardinal(n)),
        "Grundzahl, der Punkt beendet den Satz",
    );
    let mut out = Vec::new();
    if next_upper && prev_is_word {
        out.push(cardinal_choice.clone());
        out.extend(forms);
    } else {
        out.extend(forms);
        out.push(cardinal_choice);
    }
    out
}

fn roman_choices(v: u64, dot: bool, after_name: bool) -> Vec<Choice> {
    let cap = |e: &str| capitalize(&ordinal(v, e));
    let regent = vec![
        choice(
            format!("der {}", cap("e")),
            "nach einem Namen (Nominativ: „… der …te kam“)",
        ),
        choice(
            format!("des {}", cap("en")),
            "nach einem Namen (Genitiv: „die Krone …s des …ten“)",
        ),
        choice(
            format!("dem {}", cap("en")),
            "nach einem Namen (Dativ: „mit … dem …ten“)",
        ),
        choice(
            format!("den {}", cap("en")),
            "nach einem Namen (Akkusativ: „für … den …ten“)",
        ),
    ];
    let plain = choice(
        format!("{}{}", cardinal(v), if dot { "." } else { "" }),
        "Grundzahl",
    );
    let ordinals = vec![
        choice(ordinal(v, "e"), "Ordnungszahl …-e"),
        choice(ordinal(v, "en"), "Ordnungszahl …-en"),
    ];
    let mut out = Vec::new();
    if after_name {
        out.extend(regent);
        out.push(plain);
        if dot {
            out.extend(ordinals);
        }
    } else if dot {
        out.extend(ordinals);
        out.push(plain);
        out.extend(regent);
    } else {
        out.push(plain);
        out.extend(regent);
    }
    out
}

/// All spots in `text` the rules cannot decide, in text order.
pub fn find_spots(text: &str) -> Vec<Spot> {
    let mut spots = Vec::new();

    for c in ORDINAL_SPOT.captures_iter(text) {
        let m = c.get(0).unwrap();
        let num = c.get(1).unwrap();
        let next = &c[3];
        let n: u64 = match num.as_str().parse() {
            Ok(n) if n > 0 => n,
            _ => continue,
        };
        // Part of a date like 03.05.1995 or 1.2.
        if text[..num.start()].ends_with(|ch: char| ch == '.' || ch.is_ascii_digit()) {
            continue;
        }
        if is_month(next) || next == "Uhr" {
            continue;
        }
        let (p1, p2) = prev_words(text, m.start());
        if ordinal_ending(p1.as_deref(), p2.as_deref()).is_some() {
            continue; // decided by the rules
        }
        let raw_prev = prev_word_raw(text, m.start());
        // "Seite 12. Dann" / "Platz 3. Das": noun in front, sentence ends.
        if starts_upper(next) && raw_prev.is_some_and(starts_upper) {
            continue;
        }
        let original = format!("{}.", num.as_str());
        spots.push(Spot {
            start: num.start(),
            end: num.end() + 1,
            key: format!("o|{}|{original}|{next}", p1.unwrap_or_default()),
            choices: ordinal_choices(n, starts_upper(next), raw_prev.is_some()),
            original,
            kind: SpotKind::Ordinal,
        });
    }

    for c in ROMAN_SPOT.captures_iter(text) {
        let num = c.get(1).unwrap();
        let word = num.as_str();
        let dot = !c[2].is_empty();
        let Some(v) = roman_value(word) else { continue };
        if NOT_ROMAN.contains(&word) {
            continue;
        }
        let raw_prev = prev_word_raw(text, num.start());
        if raw_prev.is_some_and(|p| ROMAN_KEYWORDS.contains(&p)) {
            continue; // "Kapitel IV" is decided by the rules
        }
        let after = &text[num.end() + c[2].len()..];
        let next = NEXT_WORD
            .captures(after)
            .map(|n| n[1].to_string())
            .unwrap_or_default();
        if dot && next.starts_with("Jahrhundert") {
            let (p1, p2) = prev_words(text, num.start());
            if ordinal_ending(p1.as_deref(), p2.as_deref()).is_some() {
                continue; // "im XIX. Jahrhundert" is decided by the rules
            }
        }
        let after_name =
            raw_prev.is_some_and(|p| starts_upper(p) && !p.chars().all(char::is_uppercase));
        // Single letters are usually initials ("Hans V. Müller") or the letter
        // itself; only "Karl V. dankte ab" style is asked.
        if word.len() == 1 && (!after_name || (dot && starts_upper(&next))) {
            continue;
        }
        // Word in capitals in front: a heading or an acronym.
        if raw_prev.is_some_and(|p| p.len() > 1 && p.chars().all(char::is_uppercase)) {
            continue;
        }
        let original = format!("{word}{}", &c[2]);
        spots.push(Spot {
            start: num.start(),
            end: num.end() + c[2].len(),
            key: format!("r|{}|{original}|{next}", raw_prev.unwrap_or_default()),
            choices: roman_choices(v, dot, after_name),
            original,
            kind: SpotKind::Roman,
        });
    }

    spots.sort_by_key(|s| s.start);
    // Overlaps (a numeral matched twice) are dropped.
    let mut out: Vec<Spot> = Vec::new();
    for s in spots {
        if out.last().is_none_or(|l| s.start >= l.end) {
            out.push(s);
        }
    }
    out
}

/// A stored decision. `replacement: None` = keep the program default.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decision {
    pub segment: usize,
    pub start: usize,
    pub original: String,
    pub replacement: Option<String>,
}

/// Decisions of one project (`cache/<project>/review.json`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReviewStore {
    pub decisions: Vec<Decision>,
    #[serde(skip)]
    path: PathBuf,
}

impl ReviewStore {
    pub fn load(cache_dir: &Path) -> ReviewStore {
        let path = cache_dir.join(REVIEW_FILE);
        let mut store: ReviewStore = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        store.path = path;
        store
    }

    pub fn save(&self) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&self.path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing {}", self.path.display()))
    }

    pub fn get(&self, segment: usize, spot: &Spot) -> Option<&Decision> {
        self.decisions
            .iter()
            .find(|d| d.segment == segment && d.start == spot.start && d.original == spot.original)
    }

    pub fn is_decided(&self, segment: usize, spot: &Spot) -> bool {
        self.get(segment, spot).is_some()
    }

    pub fn set(&mut self, segment: usize, spot: &Spot, replacement: Option<String>) {
        self.unset(segment, spot);
        self.decisions.push(Decision {
            segment,
            start: spot.start,
            original: spot.original.clone(),
            replacement,
        });
    }

    pub fn unset(&mut self, segment: usize, spot: &Spot) {
        self.decisions.retain(|d| {
            !(d.segment == segment && d.start == spot.start && d.original == spot.original)
        });
    }

    /// Apply the decisions of `segment` to its text (spots whose text has
    /// changed since are ignored).
    pub fn apply(&self, segment: usize, text: &str) -> String {
        let mut ds: Vec<&Decision> = self
            .decisions
            .iter()
            .filter(|d| d.segment == segment && d.replacement.is_some())
            .filter(|d| text.get(d.start..d.start + d.original.len()) == Some(d.original.as_str()))
            .collect();
        ds.sort_by_key(|d| std::cmp::Reverse(d.start));
        let mut out = text.to_string();
        let mut limit = usize::MAX;
        for d in ds {
            let end = d.start + d.original.len();
            if end > limit {
                continue;
            }
            out.replace_range(d.start..end, d.replacement.as_deref().unwrap_or_default());
            limit = d.start;
        }
        out
    }
}

/// A spot waiting for a decision, with its context.
#[derive(Debug, Clone)]
pub struct Item {
    pub project: String,
    pub segment: usize,
    pub spot: Spot,
    /// Up to ~80 characters before and after the spot.
    pub before: String,
    pub after: String,
}

fn context(text: &str, start: usize, end: usize) -> (String, String) {
    const WIDTH: usize = 80;
    let before = &text[..start];
    let mut b_start = before.len();
    for (n, (i, _)) in before.char_indices().rev().enumerate() {
        b_start = i;
        if n >= WIDTH {
            break;
        }
    }
    let mut before = before[b_start..].to_string();
    if b_start > 0
        && let Some(sp) = before.find(' ')
    {
        before = format!("…{}", &before[sp..]);
    }
    let after_full = &text[end..];
    let cut = after_full
        .char_indices()
        .nth(WIDTH)
        .map(|(i, _)| i)
        .unwrap_or(after_full.len());
    let mut after = after_full[..cut].to_string();
    if cut < after_full.len() {
        if let Some(sp) = after.rfind(' ') {
            after.truncate(sp);
        }
        after.push('…');
    }
    (before, after)
}

/// Whether the review applies to this segment (German text, normalization on).
pub fn applies(settings: &Settings, seg: &Segment) -> bool {
    settings.normalize_numbers && resolve_language(settings, seg.language.as_deref()) == "german"
}

/// All spots of a project, with decided ones marked by `store`.
pub fn items(project: &str, segments: &[Segment], settings: &Settings) -> Vec<Item> {
    let mut out = Vec::new();
    for (i, seg) in segments.iter().enumerate() {
        if !applies(settings, seg) {
            continue;
        }
        for spot in find_spots(&seg.normalized_text) {
            let (before, after) = context(&seg.normalized_text, spot.start, spot.end);
            out.push(Item {
                project: project.to_string(),
                segment: i,
                spot,
                before,
                after,
            });
        }
    }
    out
}

/// Spots of a project that still need a decision.
pub fn pending(
    project: &str,
    segments: &[Segment],
    settings: &Settings,
    store: &ReviewStore,
) -> Vec<Item> {
    items(project, segments, settings)
        .into_iter()
        .filter(|it| !store.is_decided(it.segment, &it.spot))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn originals(text: &str) -> Vec<String> {
        find_spots(text).into_iter().map(|s| s.original).collect()
    }

    #[test]
    fn finds_only_undecidable_spots() {
        // decided by rules: am 3., der 3., 3. Mai, Kapitel IV, im XIX. Jahrhundert
        let sure = "Am 3. Tag kam der 3. Versuch am 4. Mai. Kapitel IV folgt im XIX. Jahrhundert.";
        assert!(find_spots(sure).is_empty(), "{:?}", originals(sure));
        assert_eq!(originals("Sie kam als 3. ins Ziel."), ["3."]);
        assert_eq!(originals("Er wurde 3. Dann ging er."), ["3."]);
        assert!(find_spots("Siehe Seite 12. Dann weiter.").is_empty());
        assert_eq!(originals("Heinrich IV. war König."), ["IV."]);
        assert_eq!(originals("Ludwig XIV regierte lange."), ["XIV"]);
        assert!(find_spots("Hans V. Müller und ein XL-Shirt, die CD.").is_empty());
        assert_eq!(originals("Karl V. dankte ab."), ["V."]);
        assert!(find_spots("Es war am 03.05.1995.").is_empty());
    }

    #[test]
    fn choices_depend_on_context() {
        let s = &find_spots("Er wurde 3. Dann ging er.")[0];
        assert_eq!(s.choices[0].replacement, "drei.");
        let s = &find_spots("Sie kam als 3. ins Ziel.")[0];
        assert_eq!(s.choices[0].replacement, "dritte");
        assert!(s.choices.iter().any(|c| c.replacement == "Dritte"));
        let s = &find_spots("Heinrich IV. war König.")[0];
        assert_eq!(s.choices[0].replacement, "der Vierte");
        assert!(s.choices.iter().any(|c| c.replacement == "des Vierten"));
    }

    #[test]
    fn decisions_are_stored_and_applied() {
        let dir = tempfile::tempdir().unwrap();
        let text = "Heinrich IV. war König. Sie kam als 3. ins Ziel.";
        let spots = find_spots(text);
        let mut store = ReviewStore::load(dir.path());
        store.set(0, &spots[0], Some("der Vierte".into()));
        store.set(0, &spots[1], None);
        store.save().unwrap();
        let store = ReviewStore::load(dir.path());
        assert_eq!(store.decisions.len(), 2);
        assert_eq!(
            store.apply(0, text),
            "Heinrich der Vierte war König. Sie kam als 3. ins Ziel."
        );
        assert_eq!(store.apply(1, text), text);
        // Changed text: decision no longer matches and is ignored.
        assert_eq!(store.apply(0, "Ganz anderer Text."), "Ganz anderer Text.");
    }

    #[test]
    fn pending_skips_decided_and_non_german() {
        let settings = Settings {
            normalize_numbers: true,
            ..Settings::default()
        };
        let segs = vec![
            Segment::with_text("Sie kam als 3. ins Ziel und Ludwig XIV lachte."),
            Segment::with_text("Nichts zu tun."),
        ];
        let dir = tempfile::tempdir().unwrap();
        let mut store = ReviewStore::load(dir.path());
        let all = pending("Buch", &segs, &settings, &store);
        assert_eq!(all.len(), 2);
        assert!(all[0].before.ends_with("als "));
        assert!(all[0].after.starts_with(" ins Ziel"));
        store.set(all[0].segment, &all[0].spot, Some("dritte".into()));
        assert_eq!(pending("Buch", &segs, &settings, &store).len(), 1);
        let off = Settings::default();
        assert!(pending("Buch", &segs, &off, &store).is_empty());
    }
}
