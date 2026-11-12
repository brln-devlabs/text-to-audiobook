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
    roman_is_label, roman_value,
};
use crate::number_en;
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

/// Key shared by all enumeration spots ("I.", "II.", "2." at the start of a
/// sentence), so one choice can be applied to all of them.
pub const ENUMERATION_KEY: &str = "enum";

/// Words after a number that continue a range or a pair ("1. und 2.
/// Klasse", "1. bis 3. Mai"), not an enumeration item.
const NOT_ENUMERATION: &[&str] = &["und", "oder", "bis", "sowie", "bzw"];

/// Number at the start of a sentence, after a colon or at the start of a
/// segment: an enumeration item or a heading number ("2. Die Arbeiter …",
/// "1. allseitige …"). A single letter after it is an abbreviation
/// ("V. i. S. d. P.").
fn enumeration_start(raw_prev: Option<&str>, next: &str) -> bool {
    raw_prev.is_none()
        && (starts_upper(next) || (next.chars().count() > 1 && !NOT_ENUMERATION.contains(&next)))
}

/// "Zweitens:" for an enumeration item or heading number.
fn enumeration_choice(n: u64) -> Choice {
    choice(
        format!("{}:", capitalize(&ordinal(n, "ens"))),
        "Aufzählung / Gliederung („Zweitens: …“)",
    )
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
        choice(ordinal(v, "er"), "Ordnungszahl …-er (maskulin)"),
        choice(ordinal(v, "es"), "Ordnungszahl …-es (neutral)"),
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
    // English names and titles ("Henry VIII", "World War II").
    out.extend([
        choice(
            format!("the {}", capitalize(&number_en::ordinal(v))),
            "englisch nach einem Namen („Henry the Eighth“)",
        ),
        choice(
            format!("{}{}", number_en::cardinal(v), if dot { "." } else { "" }),
            "englische Grundzahl („World War two“)",
        ),
        choice(number_en::ordinal(v), "englische Ordnungszahl („eighth“)"),
    ]);
    out
}

/// All spots in `text` the rules cannot decide, in text order.
/// Hint of the choice that keeps the spot as written.
pub const KEEP_HINT: &str = "keep as written (no conversion)";

fn keep_choice(original: &str) -> Choice {
    Choice {
        replacement: original.to_string(),
        hint: KEEP_HINT.to_string(),
    }
}

/// Hint of the choice that takes the spot out of the text.
pub const REMOVE_HINT: &str = "remove from the text (not spoken)";

/// "Remove" is stored as an empty replacement.
pub fn is_removal(replacement: &str) -> bool {
    replacement.is_empty()
}

fn remove_choice() -> Choice {
    Choice {
        replacement: String::new(),
        hint: REMOVE_HINT.to_string(),
    }
}

/// Whether the spot can be removed: something is left to speak without it.
fn removable(text: &str, start: usize, end: usize) -> bool {
    text[..start]
        .chars()
        .chain(text[end..].chars())
        .any(char::is_alphanumeric)
}

/// Byte range taken out of `text` when the spot `start..end` is removed:
/// the spot plus the spaces after it, or, with none after it, the spaces in
/// front ("Heinrich IV, der" → "Heinrich, der"), so no double space or
/// space before punctuation is left. Line breaks stay.
fn removal_range(text: &str, start: usize, end: usize) -> (usize, usize) {
    let space = |c: char| c.is_whitespace() && c != '\n' && c != '\r';
    let tail = &text[end..];
    let after = tail.len() - tail.trim_start_matches(space).len();
    if after > 0 {
        return (start, end + after);
    }
    let head = &text[..start];
    (head.trim_end_matches(space).len(), end)
}

/// `before`, the replacement, `after`: how the excerpt reads with a choice.
pub fn with_choice(before: &str, replacement: &str, after: &str, original: &str) -> String {
    if !is_removal(replacement) {
        return format!("{before}{replacement}{after}");
    }
    let text = format!("{before}{original}{after}");
    let (from, to) = removal_range(&text, before.len(), before.len() + original.len());
    format!("{}{}", &text[..from], &text[to..])
}

/// True if `before` ends with "<number>, " (the spot ends a list of numbers).
fn after_number_in_list(before: &str) -> bool {
    let Some(rest) = before.trim_end().strip_suffix(',') else {
        return false;
    };
    let word = rest.rsplit(char::is_whitespace).next().unwrap_or("");
    !word.is_empty()
        && word
            .chars()
            .all(|c| c.is_ascii_digit() || c == ',' || c == '.')
}

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
        // Part of a decimal number ("0,75.", "0.75.") or a date ("03.05.1995", "1.2.").
        let before = &text[..num.start()];
        let after_separator = before
            .strip_suffix(['.', ','])
            .is_some_and(|b| b.ends_with(|ch: char| ch.is_ascii_digit()));
        if after_separator || before.ends_with(|ch: char| ch.is_ascii_digit()) {
            continue;
        }
        if is_month(next) || next == "Uhr" {
            continue;
        }
        // Last number of a list or table row: "GPT-3, 30, 25, 13. Berühmte …"
        if after_number_in_list(before) {
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
        // "2. Die Arbeiter …", "1. allseitige …": enumeration.
        let enumeration = enumeration_start(raw_prev, next);
        let mut choices = ordinal_choices(n, starts_upper(next), raw_prev.is_some());
        let key = if enumeration {
            choices.insert(0, enumeration_choice(n));
            ENUMERATION_KEY.to_string()
        } else {
            format!("o|{}|{original}|{next}", p1.unwrap_or_default())
        };
        spots.push(Spot {
            start: num.start(),
            end: num.end() + 1,
            key,
            choices,
            original,
            kind: SpotKind::Ordinal,
        });
    }

    for c in ROMAN_SPOT.captures_iter(text) {
        let num = c.get(1).unwrap();
        let word = num.as_str();
        let dot = !c[2].is_empty();
        let Some(v) = roman_value(word) else { continue };
        if NOT_ROMAN.contains(&word) || roman_is_label(&text[num.end()..]) {
            continue; // "Anhang C.1" is a section label
        }
        let raw_prev = prev_word_raw(text, num.start());
        if raw_prev.is_some_and(|p| ROMAN_KEYWORDS.contains(&p)) {
            continue; // "Kapitel IV" is decided by the rules
        }
        // "Foster, C., Phang, J." / "Pinto, H. P.": initials in a list of
        // names ("Lastname, X."), never a Roman numeral.
        if word.len() == 1 && dot && text[..num.start()].trim_end().ends_with(',') {
            continue;
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
        // "II. Kapitalistische Revolution" at the start of a sentence: enumeration.
        let enumeration = dot && enumeration_start(raw_prev, &next);
        // Single letters are usually initials ("Hans V. Müller") or the letter
        // itself; only "Karl V. dankte ab" style and enumerations are asked.
        if word.len() == 1 && !enumeration && (!after_name || (dot && starts_upper(&next))) {
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
            key: if enumeration {
                ENUMERATION_KEY.to_string()
            } else {
                format!("r|{}|{original}|{next}", raw_prev.unwrap_or_default())
            },
            choices: {
                let mut c = roman_choices(v, dot, after_name);
                if enumeration {
                    c.insert(0, enumeration_choice(v));
                }
                c
            },
            original,
            kind: SpotKind::Roman,
        });
    }

    // Last choices everywhere: leave the text exactly as written, or take
    // the spot out ("1. Einleitung" → "Einleitung").
    for spot in &mut spots {
        if !spot.choices.iter().any(|c| c.replacement == spot.original) {
            spot.choices.push(keep_choice(&spot.original));
        }
        if removable(text, spot.start, spot.end) {
            spot.choices.push(remove_choice());
        }
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

/// A stored decision. `replacement: None` = keep the program default;
/// a replacement equal to `original` = keep the text as written.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Decision {
    pub segment: usize,
    pub start: usize,
    pub original: String,
    pub replacement: Option<String>,
    /// The excerpt around the spot when it was decided: lets the decision
    /// follow its spot when the segment numbers shift.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
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
            context: None,
        });
    }

    /// Decide `it` and remember its excerpt (see [`Decision::context`]).
    pub fn decide(&mut self, it: &Item, replacement: Option<String>) {
        self.set(it.segment, &it.spot, replacement);
        if let Some(d) = self.decisions.last_mut() {
            d.context = Some(it.context());
        }
    }

    /// Match the decisions to the current spots: a decision whose spot moved
    /// (other segment number, other offset) follows it by its excerpt; old
    /// decisions without excerpt that match no spot any more are dropped.
    /// Returns true if anything changed.
    pub fn sync(&mut self, items: &[Item]) -> bool {
        let at = |d: &Decision, it: &Item| {
            d.segment == it.segment && d.start == it.spot.start && d.original == it.spot.original
        };
        let mut changed = false;
        let mut taken: Vec<bool> = items
            .iter()
            .map(|it| self.decisions.iter().any(|d| at(d, it)))
            .collect();
        let mut keep = Vec::with_capacity(self.decisions.len());
        for mut d in std::mem::take(&mut self.decisions) {
            if let Some(it) = items.iter().find(|it| at(&d, it)) {
                // Older decisions get their excerpt now, so they can follow
                // their spot from here on.
                if d.context.is_none() {
                    d.context = Some(it.context());
                    changed = true;
                }
                keep.push(d);
                continue;
            }
            let target = d.context.as_ref().and_then(|ctx| {
                items.iter().enumerate().position(|(i, it)| {
                    !taken[i] && it.spot.original == d.original && &it.context() == ctx
                })
            });
            match (target, &d.context) {
                (Some(i), _) => {
                    d.segment = items[i].segment;
                    d.start = items[i].spot.start;
                    taken[i] = true;
                    changed = true;
                    keep.push(d);
                }
                // The text may come back: keep it, but it is not applied.
                (None, Some(_)) => keep.push(d),
                (None, None) => changed = true,
            }
        }
        self.decisions = keep;
        changed
    }

    pub fn unset(&mut self, segment: usize, spot: &Spot) {
        self.decisions.retain(|d| {
            !(d.segment == segment && d.start == spot.start && d.original == spot.original)
        });
    }

    /// Apply the decisions of `segment` to its text. Only decisions on a
    /// spot that is in the text now count, so a decision left behind by a
    /// changed text can never hit an unrelated word.
    pub fn apply(&self, segment: usize, text: &str) -> String {
        let spots = find_spots(text);
        self.apply_where(segment, text, |d| {
            spots
                .iter()
                .any(|s| s.start == d.start && s.original == d.original)
        })
    }

    fn apply_where(&self, segment: usize, text: &str, ok: impl Fn(&Decision) -> bool) -> String {
        let mut ds: Vec<&Decision> = self
            .decisions
            .iter()
            .filter(|d| d.segment == segment && d.replacement.is_some())
            .filter(|d| text.get(d.start..d.start + d.original.len()) == Some(d.original.as_str()))
            .filter(|d| ok(d))
            .collect();
        ds.sort_by_key(|d| std::cmp::Reverse(d.start));
        let mut out = text.to_string();
        let mut limit = usize::MAX;
        for d in ds {
            let end = d.start + d.original.len();
            if end > limit {
                continue;
            }
            let rep = d.replacement.as_deref().unwrap_or_default();
            if is_removal(rep) {
                // On `out`: the part from `limit` on is already rewritten.
                let (from, to) = removal_range(&out, d.start, end);
                out.replace_range(from..to, "");
                limit = from;
                continue;
            }
            // "Keep as written": hidden from the number rules (see
            // `normalize_de::protect`), restored after normalization.
            let rep = if rep == d.original {
                crate::normalize_de::protect(rep)
            } else {
                rep.to_string()
            };
            out.replace_range(d.start..end, &rep);
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

impl Item {
    /// The excerpt as shown: before, spot, after.
    pub fn context(&self) -> String {
        format!("{}{}{}", self.before, self.spot.original, self.after)
    }
}

/// Load the decisions of a project and match them to its current spots
/// ([`ReviewStore::sync`]); saved again if anything moved.
pub fn load_synced(
    cache_dir: &Path,
    project: &str,
    segments: &[Segment],
    settings: &Settings,
) -> ReviewStore {
    let mut store = ReviewStore::load(cache_dir);
    if !store.decisions.is_empty()
        && store.sync(&items(project, segments, settings))
        && let Err(e) = store.save()
    {
        log::warn!("Could not save review decisions: {e:#}");
    }
    store
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

/// Number of segments shown above and below with "more context".
pub const CONTEXT_SEGMENTS: usize = 3;

/// Segments with their numbers.
pub type Numbered<'a> = Vec<(usize, &'a str)>;

/// Up to `n` segments before and after `segment`, with their numbers.
pub fn neighbours(texts: &[String], segment: usize, n: usize) -> (Numbered<'_>, Numbered<'_>) {
    let start = segment.saturating_sub(n);
    let end = (segment + 1 + n).min(texts.len());
    let above = (start..segment.min(texts.len()))
        .map(|i| (i, texts[i].as_str()))
        .collect();
    let below = (segment + 1..end).map(|i| (i, texts[i].as_str())).collect();
    (above, below)
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

/// File with all review decisions in context (`output/<project>/text/`).
pub const REPORT_FILE: &str = "review.txt";

/// Up to `n` words before `from` / after `to`: byte offsets.
fn widen(text: &str, from: usize, to: usize, n: usize) -> (usize, usize) {
    let mut start = from;
    for _ in 0..n {
        let head = text[..start].trim_end();
        match head.rfind(char::is_whitespace) {
            Some(i) => start = i + 1,
            None => {
                start = 0;
                break;
            }
        }
    }
    let mut end = to;
    for _ in 0..n {
        let tail = &text[end..];
        let skip = tail.len() - tail.trim_start().len();
        match tail[skip..].find(char::is_whitespace) {
            Some(i) => end += skip + i,
            None => {
                end = text.len();
                break;
            }
        }
    }
    (start, end)
}

/// The excerpt shown for `it` as it is spoken: every decision of the
/// segment that lies inside it applied, then the number rules. Two words
/// more on each side are read along ("im 18." needs its "Jahrhundert") and
/// cut off again when the rules left them unchanged.
fn spoken_excerpt(text: &str, it: &Item, store: &ReviewStore) -> String {
    const DOTS: &str = "…";
    let before = it.before.strip_prefix(DOTS).unwrap_or(&it.before);
    let after = it.after.strip_suffix(DOTS).unwrap_or(&it.after);
    let from = it.spot.start - before.len();
    let to = it.spot.end + after.len();
    let (wfrom, wto) = widen(text, from, to, 2);
    let local = ReviewStore {
        decisions: store
            .decisions
            .iter()
            .filter(|d| {
                d.segment == it.segment && d.start >= wfrom && d.start + d.original.len() <= wto
            })
            .map(|d| Decision {
                start: d.start - wfrom,
                ..d.clone()
            })
            .collect(),
        path: PathBuf::new(),
    };
    let applied = local.apply_where(it.segment, &text[wfrom..wto], |_| true);
    let mut spoken = crate::normalize_de::unprotect(&crate::normalize_de::normalize(&applied));
    let (pre, post) = (&text[wfrom..from], &text[to..wto]);
    let mut cut_pre = false;
    let mut cut_post = false;
    if !pre.chars().any(|c| c.is_ascii_digit())
        && let Some(rest) = spoken.strip_prefix(pre)
    {
        spoken = rest.to_string();
        cut_pre = true;
    }
    if !post.chars().any(|c| c.is_ascii_digit())
        && let Some(rest) = spoken.strip_suffix(post)
    {
        spoken = rest.to_string();
        cut_post = true;
    }
    // "…" where the shown text does not reach the segment's start or end.
    let shown_from = if cut_pre { from } else { wfrom };
    let shown_to = if cut_post { to } else { wto };
    format!(
        "{}{}{}",
        if shown_from > 0 { DOTS } else { "" },
        spoken.trim(),
        if shown_to < text.len() { DOTS } else { "" }
    )
}

/// Readable list of every uncertain spot of a project: the sentence around
/// it, what was decided and how it is spoken.
pub fn report(
    project: &str,
    segments: &[Segment],
    settings: &Settings,
    store: &ReviewStore,
) -> String {
    let all = items(project, segments, settings);
    let decided = all
        .iter()
        .filter(|it| store.is_decided(it.segment, &it.spot))
        .count();
    let mut out = format!(
        "# Prüfentscheidungen: {project}\n\n{} Stellen, {decided} entschieden, {} offen\n",
        all.len(),
        all.len() - decided
    );
    for it in &all {
        let decision = store.get(it.segment, &it.spot);
        let how = match decision {
            None => "OFFEN (Program default)".to_string(),
            Some(d) => match d.replacement.as_deref() {
                None => "Program default".to_string(),
                Some(r) if r == it.spot.original => "keep as written".to_string(),
                Some(r) if is_removal(r) => "remove (not spoken)".to_string(),
                Some(r) => format!("„{r}“"),
            },
        };
        let spoken = spoken_excerpt(&segments[it.segment].normalized_text, it, store);
        out.push_str(&format!(
            "\n{:06} | {}[{}]{}\n       → {how}\n       gesprochen: {spoken}\n",
            it.segment, it.before, it.spot.original, it.after
        ));
    }
    // Decisions whose spot is no longer in the text (the text changed).
    let stale: Vec<&Decision> = store
        .decisions
        .iter()
        .filter(|d| {
            !all.iter().any(|it| {
                it.segment == d.segment
                    && it.spot.start == d.start
                    && it.spot.original == d.original
            })
        })
        .collect();
    if !stale.is_empty() {
        out.push_str(&format!(
            "\n## {} Entscheidung(en) ohne passende Stelle (Text geändert, werden nicht angewendet)\n\n",
            stale.len()
        ));
        for d in stale {
            out.push_str(&format!(
                "{:06} | „{}“ → {}\n",
                d.segment,
                d.original,
                match d.replacement.as_deref() {
                    None => "Program default",
                    Some(r) if is_removal(r) => "remove",
                    Some(r) => r,
                }
            ));
        }
    }
    out
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
        assert!(find_spots("BLEU(g, s) > 0,75. Dieser Wert wurde gewählt.").is_empty());
        assert!(find_spots("Der Wert lag bei 0.75. dann war es gut.").is_empty());
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
        // English readings for names and titles.
        let s = &find_spots("Henry VIII married six times.")[0];
        for form in ["the Eighth", "eight", "eighth"] {
            assert!(s.choices.iter().any(|c| c.replacement == form), "{form}");
        }
        let s = &find_spots("Lenin, VIII. Parteitag der KPR(B).")[0];
        assert!(s.choices.iter().any(|c| c.replacement == "eight."));
        // "Lenin, VIII. Parteitag": all ordinal endings, "achter" included.
        let s = &find_spots("Lenin, VIII. Parteitag der KPR(B).")[0];
        for form in ["achte", "achten", "achter", "achtes"] {
            assert!(s.choices.iter().any(|c| c.replacement == form), "{form}");
        }
    }

    #[test]
    fn enumerations_get_zweitens() {
        let spots = find_spots("II. Kapitalistische Revolution: Umwandlung der Industrie.");
        assert_eq!(spots.len(), 1);
        assert_eq!(spots[0].choices[0].replacement, "Zweitens:");
        assert_eq!(spots[0].key, ENUMERATION_KEY);
        let spots = find_spots("I. Utopischer Sozialismus. Das war so. 3. Die Arbeiter kamen.");
        assert_eq!(spots.len(), 2);
        assert_eq!(spots[0].choices[0].replacement, "Erstens:");
        assert_eq!(spots[1].choices[0].replacement, "Drittens:");
        assert_eq!(spots[1].key, ENUMERATION_KEY);
        // Initials in a sentence are still skipped.
        assert!(find_spots("Er traf Hans V. Müller.").is_empty());
        // Lower case after the number: still an enumeration item.
        let spots = find_spots("1. allseitige Agitation unter den Truppen; 2. die Arbeiter.");
        assert_eq!(spots.len(), 2);
        assert_eq!(spots[0].choices[0].replacement, "Erstens:");
        assert_eq!(spots[1].choices[0].replacement, "Zweitens:");
        assert_eq!(spots[1].key, ENUMERATION_KEY);
        let spots = find_spots("Forderungen: II. die Arbeiter.");
        assert_eq!(spots[0].choices[0].replacement, "Zweitens:");
        // Ranges and pairs are not enumerations.
        let spots = find_spots("1. und 2. Klasse waren voll.");
        assert_ne!(spots[0].choices[0].replacement, "Erstens:");
    }

    #[test]
    fn neighbours_are_clamped() {
        let texts: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        let (above, below) = neighbours(&texts, 1, 3);
        assert_eq!(above, [(0, "t0")]);
        assert_eq!(below, [(2, "t2"), (3, "t3"), (4, "t4")]);
        let (above, below) = neighbours(&texts, 4, 3);
        assert_eq!(above.len(), 3);
        assert!(below.is_empty());
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

    #[test]
    fn section_labels_are_not_asked() {
        assert!(find_spots("Anhang C.1 gibt ein Verfahren an.").is_empty());
        assert!(find_spots("Siehe C.2 und D.1.").is_empty());
    }

    #[test]
    fn keep_as_written_is_offered_and_survives_normalization() {
        let text = "2. Die Arbeiter kamen.";
        let spots = find_spots(text);
        let keep = spots[0]
            .choices
            .iter()
            .find(|c| c.hint == KEEP_HINT)
            .unwrap();
        assert_eq!(keep.replacement, "2.");

        let dir = tempfile::tempdir().unwrap();
        let mut store = ReviewStore::load(dir.path());
        // Program default: the rules read the number.
        store.set(0, &spots[0], None);
        let default =
            crate::normalize_de::unprotect(&crate::normalize_de::normalize(&store.apply(0, text)));
        assert_eq!(default, "zwei. Die Arbeiter kamen.");
        // Keep as written: the digits reach the TTS untouched.
        store.set(0, &spots[0], Some(keep.replacement.clone()));
        let kept =
            crate::normalize_de::unprotect(&crate::normalize_de::normalize(&store.apply(0, text)));
        assert_eq!(kept, text);
    }

    fn spoken(store: &ReviewStore, text: &str) -> String {
        crate::normalize_de::unprotect(&crate::normalize_de::normalize(&store.apply(0, text)))
    }

    #[test]
    fn remove_is_the_last_choice_and_takes_the_spot_out() {
        let dir = tempfile::tempdir().unwrap();
        for (text, expected) in [
            ("2. Die Arbeiter kamen.", "Die Arbeiter kamen."),
            (
                "II. Kapitalistische Revolution: Umwandlung.",
                "Kapitalistische Revolution: Umwandlung.",
            ),
            ("Sie kam als 3. ins Ziel.", "Sie kam als ins Ziel."),
            ("Heinrich IV. war König.", "Heinrich war König."),
            (
                "Damals lebte Ludwig XIV, der König.",
                "Damals lebte Ludwig, der König.",
            ),
            ("Es regierte Ludwig XIV", "Es regierte Ludwig"),
            ("Sie kam als 3.\nins Ziel.", "Sie kam als\nins Ziel."),
        ] {
            let spots = find_spots(text);
            assert_eq!(spots.len(), 1, "{text}");
            let remove = spots[0].choices.last().unwrap();
            assert_eq!(remove.hint, REMOVE_HINT, "{text}");
            assert!(is_removal(&remove.replacement));
            let mut store = ReviewStore::load(dir.path());
            store.set(0, &spots[0], Some(remove.replacement.clone()));
            assert_eq!(spoken(&store, text), expected, "{text}");
        }
    }

    #[test]
    fn remove_is_not_offered_when_nothing_would_be_left() {
        let spots = find_spots("II.");
        assert_eq!(spots.len(), 1);
        assert!(!spots[0].choices.iter().any(|c| is_removal(&c.replacement)));
    }

    #[test]
    fn removals_next_to_other_decisions() {
        let dir = tempfile::tempdir().unwrap();
        let text = "Heinrich IV. war 3. im Rennen, Ludwig XIV, der König.";
        let spots = find_spots(text);
        assert_eq!(originals(text), ["IV.", "3.", "XIV"]);
        let mut store = ReviewStore::load(dir.path());
        store.set(0, &spots[0], Some(String::new()));
        store.set(0, &spots[1], Some("Dritter".into()));
        store.set(0, &spots[2], Some(String::new()));
        assert_eq!(
            spoken(&store, text),
            "Heinrich war Dritter im Rennen, Ludwig, der König."
        );
    }

    #[test]
    fn preview_with_removal() {
        assert_eq!(
            with_choice("Sie kam als ", "", " ins Ziel.", "3."),
            "Sie kam als ins Ziel."
        );
        assert_eq!(
            with_choice("", "", " Die Arbeiter …", "2."),
            "Die Arbeiter …"
        );
        assert_eq!(
            with_choice("Sie kam als ", "Dritte", " ins Ziel.", "3."),
            "Sie kam als Dritte ins Ziel."
        );
    }

    #[test]
    fn report_shows_removals() {
        let segments = segs(&["Sie kam als 3. ins Ziel."]);
        let dir = tempfile::tempdir().unwrap();
        let mut store = ReviewStore::load(dir.path());
        let spot = &find_spots(&segments[0].normalized_text)[0];
        store.set(0, spot, Some(String::new()));
        let r = report("Buch", &segments, &german(), &store);
        assert!(
            r.contains("→ remove (not spoken)\n       gesprochen: Sie kam als ins Ziel."),
            "{r}"
        );
    }

    #[test]
    fn initials_in_reference_lists_are_not_asked() {
        for t in [
            "Gao, L., Biderman, S., Hoppe, T., Foster, C., Phang, J., He, H.",
            "Chen, M., Pinto, H. P. de O., Kaplan, J. (2021).",
            "Lin, X. V., et al. (2022).",
        ] {
            assert!(find_spots(t).is_empty(), "{t}: {:?}", find_spots(t));
        }
        // Rulers are still asked.
        assert!(!find_spots("Damals dankte Karl V. ab.").is_empty());
    }

    #[test]
    fn last_number_of_a_table_row_is_not_asked() {
        let t = "Open-Source-Lizenzen (n=42), GPT-3 DaVinci v2, 30, 25, 25, 13. Berühmte Reden (n=100), GPT-3 DaVinci Original, 20, 1, 14, 0. Berühmte Reden (n=100), GPT-3 DaVinci v2, 12, 6, 11, 3. Berühmte Monologe";
        assert!(find_spots(t).is_empty(), "{:?}", find_spots(t));
        assert!(find_spots("Werte: 0,5, 1,25, 13. Danach").is_empty());
        // Without a list in front it is still asked.
        assert!(!find_spots("Sie kam als 3. ins Ziel.").is_empty());
    }

    #[test]
    fn report_lists_decisions_in_context() {
        let settings = Settings {
            normalize_numbers: true,
            tts_language: "german".into(),
            ..Settings::default()
        };
        let seg = |t: &str| Segment {
            normalized_text: t.into(),
            ..Segment::from_value(serde_json::json!({"normalized_text": t})).unwrap()
        };
        let segments = vec![
            seg("Sie kam als 3. ins Ziel."),
            seg("2. Die Arbeiter kamen."),
            seg("Er wurde 5. und ging."),
        ];
        let dir = tempfile::tempdir().unwrap();
        let mut store = ReviewStore::load(dir.path());
        let s0 = &find_spots(&segments[0].normalized_text)[0];
        store.set(0, s0, Some("Dritte".into()));
        let s1 = &find_spots(&segments[1].normalized_text)[0];
        store.set(1, s1, Some("2.".into()));
        store.decisions.push(Decision {
            segment: 7,
            start: 0,
            original: "IV.".into(),
            replacement: None,
            context: Some("… Heinrich IV. …".into()),
        });
        let r = report("Buch", &segments, &settings, &store);
        assert!(r.contains("3 Stellen, 2 entschieden, 1 offen"), "{r}");
        assert!(r.contains("000000 | Sie kam als [3.] ins Ziel.\n       → „Dritte“\n       gesprochen: Sie kam als Dritte ins Ziel."), "{r}");
        assert!(r.contains("000001 | [2.] Die Arbeiter kamen.\n       → keep as written\n       gesprochen: 2. Die Arbeiter kamen."), "{r}");
        assert!(
            r.contains("000002 | Er wurde [5.] und ging.\n       → OFFEN (Program default)"),
            "{r}"
        );
        assert!(r.contains("000007 | „IV.“ → Program default"), "{r}");
    }

    #[test]
    fn report_speaks_all_decisions_in_the_excerpt() {
        let settings = Settings {
            normalize_numbers: true,
            tts_language: "german".into(),
            ..Settings::default()
        };
        let t = "Sie kam als 3. ins Ziel. Er wurde 5. und ging.";
        let segments =
            vec![Segment::from_value(serde_json::json!({"normalized_text": t})).unwrap()];
        let dir = tempfile::tempdir().unwrap();
        let mut store = ReviewStore::load(dir.path());
        let spots = find_spots(t);
        store.set(0, &spots[0], Some("Dritte".into()));
        store.set(0, &spots[1], Some("Fünfter".into()));
        let r = report("Buch", &segments, &settings, &store);
        assert_eq!(
            r.matches("gesprochen: Sie kam als Dritte ins Ziel. Er wurde Fünfter und ging.")
                .count(),
            2,
            "{r}"
        );
    }

    fn german() -> Settings {
        Settings {
            normalize_numbers: true,
            tts_language: "german".into(),
            ..Settings::default()
        }
    }

    fn segs(texts: &[&str]) -> Vec<Segment> {
        texts
            .iter()
            .map(|t| Segment::from_value(serde_json::json!({"normalized_text": t})).unwrap())
            .collect()
    }

    #[test]
    fn decisions_follow_their_spot_when_numbers_shift() {
        let settings = german();
        let dir = tempfile::tempdir().unwrap();
        let old = segs(&["Vorwort.", "Sie kam als 3. ins Ziel."]);
        let mut store = ReviewStore::load(dir.path());
        let it = &items("B", &old, &settings)[0];
        // Decided by an older version: no excerpt yet; sync adds it.
        store.set(it.segment, &it.spot, Some("Dritte".into()));
        assert!(store.sync(&items("B", &old, &settings)));
        assert!(store.decisions[0].context.is_some());
        // A stale legacy entry (no excerpt) that matches nothing.
        store.decisions.push(Decision {
            segment: 9,
            start: 0,
            original: "C.".into(),
            replacement: Some("C.".into()),
            context: None,
        });
        // Two paragraphs were added in front: the spot is segment 3 now.
        let new = segs(&["Titel.", "Neu.", "Vorwort.", "Sie kam als 3. ins Ziel."]);
        let now = items("B", &new, &settings);
        assert!(store.sync(&now));
        assert_eq!(store.decisions.len(), 1, "legacy entry dropped");
        assert_eq!(store.decisions[0].segment, 3);
        assert!(store.is_decided(3, &now[0].spot));
        assert_eq!(
            store.apply(3, &new[3].normalized_text),
            "Sie kam als Dritte ins Ziel."
        );
        // Nothing to move a second time.
        assert!(!store.sync(&now));
    }

    #[test]
    fn decision_without_its_spot_is_not_applied() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = ReviewStore::load(dir.path());
        let t = "Sie kam als 3. ins Ziel.";
        let spot = find_spots(t).remove(0);
        store.set(0, &spot, Some("Dritte".into()));
        // Same offset and digits, but now decided by the rules ("am 3."):
        // no spot there any more, so the old choice must not hit it.
        let other = "Sie kam am 3. Mai an.";
        assert!(find_spots(other).is_empty());
        assert_eq!(store.apply(0, other), other);
    }

    #[test]
    fn spoken_excerpt_reads_the_words_after_the_cut() {
        // The excerpt ends right after "18." – without "schon" the rules
        // would read "achtzehn." instead of "achtzehnten".
        let text = "im 16. und 17. Jahrhundert Schilderungen; im 18. schon direkt kommunistische Theorien.";
        let spot = find_spots(text)
            .into_iter()
            .find(|s| s.original == "17.")
            .unwrap();
        let it = Item {
            project: "B".into(),
            segment: 0,
            before: text[..spot.start].to_string(),
            after: format!("{}…", &text[spot.end..text.find("schon").unwrap() - 1]),
            spot,
        };
        let dir = tempfile::tempdir().unwrap();
        let mut store = ReviewStore::load(dir.path());
        store.decide(&it, Some("siebzehnten".into()));
        let spoken = spoken_excerpt(text, &it, &store);
        assert_eq!(
            spoken,
            "im sechzehnten und siebzehnten Jahrhundert Schilderungen; im achtzehnten…"
        );
    }
}
