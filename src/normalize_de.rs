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

//! Rule-based normalization of German text for the TTS (`NORMALIZE_NUMBERS`).
//!
//! Spells out numbers, years, dates, ordinals, times, money, percentages,
//! units, ranges and common abbreviations. The rules are deliberately
//! conservative: where the right form is unclear (for example an ordinal
//! number without article or preposition in front), the text is left as it
//! is and the TTS reads it itself.

use std::sync::LazyLock;

use regex::{Captures, Regex};

const UNITS: [&str; 10] = [
    "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
];
const TEENS: [&str; 10] = [
    "zehn",
    "elf",
    "zwölf",
    "dreizehn",
    "vierzehn",
    "fünfzehn",
    "sechzehn",
    "siebzehn",
    "achtzehn",
    "neunzehn",
];
const TENS: [&str; 10] = [
    "", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig",
];
pub(crate) const MONTHS: [&str; 12] = [
    "Januar",
    "Februar",
    "März",
    "April",
    "Mai",
    "Juni",
    "Juli",
    "August",
    "September",
    "Oktober",
    "November",
    "Dezember",
];

/// Unit digit inside a compound ("ein" instead of "eins").
fn unit_compound(u: u64) -> &'static str {
    if u == 1 { "ein" } else { UNITS[u as usize] }
}

/// 0..=99; `final_eins` = "eins" for a trailing 1.
fn below_100(n: u64, final_eins: bool) -> String {
    match n {
        0 => String::new(),
        1 => if final_eins { "eins" } else { "ein" }.into(),
        2..=9 => UNITS[n as usize].into(),
        10..=19 => TEENS[(n - 10) as usize].into(),
        _ => {
            let (t, u) = (n / 10, n % 10);
            if u == 0 {
                TENS[t as usize].into()
            } else {
                format!("{}und{}", unit_compound(u), TENS[t as usize])
            }
        }
    }
}

/// 0..=999. `leading`: nothing precedes (100 -> "hundert", else "einhundert").
fn below_1000(n: u64, leading: bool, final_eins: bool) -> String {
    let (h, r) = (n / 100, n % 100);
    let mut s = String::new();
    if h == 1 {
        s.push_str(if leading { "hundert" } else { "einhundert" });
    } else if h > 1 {
        s.push_str(unit_compound(h));
        s.push_str("hundert");
    }
    s.push_str(&below_100(r, final_eins));
    s
}

/// Digits one by one ("007" -> "null null sieben").
fn digits(s: &str) -> String {
    s.chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| UNITS[d as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

/// Cardinal number in words (up to 999 999 999 999).
pub fn cardinal(n: u64) -> String {
    if n == 0 {
        return "null".into();
    }
    if n >= 1_000_000_000_000 {
        return digits(&n.to_string());
    }
    let mut parts: Vec<String> = Vec::new();
    let billions = n / 1_000_000_000;
    let millions = (n / 1_000_000) % 1000;
    let thousands = (n / 1000) % 1000;
    let rest = n % 1000;
    for (value, one, many) in [
        (billions, "eine Milliarde", "Milliarden"),
        (millions, "eine Million", "Millionen"),
    ] {
        if value == 1 {
            parts.push(one.into());
        } else if value > 1 {
            parts.push(format!("{} {many}", below_1000(value, true, false)));
        }
    }
    let mut tail = String::new();
    if thousands > 0 {
        let leading = parts.is_empty();
        if thousands == 1 {
            tail.push_str(if leading { "tausend" } else { "eintausend" });
        } else {
            tail.push_str(&below_1000(thousands, leading, false));
            tail.push_str("tausend");
        }
    }
    if rest > 0 {
        let leading = parts.is_empty() && tail.is_empty();
        tail.push_str(&below_1000(rest, leading, true));
    }
    if !tail.is_empty() {
        parts.push(tail);
    }
    parts.join(" ")
}

/// Year 1100..=1999 as spoken ("neunzehnhundertfünfundneunzig"); other
/// years are read as cardinals ("zweitausendzwanzig").
pub fn year(n: u64) -> String {
    if (1100..=1999).contains(&n) {
        format!(
            "{}hundert{}",
            below_100(n / 100, false),
            below_100(n % 100, true)
        )
    } else {
        cardinal(n)
    }
}

/// Ordinal stem without ending ("dritt", "zwanzigst").
fn ordinal_stem(n: u64) -> String {
    match n {
        1 => "erst".into(),
        3 => "dritt".into(),
        7 => "siebt".into(),
        8 => "acht".into(),
        2..=19 => format!("{}t", cardinal(n)),
        _ if !n.is_multiple_of(100) && n % 100 < 20 => {
            format!("{}{}", cardinal(n - n % 100), ordinal_stem(n % 100))
        }
        _ => format!("{}st", cardinal(n)),
    }
}

/// Ordinal with an ending: "e" (der dritte), "en" (am dritten), "er" (dritter Mai).
pub fn ordinal(n: u64, ending: &str) -> String {
    format!("{}{ending}", ordinal_stem(n))
}

/// "1.234,5" -> (1234, Some("5")); None for malformed input.
fn parse_number(s: &str) -> Option<(u64, Option<String>)> {
    let (int, frac) = match s.split_once(',') {
        Some((i, f)) => (i, Some(f.to_string())),
        None => (s, None),
    };
    let int: String = int.chars().filter(|c| *c != '.').collect();
    Some((int.parse().ok()?, frac))
}

static THOUSANDS: LazyLock<Regex> = LazyLock::new(|| re(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?$"));

/// Number token in words: cardinal, decimal ("drei Komma eins vier"),
/// leading zeros digit by digit. A decimal point ("0.75") is read like a
/// decimal comma; "3.500" with groups of three is a thousands separator.
fn number_words(token: &str) -> String {
    if !token.contains(',')
        && token.contains('.')
        && !THOUSANDS.is_match(token)
        && let Some((int, frac)) = token.split_once('.')
        && let Ok(i) = int.parse::<u64>()
    {
        return format!("{} Komma {}", cardinal(i), digits(frac));
    }
    let plain = !token.contains(['.', ',']);
    if plain && token.len() > 1 && token.starts_with('0') {
        return digits(token);
    }
    if plain && token.len() > 12 {
        return digits(token);
    }
    match parse_number(token) {
        Some((int, Some(frac))) if !frac.is_empty() => {
            format!("{} Komma {}", cardinal(int), digits(&frac))
        }
        Some((int, _)) => cardinal(int),
        None => token.to_string(),
    }
}

/// Number with a following noun: "ein"/"eine" for 1.
fn count_with(token: &str, singular: &str, plural: &str, feminine: bool) -> String {
    if token == "1" {
        format!("{} {singular}", if feminine { "eine" } else { "ein" })
    } else {
        format!("{} {plural}", number_words(token))
    }
}

/// Money: "5,50" + "Euro" -> "fünf Euro fünfzig".
fn money(token: &str, currency: &str) -> String {
    // "1.50" with a decimal point reads like "1,50".
    let token = if !token.contains(',') && token.contains('.') && !THOUSANDS.is_match(token) {
        token.replace('.', ",")
    } else {
        token.to_string()
    };
    let token = token.as_str();
    let Some((whole, frac)) = parse_number(token) else {
        return format!("{token} {currency}");
    };
    let cents: u64 = match frac.as_deref() {
        None | Some("") | Some("-") | Some("–") => 0,
        Some(f) if f.len() == 1 => f.parse::<u64>().unwrap_or(0) * 10,
        Some(f) => f[..2.min(f.len())].parse().unwrap_or(0),
    };
    let main = match whole {
        0 if cents > 0 => return format!("{} Cent", cardinal(cents)),
        1 => format!("ein {currency}"),
        _ => format!("{} {currency}", cardinal(whole)),
    };
    if cents > 0 {
        format!("{main} {}", cardinal(cents))
    } else {
        main
    }
}

static WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[\p{L}]+").unwrap());

/// The word directly before byte offset `idx` as written (None after
/// sentence punctuation).
pub(crate) fn prev_word_raw(text: &str, idx: usize) -> Option<&str> {
    let before = &text[..idx];
    if before.trim_end().ends_with(['.', '!', '?', ':', ';']) {
        return None;
    }
    WORD.find_iter(before).last().map(|m| m.as_str())
}

/// The last two words before byte offset `idx`, lower case (last first).
pub(crate) fn prev_words(text: &str, idx: usize) -> (Option<String>, Option<String>) {
    let before = &text[..idx];
    let words: Vec<&str> = WORD.find_iter(before).map(|m| m.as_str()).collect();
    // Only words directly in front (no sentence punctuation in between).
    let tail = before.trim_end();
    if tail.ends_with(['.', '!', '?', ':', ';']) {
        return (None, None);
    }
    let last = words.last().map(|w| w.to_lowercase());
    let second = words.len().checked_sub(2).map(|i| words[i].to_lowercase());
    (last, second)
}

const PREPOSITIONS: &[&str] = &[
    "in",
    "an",
    "auf",
    "aus",
    "bei",
    "mit",
    "nach",
    "von",
    "zu",
    "vor",
    "hinter",
    "unter",
    "über",
    "neben",
    "zwischen",
    "seit",
    "während",
    "wegen",
    "trotz",
    "statt",
    "innerhalb",
    "außerhalb",
];

/// Ending of an ordinal from the words in front, or None if unclear.
pub(crate) fn ordinal_ending(prev: Option<&str>, prev2: Option<&str>) -> Option<&'static str> {
    let prev = prev?;
    match prev {
        "am" | "im" | "vom" | "zum" | "beim" | "zur" | "dem" | "den" | "des" | "einem"
        | "einen" | "eines" | "jedem" | "diesem" | "diesen" | "seinem" | "ihrem" | "seinen"
        | "ihren" => Some("en"),
        "der" if prev2.is_some_and(|w| PREPOSITIONS.contains(&w)) => Some("en"),
        "der" | "die" | "das" | "jeder" | "jede" | "jedes" | "dieser" | "diese" | "dieses" => {
            Some("e")
        }
        _ => None,
    }
}

/// Roman numeral for 1..=3999 ("XIV").
fn to_roman(mut n: u64) -> String {
    const TABLE: [(u64, &str); 13] = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ];
    let mut out = String::new();
    for (value, sym) in TABLE {
        while n >= value {
            out.push_str(sym);
            n -= value;
        }
    }
    out
}

/// Value of a Roman numeral in canonical form ("XIV" -> 14); None for
/// anything else ("IIII", "VX", "ABC").
pub fn roman_value(s: &str) -> Option<u64> {
    if s.is_empty() || s.len() > 15 {
        return None;
    }
    let val = |c: char| match c {
        'I' => Some(1i64),
        'V' => Some(5),
        'X' => Some(10),
        'L' => Some(50),
        'C' => Some(100),
        'D' => Some(500),
        'M' => Some(1000),
        _ => None,
    };
    let vals: Vec<i64> = s.chars().map(val).collect::<Option<_>>()?;
    let mut total = 0i64;
    for (i, v) in vals.iter().enumerate() {
        if vals.get(i + 1).is_some_and(|next| next > v) {
            total -= v;
        } else {
            total += v;
        }
    }
    let total = u64::try_from(total)
        .ok()
        .filter(|t| (1..=3999).contains(t))?;
    (to_roman(total) == s).then_some(total)
}

/// True if the text after a letter is a section number ("C.1"): the letter
/// is a label, not a Roman numeral.
pub(crate) fn roman_is_label(after: &str) -> bool {
    after
        .strip_prefix('.')
        .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
}

/// Words after which a Roman numeral is a plain number ("Kapitel IV").
pub(crate) const ROMAN_KEYWORDS: &[&str] = &[
    "Kapitel",
    "Teil",
    "Band",
    "Buch",
    "Akt",
    "Szene",
    "Abschnitt",
    "Anhang",
    "Folge",
    "Staffel",
    "Klasse",
    "Stufe",
    "Phase",
    "Artikel",
    "Tafel",
    "Abbildung",
    "Paragraph",
    "Tabelle",
    "Bd",
];

/// Replace every match; `f` gets the captures and the byte offset of the
/// match and returns None to keep the original text.
fn replace_ctx(re: &Regex, text: &str, f: impl Fn(&Captures, usize) -> Option<String>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for caps in re.captures_iter(text) {
        let m = caps.get(0).unwrap();
        if let Some(rep) = f(&caps, m.start()) {
            out.push_str(&text[last..m.start()]);
            out.push_str(&rep);
            last = m.end();
        }
    }
    out.push_str(&text[last..]);
    out
}

fn re(pattern: &str) -> Regex {
    Regex::new(pattern).expect("valid regex")
}

static ABBREVIATIONS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"\bz\.\s?B\.", "zum Beispiel"),
        (r"\bd\.\s?h\.", "das heißt"),
        (r"\bu\.\s?a\.", "unter anderem"),
        (r"\bu\.\s?U\.", "unter Umständen"),
        (r"\bz\.\s?T\.", "zum Teil"),
        (r"\bv\.\s?a\.", "vor allem"),
        (r"\bo\.\s?Ä\.", "oder Ähnliches"),
        (r"\bbzw\.", "beziehungsweise"),
        (r"\bca\.", "circa"),
        (r"\bevtl\.", "eventuell"),
        (r"\bggf\.", "gegebenenfalls"),
        (r"\bvgl\.", "vergleiche"),
        (r"\binkl\.", "inklusive"),
        (r"\bzzgl\.", "zuzüglich"),
        (r"\bDr\.", "Doktor"),
        (r"\bProf\.", "Professor"),
        (r"\bNr\.", "Nummer"),
        (r"\bStr\.", "Straße"),
        (r"\bJh\.", "Jahrhundert"),
        (r"\bMio\.", "Millionen"),
        (r"\bMrd\.", "Milliarden"),
        (r"\bAbb\.", "Abbildung"),
        (r"\bTab\.", "Tabelle"),
        (r"\bKap\.", "Kapitel"),
        (r"\bAbschn\.", "Abschnitt"),
    ]
    .into_iter()
    .map(|(p, r)| (re(p), r))
    .collect()
});

/// "usw." / "etc." keep the full stop when they end a sentence.
static SENTENCE_ABBREVIATIONS: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    [
        (r"\busw\.(\s+\p{Lu}|\s*$)?", "und so weiter"),
        (r"\betc\.(\s+\p{Lu}|\s*$)?", "et cetera"),
    ]
    .into_iter()
    .map(|(p, r)| (re(p), r))
    .collect()
});

static PAGE: LazyLock<Regex> = LazyLock::new(|| re(r"\bS\.\s?(\d)"));
static PAGES: LazyLock<Regex> = LazyLock::new(|| re(r"\bpp\.\s?(\d)"));

/// Words in front of a section number ("Abschnitt 3.3", "Anhang B.1").
const SECTION_KEYWORDS: &str = "Abschnitt|Abschnitte|Abschnitten|Kapitel|Kapiteln|Anhang|Anhänge|Anhängen|Tabelle|Tabellen|Abbildung|Abbildungen|Teil|Gleichung|Formel|Algorithmus|Definition|Satz|Lemma|Theorem|Version|Section|Appendix|Table|Figure|Fig|Eq";
static SECTION: LazyLock<Regex> = LazyLock::new(|| {
    re(&format!(
        r"\b({SECTION_KEYWORDS})(\.?\s+)((?:[A-Z]|\d{{1,3}})(?:\.\d{{1,3}})+)\b"
    ))
});
/// "B.1" / "C.2.3": appendix-style numbers without a keyword.
static LETTER_SECTION: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b([A-Z])\.(\d{1,3}(?:\.\d{1,3})*)\b"));
/// "arXiv:2108.01624v2" is read digit by digit.
static ARXIV: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b(arXiv):?\s?(\d{4})\.(\d{4,5})(?:v(\d+))?\b"));
/// Model sizes: "125M", "6B", "1.3B" (millions, billions, trillions).
static MAGNITUDE: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d+(?:[.,]\d+)?)([MBT])\b"));
/// Version numbers glued to a name: "SQuAD2.0", "GPT-3.5".
static VERSION: LazyLock<Regex> = LazyLock::new(|| re(r"(\p{L})(-?)(\d+(?:\.\d+)+)\b"));
/// Superscript digits after a number are an exponent: "10²".
static EXPONENT: LazyLock<Regex> = LazyLock::new(|| re(r"(\d)([¹²³⁰⁴⁵⁶⁷⁸⁹]+)"));
static LESS: LazyLock<Regex> = LazyLock::new(|| re(r"(^|[\s(\[])<\s?"));
static GREATER: LazyLock<Regex> = LazyLock::new(|| re(r"(^|[\s(\[])>\s?"));
static APPROX: LazyLock<Regex> = LazyLock::new(|| re(r"(^|[\s(\[])~\s?(\d)"));
static PLUS: LazyLock<Regex> = LazyLock::new(|| re(r"\s\+\s"));
/// "%" without a number in front ("% Beispiele mit Treffer").
static LONE_PERCENT: LazyLock<Regex> = LazyLock::new(|| re(r"(^|[\s(])%"));
/// "0,53–0,59" -> "0,53 bis 0,59"
static DECIMAL_RANGE: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d+,\d+)\s?[–-]\s?(\d+,\d+)\b"));
/// Journal references "33:1877–1901", "13(7):422–426": volume, issue, pages.
static JOURNAL: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b(\d{1,4})(?:\((\d{1,4})\))?:(\d{1,6})\s?[–-]\s?(\d{1,6})\b"));
/// Page numbers are cardinals, never years ("Seite 1173").
static PAGE_NUMBERS: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b(Seiten?)\s+(\d{1,6})(?:\s?[–-]\s?(\d{1,6}))?\b"));
/// "600 000", "1 250 000": thousands separated by spaces.
static SPACED_THOUSANDS: LazyLock<Regex> =
    LazyLock::new(|| re(r"(^|[^\d.,])(\d{1,3}(?:[ \u{00A0}\u{2009}\u{202F}]\d{3})+)($|[^\d.,])"));
static MULTI_SPACE: LazyLock<Regex> = LazyLock::new(|| re(r"[ \t]{2,}"));
static DAY_RANGE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"\b(\d{1,2})\.\s?[–-]\s?(\d{1,2})\.\s+(Januar|Jänner|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\b",
    )
});

/// Mathematical symbols in words: "n=11", "s ∈ D", "BLEU > 0,75".
fn symbols(text: &str) -> String {
    let mut s = text.to_string();
    for (sym, word) in [
        ("≠", " ungleich "),
        ("≤", " kleiner gleich "),
        ("≥", " größer gleich "),
        ("≈", " ungefähr "),
        ("∉", " nicht Element von "),
        ("∈", " Element von "),
        ("×", " mal "),
        ("±", " plus minus "),
        ("^", " hoch "),
        ("=", " gleich "),
        ("©", " Copyright "),
        ("⇒", " zu "),
        ("→", " zu "),
    ] {
        s = s.replace(sym, word);
    }
    s = LESS.replace_all(&s, "${1}kleiner als ").into_owned();
    s = GREATER.replace_all(&s, "${1}größer als ").into_owned();
    s = APPROX.replace_all(&s, "${1}ungefähr $2").into_owned();
    s = PLUS.replace_all(&s, " plus ").into_owned();
    s = LONE_PERCENT.replace_all(&s, "${1}Prozent").into_owned();
    let s = MULTI_SPACE.replace_all(&s, " ");
    // Keep the original edges ("© 2023" at the start must not gain a space).
    let lead = text.len() - text.trim_start().len();
    let trail = text.len() - text.trim_end().len();
    format!(
        "{}{}{}",
        &text[..lead],
        s.trim(),
        &text[text.len() - trail..]
    )
}

/// "3.3" -> "drei Punkt drei", "B.1" -> "B Punkt eins".
fn section_number(s: &str) -> String {
    s.split('.')
        .map(|p| match p.parse::<u64>() {
            Ok(n) => cardinal(n),
            Err(_) => p.to_string(),
        })
        .collect::<Vec<_>>()
        .join(" Punkt ")
}

/// "125" + "M" -> "hundertfünfundzwanzig Millionen".
fn magnitude(num: &str, unit: &str) -> String {
    let (one, many) = match unit {
        "M" => ("eine Million", "Millionen"),
        "B" => ("eine Milliarde", "Milliarden"),
        _ => ("eine Billion", "Billionen"),
    };
    if num == "1" {
        one.to_string()
    } else {
        format!("{} {many}", number_words(num))
    }
}

/// Superscript digits as ASCII digits.
fn superscript_digits(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '⁰' => '0',
            '¹' => '1',
            '²' => '2',
            '³' => '3',
            '⁴' => '4',
            '⁵' => '5',
            '⁶' => '6',
            '⁷' => '7',
            '⁸' => '8',
            '⁹' => '9',
            o => o,
        })
        .collect()
}
static ROMAN_AFTER_KEYWORD: LazyLock<Regex> = LazyLock::new(|| {
    re(&format!(
        r"\b({})(\.?\s+)([IVXLC]+)\b",
        ROMAN_KEYWORDS.join("|")
    ))
});
static ROMAN_CENTURY: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b([IVXLC]+)\.(\s+)(Jahrhundert\p{L}*|Jh\.)"));
static NUMERIC_DATE: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"));
/// English dates in headers and bylines: "Nov 7, 2026", "September 11, 2023".
static ENGLISH_DATE: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),\s+(\d{4})\b",
    )
});
/// Month index (0-based) of an English month name or abbreviation.
fn english_month(name: &str) -> Option<usize> {
    const ABBR: [&str; 12] = [
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    ];
    let key = name.get(..3)?.to_ascii_lowercase();
    ABBR.iter().position(|m| *m == key)
}
static DAY_MONTH: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"\b(\d{1,2})\.\s+(Januar|Jänner|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\b",
    )
});
static DAY_MONTH_NUMERIC: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d{1,2})\.(\d{1,2})\.(\s|$)"));
static TIME: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b([01]?\d|2[0-3])(?:[:.]([0-5]\d))?\s?Uhr\b"));
static ORDINAL: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d{1,4})\.(\s+)(\p{L})"));
static RANGE: LazyLock<Regex> =
    LazyLock::new(|| re(r"(^|[^\d.,\-–])(\d{1,4})\s?[–-]\s?(\d{1,4})($|[^\d.,\-–]|[.,](?:\s|$))"));
static CURRENCY_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| re(r"(€|\$)\s?(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+\.\d+|\d+(?:,\d+)?)"));
static WITH_UNIT: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"\b(\d{1,3}(?:\.\d{3})+(?:,(?:\d+|–|-))?|\d+\.\d+|\d+(?:,(?:\d+|–|-))?)\s?(?:(%|€|\$|°C|°)|(km/h|km|kg|cm|mm|ml|mg|ms|m|g|l|h|min|GB|MB|TB|KB|kB|GHz|MHz|EUR|Euro|Prozent|Dollar)\b)",
    )
});
static NEGATIVE: LazyLock<Regex> = LazyLock::new(|| re(r"(^|[\s(])[-–−](\d)"));
static DECADE: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d+)er\b"));
static NUMBER: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+\.\d+|\d+,\d+|\d+)\b"));

/// Characters of protected text live in a private-use plane, where no rule
/// matches them; `unprotect` brings them back after `normalize`.
const PROTECT_OFFSET: u32 = 0xF0000;

/// Hide `text` from `normalize` (review choice "keep as written").
pub fn protect(text: &str) -> String {
    text.chars()
        .map(|c| match char::from_u32(PROTECT_OFFSET + c as u32) {
            Some(p) if (c as u32) < 0xFFFE => p,
            _ => c,
        })
        .collect()
}

/// Undo `protect`.
pub fn unprotect(text: &str) -> String {
    text.chars()
        .map(|c| match c as u32 {
            v if (PROTECT_OFFSET..PROTECT_OFFSET + 0xFFFE).contains(&v) => {
                char::from_u32(v - PROTECT_OFFSET).unwrap_or(c)
            }
            _ => c,
        })
        .collect()
}

/// Spell out numbers, dates, times, units and abbreviations in German text.
pub fn normalize(text: &str) -> String {
    // 10² -> 10 hoch 2
    let s = EXPONENT
        .replace_all(text, |c: &Captures| {
            format!("{} hoch {}", &c[1], superscript_digits(&c[2]))
        })
        .into_owned();
    let mut s = symbols(&s);

    // 600 000 -> 600000 (space, no-break or thin space between groups of three)
    s = SPACED_THOUSANDS
        .replace_all(&s, |c: &Captures| {
            let digits: String = c[2].chars().filter(char::is_ascii_digit).collect();
            format!("{}{digits}{}", &c[1], &c[3])
        })
        .into_owned();

    for (r, word) in SENTENCE_ABBREVIATIONS.iter() {
        s = r
            .replace_all(&s, |c: &Captures| match c.get(1) {
                Some(rest) => format!("{word}.{}", rest.as_str()),
                None => word.to_string(),
            })
            .into_owned();
    }
    for (r, word) in ABBREVIATIONS.iter() {
        s = r.replace_all(&s, *word).into_owned();
    }
    s = PAGE.replace_all(&s, "Seite $1").into_owned();
    s = PAGES.replace_all(&s, "Seiten $1").into_owned();

    // Communications of the ACM, 13(7):422–426 -> Band dreizehn, Heft sieben, Seiten …
    s = JOURNAL
        .replace_all(&s, |c: &Captures| {
            let n = |i: usize| cardinal(c[i].parse().unwrap_or(0));
            let issue = c
                .get(2)
                .map(|m| format!(" Heft {},", cardinal(m.as_str().parse().unwrap_or(0))))
                .unwrap_or_default();
            format!("Band {},{issue} Seiten {} bis {}", n(1), n(3), n(4))
        })
        .into_owned();
    s = PAGE_NUMBERS
        .replace_all(&s, |c: &Captures| {
            let n = |m: &str| cardinal(m.parse().unwrap_or(0));
            match c.get(3) {
                Some(to) => format!("{} {} bis {}", &c[1], n(&c[2]), n(to.as_str())),
                None => format!("{} {}", &c[1], n(&c[2])),
            }
        })
        .into_owned();

    // arXiv:2108.01624 -> arXiv zwei eins null acht Punkt null eins sechs zwei vier
    s = ARXIV
        .replace_all(&s, |c: &Captures| {
            let mut out = format!("{} {} Punkt {}", &c[1], digits(&c[2]), digits(&c[3]));
            if let Some(v) = c.get(4) {
                out.push_str(&format!(" Version {}", number_words(v.as_str())));
            }
            out
        })
        .into_owned();

    // Abschnitt 3.3 -> Abschnitt drei Punkt drei; Anhang B.1 -> Anhang B Punkt eins
    s = SECTION
        .replace_all(&s, |c: &Captures| {
            format!("{}{}{}", &c[1], &c[2], section_number(&c[3]))
        })
        .into_owned();
    s = LETTER_SECTION
        .replace_all(&s, |c: &Captures| {
            format!("{} Punkt {}", &c[1], section_number(&c[2]))
        })
        .into_owned();

    // 1.3B -> eins Komma drei Milliarden
    s = MAGNITUDE
        .replace_all(&s, |c: &Captures| magnitude(&c[1], &c[2]))
        .into_owned();

    // SQuAD2.0 -> SQuAD zwei Punkt null, GPT-3.5 -> GPT-drei Punkt fünf
    s = VERSION
        .replace_all(&s, |c: &Captures| {
            let sep = if c[2].is_empty() { " " } else { "-" };
            format!("{}{sep}{}", &c[1], section_number(&c[3]))
        })
        .into_owned();

    // Kapitel IV -> Kapitel vier. "Anhang C" and "Anhang C.1" are letters.
    let t = s.clone();
    s = replace_ctx(&ROMAN_AFTER_KEYWORD, &t, |c, _| {
        let numeral = &c[3];
        let after = &t[c.get(0).unwrap().end()..];
        if matches!(numeral, "C" | "L") || roman_is_label(after) {
            return None;
        }
        let v = roman_value(numeral)?;
        Some(format!("{}{}{}", &c[1], &c[2], cardinal(v)))
    });

    // im XIX. Jahrhundert -> im neunzehnten Jahrhundert (with clear context)
    let t = s.clone();
    s = replace_ctx(&ROMAN_CENTURY, &t, |c, at| {
        let v = roman_value(&c[1])?;
        let (p1, p2) = prev_words(&t, at);
        let ending = ordinal_ending(p1.as_deref(), p2.as_deref())?;
        Some(format!("{}{}{}", ordinal(v, ending), &c[2], &c[3]))
    });

    // Nov 7, 2026 -> 7. November 2026, then read like a German date below
    s = ENGLISH_DATE
        .replace_all(&s, |c: &Captures| {
            let day: u64 = c[2].parse().unwrap_or(0);
            match english_month(&c[1]) {
                Some(m) if (1..=31).contains(&day) => format!("{day}. {} {}", MONTHS[m], &c[3]),
                _ => c[0].to_string(),
            }
        })
        .into_owned();

    // 03.05.1995 -> dritter Mai neunzehnhundertfünfundneunzig
    let t = s.clone();
    s = replace_ctx(&NUMERIC_DATE, &t, |c, at| {
        let day: u64 = c[1].parse().ok()?;
        let month: usize = c[2].parse().ok()?;
        if !(1..=31).contains(&day) || !(1..=12).contains(&month) {
            return None;
        }
        let (p1, p2) = prev_words(&t, at);
        let ending = ordinal_ending(p1.as_deref(), p2.as_deref()).unwrap_or("er");
        let y: u64 = c[3].parse().ok()?;
        Some(format!(
            "{} {} {}",
            ordinal(day, ending),
            MONTHS[month - 1],
            year(y)
        ))
    });

    // 11.–15. September -> elfter bis fünfzehnter September
    let t = s.clone();
    s = replace_ctx(&DAY_RANGE, &t, |c, at| {
        let from: u64 = c[1].parse().ok()?;
        let to: u64 = c[2].parse().ok()?;
        if !(1..=31).contains(&from) || !(1..=31).contains(&to) {
            return None;
        }
        let (p1, p2) = prev_words(&t, at);
        let ending = ordinal_ending(p1.as_deref(), p2.as_deref()).unwrap_or("er");
        Some(format!(
            "{} bis {} {}",
            ordinal(from, ending),
            ordinal(to, ending),
            &c[3]
        ))
    });

    // 3. Mai -> dritter Mai / am dritten Mai
    let t = s.clone();
    s = replace_ctx(&DAY_MONTH, &t, |c, at| {
        let day: u64 = c[1].parse().ok()?;
        if !(1..=31).contains(&day) {
            return None;
        }
        let (p1, p2) = prev_words(&t, at);
        let ending = ordinal_ending(p1.as_deref(), p2.as_deref()).unwrap_or("er");
        Some(format!("{} {}", ordinal(day, ending), &c[2]))
    });

    // am 3.5. -> am dritten fünften (only with a clear article/preposition)
    let t = s.clone();
    s = replace_ctx(&DAY_MONTH_NUMERIC, &t, |c, at| {
        let day: u64 = c[1].parse().ok()?;
        let month: u64 = c[2].parse().ok()?;
        if !(1..=31).contains(&day) || !(1..=12).contains(&month) {
            return None;
        }
        let (p1, p2) = prev_words(&t, at);
        let ending = ordinal_ending(p1.as_deref(), p2.as_deref())?;
        Some(format!(
            "{} {}{}",
            ordinal(day, ending),
            ordinal(month, ending),
            &c[3]
        ))
    });

    // 14:30 Uhr -> vierzehn Uhr dreißig
    s = TIME
        .replace_all(&s, |c: &Captures| {
            let h: u64 = c[1].parse().unwrap_or(0);
            let hour = if h == 1 {
                "ein".to_string()
            } else {
                cardinal(h)
            };
            match c.get(2).and_then(|m| m.as_str().parse::<u64>().ok()) {
                Some(m) if m > 0 => format!("{hour} Uhr {}", cardinal(m)),
                _ => format!("{hour} Uhr"),
            }
        })
        .into_owned();

    // am 3. Tag -> am dritten Tag (only with a clear article/preposition)
    let t = s.clone();
    s = replace_ctx(&ORDINAL, &t, |c, at| {
        let n: u64 = c[1].parse().ok()?;
        if n == 0 {
            return None;
        }
        let (p1, p2) = prev_words(&t, at);
        let ending = ordinal_ending(p1.as_deref(), p2.as_deref())?;
        Some(format!("{}{}{}", ordinal(n, ending), &c[2], &c[3]))
    });

    s = DECIMAL_RANGE.replace_all(&s, "$1 bis $2").into_owned();

    // 1990–1995 -> 1990 bis 1995
    for _ in 0..2 {
        s = RANGE.replace_all(&s, "$1$2 bis $3$4").into_owned();
    }

    s = NEGATIVE.replace_all(&s, "${1}minus $2").into_owned();

    // €5 -> 5 €
    s = CURRENCY_PREFIX.replace_all(&s, "$2 $1").into_owned();

    // Numbers with units, money and percent
    s = WITH_UNIT
        .replace_all(&s, |c: &Captures| {
            let num = &c[1];
            let unit = c.get(2).or(c.get(3)).map(|m| m.as_str()).unwrap_or("");
            match unit {
                "€" | "EUR" | "Euro" => money(num, "Euro"),
                "$" | "Dollar" => money(num, "Dollar"),
                "%" | "Prozent" => count_with(num, "Prozent", "Prozent", false),
                "°C" => count_with(num, "Grad Celsius", "Grad Celsius", false),
                "°" => count_with(num, "Grad", "Grad", false),
                "km/h" => count_with(num, "Kilometer pro Stunde", "Kilometer pro Stunde", false),
                "km" => count_with(num, "Kilometer", "Kilometer", false),
                "kg" => count_with(num, "Kilogramm", "Kilogramm", false),
                "cm" => count_with(num, "Zentimeter", "Zentimeter", false),
                "mm" => count_with(num, "Millimeter", "Millimeter", false),
                "ml" => count_with(num, "Milliliter", "Milliliter", false),
                "mg" => count_with(num, "Milligramm", "Milligramm", false),
                "m" => count_with(num, "Meter", "Meter", false),
                "g" => count_with(num, "Gramm", "Gramm", false),
                "l" => count_with(num, "Liter", "Liter", false),
                "h" => count_with(num, "Stunde", "Stunden", true),
                "min" => count_with(num, "Minute", "Minuten", true),
                "ms" => count_with(num, "Millisekunde", "Millisekunden", true),
                "GB" => count_with(num, "Gigabyte", "Gigabyte", false),
                "MB" => count_with(num, "Megabyte", "Megabyte", false),
                "TB" => count_with(num, "Terabyte", "Terabyte", false),
                "KB" | "kB" => count_with(num, "Kilobyte", "Kilobyte", false),
                "GHz" => count_with(num, "Gigahertz", "Gigahertz", false),
                "MHz" => count_with(num, "Megahertz", "Megahertz", false),
                _ => c[0].to_string(),
            }
        })
        .into_owned();

    s = DECADE
        .replace_all(&s, |c: &Captures| format!("{}er", number_words(&c[1])))
        .into_owned();

    // Remaining numbers: years 1100-1999, decimals, cardinals. A number
    // followed by "." and a lower-case word is an ordinal without clear
    // context ("sie wurde 3. im Rennen") and stays for the TTS.
    let t = s.clone();
    s = replace_ctx(&NUMBER, &t, |c, _| {
        let m = c.get(1).unwrap();
        let after = &t[m.end()..];
        if let Some(rest) = after.strip_prefix('.') {
            let rest = rest.trim_start();
            if rest.len() < after.len() - 1 && rest.starts_with(char::is_lowercase) {
                return None;
            }
        }
        let tok = m.as_str();
        if tok.len() == 4
            && !tok.contains(['.', ','])
            && let Ok(n) = tok.parse::<u64>()
            && (1100..=1999).contains(&n)
        {
            return Some(year(n));
        }
        Some(number_words(tok))
    });
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinals() {
        let cases = [
            (0, "null"),
            (1, "eins"),
            (7, "sieben"),
            (16, "sechzehn"),
            (17, "siebzehn"),
            (21, "einundzwanzig"),
            (30, "dreißig"),
            (99, "neunundneunzig"),
            (100, "hundert"),
            (101, "hunderteins"),
            (111, "hundertelf"),
            (200, "zweihundert"),
            (1000, "tausend"),
            (1001, "tausendeins"),
            (1100, "tausendeinhundert"),
            (2021, "zweitausendeinundzwanzig"),
            (21000, "einundzwanzigtausend"),
            (100_000, "hunderttausend"),
            (1_000_000, "eine Million"),
            (2_500_000, "zwei Millionen fünfhunderttausend"),
            (3_000_000_001, "drei Milliarden eins"),
        ];
        for (n, w) in cases {
            assert_eq!(cardinal(n), w, "{n}");
        }
    }

    #[test]
    fn years_and_ordinals() {
        assert_eq!(year(1995), "neunzehnhundertfünfundneunzig");
        assert_eq!(year(1900), "neunzehnhundert");
        assert_eq!(year(1101), "elfhunderteins");
        assert_eq!(year(2024), "zweitausendvierundzwanzig");
        assert_eq!(ordinal(1, "e"), "erste");
        assert_eq!(ordinal(3, "en"), "dritten");
        assert_eq!(ordinal(7, "er"), "siebter");
        assert_eq!(ordinal(8, "e"), "achte");
        assert_eq!(ordinal(12, "e"), "zwölfte");
        assert_eq!(ordinal(20, "e"), "zwanzigste");
        assert_eq!(ordinal(31, "en"), "einunddreißigsten");
        assert_eq!(ordinal(100, "e"), "hundertste");
        assert_eq!(ordinal(103, "e"), "hundertdritte");
    }

    #[test]
    fn english_dates() {
        let cases = [
            (
                "Nov 7, 2026 · @name",
                "siebter November zweitausendsechsundzwanzig · @name",
            ),
            (
                "am Sept. 11, 2001",
                "am elften September zweitausendeins",
            ),
            (
                "Stand: March 3, 1995.",
                "Stand: dritter März neunzehnhundertfünfundneunzig.",
            ),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize(input), expected, "{input}");
        }
        // Not a date: day out of range, or no comma before the year.
        assert!(normalize("Nov 42, 2026").starts_with("Nov "));
        assert!(normalize("Jan 2026").starts_with("Jan "));
    }

    #[test]
    fn sentences() {
        let cases = [
            (
                "Im Jahr 1995 kam er an.",
                "Im Jahr neunzehnhundertfünfundneunzig kam er an.",
            ),
            ("Das war 2003.", "Das war zweitausenddrei."),
            (
                "Am 3. Mai 1995 regnete es.",
                "Am dritten Mai neunzehnhundertfünfundneunzig regnete es.",
            ),
            ("Der 3. Mai war kalt.", "Der dritte Mai war kalt."),
            (
                "Bis zum 03.05.2001 blieb er.",
                "Bis zum dritten Mai zweitausendeins blieb er.",
            ),
            ("Es war der 3. Versuch.", "Es war der dritte Versuch."),
            ("Sie saß in der 3. Reihe.", "Sie saß in der dritten Reihe."),
            ("Am 2. Tag ging es los.", "Am zweiten Tag ging es los."),
            ("Er kam um 14:30 Uhr.", "Er kam um vierzehn Uhr dreißig."),
            ("Um 1 Uhr schlief sie.", "Um ein Uhr schlief sie."),
            ("Um 8.00 Uhr.", "Um acht Uhr."),
            ("Es kostet 5,50 €.", "Es kostet fünf Euro fünfzig."),
            ("Nur 1 Euro!", "Nur ein Euro!"),
            ("Für 0,99 € zu haben.", "Für neunundneunzig Cent zu haben."),
            ("Es waren $20.", "Es waren zwanzig Dollar."),
            ("Zu 25 % fertig.", "Zu fünfundzwanzig Prozent fertig."),
            (
                "Die Rate lag bei 3,5%.",
                "Die Rate lag bei drei Komma fünf Prozent.",
            ),
            (
                "Es war 1 h und 15 min.",
                "Es war eine Stunde und fünfzehn Minuten.",
            ),
            (
                "Noch 12 km bis zum Ziel.",
                "Noch zwölf Kilometer bis zum Ziel.",
            ),
            ("Es hatte -5 °C.", "Es hatte minus fünf Grad Celsius."),
            (
                "Von 1939–1945 herrschte Krieg.",
                "Von neunzehnhundertneununddreißig bis neunzehnhundertfünfundvierzig herrschte Krieg.",
            ),
            ("Er las 5-6 Bücher.", "Er las fünf bis sechs Bücher."),
            ("In den 20er Jahren.", "In den zwanziger Jahren."),
            ("Über 10.000 Menschen.", "Über zehntausend Menschen."),
            ("Agent 007 kam.", "Agent null null sieben kam."),
            ("Er wurde 3. Dann ging er.", "Er wurde drei. Dann ging er."),
            ("Sie wurde 3. im Rennen.", "Sie wurde 3. im Rennen."),
            ("Siehe S. 12.", "Siehe Seite zwölf."),
            (
                "Äpfel, Birnen usw. Dann Brot.",
                "Äpfel, Birnen und so weiter. Dann Brot.",
            ),
            ("Obst, z. B. Äpfel.", "Obst, zum Beispiel Äpfel."),
            (
                "Dr. Müller kam ca. 3 Stunden später.",
                "Doktor Müller kam circa drei Stunden später.",
            ),
            ("Platz 1.", "Platz eins."),
            ("Kein Covid19 hier.", "Kein Covid19 hier."),
            (
                "BLEU > 0,75. Dieser Wert.",
                "BLEU größer als null Komma sieben fünf. Dieser Wert.",
            ),
            (
                "BLEU > 0.75. Dieser Wert.",
                "BLEU größer als null Komma sieben fünf. Dieser Wert.",
            ),
            (
                "Der Faktor 1.5 reicht.",
                "Der Faktor eins Komma fünf reicht.",
            ),
            (
                "Es kostet 3.500 Euro.",
                "Es kostet dreitausendfünfhundert Euro.",
            ),
            (
                "Um 0.5 % gestiegen.",
                "Um null Komma fünf Prozent gestiegen.",
            ),
            ("Am 3.5. kam er.", "Am dritten fünften kam er."),
            ("Für $1.50 zu haben.", "Für ein Dollar fünfzig zu haben."),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize(input), expected, "input: {input}");
        }
    }

    #[test]
    fn roman_numerals() {
        assert_eq!(roman_value("XIV"), Some(14));
        assert_eq!(roman_value("MCMXCV"), Some(1995));
        assert_eq!(roman_value("IIII"), None);
        assert_eq!(roman_value("VX"), None);
        assert_eq!(roman_value("Hallo"), None);
        assert_eq!(normalize("Kapitel IV beginnt."), "Kapitel vier beginnt.");
        assert_eq!(normalize("Im Teil II."), "Im Teil zwei.");
        assert_eq!(
            normalize("Im XIX. Jahrhundert war es so."),
            "Im neunzehnten Jahrhundert war es so."
        );
        // Without clear context the numeral stays (see review).
        assert_eq!(normalize("Heinrich IV. kam."), "Heinrich IV. kam.");
        assert_eq!(normalize("Kapitel CD"), "Kapitel CD");
    }

    #[test]
    fn text_without_numbers_is_unchanged() {
        let t = "Es war einmal ein kleiner Drache. „Hallo“, sagte er.";
        assert_eq!(normalize(t), t);
    }

    #[test]
    fn scientific_text() {
        let cases = [
            (
                "Seiten 28–53, 11.–15. September 2023.",
                "Seiten achtundzwanzig bis dreiundfünfzig, elfter bis fünfzehnter September zweitausenddreiundzwanzig.",
            ),
            ("pp. 28–53", "Seiten achtundzwanzig bis dreiundfünfzig"),
            (
                "eine Bevölkerung von 600 000 erzeugen",
                "eine Bevölkerung von sechshunderttausend erzeugen",
            ),
            (
                "mehr als 300 000 Pfund Sterling",
                "mehr als dreihunderttausend Pfund Sterling",
            ),
            (
                "1\u{202F}250\u{00A0}000 Menschen",
                "eine Million zweihundertfünfzigtausend Menschen",
            ),
            (
                "EMNLP 2020, pp. 1173–1186.",
                "EMNLP zweitausendzwanzig, Seiten tausendeinhundertdreiundsiebzig bis tausendeinhundertsechsundachtzig.",
            ),
            (
                "Communications of the ACM, 13(7):422–426.",
                "Communications of the ACM, Band dreizehn, Heft sieben, Seiten vierhundertzweiundzwanzig bis vierhundertsechsundzwanzig.",
            ),
            (
                "Systems, 33:1877–1901.",
                "Systems, Band dreiunddreißig, Seiten tausendachthundertsiebenundsiebzig bis tausendneunhunderteins.",
            ),
            (
                "Seite 997–1005.",
                "Seite neunhundertsiebenundneunzig bis tausendfünf.",
            ),
            ("siehe Abschnitt 3.3", "siehe Abschnitt drei Punkt drei"),
            ("Anhang B.1 gibt", "Anhang B Punkt eins gibt"),
            (
                "Anhang C.1 und Anhang C",
                "Anhang C Punkt eins und Anhang C",
            ),
            ("Kapitel IV", "Kapitel vier"),
            (
                "für 125M, 1.3B und 6B",
                "für hundertfünfundzwanzig Millionen, eins Komma drei Milliarden und sechs Milliarden",
            ),
            ("Pythia-6.9B", "Pythia-sechs Komma neun Milliarden"),
            ("1B Parameter", "eine Milliarde Parameter"),
            (
                "den 800GB großen Pile",
                "den achthundert Gigabyte großen Pile",
            ),
            (
                "SQuAD2.0 und GPT-3.5",
                "SQuAD zwei Punkt null und GPT-drei Punkt fünf",
            ),
            (
                "arXiv:2108.01624.",
                "arXiv zwei eins null acht Punkt null eins sechs zwei vier.",
            ),
            ("(n=77)", "(n gleich siebenundsiebzig)"),
            ("wenn LM(p) = s gilt", "wenn LM(p) gleich s gilt"),
            ("die Abfrage s ∈ D", "die Abfrage s Element von D"),
            ("(< 5) Änderungen", "(kleiner als fünf) Änderungen"),
            ("etwa ~150 Zeichen", "etwa ungefähr hundertfünfzig Zeichen"),
            ("10² Fälle", "zehn hoch zwei Fälle"),
            ("© 2023 ACL", "Copyright zweitausenddreiundzwanzig ACL"),
            ("Abb. 3 und Tab. 2", "Abbildung drei und Tabelle zwei"),
            (
                "bei etwa 0,53–0,59, GPT-Neo",
                "bei etwa null Komma fünf drei bis null Komma fünf neun, GPT-Neo",
            ),
            ("% Beispiele mit Treffer", "Prozent Beispiele mit Treffer"),
            ("WikiAuto + Turk", "WikiAuto plus Turk"),
            ("„claim“ ⇒ „claims“", "„claim“ zu „claims“"),
        ];
        for (input, expected) in cases {
            assert_eq!(normalize(input), expected, "input: {input}");
        }
    }
}
