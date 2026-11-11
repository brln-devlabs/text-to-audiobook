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

/// Number token in words: cardinal, decimal ("drei Komma eins vier"),
/// leading zeros digit by digit.
fn number_words(token: &str) -> String {
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
static ROMAN_AFTER_KEYWORD: LazyLock<Regex> = LazyLock::new(|| {
    re(&format!(
        r"\b({})(\.?\s+)([IVXLC]+)\b",
        ROMAN_KEYWORDS.join("|")
    ))
});
static ROMAN_CENTURY: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b([IVXLC]+)\.(\s+)(Jahrhundert\p{L}*|Jh\.)"));
static NUMERIC_DATE: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"));
static DAY_MONTH: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"\b(\d{1,2})\.\s+(Januar|Jänner|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\b",
    )
});
static TIME: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b([01]?\d|2[0-3])(?:[:.]([0-5]\d))?\s?Uhr\b"));
static ORDINAL: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d{1,4})\.(\s+)(\p{L})"));
static RANGE: LazyLock<Regex> =
    LazyLock::new(|| re(r"(^|[^\d.,\-–])(\d{1,4})\s?[–-]\s?(\d{1,4})($|[^\d.,\-–])"));
static CURRENCY_PREFIX: LazyLock<Regex> =
    LazyLock::new(|| re(r"(€|\$)\s?(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?)"));
static WITH_UNIT: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"\b(\d{1,3}(?:\.\d{3})+(?:,(?:\d+|–|-))?|\d+(?:,(?:\d+|–|-))?)\s?(?:(%|€|\$|°C|°)|(km/h|km|kg|cm|mm|ml|mg|m|g|l|h|min|EUR|Euro|Prozent|Dollar)\b)",
    )
});
static NEGATIVE: LazyLock<Regex> = LazyLock::new(|| re(r"(^|[\s(])[-–−](\d)"));
static DECADE: LazyLock<Regex> = LazyLock::new(|| re(r"\b(\d+)er\b"));
static NUMBER: LazyLock<Regex> =
    LazyLock::new(|| re(r"\b(\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+,\d+|\d+)\b"));

/// Spell out numbers, dates, times, units and abbreviations in German text.
pub fn normalize(text: &str) -> String {
    let mut s = text.to_string();

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

    // Kapitel IV -> Kapitel vier
    s = ROMAN_AFTER_KEYWORD
        .replace_all(&s, |c: &Captures| match roman_value(&c[3]) {
            Some(v) => format!("{}{}{}", &c[1], &c[2], cardinal(v)),
            None => c[0].to_string(),
        })
        .into_owned();

    // im XIX. Jahrhundert -> im neunzehnten Jahrhundert (with clear context)
    let t = s.clone();
    s = replace_ctx(&ROMAN_CENTURY, &t, |c, at| {
        let v = roman_value(&c[1])?;
        let (p1, p2) = prev_words(&t, at);
        let ending = ordinal_ending(p1.as_deref(), p2.as_deref())?;
        Some(format!("{}{}{}", ordinal(v, ending), &c[2], &c[3]))
    });

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
}
