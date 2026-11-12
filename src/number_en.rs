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

//! English number words for the number review ("Henry VIII" → "Henry the
//! Eighth", "World War II" → "World War two").

const ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];

const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

fn below_100(n: u64) -> String {
    let n = n as usize;
    if n < 20 {
        return ONES[n].to_string();
    }
    match n % 10 {
        0 => TENS[n / 10].to_string(),
        r => format!("{}-{}", TENS[n / 10], ONES[r]),
    }
}

fn below_1000(n: u64) -> String {
    match (n / 100, n % 100) {
        (0, r) => below_100(r),
        (h, 0) => format!("{} hundred", ONES[h as usize]),
        (h, r) => format!("{} hundred and {}", ONES[h as usize], below_100(r)),
    }
}

/// "three hundred and forty-two"; numbers from a million on stay digits.
pub fn cardinal(n: u64) -> String {
    if n >= 1_000_000 {
        return n.to_string();
    }
    match (n / 1000, n % 1000) {
        (0, r) => below_1000(r),
        (t, 0) => format!("{} thousand", below_1000(t)),
        (t, r) if r < 100 => format!("{} thousand and {}", below_1000(t), below_100(r)),
        (t, r) => format!("{} thousand {}", below_1000(t), below_1000(r)),
    }
}

/// "first", "twenty-second", "one hundred and eighth".
pub fn ordinal(n: u64) -> String {
    let words = cardinal(n);
    if n >= 1_000_000 {
        return format!("{words}th");
    }
    let cut = words.rfind([' ', '-']).map_or(0, |i| i + 1);
    let (head, last) = words.split_at(cut);
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    format!("{head}{last}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinals() {
        assert_eq!(cardinal(2), "two");
        assert_eq!(cardinal(14), "fourteen");
        assert_eq!(cardinal(20), "twenty");
        assert_eq!(cardinal(42), "forty-two");
        assert_eq!(cardinal(100), "one hundred");
        assert_eq!(cardinal(342), "three hundred and forty-two");
        assert_eq!(cardinal(1066), "one thousand and sixty-six");
        assert_eq!(cardinal(1999), "one thousand nine hundred and ninety-nine");
        assert_eq!(cardinal(2000), "two thousand");
    }

    #[test]
    fn ordinals() {
        assert_eq!(ordinal(1), "first");
        assert_eq!(ordinal(2), "second");
        assert_eq!(ordinal(3), "third");
        assert_eq!(ordinal(4), "fourth");
        assert_eq!(ordinal(8), "eighth");
        assert_eq!(ordinal(9), "ninth");
        assert_eq!(ordinal(12), "twelfth");
        assert_eq!(ordinal(14), "fourteenth");
        assert_eq!(ordinal(20), "twentieth");
        assert_eq!(ordinal(22), "twenty-second");
        assert_eq!(ordinal(100), "one hundredth");
        assert_eq!(ordinal(108), "one hundred and eighth");
    }
}
