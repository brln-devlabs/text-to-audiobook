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

//! Natural sort order ("2" before "10"), matching Python's `natsort.natsorted`
//! with default options: case-sensitive text parts, unsigned integer parts.

use std::cmp::Ordering;

#[derive(Debug, PartialEq, Eq)]
enum Part<'a> {
    Text(&'a str),
    Num(&'a str),
}

/// Split into alternating text/number parts. Like natsort, a key always
/// starts with a text part (possibly empty) so positions line up.
fn parts(s: &str) -> Vec<Part<'_>> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut in_num = false;
    for (i, c) in s.char_indices() {
        let is_digit = c.is_ascii_digit();
        if i == 0 {
            in_num = is_digit;
            if in_num {
                out.push(Part::Text(""));
            }
            continue;
        }
        if is_digit != in_num {
            let piece = &s[start..i];
            out.push(if in_num {
                Part::Num(piece)
            } else {
                Part::Text(piece)
            });
            start = i;
            in_num = is_digit;
        }
    }
    if !s.is_empty() {
        let piece = &s[start..];
        out.push(if in_num {
            Part::Num(piece)
        } else {
            Part::Text(piece)
        });
    }
    out
}

fn cmp_num(a: &str, b: &str) -> Ordering {
    let a = a.trim_start_matches('0');
    let b = b.trim_start_matches('0');
    a.len().cmp(&b.len()).then_with(|| a.cmp(b))
}

/// Compare two strings in natural order.
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let pa = parts(a);
    let pb = parts(b);
    for (x, y) in pa.iter().zip(pb.iter()) {
        let ord = match (x, y) {
            (Part::Text(x), Part::Text(y)) => x.cmp(y),
            (Part::Num(x), Part::Num(y)) => cmp_num(x, y),
            (Part::Text(_), Part::Num(_)) => Ordering::Greater,
            (Part::Num(_), Part::Text(_)) => Ordering::Less,
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    pa.len().cmp(&pb.len())
}

/// Stable natural sort of strings.
pub fn natsorted<S: AsRef<str>>(items: impl IntoIterator<Item = S>) -> Vec<String> {
    let mut v: Vec<String> = items.into_iter().map(|s| s.as_ref().to_string()).collect();
    v.sort_by(|a, b| natural_cmp(a, b));
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_sort_numerically() {
        assert_eq!(
            natsorted([
                "10_Chapter.md",
                "2_Chapter.md",
                "0_Prologue.md",
                "1_Start.md"
            ]),
            [
                "0_Prologue.md",
                "1_Start.md",
                "2_Chapter.md",
                "10_Chapter.md"
            ]
        );
        assert_eq!(
            natsorted(["chapter10", "chapter2", "chapter1"]),
            ["chapter1", "chapter2", "chapter10"]
        );
    }

    #[test]
    fn numbers_before_text_and_case_sensitive() {
        assert_eq!(natsorted(["b", "a", "1", "B"]), ["1", "B", "a", "b"]);
        assert_eq!(natsorted(["a", ""]), ["", "a"]);
    }
}
