# This file is part of text-to-audiobook
#
# text-to-audiobook is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# text-to-audiobook is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from src.chunker import split_to_chunks


def test_chunks_under_max_words():
    text = "First sentence. Second sentence. Third sentence."
    chunks = split_to_chunks(text, max_words=10)
    for chunk in chunks:
        word_count = len(chunk.text.split())
        assert word_count < 11, f"Chunk exceeds max_words: {word_count}"


def test_chunks_end_with_sentence_boundary():
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = split_to_chunks(text)
    for chunk in chunks:
        text_stripped = chunk.text.rstrip()
        last_char = text_stripped[-1] if text_stripped else ""
        assert last_char in [
            ".",
            "!",
            "?",
        ], f"Chunk doesn't end in sentence boundary: {text_stripped}"


def test_single_short_text():
    text = "Short text."
    chunks = split_to_chunks(text)
    assert len(chunks) >= 1
    assert chunks[0].text.strip() == "Short text."


def test_paragraph_flag():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = split_to_chunks(text)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk.ends_paragraph, bool)


def test_empty_text():
    text = ""
    chunks = split_to_chunks(text)
    assert len(chunks) == 0


def test_very_long_sentence():
    long_sentence = "This is a very long sentence that contains many words " * 30 + "."
    chunks = split_to_chunks(long_sentence, max_words=50)
    assert len(chunks) > 0


def test_chunks_never_span_paragraphs():
    text = "Erster Absatz. Noch ein Satz.\n\nZweiter Absatz.\n\nDritter Absatz. Ende."
    chunks = split_to_chunks(text, max_words=499)
    assert [c.text for c in chunks] == [
        "Erster Absatz. Noch ein Satz.",
        "Zweiter Absatz.",
        "Dritter Absatz. Ende.",
    ]
    assert all(c.ends_paragraph for c in chunks)


def test_long_paragraph_only_last_chunk_ends_paragraph():
    paragraph = " ".join(f"Satz Nummer {i} hat fünf Wörter." for i in range(20))
    chunks = split_to_chunks(paragraph + "\n\nKurz.", max_words=20)
    assert len(chunks) > 2
    *first_para, last_of_para, short = chunks
    assert not any(c.ends_paragraph for c in first_para)
    assert last_of_para.ends_paragraph
    assert short.text == "Kurz." and short.ends_paragraph
    for c in chunks:
        assert len(c.text.split()) <= 20


def test_german_sentence_starts_are_split():
    """Sentences starting with umlauts or German quotes are sentence starts."""
    text = "Erster Satz. Über alles. „Zitat“, sagte er. Ärger folgt."
    chunks = split_to_chunks(text, max_words=3)
    assert [c.text for c in chunks] == [
        "Erster Satz.",
        "Über alles.",
        "„Zitat“, sagte er.",
        "Ärger folgt.",
    ]


def test_line_breaks_inside_paragraph_are_joined():
    chunks = split_to_chunks("Eine Zeile\nund die nächste.")
    assert [c.text for c in chunks] == ["Eine Zeile und die nächste."]
