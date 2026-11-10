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

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    ends_paragraph: bool
    source_file: str = "default"


# Paragraphs are separated by one or more blank lines.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

# Sentence boundary: text ending in end punctuation (optionally followed by
# closing quotes/brackets), then whitespace, then something that can start a
# sentence. Covers German umlauts and German/French quotes as well as ASCII.
_WS_RE = re.compile(r"\s+")
_SENTENCE_END_RE = re.compile(r"[.!?…][\"'“”»«’)\]]*$")
_SENTENCE_START_RE = re.compile(r"[A-ZÄÖÜ0-9\"'„“»«‚(\[]")


def _split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    start = 0
    for ws in _WS_RE.finditer(paragraph):
        before = paragraph[start : ws.start()]
        after = paragraph[ws.end() : ws.end() + 1]
        if after and _SENTENCE_END_RE.search(before) and _SENTENCE_START_RE.match(after):
            sentences.append(before.strip())
            start = ws.end()
    tail = paragraph[start:].strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s]


def split_to_chunks(
    text: str, max_words: int = 499, source_file: str = "default"
) -> list[Chunk]:
    """Split text into chunks of at most ``max_words`` words.

    Chunks never span a paragraph break, so every paragraph ends a chunk and
    the last chunk of each paragraph is flagged ``ends_paragraph=True``. The
    assembler uses that flag to insert the longer paragraph pause. Within a
    paragraph, sentences are packed greedily; a single sentence longer than
    ``max_words`` is kept whole (splitting mid-sentence damages prosody).
    """
    result: list[Chunk] = []

    for paragraph in _PARAGRAPH_RE.split(text):
        # Line breaks inside a paragraph are just wrapping.
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue

        packed: list[str] = []
        current: list[str] = []
        current_words = 0
        for sentence in _split_sentences(paragraph):
            words = len(sentence.split())
            if current and current_words + words > max_words:
                packed.append(" ".join(current))
                current = [sentence]
                current_words = words
            else:
                current.append(sentence)
                current_words += words
        if current:
            packed.append(" ".join(current))

        for i, chunk_text in enumerate(packed):
            result.append(
                Chunk(
                    text=chunk_text,
                    ends_paragraph=(i == len(packed) - 1),
                    source_file=source_file,
                )
            )

    return result
