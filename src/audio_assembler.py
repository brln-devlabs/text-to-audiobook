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

import logging
import re
from pathlib import Path

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

from . import config

logger = logging.getLogger(__name__)

# chunk_000012.wav (single part) or chunk_000012_3.wav (sub-part 3 of chunk 12)
_CHUNK_NAME_RE = re.compile(r"^chunk_(\d+)(?:_(\d+))?$")


def _ends_paragraph(meta) -> bool:
    if isinstance(meta, dict):
        return bool(meta.get("ends_paragraph", False))
    return bool(getattr(meta, "ends_paragraph", False))


def _silence_plan(chunk_wavs: list, chunk_meta: list) -> list:
    """Silence (ms, or None) to insert after each WAV.

    One segment can be synthesized as several sub-part WAVs
    (chunk_NNN_0.wav, chunk_NNN_1.wav, ...). Consecutive WAVs of the same
    chunk form one group, and group k belongs to chunk_meta[k]. Between
    sub-parts of a group the sentence pause is used; after the last part the
    paragraph pause if the segment ends a paragraph. WAVs whose names do not
    follow the chunk pattern are treated as one group each.
    """
    keys = []
    for pos, wav in enumerate(chunk_wavs):
        m = _CHUNK_NAME_RE.match(Path(wav).stem)
        keys.append(int(m.group(1)) if m else ("pos", pos))

    plan = []
    group = -1
    for i, key in enumerate(keys):
        if i == 0 or key != keys[i - 1]:
            group += 1
        if group >= len(chunk_meta):
            plan.append(None)
            continue
        last_of_group = i == len(keys) - 1 or keys[i + 1] != key
        if last_of_group and _ends_paragraph(chunk_meta[group]):
            plan.append(config.SILENCE_AFTER_PARAGRAPH_MS)
        else:
            plan.append(config.SILENCE_AFTER_PERIOD_MS)
    return plan


def assemble(chunk_wavs: list[Path], chunk_meta: list, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if AudioSegment is None:
        logger.warning("pydub not available, skipping assembly")
        return out_path

    logger.info(f"Assembling {len(chunk_wavs)} audio chunks")

    try:
        combined = None
        silence_plan = _silence_plan(chunk_wavs, chunk_meta)

        for i, wav_path in enumerate(chunk_wavs):
            logger.debug(f"Loading chunk {i}: {wav_path}")
            audio = AudioSegment.from_wav(str(wav_path))

            if combined is None:
                combined = audio
            else:
                combined = combined.append(audio, crossfade=config.CROSSFADE_MS)

            silence_ms = silence_plan[i]
            if silence_ms is not None:
                silence = AudioSegment.silent(duration=silence_ms)
                combined = combined.append(silence)

        if combined is None:
            logger.warning("No chunks to combine")
            return out_path

        wav_out = out_path.with_suffix(".wav")
        combined.export(str(wav_out), format="wav")
        logger.info(f"Exported WAV: {wav_out}")

        mp3_out = out_path.with_suffix(".mp3")
        combined.export(str(mp3_out), format="mp3", bitrate="192k")
        logger.info(f"Exported MP3: {mp3_out}")

        return mp3_out

    except Exception as e:
        logger.error(f"Assembly failed: {e}")
        raise
