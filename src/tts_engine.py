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

"""TTS engine module for synthesizing audio from text using Qwen3-TTS."""

import gc
import logging
import re
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:
    torch: Any = None  # type: ignore[no-redef]

try:
    import numpy as np
except ImportError:
    np: Any = None  # type: ignore[no-redef]

try:
    import soundfile as sf
except ImportError:
    sf: Any = None  # type: ignore[no-redef]

try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel: Any = None  # type: ignore[no-redef]

from . import config

logger = logging.getLogger(__name__)

_MAX_SUB_PARTS = 10

# Short fixed texts used to generate the voice reference WAV when none exists.
_VOICE_REF_TEXTS = {
    "english": (
        "This is a short voice sample. It defines how the narrator "
        "will sound throughout the entire audiobook."
    ),
    "german": (
        "Dies ist eine kurze Sprachprobe. Sie legt fest, wie die Stimme "
        "des Erzählers im ganzen Hörbuch klingen wird."
    ),
}


def resolve_language(segment_language: str | None = None) -> str:
    """Language for Qwen3-TTS: config.TTS_LANGUAGE, or the segment's own
    detected language when TTS_LANGUAGE is "auto"."""
    fixed = str(getattr(config, "TTS_LANGUAGE", "auto") or "auto").strip().lower()
    if fixed != "auto":
        return fixed
    return segment_language or TTSEngine.DEFAULT_LANGUAGE


def _resolve_dtype():
    """torch dtype for GPU loading from config.TTS_DTYPE (bf16 -> fp16 fallback)."""
    name = str(getattr(config, "TTS_DTYPE", "bfloat16")).strip().lower()
    if name == "bfloat16":
        try:
            if not torch.cuda.is_bf16_supported():
                logger.warning("GPU has no bfloat16 support, falling back to float16")
                name = "float16"
        except Exception:
            pass
    dtype = getattr(torch, name, None)
    if dtype is None:
        logger.warning(f"Unknown TTS_DTYPE '{name}', using bfloat16")
        dtype = torch.bfloat16
    return dtype


class TTSEngine:
    """Wrapper around Qwen3-TTS for voice-cloned audio synthesis."""

    def __init__(
        self,
        model_name: str = config.TTS_MODEL_NAME,
        voice_design_model_name: str = config.TTS_VOICE_DESIGN_MODEL_NAME,
    ):
        self.model_name = model_name
        self.voice_design_model_name = voice_design_model_name
        self.engine: Any = None
        self.anchor_path: Path | None = None
        self._voice_ref_path: Path | None = None

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def _make_load_kwargs(self):
        kwargs = {"trust_remote_code": True}
        if torch is not None and torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["dtype"] = _resolve_dtype()
        else:
            kwargs["device_map"] = "cpu"
        return kwargs

    def _generate_voice_reference(self, ref_path: Path, narrator_prompt: str):
        """Generate voice reference using base model."""
        logger.info(f"Generating voice reference with {self.model_name} ...")
        kwargs = self._make_load_kwargs()
        engine = Qwen3TTSModel.from_pretrained(self.model_name, **kwargs)
        try:
            if (
                torch is not None
                and hasattr(engine, "model")
                and engine.model is not None
            ):
                try:
                    engine.model = torch.compile(engine.model, mode="reduce-overhead")
                except Exception as e:
                    logger.debug(f"torch.compile not available or failed: {e}")

            self._set_seed()
            ref_language = resolve_language("english")
            wavs, sr = engine.generate(
                text=_VOICE_REF_TEXTS.get(ref_language, _VOICE_REF_TEXTS["english"]),
                language=ref_language,
                temperature=0.3,
                top_p=0.9,
            )
            if not wavs:
                raise RuntimeError(
                    "Base model returned no audio for reference generation"
                )
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(ref_path), wavs[0], sr)
            logger.info(f"Voice reference saved: {ref_path}")
        finally:
            del engine
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load(self, narrator_prompt: str | None = None):
        if Qwen3TTSModel is None:
            logger.warning("qwen_tts not available, using mock mode")
            return

        prompt = narrator_prompt or config.NARRATOR_PROMPT
        ref_path = Path(config.VOICE_REF_PATH)

        if not ref_path.exists():
            self._generate_voice_reference(ref_path, prompt)

        kwargs = self._make_load_kwargs()
        gpu = torch is not None and torch.cuda.is_available()
        logger.info(f"Loading TTS model: {self.model_name} ({'GPU' if gpu else 'CPU'})")
        self.engine = Qwen3TTSModel.from_pretrained(self.model_name, **kwargs)

        if (
            torch is not None
            and hasattr(self.engine, "model")
            and self.engine.model is not None
        ):
            try:
                logger.info(
                    "Compiling model with torch.compile for optimized inference..."
                )
                self.engine.model = torch.compile(
                    self.engine.model, mode="reduce-overhead"
                )
            except Exception as e:
                logger.debug(f"torch.compile not available or failed: {e}")

        if ref_path.exists():
            self._voice_ref_path = ref_path
            logger.info(f"Voice reference ready: {ref_path.name}")

    @property
    def model(self):
        return self.engine.model if self.engine is not None else None

    @property
    def processor(self):
        return self.engine.processor if self.engine is not None else None

    def unload(self):
        if self.engine is not None:
            logger.info("Unloading TTS model")
            del self.engine
            self.engine = None
            self._voice_ref_path = None
            gc.collect()
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_seed(self):
        if torch is not None:
            torch.manual_seed(config.VOICE_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.VOICE_SEED)

    def _write_silence(self, out_path: Path) -> Path:
        if np is not None and sf is not None:
            silence = np.zeros((config.SAMPLE_RATE * 2,), dtype=np.float32)
            sf.write(str(out_path), silence, config.SAMPLE_RATE)
        if not out_path.exists():
            out_path.write_bytes(b"")
        return out_path

    DEFAULT_LANGUAGE = "english"

    # ------------------------------------------------------------------
    # Text splitting
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        try:
            tokenizer = self.engine.processor.tokenizer
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return len(text.split())

    def _split_if_too_long(self, text: str) -> list[str]:
        tokenizer_available = (
            self.engine is not None
            and hasattr(self.engine, "processor")
            and self.engine.processor is not None
            and hasattr(self.engine.processor, "tokenizer")
        )

        if tokenizer_available:
            limit = config.TTS_MAX_TOKENS
            unit = "tokens"
        else:
            limit = config.TTS_MAX_WORDS_FALLBACK
            unit = "words"

        # Pack with 15% headroom so sub-parts stay clear of the model's hard ceiling,
        # where Qwen3-TTS gets unstable (truncation, hallucinated tail noise).
        target = int(limit * 0.85)

        def measure(s: str) -> int:
            return self._count_tokens(s) if tokenizer_available else len(s.split())

        total = measure(text)
        if total <= limit:
            return [text]

        # Build atomic units hierarchically: split on paragraph breaks first
        # (preserves natural pacing), then on sentence boundaries within any
        # paragraph that's still over target. A single sentence over target
        # is kept whole — splitting mid-sentence would damage prosody.
        atoms: list[tuple[str, int]] = []
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if not para:
                continue
            p_count = measure(para)
            if p_count <= target:
                atoms.append((para, p_count))
                continue
            for sent in re.split(r"(?<=[.!?])\s+", para):
                sent = sent.strip()
                if not sent:
                    continue
                atoms.append((sent, measure(sent)))

        # Greedy pack atoms into parts, never exceeding the hard limit and
        # preferring to stay under target.
        parts: list[str] = []
        current: list[str] = []
        current_count = 0
        for atom, atom_count in atoms:
            if current and (
                current_count + atom_count > limit or current_count >= target
            ):
                parts.append(" ".join(current))
                current = [atom]
                current_count = atom_count
            else:
                current.append(atom)
                current_count += atom_count

        if current:
            parts.append(" ".join(current))

        oversize = [i for i, p in enumerate(parts) if measure(p) > limit]
        if oversize:
            logger.warning(
                f"Split produced {len(oversize)} sub-part(s) still over the {limit}-{unit} "
                f"limit (atomic sentences too long); TTS may truncate or hallucinate"
            )

        logger.debug(
            f"Split long segment ({total} {unit}, target {target}, limit {limit}) "
            f"into {len(parts)} sub-parts"
        )
        return parts

    # ------------------------------------------------------------------
    # Core synthesis
    # ------------------------------------------------------------------

    def _synthesize_single(
        self, text: str, out_path: Path, language: str = DEFAULT_LANGUAGE
    ) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.engine is None:
            return self._write_silence(out_path)

        self._set_seed()
        wavs, sr = self.engine.generate_voice_clone(
            text=text,
            language=language or self.DEFAULT_LANGUAGE,
            ref_audio=str(self._voice_ref_path),
            x_vector_only_mode=True,
            temperature=0.6,
            top_p=0.9,
        )
        if not wavs:
            raise RuntimeError("generate_voice_clone returned no audio")
        sf.write(str(out_path), wavs[0], sr)
        return out_path

    def _synthesize(
        self, text: str, out_path: Path, language: str = DEFAULT_LANGUAGE
    ) -> list[Path]:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.engine is None:
            return [self._write_silence(out_path)]

        try:
            parts = self._split_if_too_long(text)

            if len(parts) == 1:
                self._synthesize_single(parts[0], out_path, language)
                return [out_path]

            if len(parts) > _MAX_SUB_PARTS:
                raise RuntimeError(
                    f"Segment split produced {len(parts)} sub-parts; "
                    f"max supported is {_MAX_SUB_PARTS}"
                )

            # Multi-part chunks: every part gets a _N suffix (0-indexed). The
            # base unsuffixed filename is reserved for single-part chunks
            # exclusively, so the assembler can distinguish the two cases by
            # checking which file exists.
            output_paths = []
            for idx, part in enumerate(parts):
                part_out_path = (
                    out_path.parent / f"{out_path.stem}_{idx}{out_path.suffix}"
                )
                if part_out_path.exists():
                    logger.info(
                        f"Part {idx + 1}/{len(parts)} already exists, reusing: {part_out_path.name}"
                    )
                    output_paths.append(part_out_path)
                    continue
                self._synthesize_single(part, part_out_path, language)
                output_paths.append(part_out_path)
                logger.warning(
                    f"Text split into {len(parts)} parts; saved as {part_out_path.name}"
                )
            return output_paths

        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

    def synthesize_first(
        self,
        text: str,
        narrator_prompt: str,
        out_path: Path,
        language: str = DEFAULT_LANGUAGE,
    ) -> list[Path]:
        """Synthesize the first chunk using voice clone with the voice design reference.

        The voice reference is created with voice design during load() for narrator consistency.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.engine is None:
            return [self._write_silence(out_path)]

        try:
            parts = self._split_if_too_long(text)

            if len(parts) == 1:
                self._set_seed()
                wavs, sr = self.engine.generate_voice_clone(
                    text=parts[0],
                    language=language,
                    ref_audio=str(self._voice_ref_path),
                    x_vector_only_mode=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                if not wavs:
                    raise RuntimeError("generate_voice_clone returned no audio")
                sf.write(str(out_path), wavs[0], sr)
                self.anchor_path = out_path
                return [out_path]

            if len(parts) > _MAX_SUB_PARTS:
                raise RuntimeError(
                    f"First chunk split produced {len(parts)} sub-parts; "
                    f"max supported is {_MAX_SUB_PARTS}"
                )

            output_paths = []
            for idx, part in enumerate(parts):
                part_out_path = (
                    out_path.parent / f"{out_path.stem}_{idx}{out_path.suffix}"
                )
                if part_out_path.exists():
                    logger.info(
                        f"First chunk part {idx + 1}/{len(parts)} already exists, reusing: {part_out_path.name}"
                    )
                    output_paths.append(part_out_path)
                    if idx == 0:
                        self.anchor_path = part_out_path
                    continue
                self._set_seed()
                wavs, sr = self.engine.generate_voice_clone(
                    text=part,
                    language=language,
                    ref_audio=str(self._voice_ref_path),
                    x_vector_only_mode=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                if not wavs:
                    raise RuntimeError("generate_voice_clone returned no audio")
                sf.write(str(part_out_path), wavs[0], sr)
                output_paths.append(part_out_path)
                if idx == 0:
                    self.anchor_path = part_out_path
                logger.warning(
                    f"First chunk split into {len(parts)} parts; saved as {part_out_path.name}"
                )
            return output_paths
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

    def synthesize_with_anchor(
        self,
        text: str,
        anchor_path: Path,
        out_path: Path,
        language: str = DEFAULT_LANGUAGE,
    ) -> list[Path]:
        """Synthesize a chunk using the anchor voice reference.

        This uses generate_voice_clone with the anchor WAV to clone the voice from the first chunk.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.engine is None:
            return [self._write_silence(out_path)]

        try:
            parts = self._split_if_too_long(text)

            if len(parts) == 1:
                self._set_seed()
                wavs, sr = self.engine.generate_voice_clone(
                    text=parts[0],
                    language=language,
                    ref_audio=str(anchor_path),
                    x_vector_only_mode=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                if not wavs:
                    raise RuntimeError("generate_voice_clone returned no audio")
                sf.write(str(out_path), wavs[0], sr)
                return [out_path]

            if len(parts) > _MAX_SUB_PARTS:
                raise RuntimeError(
                    f"Chunk split produced {len(parts)} sub-parts; "
                    f"max supported is {_MAX_SUB_PARTS}"
                )

            output_paths = []
            for idx, part in enumerate(parts):
                part_out_path = (
                    out_path.parent / f"{out_path.stem}_{idx}{out_path.suffix}"
                )
                if part_out_path.exists():
                    logger.info(
                        f"Part {idx + 1}/{len(parts)} already exists, reusing: {part_out_path.name}"
                    )
                    output_paths.append(part_out_path)
                    continue
                self._set_seed()
                wavs, sr = self.engine.generate_voice_clone(
                    text=part,
                    language=language,
                    ref_audio=str(anchor_path),
                    x_vector_only_mode=True,
                    temperature=0.6,
                    top_p=0.9,
                )
                if not wavs:
                    raise RuntimeError("generate_voice_clone returned no audio")
                sf.write(str(part_out_path), wavs[0], sr)
                output_paths.append(part_out_path)
                logger.warning(
                    f"Text split into {len(parts)} parts; saved as {part_out_path.name}"
                )
            return output_paths
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            raise

    # ------------------------------------------------------------------
    # WAV metadata tagging
    # ------------------------------------------------------------------

    def _tag_wav(self, wav_path: Path, payload: dict) -> None:
        """Add ID3 metadata tags to a WAV file.

        Tags added:
          - TIT2 (Title): "{chapter_title} - Part {chunk_number}"
          - TALB (Album): story_title
          - TRCK (Track): chapter_number
          - TPE1 (Artist): "Narrator"
          - TCON (Genre): "Audiobook"
          - TDRC (Year): current year
          - COMM (Comment): "Chunk {chunk_number}, Chapter {chapter_number}"
        """
        try:
            from datetime import datetime

            from mutagen.id3 import COMM, ID3, TALB, TCON, TDRC, TIT2, TPE1, TRCK

            chunk_number = payload.get("chunk_number", 0)
            chapter_number = payload.get("chapter_number", 1)
            chapter_title = payload.get("chapter_title", f"Chapter {chapter_number}")
            story_title = payload.get("story_title", "Audiobook")

            try:
                tags = ID3(str(wav_path))
            except Exception:
                tags = ID3()

            tags[TIT2.FrameID] = TIT2(text=[f"{chapter_title} - Part {chunk_number}"])
            tags[TALB.FrameID] = TALB(text=[story_title])
            tags[TRCK.FrameID] = TRCK(text=[str(chapter_number)])
            tags[TPE1.FrameID] = TPE1(text=["Narrator"])
            tags[TCON.FrameID] = TCON(text=["Audiobook"])
            tags[TDRC.FrameID] = TDRC(text=[str(datetime.now().year)])
            tags[COMM.FrameID] = COMM(
                desc="",
                lang="eng",
                text=[f"Chunk {chunk_number}, Chapter {chapter_number}"],
            )

            tags.save(str(wav_path), v2_version=3)
        except ImportError:
            logger.warning("mutagen not available, skipping WAV metadata tagging")
        except Exception as e:
            logger.warning(f"Failed to tag WAV file {wav_path}: {e}")

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _migrate_legacy_chunk_names(self, output_dir: Path, total_chunks: int) -> None:
        """Unify legacy 1-indexed multi-part naming to the 0-indexed scheme.

        Convention:
          - single-part chunks  -> chunk_NNN.wav (base, no suffix)
          - multi-part chunks   -> chunk_NNN_0.wav, chunk_NNN_1.wav, ...

        An earlier code variant saved multi-part chunks as _1, _2, ... with no
        base. Rename those down by one (_1 -> _0, _2 -> _1, ...) so they fit
        the current scheme. Idempotent: a no-op once names are unified.
        """
        renamed = 0
        for i in range(total_chunks):
            stem = f"chunk_{i:06d}"
            # If a single-part base file exists, leave it alone.
            if (output_dir / f"{stem}.wav").exists():
                continue
            # Already in 0-indexed form -> nothing to do.
            if (output_dir / f"{stem}_0.wav").exists():
                continue

            # Collect 1-indexed legacy parts in ascending order.
            legacy: list[tuple[int, Path]] = []
            for k in range(1, _MAX_SUB_PARTS + 1):
                p = output_dir / f"{stem}_{k}.wav"
                if p.exists():
                    legacy.append((k, p))

            if not legacy:
                continue

            # Shift each _K down to _(K-1).
            for k, src in legacy:
                dst = output_dir / f"{stem}_{k - 1}.wav"
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
            renamed += 1

        if renamed:
            logger.info(f"Migrated {renamed} chunk(s) to 0-indexed multi-part naming")

    def process_all(
        self, json_payloads: list, output_dir: Path, stop_event=None, chunk_done_cb=None
    ) -> list:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._migrate_legacy_chunk_names(output_dir, len(json_payloads))

        wav_paths = []

        for i, payload in enumerate(json_payloads):
            if stop_event is not None and stop_event.is_set():
                logger.info(
                    f"Stop requested after {i}/{len(json_payloads)} chunks generated"
                )
                break

            out_path = output_dir / f"chunk_{i:06d}.wav"

            # Two valid completion states:
            #   single-part: chunk_NNN.wav exists alone
            #   multi-part:  chunk_NNN_0.wav exists (and any _1, _2, ...)
            existing_parts: list[Path] = []
            if out_path.exists():
                existing_parts.append(out_path)
            else:
                for part_num in range(_MAX_SUB_PARTS):
                    p = out_path.parent / f"{out_path.stem}_{part_num}{out_path.suffix}"
                    if not p.exists():
                        break
                    existing_parts.append(p)

            if existing_parts:
                logger.debug(
                    f"Chunk {i} already exists ({len(existing_parts)} part(s)), skipping"
                )
                wav_paths.extend(existing_parts)
                # Tag existing WAV files with metadata from payload
                for wav_path in existing_parts:
                    self._tag_wav(wav_path, payload)
                if i == 0:
                    self.anchor_path = existing_parts[0]
                if chunk_done_cb:
                    chunk_done_cb()
                continue

            text = payload.get("normalized_text", "")
            language = resolve_language(payload.get("language"))

            try:
                if i == 0:
                    chunk_wav_paths = self.synthesize_first(
                        text, config.NARRATOR_PROMPT, out_path, language
                    )
                elif self.anchor_path:
                    chunk_wav_paths = self.synthesize_with_anchor(
                        text, self.anchor_path, out_path, language
                    )
                else:
                    chunk_wav_paths = self._synthesize(
                        text, out_path, language=language
                    )
                if i == 0:
                    self.anchor_path = chunk_wav_paths[0]
            except KeyboardInterrupt:
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except OSError:
                        pass
                logger.warning(
                    f"Force-stopped at chunk {i}. Partial file removed. "
                    "CUDA state may be inconsistent — consider restarting the app."
                )
                raise

            wav_paths.extend(chunk_wav_paths)

            # Tag WAV files with metadata from payload
            for wav_path in chunk_wav_paths:
                self._tag_wav(Path(wav_path), payload)

            if chunk_done_cb:
                chunk_done_cb()

            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (i + 1) % 20 == 0:
                logger.info(
                    f"Checkpoint at chunk {i + 1}: clearing model cache and CUDA memory"
                )
                if (
                    self.engine is not None
                    and hasattr(self.engine, "model")
                    and self.engine.model is not None
                ):
                    if hasattr(self.engine.model, "reset_cache"):
                        try:
                            self.engine.model.reset_cache()
                        except Exception as e:
                            logger.debug(f"Could not reset model cache: {e}")
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

        return wav_paths
