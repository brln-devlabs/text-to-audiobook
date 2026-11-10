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

import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    cuda_path = (
        os.environ.get("CUDA_PATH")
        or r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    )
    cuda_bin = Path(cuda_path) / "bin"
    if cuda_bin.is_dir():
        os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(cuda_bin))

try:
    import llama_cpp
    import torch
    from huggingface_hub import hf_hub_download
except ImportError:
    torch: Any = None  # type: ignore[no-redef]
    llama_cpp: Any = None  # type: ignore[no-redef]
    hf_hub_download: Any = None  # type: ignore[no-redef]

from . import config

logger = logging.getLogger(__name__)


class LLMNormalizer:
    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.llm = None

    def _download_model_if_needed(self):
        model_path = Path(self.model_path)
        if model_path.exists():
            return self.model_path

        if hf_hub_download is None:
            raise ImportError("huggingface_hub required for auto-download")

        model_path.parent.mkdir(parents=True, exist_ok=True)
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning(
                "HF_TOKEN not set in environment; download may fail for private models"
            )

        logger.info("Downloading Llama model from HuggingFace...")
        cache_path = hf_hub_download(
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
            token=hf_token,
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False,
        )
        logger.info(f"Model downloaded to {cache_path}")
        return cache_path

    def load(self):
        if llama_cpp is None:
            logger.warning("llama_cpp not available, using mock mode")
            return

        try:
            model_path = self._download_model_if_needed()
            logger.info(f"Loading LLM from {model_path}")
            self.llm = llama_cpp.Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=getattr(config, "LLM_GPU_LAYERS", -1),
                verbose=False,
            )
        except Exception as e:
            logger.warning(f"Failed to load LLM: {e}. Using mock mode.")
            self.llm = None

    def unload(self):
        if self.llm is not None:
            logger.info("Unloading LLM")
            del self.llm
            self.llm = None
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # JSON extraction
    # ------------------------------------------------------------------

    def _extract_json(self, text: str, opener: str = "{", closer: str = "}"):
        """Extract first balanced JSON value from messy LLM output.

        Walks the string with a brace counter so we get the first complete
        object/array, skipping characters inside string literals.
        """
        text = text.strip()
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
        return None

    def _call_llm(self, prompt: str, max_tokens: int, stops: list):
        response = self.llm(prompt, max_tokens=max_tokens, stop=stops)
        return response["choices"][0]["text"].strip()

    # ------------------------------------------------------------------
    # Step A: emotion analysis
    # ------------------------------------------------------------------

    EMOTIONS = [
        # Basic / core emotions
        "neutral",
        "happy",
        "sad",
        "angry",
        "fearful",
        "surprised",
        "disgusted",
        # Calm / low-arousal positive
        "calm",
        "content",
        "relieved",
        "serene",
        "peaceful",
        "tranquil",
        "satisfied",
        "relaxed",
        # High-arousal positive
        "ecstatic",
        "elated",
        "euphoric",
        "joyful",
        "excited",
        "thrilled",
        "exhilarated",
        "enthusiastic",
        "playful",
        "amused",
        "cheerful",
        "delighted",
        "gleeful",
        "jubilant",
        # Hope / aspiration
        "hopeful",
        "inspired",
        "optimistic",
        "motivated",
        "determined",
        "confident",
        "empowered",
        "eager",
        "anticipating",
        # Affection / connection
        "affectionate",
        "loving",
        "tender",
        "compassionate",
        "empathetic",
        "sympathetic",
        "warm",
        "nurturing",
        "protective",
        "trusting",
        "admiring",
        "grateful",
        "appreciative",
        # Pride / accomplishment
        "proud",
        "accomplished",
        "triumphant",
        "victorious",
        "validated",
        # Sadness spectrum
        "melancholic",
        "sorrowful",
        "grief_stricken",
        "heartbroken",
        "disappointed",
        "dejected",
        "despondent",
        "hopeless",
        "despairing",
        "miserable",
        "gloomy",
        "wistful",
        "nostalgic",
        "homesick",
        "regretful",
        # Loneliness / isolation
        "lonely",
        "isolated",
        "abandoned",
        "rejected",
        "excluded",
        "alienated",
        # Anger spectrum
        "frustrated",
        "irritated",
        "annoyed",
        "agitated",
        "resentful",
        "bitter",
        "indignant",
        "outraged",
        "furious",
        "enraged",
        "hostile",
        "contemptuous",
        "spiteful",
        "vengeful",
        # Fear / anxiety spectrum
        "tense",
        "anxious",
        "worried",
        "nervous",
        "uneasy",
        "apprehensive",
        "panicked",
        "terrified",
        "horrified",
        "dread",
        "paranoid",
        "insecure",
        "vulnerable",
        "intimidated",
        # Shame / guilt family
        "guilty",
        "ashamed",
        "embarrassed",
        "humiliated",
        "remorseful",
        "self_conscious",
        "awkward",
        "mortified",
        # Envy / jealousy
        "envious",
        "jealous",
        "covetous",
        "possessive",
        # Disgust spectrum
        "repulsed",
        "revolted",
        "appalled",
        "horrified",
        "contempt",
        # Surprise spectrum
        "astonished",
        "amazed",
        "shocked",
        "stunned",
        "bewildered",
        "dumbfounded",
        "startled",
        # Confusion / cognitive
        "confused",
        "perplexed",
        "puzzled",
        "uncertain",
        "doubtful",
        "skeptical",
        "suspicious",
        "curious",
        "intrigued",
        "fascinated",
        "contemplative",
        "pensive",
        "reflective",
        "focused",
        "absorbed",
        # Low-energy / disengaged
        "bored",
        "apathetic",
        "indifferent",
        "numb",
        "detached",
        "dissociated",
        "exhausted",
        "weary",
        "fatigued",
        "drained",
        "listless",
        "lethargic",
        "sluggish",
        # Overwhelmed
        "overwhelmed",
        "stressed",
        "burdened",
        "conflicted",
        "torn",
        "ambivalent",
        "restless",
        # Mischievous / complex social
        "mischievous",
        "smug",
        "sarcastic",
        "defiant",
        "rebellious",
        "stubborn",
        "impatient",
        "flirtatious",
        "sentimental",
        "bittersweet",
        # Awe / transcendent
        "awestruck",
        "reverent",
        "humbled",
        "moved",
        "touched",
        "vulnerable_open",
    ]

    # Languages Qwen3-TTS-VoiceDesign accepts as the `language=` argument.
    SUPPORTED_LANGUAGES = [
        "english",
        "chinese",
        "spanish",
        "french",
        "german",
        "italian",
        "portuguese",
        "russian",
        "japanese",
        "korean",
    ]
    DEFAULT_LANGUAGE = "english"

    # Speaking speed options for TTS
    SPEAKING_SPEEDS = [
        "slow",
        "normal",
        "fast",
    ]
    DEFAULT_SPEAKING_SPEED = "normal"

    @staticmethod
    def _script_candidates(text: str) -> set:
        """Return the set of SUPPORTED_LANGUAGES plausible given the Unicode
        scripts present in `text`. Used to filter LLM hallucinations like
        labelling pure ASCII text as 'japanese'.
        """
        latin = cyrillic = hiragana_katakana = hangul = han = 0
        for ch in text:
            cp = ord(ch)
            if 0x0041 <= cp <= 0x024F:  # Latin (basic + extended)
                latin += 1
            elif 0x0400 <= cp <= 0x04FF:  # Cyrillic
                cyrillic += 1
            elif 0x3040 <= cp <= 0x30FF:  # Hiragana + Katakana
                hiragana_katakana += 1
            elif 0xAC00 <= cp <= 0xD7AF:  # Hangul syllables
                hangul += 1
            elif 0x4E00 <= cp <= 0x9FFF:  # CJK unified ideographs (Han)
                han += 1

        # Decide what scripts dominate.
        if hangul > 5:
            return {"korean"}
        if hiragana_katakana > 5:
            return {"japanese"}
        if han > 20 and hiragana_katakana == 0 and hangul == 0:
            return {"chinese"}
        if cyrillic > 20:
            return {"russian"}
        if latin > 20:
            # Pure Latin script — must be one of the Western languages.
            return {"english", "spanish", "french", "german", "italian", "portuguese"}
        # Too short / mixed / unknown: trust the LLM.
        return set(LLMNormalizer.SUPPORTED_LANGUAGES)

    @staticmethod
    def format_chapter_title(source_file: str) -> str:
        """Convert source filename to a readable chapter title.

        Examples: "Chapter_01.md" -> "Chapter 01"
                  "intro.txt" -> "Intro"
                  "01_prologue.md" -> "Prologue"
        """
        # Remove file extension
        name = Path(source_file).stem
        # Replace underscores and hyphens with spaces
        name = name.replace("_", " ").replace("-", " ")
        # Capitalize each word
        name = " ".join(w.capitalize() for w in name.split() if w)
        return name or "Chapter"

    def detect_speaking_speed(self, text: str) -> str:
        """Detect the appropriate speaking speed for the given text.

        Analyzes text characteristics (punctuation density, sentence length,
        emotional tone indicators) to recommend a speaking speed.
        Returns one of: "slow", "normal", "fast".
        """
        if self.llm is None or not text.strip():
            return self.DEFAULT_SPEAKING_SPEED

        sample = text[:500]
        speeds_csv = ", ".join(self.SPEAKING_SPEEDS)
        system_prompt = (
            "Analyze the text and recommend a speaking speed for narration.\n"
            f"Options: {speeds_csv}\n"
            "Consider: sentence length, punctuation density, emotional indicators, complexity.\n"
            "Output ONLY a JSON object with one key: speed (lowercase).\n"
            f"If unsure, use default: {self.DEFAULT_SPEAKING_SPEED}\n"
            'Example: {"speed": "normal"}'
        )

        last_error = None
        for attempt in range(1, self.MAX_LLM_RETRIES + 1):
            try:
                prompt = f"{system_prompt}\n\nText:\n{sample}\n\nJSON:"
                output = self._call_llm(
                    prompt,
                    max_tokens=64,
                    stops=["\n\nText:", "\n\nJSON:", "</s>", "<|eot_id|>"],
                )
                data = self._extract_json(output)
                if not isinstance(data, dict):
                    raise ValueError(f"non-dict response: {output!r}")
                speed = str(data.get("speed", "")).strip().lower()
                if not speed or speed not in self.SPEAKING_SPEEDS:
                    return self.DEFAULT_SPEAKING_SPEED
                return speed
            except Exception as e:
                last_error = e
                logger.warning(
                    f"detect_speaking_speed attempt {attempt}/{self.MAX_LLM_RETRIES} failed: {e}"
                )

        logger.warning(
            f"detect_speaking_speed failed after {self.MAX_LLM_RETRIES} attempts: {last_error}; using default"
        )
        return self.DEFAULT_SPEAKING_SPEED

    def detect_chapter_info(self, text: str) -> dict:
        """Detect chapter number and title from text.

        Looks for patterns like "CHAPTER 2. The Beginning" and extracts
        the chapter number and title. Returns {chapter_number, chapter_title}
        with None values on failure.
        """
        if self.llm is None or not text.strip():
            return {"chapter_number": None, "chapter_title": None}

        sample = text[:1000]
        system_prompt = (
            "Extract chapter information from the text.\n"
            "Look for patterns like: 'CHAPTER 2. The Beginning', 'Chapter II - Title', etc.\n"
            "Output ONLY a JSON object with two keys:\n"
            "  - chapter_number: extracted integer chapter number, or null if not found\n"
            "  - chapter_title: extracted chapter title string, or null if not found\n"
            'Example: {"chapter_number": 2, "chapter_title": "The Beginning"}'
        )

        last_error = None
        for attempt in range(1, self.MAX_LLM_RETRIES + 1):
            try:
                prompt = f"{system_prompt}\n\nText:\n{sample}\n\nJSON:"
                output = self._call_llm(
                    prompt,
                    max_tokens=128,
                    stops=["\n\nText:", "\n\nJSON:", "</s>", "<|eot_id|>"],
                )
                data = self._extract_json(output)
                if not isinstance(data, dict):
                    raise ValueError(f"non-dict response: {output!r}")
                chapter_number = data.get("chapter_number")
                chapter_title = data.get("chapter_title")
                if chapter_number is not None:
                    try:
                        chapter_number = int(chapter_number)
                    except (TypeError, ValueError):
                        chapter_number = None
                if chapter_title is not None:
                    chapter_title = str(chapter_title).strip()
                    if not chapter_title:
                        chapter_title = None
                return {
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                }
            except Exception as e:
                last_error = e
                logger.warning(
                    f"detect_chapter_info attempt {attempt}/{self.MAX_LLM_RETRIES} failed: {e}"
                )

        logger.warning(
            f"detect_chapter_info failed after {self.MAX_LLM_RETRIES} attempts: {last_error}"
        )
        return {"chapter_number": None, "chapter_title": None}

    def detect_story_title(self, text: str) -> str:
        """Detect the story/book title from text context.

        Analyzes the text to infer the story or book title (from title pages,
        headers, or overall context). Returns the title string or None on failure.
        """
        if self.llm is None or not text.strip():
            return None

        sample = text[:1000]
        system_prompt = (
            "Infer the story, book, or novel title from the provided text.\n"
            "Look for: title pages, chapter headers, author references, or contextual clues.\n"
            "Output ONLY a JSON object with one key:\n"
            "  - title: the inferred title string, or null if not determinable\n"
            'Example: {"title": "The Fellowship of the Ring"}'
        )

        last_error = None
        for attempt in range(1, self.MAX_LLM_RETRIES + 1):
            try:
                prompt = f"{system_prompt}\n\nText:\n{sample}\n\nJSON:"
                output = self._call_llm(
                    prompt,
                    max_tokens=128,
                    stops=["\n\nText:", "\n\nJSON:", "</s>", "<|eot_id|>"],
                )
                data = self._extract_json(output)
                if not isinstance(data, dict):
                    raise ValueError(f"non-dict response: {output!r}")
                title = data.get("title")
                if title is not None:
                    title = str(title).strip()
                    if title:
                        return title
                return None
            except Exception as e:
                last_error = e
                logger.warning(
                    f"detect_story_title attempt {attempt}/{self.MAX_LLM_RETRIES} failed: {e}"
                )

        logger.warning(
            f"detect_story_title failed after {self.MAX_LLM_RETRIES} attempts: {last_error}"
        )
        return None

    def detect_language(self, text: str) -> str:
        """Ask the LLM which language `text` is written in.

        Combines an LLM call with a Unicode-script heuristic that rejects
        answers contradicted by the actual characters in the text (e.g. LLM
        says 'japanese' for pure ASCII English). Retries up to
        MAX_LLM_RETRIES; on persistent failure raises RuntimeError.
        """
        if self.llm is None or not text.strip():
            return self.DEFAULT_LANGUAGE
        sample = text[:1000]  # 1000 chars for more reliable LLM detection
        plausible = self._script_candidates(sample)

        # Fast path: when the script makes the answer unambiguous, skip LLM.
        if len(plausible) == 1:
            return next(iter(plausible))

        options = ", ".join(sorted(plausible))
        system_prompt = (
            f"Identify the language of the text. Choose ONE from: {options}.\n"
            "Output ONLY a JSON object with one key: language (lowercase).\n"
            f"If your confidence is below 80%, use the default: {self.DEFAULT_LANGUAGE}.\n"
            'Example: {"language": "english"}'
        )

        last_error = None
        for attempt in range(1, self.MAX_LLM_RETRIES + 1):
            try:
                prompt = f"{system_prompt}\n\nText:\n{sample}\n\nJSON:"
                output = self._call_llm(
                    prompt,
                    max_tokens=64,
                    stops=["\n\nText:", "\n\nJSON:", "</s>", "<|eot_id|>"],
                )
                data = self._extract_json(output)
                if not isinstance(data, dict):
                    raise ValueError(f"non-dict response: {output!r}")
                lang = str(data.get("language", "")).strip().lower()
                if not lang or lang not in self.SUPPORTED_LANGUAGES:
                    return self.DEFAULT_LANGUAGE
                if lang not in plausible:
                    raise ValueError(
                        f"LLM said {lang!r} but Unicode scripts in text only support "
                        f"{sorted(plausible)} — rejecting hallucination"
                    )
                return lang
            except Exception as e:
                last_error = e
                logger.warning(
                    f"detect_language attempt {attempt}/{self.MAX_LLM_RETRIES} failed: {e}"
                )

        raise RuntimeError(
            f"FATAL ERROR: detect_language failed after {self.MAX_LLM_RETRIES} attempts: {last_error}"
        )

    def sort_files(self, files: list) -> list:
        """Ask the LLM to order files by narrative sequence.

        Input: list of dicts {"name": filename, "preview": first ~200 chars}.
        Files are expected to arrive pre-sorted with natsort as the baseline.
        Returns: list of file names in narrative reading order.
        Falls back to natsort on failure.
        """
        from natsort import natsorted

        names_in = [f["name"] for f in files]
        if self.llm is None or len(files) <= 1:
            return natsorted(names_in)

        listing = "\n".join(
            f'- {f["name"]}: "{(f.get("preview") or "")[:200].strip()}"' for f in files
        )
        system_prompt = (
            "You are sorting book chapter files into correct reading order.\n"
            "Rules:\n"
            "  1. Files are shown in their default order. Only reorder when the names or "
            "     content clearly indicate a different narrative sequence.\n"
            "  2. Recognise ordinal words: 'first'<'second'<'third'; "
            "     'one'<'two'<'three'<'four'<'five'<'six'<'seven'<'eight'<'nine'<'ten' etc.\n"
            "  3. Recognise structural keywords: 'prologue'/'preface'/'introduction' come "
            "     BEFORE chapters; 'epilogue'/'afterword'/'appendix' come AFTER chapters.\n"
            "  4. Output ONLY a JSON array of ALL file names in reading order. No commentary.\n"
            "Example:\n"
            "  Input:\n"
            '    - Chapter_Two.md: "The morning was cold..."\n'
            '    - Prologue.md: "Long before the war began..."\n'
            '    - Chapter_One.md: "She stepped off the train..."\n'
            '  Output: ["Prologue.md", "Chapter_One.md", "Chapter_Two.md"]'
        )
        try:
            prompt = f"{system_prompt}\n\nInput:\n{listing}\n\nOutput:"
            output = self._call_llm(
                prompt,
                max_tokens=2048,
                stops=["\n\nInput:", "\n\nOutput:", "</s>", "<|eot_id|>"],
            )
            ordered = self._extract_json(output, opener="[", closer="]")
            if not isinstance(ordered, list):
                raise ValueError("LLM did not return a JSON array")
            valid = [n for n in ordered if isinstance(n, str) and n in names_in]
            missing = [n for n in names_in if n not in valid]
            if missing:
                logger.warning(
                    f"LLM sort missed {len(missing)} files; appending in natsort order"
                )
                valid.extend(natsorted(missing))
            if not valid:
                raise ValueError("no valid file names returned")
            logger.info(f"LLM ordered {len(valid)} files")
            return valid
        except Exception as e:
            logger.warning(f"File sort failed: {e}; falling back to natsort")
            return natsorted(names_in)

    def analyze_emotions(self, text: str) -> list:
        """Return [{text, emotion}, ...] — one segment per emotional span."""
        if self.llm is None:
            return [{"text": text, "emotion": "neutral"}]

        # Sentence-index schema: we pre-split the text into numbered sentences
        # and ask the LLM only for integer index ranges. The model never copies
        # any prose — it cannot hallucinate text. Validation reduces to
        # checking integer ranges cover [1..N] without gaps or overlaps.
        import re as _re

        sentences = _re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        n_sent = len(sentences)

        if n_sent <= 1:
            # Trivial case: nothing to tag.
            return [{"text": text, "emotion": "neutral"}]

        emotions_csv = ", ".join(self.EMOTIONS)
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))

        system_prompt = (
            "You tag an audiobook passage with emotion spans by SENTENCE NUMBER.\n"
            f"Allowed emotions (pick exactly one): {emotions_csv}.\n"
            "The Input is a numbered list of sentences (1..N). Group consecutive "
            "sentences that share the same emotion into spans.\n"
            "Output a JSON array of objects with these EXACT keys:\n"
            '  "Start"   — INTEGER sentence number where the span begins (1-based).\n'
            '  "Stop"    — INTEGER sentence number where the span ends (inclusive).\n'
            '  "Emotion" — one allowed emotion value.\n'
            "HARD RULES:\n"
            f"  1. Start and Stop are integers in [1, {n_sent}], with Start <= Stop.\n"
            "  2. Spans MUST be in order: each span's Start = previous span's Stop + 1.\n"
            f"  3. The first span's Start = 1; the last span's Stop = {n_sent}. NO gaps, NO overlaps.\n"
            "  4. If unsure, output ONE object: "
            f'{{"Start": 1, "Stop": {n_sent}, "Emotion": "neutral"}}.\n'
            "  5. Output ONLY the JSON array — no prose, no markdown, no comments.\n"
            "Schema example (numbers are placeholders):\n"
            '  [{"Start": 1, "Stop": 3, "Emotion": "neutral"}, '
            '{"Start": 4, "Stop": 5, "Emotion": "tense"}]'
        )

        last_error = None
        last_raw = ""
        for attempt in range(1, self.MAX_LLM_RETRIES + 1):
            try:
                prompt = f"{system_prompt}\n\nInput:\n{numbered}\n\nJSON array:"
                output = self._call_llm(
                    prompt,
                    max_tokens=1024,
                    stops=["\n\nInput:", "\n\nJSON array:", "</s>", "<|eot_id|>"],
                )
                last_raw = output
                segments = self._extract_json(output, opener="[", closer="]")
                if not isinstance(segments, list) or not segments:
                    raise ValueError(
                        f"no segments parsed | raw[:200]={last_raw[:200]!r}"
                    )

                # Sanity: cannot have more emotion spans than sentences. If the
                # LLM returned more, it has hallucinated or split sentences it
                # was not asked to split — retry rather than try to repair.
                if len(segments) > n_sent:
                    raise ValueError(
                        f"too many spans: LLM returned {len(segments)} but only "
                        f"{n_sent} sentence(s) available — likely hallucinated"
                    )

                # Parse + validate ranges
                spans = []
                rejected = []
                for seg in segments:
                    if not isinstance(seg, dict):
                        rejected.append(("not-a-dict", seg))
                        continue
                    try:
                        raw_start = (
                            seg.get("Start")
                            if seg.get("Start") is not None
                            else seg.get("start")
                        )
                        raw_stop = (
                            seg.get("Stop")
                            if seg.get("Stop") is not None
                            else seg.get("stop")
                        )
                        if raw_start is None or raw_stop is None:
                            rejected.append(("missing-range", seg))
                            continue
                        start = int(raw_start)
                        stop = int(raw_stop)
                    except (TypeError, ValueError):
                        rejected.append(("non-int-range", seg))
                        continue
                    emotion = (
                        str(seg.get("Emotion") or seg.get("emotion") or "neutral")
                        .strip()
                        .lower()
                    )
                    if emotion not in self.EMOTIONS:
                        emotion = "neutral"
                    if not (1 <= start <= stop <= n_sent):
                        rejected.append((f"out-of-range[1..{n_sent}]", (start, stop)))
                        continue
                    spans.append((start, stop, emotion))

                if not spans:
                    raise ValueError(
                        f"no valid spans | rejected: {rejected[:3]} | raw[:200]={last_raw[:200]!r}"
                    )

                # Order & gap analysis (the "dead spaces" check)
                spans.sort(key=lambda s: s[0])
                gaps = []
                overlaps = []
                expected = 1
                for st, sp, _ in spans:
                    if st > expected:
                        gaps.append((expected, st - 1))
                    elif st < expected:
                        overlaps.append((st, expected - 1))
                    expected = max(expected, sp + 1)
                if expected <= n_sent:
                    gaps.append((expected, n_sent))

                if overlaps:
                    raise ValueError(f"overlapping spans at sentences {overlaps[:3]}")

                # Fill any gaps with neutral spans (these are the "dead spaces").
                if gaps:
                    logger.debug(
                        f"  filling {len(gaps)} dead-space gap(s) with neutral: {gaps[:5]}"
                    )
                    for gs, ge in gaps:
                        spans.append((gs, ge, "neutral"))
                    spans.sort(key=lambda s: s[0])

                # Build output segments by joining the actual sentences.
                result = []
                for st, sp, emotion in spans:
                    seg_text = " ".join(sentences[st - 1 : sp]).strip()
                    if seg_text:
                        result.append({"text": seg_text, "emotion": emotion})
                if not result:
                    raise ValueError("empty result after assembly")
                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    f"analyze_emotions attempt {attempt}/{self.MAX_LLM_RETRIES} failed: {e}"
                )
                if last_raw:
                    logger.debug(f"  raw LLM output[:300]: {last_raw[:300]!r}")

        raise RuntimeError(
            f"FATAL ERROR: analyze_emotions failed after {self.MAX_LLM_RETRIES} attempts: "
            f"{last_error} | last_raw[:200]={last_raw[:200]!r}"
        )

    def _reconstruct_from_anchors(self, source: str, anchors: list) -> list:
        """Use LLM segments only as emotion *anchors*; build final segments from source text.

        For each LLM anchor we locate where it starts in the source. Then we cover the
        whole source by slicing it at those start positions, assigning each slice the
        preceding anchor's emotion (or 'neutral' for content before the first anchor).
        Guarantees 100% source coverage with no LLM-injected paraphrasing.
        """
        positions = []  # list of (start_index, emotion)
        cursor = 0
        for a in anchors:
            anchor_text = a["text"]
            head = anchor_text[
                :60
            ]  # match by leading window — robust against LLM minor edits
            idx = source.find(head, cursor)
            if idx == -1:
                # fall back to a shorter prefix
                head = anchor_text[:30]
                idx = source.find(head, cursor) if head else -1
            if idx == -1 or idx < cursor:
                continue
            positions.append((idx, a["emotion"]))
            cursor = idx + max(1, len(head))

        if not positions:
            logger.warning(
                "Could not anchor any LLM segment to source; using single neutral segment"
            )
            return [{"text": source, "emotion": "neutral"}]

        # Build final segments by cutting source at each position
        result = []
        if positions[0][0] > 0:
            head_text = source[: positions[0][0]].strip()
            if head_text:
                result.append({"text": head_text, "emotion": "neutral"})
        for k, (start, emotion) in enumerate(positions):
            end = positions[k + 1][0] if k + 1 < len(positions) else len(source)
            slice_text = source[start:end].strip()
            if slice_text:
                result.append({"text": slice_text, "emotion": emotion})

        # Sanity check: total coverage must reconstruct the source modulo whitespace
        joined = " ".join(s["text"] for s in result)
        if len(joined) < int(0.9 * len(source.strip())):
            logger.warning(
                f"Reconstruction covered only {len(joined)}/{len(source.strip())} chars; "
                f"falling back to single neutral segment"
            )
            return [{"text": source, "emotion": "neutral"}]

        return result

    # ------------------------------------------------------------------
    # Step C: text normalization
    # ------------------------------------------------------------------

    @property
    def MAX_LLM_RETRIES(self) -> int:
        """Read live from config so settings changes apply without restart."""
        from . import config

        return int(getattr(config, "LLM_MAX_RETRIES", 3))

    def _validate_replacements(self, source: str, replacements: list) -> list:
        """Verify each {find, occurrence} actually exists in source at that occurrence.

        Returns a sanitized list of dicts with offsets resolved:
        [{"start": int, "end": int, "replace": str}, ...] sorted by start.
        Raises ValueError if any replacement cannot be located.
        """
        resolved = []
        for r in replacements:
            if not isinstance(r, dict):
                raise ValueError(f"replacement is not a dict: {r!r}")
            find = r.get("find")
            replace = r.get("replace")
            occurrence = r.get("occurrence", 1)
            if not isinstance(find, str) or not find:
                raise ValueError(f"missing/empty 'find' in {r!r}")
            if not isinstance(replace, str):
                raise ValueError(f"missing 'replace' in {r!r}")
            try:
                occurrence = int(occurrence)
            except (TypeError, ValueError):
                raise ValueError(f"invalid 'occurrence' in {r!r}")
            if occurrence < 1:
                raise ValueError(f"occurrence must be >= 1 in {r!r}")

            start = -1
            cursor = 0
            for _ in range(occurrence):
                start = source.find(find, cursor)
                if start == -1:
                    break
                cursor = start + 1
            if start == -1:
                raise ValueError(
                    f"'find'={find!r} occurrence #{occurrence} not present in source"
                )
            resolved.append(
                {"start": start, "end": start + len(find), "replace": replace}
            )

        resolved.sort(key=lambda x: x["start"])
        for a, b in zip(resolved, resolved[1:]):
            if a["end"] > b["start"]:
                raise ValueError("overlapping replacements")
        return resolved

    def _apply_replacements(self, source: str, resolved: list) -> str:
        out = []
        cursor = 0
        for r in resolved:
            out.append(source[cursor : r["start"]])
            out.append(r["replace"])
            cursor = r["end"]
        out.append(source[cursor:])
        return "".join(out)

    def normalize_text(self, text: str) -> str:
        """Expand numbers, dates, abbreviations for TTS via location-specific edits.

        The LLM returns only the substrings that need to change (with their
        occurrence index). We locate each in the source, validate, and apply.
        Retries up to MAX_LLM_RETRIES times; on persistent failure raises
        RuntimeError to stop the pipeline.
        """
        if self.llm is None:
            return text

        system_prompt = (
            "You normalize text for a TTS narrator. Do NOT rewrite the passage. "
            "Return ONLY the substrings that must change (abbreviations, numbers, "
            "dates, times, symbols) so they read naturally aloud.\n"
            "For each edit output: 'find' = the EXACT substring copied verbatim "
            "from the input, 'occurrence' = which appearance (1 = first), "
            "'replace' = the spoken form.\n"
            "Rules: do not paraphrase; do not merge or reorder words; if nothing "
            "needs changing return an empty list.\n"
            "Output ONLY a JSON object with one key: replacements (a list).\n"
            "Example:\n"
            '  Input: "Dr. Smith arrived at 3:45 PM in 1995. Dr. Smith smiled."\n'
            '  Output: {"replacements": ['
            '{"find": "Dr.", "occurrence": 1, "replace": "Doctor"}, '
            '{"find": "3:45 PM", "occurrence": 1, "replace": "three forty-five PM"}, '
            '{"find": "1995", "occurrence": 1, "replace": "nineteen ninety-five"}, '
            '{"find": "Dr.", "occurrence": 2, "replace": "Doctor"}'
            "]}"
        )

        last_error = None
        for attempt in range(1, self.MAX_LLM_RETRIES + 1):
            try:
                prompt = f"{system_prompt}\n\nText:\n{text}\n\nJSON:"
                output = self._call_llm(
                    prompt,
                    max_tokens=1024,
                    stops=["\n\nText:", "\n\nJSON:", "</s>", "<|eot_id|>"],
                )
                data = self._extract_json(output)
                if not isinstance(data, dict) or "replacements" not in data:
                    raise ValueError("no 'replacements' key in response")
                reps = data["replacements"]
                if not isinstance(reps, list):
                    raise ValueError("'replacements' is not a list")
                resolved = self._validate_replacements(text, reps)
                return self._apply_replacements(text, resolved)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"normalize_text attempt {attempt}/{self.MAX_LLM_RETRIES} failed: {e}"
                )

        raise RuntimeError(
            f"FATAL ERROR: normalize_text failed after {self.MAX_LLM_RETRIES} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Step D: context fill
    # ------------------------------------------------------------------

    def fill_context(self, text: str, prev_text: str = "", next_text: str = "") -> str:
        """Resolve pronouns / elided references via location-specific edits only.

        The LLM returns substrings within SEGMENT to replace (e.g. "she" -> "Sarah").
        Each edit is validated against the source by find+occurrence; on failure
        retries up to MAX_LLM_RETRIES, then raises RuntimeError.
        """
        if self.llm is None:
            return text

        system_prompt = (
            "Resolve pronouns and elided references in the SEGMENT using PREVIOUS "
            "and NEXT context. Do NOT rewrite the segment. Return ONLY the "
            "substrings inside SEGMENT that must change (e.g. an ambiguous pronoun "
            "like 'she' replaced with the actual name from context).\n"
            "For each edit output: 'find' = the EXACT substring copied verbatim "
            "from SEGMENT, 'occurrence' = which appearance in SEGMENT (1 = first), "
            "'replace' = the resolved form.\n"
            "Rules: do not paraphrase, shorten, or reorder; if SEGMENT needs no "
            "edits return an empty list.\n"
            "Output ONLY a JSON object with one key: replacements (a list).\n"
            "Example:\n"
            '  PREVIOUS: "Sarah opened the door."\n'
            '  SEGMENT: "Then she walked inside."\n'
            '  NEXT: "The room was dark."\n'
            '  Output: {"replacements": [{"find": "she", "occurrence": 1, "replace": "Sarah"}]}'
        )

        last_error = None
        for attempt in range(1, self.MAX_LLM_RETRIES + 1):
            try:
                prompt = (
                    f"{system_prompt}\n\n"
                    f"PREVIOUS:\n{prev_text or '(none)'}\n\n"
                    f"SEGMENT:\n{text}\n\n"
                    f"NEXT:\n{next_text or '(none)'}\n\n"
                    f"JSON:"
                )
                output = self._call_llm(
                    prompt,
                    max_tokens=1024,
                    stops=[
                        "\n\nPREVIOUS:",
                        "\n\nSEGMENT:",
                        "\n\nNEXT:",
                        "\n\nJSON:",
                        "</s>",
                        "<|eot_id|>",
                    ],
                )
                data = self._extract_json(output)
                if not isinstance(data, dict) or "replacements" not in data:
                    raise ValueError("no 'replacements' key in response")
                reps = data["replacements"]
                if not isinstance(reps, list):
                    raise ValueError("'replacements' is not a list")
                resolved = self._validate_replacements(text, reps)
                return self._apply_replacements(text, resolved)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"fill_context attempt {attempt}/{self.MAX_LLM_RETRIES} failed: {e}"
                )

        raise RuntimeError(
            f"FATAL ERROR: fill_context failed after {self.MAX_LLM_RETRIES} attempts: {last_error}"
        )

    # ------------------------------------------------------------------
    # Backward-compat single-shot wrapper used by tests
    # ------------------------------------------------------------------

    def normalize(self, chunk_text: str) -> dict:
        """Single-shot normalization for the legacy test interface."""
        if self.llm is None:
            return {
                "normalized_text": chunk_text,
                "emotion": "neutral",
                "detected_emotions": ["neutral"],
                "language": self.DEFAULT_LANGUAGE,
            }

        try:
            normalized = self.normalize_text(chunk_text)
        except RuntimeError:
            logger.warning("LLM normalization failed, returning text unchanged")
            normalized = chunk_text

        try:
            language = self.detect_language(chunk_text)
        except RuntimeError:
            logger.warning("LLM language detection failed, using default")
            language = self.DEFAULT_LANGUAGE

        return {
            "normalized_text": normalized,
            "emotion": "neutral",
            "detected_emotions": ["neutral"],
            "language": language,
        }

    # ------------------------------------------------------------------
    # Orchestrator: A -> B -> C -> D with per-step caching
    # ------------------------------------------------------------------

    def process_all(self, chunks, cache_dir, progress_cb=None, stop_event=None) -> list:
        """Run analyze -> rechunk -> normalize -> context fill.

        Cache layout (varies by STORAGE_BACKEND):
          JSON: <cache_dir>/analyze/chunk_NNN.json, <cache_dir>/segments/seg_GGGG.json
          SQLite: <cache_dir>/storage.db with chunks and segments tables
        """
        from . import config
        from .storage_factory import create_storage

        cache_dir = Path(cache_dir)
        storage = create_storage(cache_dir)
        storage.init_schema()

        def report(msg):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg)

        # --- Detect project language by majority vote over the first 10 chunks.
        # The 1B model hallucinates on individual samples (English novel labelled
        # 'japanese' on chunk 6, 'korean' on chunk 7), but the *mode* of 10
        # independent detections is reliable. The script-gate inside
        # detect_language already filters obvious script contradictions.
        project_language = None
        fixed_language = (
            str(getattr(config, "TTS_LANGUAGE", "auto") or "auto").strip().lower()
        )
        if chunks and fixed_language != "auto":
            # Language fixed in the settings: no need to ask the LLM.
            project_language = fixed_language
            report(f"[A] project language: {project_language} (fixed in settings)")
        elif chunks:
            from collections import Counter

            votes: list[str] = []
            n_probe = min(10, len(chunks))
            for probe_chunk in chunks[:n_probe]:
                try:
                    votes.append(self.detect_language(probe_chunk.text))
                except Exception as e:
                    logger.debug(f"language probe failed on a chunk: {e}")
            if votes:
                tally = Counter(votes)
                project_language, top_count = tally.most_common(1)[0]
                report(
                    f"[A] project language: {project_language} "
                    f"({top_count}/{len(votes)} votes; tally={dict(tally)})"
                )
            else:
                logger.warning(
                    f"All {n_probe} project-language probes failed; "
                    f"falling back to per-chunk detection"
                )

        # --- Step A: per-chunk emotion analysis -------------------------------
        analyzed = []  # list[dict]: {text, emotion, ends_paragraph}

        for i, chunk in enumerate(chunks):
            if stop_event is not None and stop_event.is_set():
                report(f"Stop requested after {i}/{len(chunks)} chunks analyzed")
                break

            payload = storage.load_chunk_meta(i)
            if payload is not None:
                segs = payload.get("segments", [])
                ends_paragraph = payload.get("ends_paragraph", chunk.ends_paragraph)
                language = payload.get("language") or self.DEFAULT_LANGUAGE
            else:
                report(f"[A] analyzing chunk {i + 1}/{len(chunks)}")
                try:
                    segs = self.analyze_emotions(chunk.text)
                except RuntimeError:
                    logger.warning(
                        f"LLM emotion analysis failed for chunk {i}, using neutral fallback"
                    )
                    segs = [{"text": chunk.text, "emotion": "neutral"}]
                ends_paragraph = chunk.ends_paragraph
                if project_language is not None:
                    language = project_language
                else:
                    try:
                        language = self.detect_language(chunk.text)
                    except RuntimeError:
                        logger.warning(
                            f"LLM language detection failed for chunk {i}, using default"
                        )
                        language = self.DEFAULT_LANGUAGE
                    report(f"[A] chunk {i + 1} language: {language}")
                storage.save_chunk_meta(
                    i,
                    {
                        "segments": segs,
                        "ends_paragraph": ends_paragraph,
                        "language": language,
                    },
                )
                # back-compat shim: legacy JSON files in cache root for older tests
                legacy_file = cache_dir / f"chunk_{i:03d}.json"
                legacy_file.write_text(
                    json.dumps(
                        {
                            "normalized_text": chunk.text,
                            "emotion": segs[0]["emotion"] if segs else "neutral",
                            "detected_emotions": [s["emotion"] for s in segs]
                            or ["neutral"],
                            "language": language,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )

            for j, seg in enumerate(segs):
                analyzed.append(
                    {
                        "text": seg["text"],
                        "emotion": seg["emotion"],
                        "ends_paragraph": (
                            ends_paragraph if j == len(segs) - 1 else False
                        ),
                        "source_chunk": i,
                        "source_file": chunk.source_file,
                        "language": language,
                    }
                )

        # --- Step B: re-chunk over-budget segments ----------------------------
        from .chunker import split_to_chunks

        rechunked = []
        max_words = config.MAX_WORDS_PER_CHUNK
        for seg in analyzed:
            words = seg["text"].split()
            if len(words) <= max_words:
                rechunked.append(seg)
                continue
            sub = split_to_chunks(
                seg["text"], max_words, source_file=seg.get("source_file", "default")
            )
            for k, s in enumerate(sub):
                rechunked.append(
                    {
                        "text": s.text,
                        "emotion": seg["emotion"],
                        "ends_paragraph": (
                            seg["ends_paragraph"] if k == len(sub) - 1 else False
                        ),
                        "source_chunk": seg["source_chunk"],
                        "source_file": seg.get("source_file", "default"),
                        "language": seg.get("language", self.DEFAULT_LANGUAGE),
                    }
                )

        report(
            f"After A+B: {len(analyzed)} emotion segments -> {len(rechunked)} TTS-sized segments"
        )

        # --- Steps C/D skipped: use the verbatim split text as segment ------
        # The 1B model copies few-shot examples instead of processing the
        # input, so normalize + context-fill produce garbage at this scale.
        # We pass the original split text straight through to TTS.
        final = []
        current_source = None
        chapter_title_added = False
        current_chapter_number = 1
        current_chapter_title = "Chapter 1"
        story_title = None
        story_title_detected = False

        for i, seg in enumerate(rechunked):
            record = storage.load_segment_meta(i)
            if record is not None:
                final.append(record)
                current_source = seg.get("source_file", "default")
                continue

            text = seg["text"]
            source_file = seg.get("source_file", "default")

            # Detect story title once on the first segment
            if not story_title_detected:
                story_title = self.detect_story_title(seg["text"]) or "Audiobook"
                story_title_detected = True
                if story_title != "Audiobook":
                    report(f"  detected story title: '{story_title}'")

            # Detect chapter info when source file changes
            if source_file != current_source:
                chapter_info = self.detect_chapter_info(seg["text"])
                detected_num = chapter_info.get("chapter_number")
                detected_title = chapter_info.get("chapter_title")
                if detected_num is not None:
                    current_chapter_number = detected_num
                    report(f"  detected chapter number: {current_chapter_number}")
                if detected_title:
                    current_chapter_title = detected_title
                    report(f"  detected chapter title: '{current_chapter_title}'")
                else:
                    current_chapter_title = f"Chapter {current_chapter_number}"

            # Add chapter title to the first segment of each new chapter if enabled
            if config.VOCALIZE_CHAPTER_NAMES and source_file != current_source:
                chapter_title = self.format_chapter_title(source_file)
                text = f"{chapter_title}. {text}"
                report(f"  added chapter title: '{chapter_title}' to segment {i + 1}")
                chapter_title_added = True
            else:
                chapter_title_added = False

            current_source = source_file

            # Detect speaking speed for the segment
            try:
                speaking_speed = self.detect_speaking_speed(seg["text"])
            except Exception as e:
                logger.debug(f"Speaking speed detection failed for segment {i}: {e}")
                speaking_speed = self.DEFAULT_SPEAKING_SPEED

            record = {
                "normalized_text": text,
                "emotion": seg["emotion"],
                "detected_emotions": [seg["emotion"]],
                "language": seg.get("language", self.DEFAULT_LANGUAGE),
                "speaking_speed": speaking_speed,
                "ends_paragraph": seg["ends_paragraph"],
                "source_chunk": seg["source_chunk"],
                "source_file": source_file,
                "chapter_title_added": chapter_title_added,
                "chunk_number": i,
                "chapter_number": current_chapter_number,
                "chapter_title": current_chapter_title,
                "story_title": story_title,
            }
            storage.save_segment_meta(i, record)
            final.append(record)

        report(
            f"Stage 1 complete: {len(chunks)} input chunks -> {len(final)} TTS segments"
        )
        return final
