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

import json
import sys as _sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
CACHE_DIR = ROOT / "cache"
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"

for d in [INPUT_DIR, OUTPUT_DIR, CACHE_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True)

MAX_WORDS_PER_CHUNK = 499
VOICE_SEED = 42
SAMPLE_RATE = 24000
CROSSFADE_MS = 80
SILENCE_AFTER_PERIOD_MS = 500
SILENCE_AFTER_PARAGRAPH_MS = 1200
TTS_MAX_TOKENS = 100
TTS_MAX_WORDS_FALLBACK = 100
CHAPTER_BY_CHAPTER = False
VOCALIZE_CHAPTER_NAMES = False

# Language passed to Qwen3-TTS for every segment. "auto" restores the upstream
# behaviour (per-segment language from the LLM's detection). A fixed language
# avoids mispronunciation when the small LLM mislabels a chunk, and skips the
# 10 LLM language probes in Stage 1.
TTS_LANGUAGE = "german"
# Precision for the TTS model on GPU: "bfloat16" (recommended, falls back to
# float16 on GPUs without bf16 support), "float16" or "float32". float16 can
# produce NaNs in Qwen3-TTS sampling ("device-side assert triggered").
TTS_DTYPE = "bfloat16"
# Layers of the Llama helper model offloaded to the GPU: 0 = CPU only (keeps
# VRAM free for TTS on small GPUs), -1 = all layers on the GPU.
LLM_GPU_LAYERS = 0

LOG_LEVEL_CONSOLE = "INFO"
LOG_LEVEL_FILE = "DEBUG"

LLM_MAX_RETRIES = 3
STORAGE_BACKEND = "json"

NARRATOR_PROMPT = (
    "A calm, clear and warm narrator voice. "
    "Speaks at a measured, unhurried pace with natural intonation, "
    "like an experienced audiobook reader presenting a non-fiction text."
)

LLM_MODEL_NAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
LLM_MODEL_PATH = MODELS_DIR / LLM_MODEL_NAME
# 0.6B fits GPUs with 4 GB VRAM; use "Qwen/Qwen3-TTS-12Hz-1.7B-Base" with ~6 GB+.
TTS_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
TTS_VOICE_DESIGN_MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
VOICE_REF_PATH = CACHE_DIR / "voice_reference.wav"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE = LOGS_DIR / "pipeline.log"
LOG_MAX_BYTES = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 5

SETTINGS_FILE = ROOT / "settings.json"

_SETTINGS_KEYS = {
    "MAX_WORDS_PER_CHUNK": int,
    "VOICE_SEED": int,
    "SAMPLE_RATE": int,
    "CROSSFADE_MS": int,
    "SILENCE_AFTER_PERIOD_MS": int,
    "SILENCE_AFTER_PARAGRAPH_MS": int,
    "LLM_MAX_RETRIES": int,
    "LOG_LEVEL_CONSOLE": str,
    "LOG_LEVEL_FILE": str,
    "NARRATOR_PROMPT": str,
    "STORAGE_BACKEND": str,
    "TTS_MAX_TOKENS": int,
    "TTS_MAX_WORDS_FALLBACK": int,
    "CHAPTER_BY_CHAPTER": bool,
    "VOCALIZE_CHAPTER_NAMES": bool,
    "TTS_LANGUAGE": str,
    "TTS_DTYPE": str,
    "LLM_GPU_LAYERS": int,
    "TTS_MODEL_NAME": str,
}

# Snapshot the values defined above as the single source of truth for
# "code defaults". load_settings_from_file may overwrite the module
# attributes, but this dict stays pinned to the in-code defaults so
# the GUI's "Reset to Defaults" button has something stable to read.
_DEFAULTS: dict = {k: getattr(_sys.modules[__name__], k) for k in _SETTINGS_KEYS}


def get_default(key: str):
    """Return the in-code default for a settings key (None if unknown)."""
    return _DEFAULTS.get(key)


def get_all_defaults() -> dict:
    """Return a copy of all in-code defaults for settings keys."""
    return dict(_DEFAULTS)


def reset_to_defaults(path: Path | None = None) -> dict:
    """Revert module attrs to in-code defaults and rewrite settings file."""
    mod = _sys.modules[__name__]
    for key, val in _DEFAULTS.items():
        setattr(mod, key, val)
    path = path or SETTINGS_FILE
    try:
        path.write_text(json.dumps(_DEFAULTS, indent=2), encoding="utf-8")
    except Exception:
        pass
    return dict(_DEFAULTS)


def load_settings_from_file(path: Path | None = None) -> dict:
    """Load persisted settings and apply them to this module's attributes."""
    import sys

    mod = sys.modules[__name__]
    path = path or SETTINGS_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    applied = {}
    for key, cast in _SETTINGS_KEYS.items():
        if key in data:
            try:
                val = cast(data[key])
                setattr(mod, key, val)
                applied[key] = val
            except (TypeError, ValueError):
                pass
    return applied


def save_settings_to_file(settings: dict, path: Path | None = None) -> None:
    path = path or SETTINGS_FILE
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing.update(settings)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
