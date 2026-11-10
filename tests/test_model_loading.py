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

"""Model-loading regression tests.

Today's bug: AutoModel.from_pretrained("...qwen3_tts...") fails because
transformers doesn't ship the architecture. The fix routes loading
through `qwen_tts.Qwen3TTSModel.from_pretrained` which registers the
model class with AutoConfig/AutoModel/AutoProcessor before delegating
to transformers. These tests pin that contract in place.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import config
from src.tts_engine import TTSEngine


def _fake_wrapper():
    wrapper = MagicMock()
    wrapper.model = MagicMock(name="Qwen3TTSForConditionalGeneration")
    wrapper.processor = MagicMock(name="Qwen3TTSProcessor")
    wrapper.generate = MagicMock(
        return_value=(
            [np.zeros(config.SAMPLE_RATE * 1, dtype=np.float32)],
            config.SAMPLE_RATE,
        )
    )
    return wrapper


def test_tts_load_uses_qwen_wrapper():
    """Regression test: TTSEngine.load() must go through Qwen3TTSModel.from_pretrained."""
    engine = TTSEngine()
    with patch("src.tts_engine.Qwen3TTSModel") as mock_cls:
        mock_cls.from_pretrained.return_value = _fake_wrapper()
        with patch("src.tts_engine.Path") as mock_path:
            mock_voice_ref = MagicMock()
            mock_voice_ref.exists.return_value = True
            mock_path.return_value = mock_voice_ref
            engine.load()

            mock_cls.from_pretrained.assert_called_once()
            args, kwargs = mock_cls.from_pretrained.call_args
            assert args[0] == config.TTS_MODEL_NAME
            assert kwargs.get("trust_remote_code") is True


def test_tts_load_propagates_registration_error():
    """If qwen_tts ever stops registering qwen3_tts, the failure must surface immediately."""
    engine = TTSEngine()
    with patch("src.tts_engine.Qwen3TTSModel") as mock_cls:
        mock_cls.from_pretrained.side_effect = ValueError(
            "The checkpoint you are trying to load has model type `qwen3_tts` "
            "but Transformers does not recognize this architecture."
        )
        with pytest.raises(ValueError, match="qwen3_tts"):
            engine.load()


def test_tts_load_mock_mode_when_qwen_missing():
    """When the qwen_tts package isn't importable, load() must degrade to mock mode (no crash)."""
    engine = TTSEngine()
    with patch("src.tts_engine.Qwen3TTSModel", None):
        engine.load()
        assert engine.engine is None
        assert engine.model is None
        assert engine.processor is None


def test_tts_engine_exposes_back_compat_properties():
    """Pipeline + tests still read engine.model / engine.processor."""
    engine = TTSEngine()
    assert engine.model is None
    assert engine.processor is None
    with patch("src.tts_engine.Qwen3TTSModel") as mock_cls:
        wrapper = _fake_wrapper()
        mock_cls.from_pretrained.return_value = wrapper
        with patch("src.tts_engine.Path") as mock_path:
            mock_voice_ref = MagicMock()
            mock_voice_ref.exists.return_value = True
            mock_path.return_value = mock_voice_ref
            engine.load()
            assert engine.model is wrapper.model
            assert engine.processor is wrapper.processor


def test_llm_load_instantiates_llama_with_configured_path():
    from src.llm_normalizer import LLMNormalizer

    captured = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    with patch("src.llm_normalizer.llama_cpp") as mock_llama_cpp, patch(
        "src.llm_normalizer.hf_hub_download"
    ) as mock_dl:
        mock_llama_cpp.Llama = FakeLlama
        mock_dl.return_value = "/tmp/model.gguf"
        norm = LLMNormalizer(config.LLM_MODEL_PATH)
        norm.load()

    assert "model_path" in captured
    assert captured["model_path"]


def test_llm_load_mock_mode_when_llama_missing():
    from src.llm_normalizer import LLMNormalizer

    with patch("src.llm_normalizer.llama_cpp", None):
        norm = LLMNormalizer(config.LLM_MODEL_PATH)
        norm.load()
        assert norm.llm is None


def test_config_has_expected_constants():
    assert config.TTS_MODEL_NAME == "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    assert config.SAMPLE_RATE == 24000
    assert config.LLM_MODEL_PATH.parent.exists()
    assert config.NARRATOR_PROMPT.strip() != ""


@pytest.mark.live
def test_tts_loads_real_qwen_model():
    """Skipped by default — run with `pytest -m live` to exercise a real model load."""
    import importlib
    import sys

    for mod in ("torch", "transformers", "qwen_tts"):
        sys.modules.pop(mod, None)
    qwen_tts = importlib.import_module("qwen_tts")
    wrapper = qwen_tts.Qwen3TTSModel.from_pretrained(
        config.TTS_MODEL_NAME,
        trust_remote_code=True,
    )
    assert wrapper.model is not None
    assert wrapper.processor is not None


def test_tts_model_label_round_trip():
    for label, model_id in config.TTS_MODEL_CHOICES.items():
        assert config.tts_model_label(model_id) == label
        assert config.tts_model_id(label) == model_id
    # Unknown/custom models pass through unchanged in both directions.
    assert config.tts_model_label("my/custom-model") == "my/custom-model"
    assert config.tts_model_id("my/custom-model") == "my/custom-model"


def test_default_model_is_a_gui_choice():
    assert config.TTS_MODEL_NAME in config.TTS_MODEL_CHOICES.values()


def test_tts_engine_reads_model_at_runtime(monkeypatch):
    """A model chosen in the GUI after import must be used by new engines."""
    from src.tts_engine import TTSEngine

    monkeypatch.setattr(config, "TTS_MODEL_NAME", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    assert TTSEngine().model_name == "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    assert TTSEngine("explicit/model").model_name == "explicit/model"
