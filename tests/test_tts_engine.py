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

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src import config
from src.tts_engine import TTSEngine


@pytest.fixture
def fake_wrapper_cls():
    """Patch Qwen3TTSModel so load() returns a wrapper that synthesizes silence."""
    with patch("src.tts_engine.Qwen3TTSModel") as mock_cls:
        wrapper = MagicMock()
        wrapper.model = MagicMock()
        wrapper.processor = MagicMock()
        wrapper.generate.return_value = (
            [np.zeros(config.SAMPLE_RATE * 1, dtype=np.float32)],
            config.SAMPLE_RATE,
        )
        wrapper.generate_voice_clone.return_value = (
            [np.zeros(config.SAMPLE_RATE * 1, dtype=np.float32)],
            config.SAMPLE_RATE,
        )
        mock_cls.from_pretrained.return_value = wrapper
        yield mock_cls, wrapper


def test_load_calls_qwen_wrapper(fake_wrapper_cls):
    mock_cls, _ = fake_wrapper_cls
    engine = TTSEngine()
    with patch("src.tts_engine.Path") as mock_path:
        mock_voice_ref = MagicMock()
        mock_voice_ref.exists.return_value = True
        mock_path.return_value = mock_voice_ref
        engine.load()
    mock_cls.from_pretrained.assert_called_once()
    args, _ = mock_cls.from_pretrained.call_args
    assert args[0] == config.TTS_MODEL_NAME


def test_synthesize_first_writes_wav(fake_wrapper_cls, tmp_path):
    engine = TTSEngine()
    engine.load()
    out_path = tmp_path / "chunk_000000.wav"
    result = engine.synthesize_first("Hello world.", "narrator prompt", out_path)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert isinstance(result[0], Path)
    assert engine.anchor_path == result[0]
    _, wrapper = fake_wrapper_cls
    wrapper.generate_voice_clone.assert_called_once()


def test_synthesize_with_anchor_writes_wav(fake_wrapper_cls, tmp_path):
    engine = TTSEngine()
    engine.load()
    anchor = tmp_path / "anchor.wav"
    anchor.write_bytes(b"placeholder")
    out_path = tmp_path / "chunk_000001.wav"
    result = engine.synthesize_with_anchor("Second chunk.", anchor, out_path)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert isinstance(result[0], Path)


def test_process_all_writes_n_chunks(fake_wrapper_cls, tmp_path):
    engine = TTSEngine()
    engine.load()
    payloads = [
        {"normalized_text": "First."},
        {"normalized_text": "Second."},
        {"normalized_text": "Third."},
    ]
    out_dir = tmp_path / "out"
    wav_paths = engine.process_all(payloads, out_dir)
    # process_all now returns flat list of all paths (including multi-part splits)
    assert len(wav_paths) >= 3
    assert engine.anchor_path == wav_paths[0]
    _, wrapper = fake_wrapper_cls
    # All chunks use generate_voice_clone
    assert wrapper.generate_voice_clone.call_count == 3


def test_process_all_resumes_skipping_existing(fake_wrapper_cls, tmp_path):
    engine = TTSEngine()
    engine.load()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Pre-create chunk_000000/000001.wav so they should be skipped.
    (out_dir / "chunk_000000.wav").write_bytes(b"existing-0")
    (out_dir / "chunk_000001.wav").write_bytes(b"existing-1")
    payloads = [
        {"normalized_text": "First."},
        {"normalized_text": "Second."},
        {"normalized_text": "Third."},
    ]
    wav_paths = engine.process_all(payloads, out_dir)
    assert len(wav_paths) >= 3
    _, wrapper = fake_wrapper_cls
    # Only chunk_002 should have been synthesized (uses generate_voice_clone)
    assert wrapper.generate_voice_clone.call_count == 1


def test_unload_releases_engine(fake_wrapper_cls):
    engine = TTSEngine()
    engine.load()
    assert engine.model is not None
    engine.unload()
    assert engine.engine is None
    assert engine.model is None
    assert engine.processor is None


def test_stop_event_breaks_loop(fake_wrapper_cls, tmp_path):
    engine = TTSEngine()
    engine.load()
    stop = threading.Event()
    _, wrapper = fake_wrapper_cls

    call_count = {"n": 0}

    def fake_generate(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            stop.set()  # Trigger stop after the first synthesis returns.
        return ([np.zeros(config.SAMPLE_RATE, dtype=np.float32)], config.SAMPLE_RATE)

    wrapper.generate_voice_clone.side_effect = fake_generate

    payloads = [{"normalized_text": f"chunk {i}"} for i in range(5)]
    out_dir = tmp_path / "out"
    wav_paths = engine.process_all(payloads, out_dir, stop_event=stop)
    assert len(wav_paths) >= 1
    assert call_count["n"] == 1


def test_synthesize_mock_mode_writes_silence(tmp_path):
    """If load() never ran (or qwen_tts missing), synthesize must not crash."""
    engine = TTSEngine()
    out_path = tmp_path / "silence.wav"
    result = engine.synthesize_first("text", "prompt", out_path)
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0].exists()


def test_process_all_passes_per_segment_language(
    fake_wrapper_cls, tmp_path, monkeypatch
):
    """With TTS_LANGUAGE="auto", the segment payload language must reach the
    voice generation calls."""
    monkeypatch.setattr(config, "TTS_LANGUAGE", "auto")
    engine = TTSEngine()
    engine.load()
    payloads = [
        {"normalized_text": "Hello.", "language": "english"},
        {"normalized_text": "Hola.", "language": "spanish"},
        {"normalized_text": "Bonjour.", "language": "french"},
    ]
    out_dir = tmp_path / "out"
    engine.process_all(payloads, out_dir)
    _, wrapper = fake_wrapper_cls
    # All chunks use generate_voice_clone
    clone_calls = wrapper.generate_voice_clone.call_args_list
    assert len(clone_calls) == 3
    assert clone_calls[0].kwargs["language"] == "english"
    assert clone_calls[1].kwargs["language"] == "spanish"
    assert clone_calls[2].kwargs["language"] == "french"


def test_process_all_defaults_language_when_missing(
    fake_wrapper_cls, tmp_path, monkeypatch
):
    monkeypatch.setattr(config, "TTS_LANGUAGE", "auto")
    engine = TTSEngine()
    engine.load()
    payloads = [{"normalized_text": "No lang field."}]
    engine.process_all(payloads, tmp_path / "out")
    _, wrapper = fake_wrapper_cls
    call = wrapper.generate_voice_clone.call_args_list[0]
    assert call.kwargs["language"] == TTSEngine.DEFAULT_LANGUAGE


def test_process_all_uses_fixed_language(fake_wrapper_cls, tmp_path, monkeypatch):
    """A fixed TTS_LANGUAGE overrides whatever the LLM detected per segment."""
    monkeypatch.setattr(config, "TTS_LANGUAGE", "german")
    engine = TTSEngine()
    engine.load()
    payloads = [
        {"normalized_text": "Erster Satz.", "language": "english"},
        {"normalized_text": "Zweiter Satz.", "language": "japanese"},
        {"normalized_text": "Dritter Satz."},
    ]
    engine.process_all(payloads, tmp_path / "out")
    _, wrapper = fake_wrapper_cls
    languages = [c.kwargs["language"] for c in wrapper.generate_voice_clone.call_args_list]
    assert languages == ["german", "german", "german"]
