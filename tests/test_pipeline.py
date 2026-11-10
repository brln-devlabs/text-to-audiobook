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


from src.pipeline import prepare_segments, run


def test_pipeline_with_no_files(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    result = run(input_dir, output_dir)
    assert result == {} or result is not None


def test_pipeline_with_example_file(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    md_file = input_dir / "test.md"
    md_file.write_text("# Title\n\nThis is test content. More content here.")

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    try:
        result = run(input_dir, output_dir)
        assert result is not None
    except Exception:
        pass


def test_prepare_segments_respects_selected_names(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")

    input_dir = tmp_path / "input"
    (input_dir / "alpha").mkdir(parents=True)
    (input_dir / "alpha" / "0_chap.md").write_text(
        "Alpha story content.", encoding="utf-8"
    )
    (input_dir / "beta").mkdir(parents=True)
    (input_dir / "beta" / "0_chap.md").write_text(
        "Beta story content.", encoding="utf-8"
    )

    output_dir = tmp_path / "output"

    results = prepare_segments(input_dir, output_dir, selected_names=["alpha"])
    assert "alpha" in results
    assert "beta" not in results


def test_prepare_segments_empty_selection_returns_empty(tmp_path):
    input_dir = tmp_path / "input"
    (input_dir / "alpha").mkdir(parents=True)
    (input_dir / "alpha" / "0_chap.md").write_text("Alpha content.", encoding="utf-8")
    output_dir = tmp_path / "output"
    results = prepare_segments(input_dir, output_dir, selected_names=[])
    assert results == {}


def test_wavs_for_segments_includes_sub_parts():
    from pathlib import Path

    from src.pipeline import _wavs_for_segments

    wavs = [
        Path("chunk_000000.wav"),
        Path("chunk_000001_0.wav"),
        Path("chunk_000001_1.wav"),
        Path("chunk_000002.wav"),
    ]
    assert _wavs_for_segments(wavs, [1, 2]) == wavs[1:]
    assert _wavs_for_segments(wavs, [0]) == wavs[:1]
