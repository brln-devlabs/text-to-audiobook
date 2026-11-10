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
import logging
import re
from pathlib import Path

from natsort import natsorted

from . import config
from .audio_assembler import assemble
from .audio_postprocess import denoise
from .chunker import split_to_chunks
from .llm_normalizer import LLMNormalizer
from .markdown_cleaner import clean
from .tts_engine import TTSEngine

logger = logging.getLogger(__name__)


def _emit(progress_cb, msg):
    logger.info(msg)
    if progress_cb:
        progress_cb(msg)


def _make_progress(progress_cb):
    """Create a progress reporter bound to the given callback."""

    def progress(msg):
        _emit(progress_cb, msg)

    return progress


_NUMERIC_PREFIX_RE = re.compile(r"^\d")


def _all_have_numeric_prefix(names: list) -> bool:
    return bool(names) and all(_NUMERIC_PREFIX_RE.match(n) for n in names)


def _order_files(
    input_dir: Path, llm: LLMNormalizer, cache_dir: Path, progress_cb=None
):
    """Determine narrative reading order for .md files.

    Strategy:
    - If every filename starts with a digit (e.g. '0_Prologue.md', '10_Chapter_10.md'),
      use natsort directly — the 1B LLM cannot handle numeric collation reliably.
    - Otherwise send content previews to the LLM so it can recognise ordinal words
      ('First', 'Prologue', 'Chapter_One') and narrative cues.

    Result is cached at <cache_dir>/file_order.json.
    """
    progress = _make_progress(progress_cb)

    md_paths = list(input_dir.glob("**/*.md"))
    if not md_paths:
        return []

    name_to_path = {p.name: p for p in md_paths}
    cache_file = cache_dir / "file_order.json"
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            order = [
                name_to_path[n] for n in cached.get("order", []) if n in name_to_path
            ]
            extras = [p for p in md_paths if p not in order]
            if not extras and order:
                progress(f"Using cached file order ({len(order)} files)")
                return order
        except Exception as e:
            logger.warning(f"Could not read cached file order: {e}")

    if len(md_paths) == 1:
        return md_paths

    names = [p.name for p in md_paths]

    if _all_have_numeric_prefix(names):
        # Numeric-prefixed filenames already encode order; natsort handles 1, 2, ..., 10, 11
        ordered_names = natsorted(names)
        progress(
            f"Numeric-prefixed files: using natural sort ({len(ordered_names)} files)"
        )
    else:
        # Pre-sort with natsort so LLM sees a sensible baseline, then let it
        # reorder by narrative cues (ordinal words, prologue/epilogue keywords, content)
        presorted_paths = natsorted(md_paths, key=lambda p: p.name)
        previews = []
        for p in presorted_paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    head = f.read(400)
            except Exception:
                head = ""
            previews.append({"name": p.name, "preview": clean(head)[:200]})

        progress(f"Asking LLM to order {len(md_paths)} files by narrative sequence...")
        ordered_names = llm.sort_files(previews)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"order": ordered_names}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return [name_to_path[n] for n in ordered_names if n in name_to_path]


def _build_chunks(md_files, progress_cb=None):
    progress = _make_progress(progress_cb)
    progress(f"Found {len(md_files)} file(s)")
    progress("Cleaning and chunking text...")

    chunks = []
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = clean(raw)
        source_name = md_file.stem
        file_chunks = split_to_chunks(
            cleaned, config.MAX_WORDS_PER_CHUNK, source_file=source_name
        )
        chunks.extend(file_chunks)
        progress(f"  {source_name}: {len(file_chunks)} chunk(s)")

    progress(f"Total: {len(chunks)} chunks from {len(md_files)} file(s)")
    return chunks


def discover_projects(input_dir):
    """Each subfolder of input_dir is one project; each loose .md file is its own project.

    Returns: list of {name, files} sorted alphabetically by name.
    """
    input_dir = Path(input_dir)
    projects = []
    if not input_dir.exists():
        return projects

    for sub in sorted(input_dir.iterdir()):
        if sub.is_dir():
            files = list(sub.glob("**/*.md"))
            if files:
                projects.append({"name": sub.name, "files": files, "input_dir": sub})

    for f in sorted(input_dir.glob("*.md")):
        projects.append({"name": f.stem, "files": [f], "input_dir": input_dir})

    return projects


def load_cached_segments(cache_dir: Path) -> list:
    """Load all valid TTS-ready segments from a completed (or partial) Stage 1 cache.

    Returns empty list if no cache exists or segments are malformed.
    """
    from .storage_factory import create_storage

    cache_dir = Path(cache_dir)
    try:
        storage = create_storage(cache_dir)
        storage.init_schema()
        segments = storage.list_all_segments()
        # Filter to only valid segments with normalized_text
        result = [
            s for s in segments if isinstance(s, dict) and s.get("normalized_text")
        ]
        return result
    except Exception as e:
        logger.warning(f"Could not load cached segments: {e}")
        return []


def prepare_segments_for_project(
    project, output_root, progress_cb=None, llm=None, stop_event=None
):
    """Stage 1 for a single project. Returns segment list.

    `llm` may be passed in pre-loaded so multiple projects share one LLM load.
    """
    progress = _make_progress(progress_cb)
    name = project["name"]
    files = project["files"]
    output_root = Path(output_root)
    cache_dir = config.CACHE_DIR / name

    progress(f"--- Project '{name}' ({len(files)} file(s)) ---")

    own_llm = llm is None
    if own_llm:
        progress("Loading LLM...")
        llm = LLMNormalizer(config.LLM_MODEL_PATH)
        llm.load()

    try:
        if stop_event and stop_event.is_set():
            return []

        if len(files) > 1:
            ordered_files = _order_files(
                project["input_dir"], llm, cache_dir, progress_cb
            )
        else:
            ordered_files = files
        progress(f"[{name}] file order: " + ", ".join(p.name for p in ordered_files))

        chunks = _build_chunks(ordered_files, progress_cb)
        if not chunks:
            return []

        progress(f"[{name}] emotion analysis...")
        segments = llm.process_all(
            chunks, cache_dir, progress_cb=progress_cb, stop_event=stop_event
        )
        progress(f"[{name}] {len(segments)} TTS segments ready")
        return segments
    finally:
        if own_llm:
            llm.unload()


def prepare_segments(
    input_dir, output_dir, progress_cb=None, stop_event=None, selected_names=None
):
    """Stage 1 for projects under input_dir. Returns dict {project_name: segments}.

    `selected_names` (optional iterable): when given, restrict processing to
    projects whose name is in the set. None = process every discovered project.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    progress = _make_progress(progress_cb)

    progress("Discovering projects in input directory...")
    projects = discover_projects(input_dir)
    if not projects:
        progress("No projects found (no folders with .md files and no loose .md files)")
        return {}
    if selected_names is not None:
        wanted = set(selected_names)
        projects = [p for p in projects if p["name"] in wanted]
        if not projects:
            progress("No selected projects to process")
            return {}
    progress(
        f"Found {len(projects)} project(s): " + ", ".join(p["name"] for p in projects)
    )

    progress("Loading LLM...")
    llm = LLMNormalizer(config.LLM_MODEL_PATH)
    llm.load()

    results = {}
    try:
        for project in projects:
            if stop_event and stop_event.is_set():
                progress("Stopped by user")
                break
            segments = prepare_segments_for_project(
                project, output_dir, progress_cb, llm=llm, stop_event=stop_event
            )
            results[project["name"]] = segments
    finally:
        llm.unload()

    progress(f"Stage 1 complete for {len(results)}/{len(projects)} project(s)")
    return results


_CHUNK_WAV_RE = re.compile(r"^chunk_(\d+)(?:_(\d+))?$")


def _wavs_for_segments(wav_paths: list, seg_indices: list) -> list:
    """All WAVs (including sub-parts chunk_NNN_K.wav) of the given segments.

    Segment i is synthesized as chunk_{i:06d}.wav or as several sub-parts
    chunk_{i:06d}_0.wav, _1.wav, ..., so WAV list positions do not match
    segment indices once any segment was split. WAVs with other names fall
    back to positional matching.
    """
    by_chunk: dict = {}
    positional = []
    for wav in wav_paths:
        m = _CHUNK_WAV_RE.match(Path(wav).stem)
        if m:
            by_chunk.setdefault(int(m.group(1)), []).append(wav)
        else:
            positional.append(wav)
    if not by_chunk:
        return [positional[i] for i in seg_indices if i < len(positional)]
    return [w for i in seg_indices for w in by_chunk.get(i, [])]


def _assemble_chapters(
    wav_paths: list,
    segments: list,
    final_path: Path,
    project_name: str,
    progress_cb=None,
) -> list:
    """Assemble WAV files into per-chapter MP3s based on source_file metadata.

    Returns list of output file paths.
    """
    progress = _make_progress(progress_cb)
    final_path = Path(final_path)
    final_path.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict

    by_source = defaultdict(list)
    for i, seg in enumerate(segments):
        source = seg.get("source_file", "default")
        by_source[source].append((i, seg))

    output_files = []

    if len(by_source) == 1:
        source_name = list(by_source.keys())[0]
        seg_indices = [i for i, _ in by_source[source_name]]
        chapter_wavs = _wavs_for_segments(wav_paths, seg_indices)
        if chapter_wavs:
            assembled = assemble(
                chapter_wavs,
                [seg for _, seg in by_source[source_name]],
                final_path / "audiobook.mp3",
            )
            progress(f"[{project_name}] done -> {assembled}")
            output_files.append(assembled)
    else:
        sorted_sources = sorted(by_source.keys())
        progress(f"[{project_name}] {len(sorted_sources)} chapter(s) detected")
        for chapter_num, source_name in enumerate(sorted_sources, start=1):
            seg_indices = [i for i, _ in by_source[source_name]]
            chapter_wavs = _wavs_for_segments(wav_paths, seg_indices)
            if not chapter_wavs:
                progress(f"  chapter {chapter_num} ({source_name}): no WAVs found")
                continue
            out_file = final_path / f"chapter_{chapter_num:02d}.mp3"
            assembled = assemble(
                chapter_wavs, [seg for _, seg in by_source[source_name]], out_file
            )
            progress(
                f"  chapter {chapter_num} ({source_name}): {len(chapter_wavs)} chunks -> {assembled}"
            )
            output_files.append(assembled)

    return output_files


def generate_audio_for_project(
    project_name,
    segments,
    output_root,
    progress_cb=None,
    tts=None,
    stop_event=None,
    narrator_prompt=None,
    chunk_done_cb=None,
):
    """Stage 2 for a single project. Returns path to assembled audiobook."""
    progress = _make_progress(progress_cb)
    output_root = Path(output_root)
    project_out = output_root / project_name

    if not segments:
        progress(f"[{project_name}] no segments, skipping")
        return None

    progress(
        f"--- Audio gen for project '{project_name}' ({len(segments)} segments) ---"
    )

    own_tts = tts is None
    if own_tts:
        progress("Loading TTS engine...")
        tts = TTSEngine(config.TTS_MODEL_NAME)
        tts.load(narrator_prompt=narrator_prompt)

    try:
        chunk_output_dir = project_out / "chunks"
        wav_paths = tts.process_all(
            segments,
            chunk_output_dir,
            stop_event=stop_event,
            chunk_done_cb=chunk_done_cb,
        )

        if stop_event and stop_event.is_set():
            progress(
                f"[{project_name}] stopped — {len(wav_paths)} chunk(s) generated so far"
            )
            return None

        progress(f"[{project_name}] denoising {len(wav_paths)} chunks...")
        for wav_path in wav_paths:
            if stop_event and stop_event.is_set():
                break
            denoise(wav_path)

        if stop_event and stop_event.is_set():
            progress(f"[{project_name}] stopped during denoising")
            return None

        progress(f"[{project_name}] assembling final audio...")
        final_path = project_out / "final"
        final_path.mkdir(parents=True, exist_ok=True)

        if config.CHAPTER_BY_CHAPTER:
            output_files = _assemble_chapters(
                wav_paths, segments, final_path, project_name, progress_cb
            )
            if output_files:
                progress(f"[{project_name}] done -> chapter files generated")
                return output_files[0]
        else:
            assembled = assemble(wav_paths, segments, final_path / "audiobook.mp3")
            progress(f"[{project_name}] done -> {assembled}")
            return assembled
        return None
    finally:
        if own_tts:
            tts.unload()


def generate_audio(
    segments_by_project,
    output_dir,
    progress_cb=None,
    stop_event=None,
    narrator_prompt=None,
    chunk_done_cb=None,
):
    """Stage 2 for all projects.

    Accepts either {project_name: segments} (preferred) or a flat list (legacy single project).
    """
    output_dir = Path(output_dir)
    progress = _make_progress(progress_cb)

    if isinstance(segments_by_project, list):
        return generate_audio_for_project(
            output_dir.name or "project",
            segments_by_project,
            output_dir.parent or output_dir,
            progress_cb,
            stop_event=stop_event,
            narrator_prompt=narrator_prompt,
            chunk_done_cb=chunk_done_cb,
        )

    if not segments_by_project:
        progress("No segments to synthesize")
        return {}

    progress("Loading TTS engine...")
    tts = TTSEngine(config.TTS_MODEL_NAME)
    tts.load()

    results = {}
    try:
        for name, segments in segments_by_project.items():
            if stop_event and stop_event.is_set():
                progress("Stopped by user")
                break
            results[name] = generate_audio_for_project(
                name,
                segments,
                output_dir,
                progress_cb,
                tts=tts,
                stop_event=stop_event,
                narrator_prompt=narrator_prompt,
                chunk_done_cb=chunk_done_cb,
            )
    finally:
        tts.unload()

    progress(f"Stage 2 complete for {len(results)} project(s)")
    return results


def _mfcc_fingerprint(samples, sample_rate):
    """Return a fixed-length voice fingerprint vector (MFCC mean) for `samples`.

    Returns None if librosa unavailable or audio too short.
    """
    try:
        import librosa
    except ImportError:
        return None
    samples = samples.astype("float32")
    if samples.size < sample_rate // 2:  # need at least 0.5 s
        return None
    if abs(samples).max() > 1.0:
        samples = samples / 32768.0
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=20)
    return mfcc.mean(axis=1)


def _cosine(a, b):
    import numpy as np

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-10
    return float(np.dot(a, b) / denom)


def cleanup_noisy_chunks(
    output_dir,
    project_names=None,
    progress_cb=None,
    max_noise_ms=1500,
    voice_ref_path=None,
    min_speech_ratio=0.40,
    min_avg_rms=300.0,
    max_clip_ratio=0.02,
    min_voice_similarity=0.55,
):
    """Audit chunk WAVs for TTS failures and either clean or delete them.

    Three checks per chunk:
      1. Discriminator (speech ratio): fraction of frames classified as voiced
         speech must be >= min_speech_ratio.
      2. Volume level: mean RMS of voiced frames in [min_avg_rms, soft cap];
         clipped-sample fraction <= max_clip_ratio.
      3. Voice match: cosine similarity of MFCC fingerprint vs the voice
         reference WAV must be >= min_voice_similarity.

    Chunks that fail any check are DELETED so Stage 2 can regenerate them.
    Chunks that pass but contain long noisy regions are cleaned in-place
    (existing flatness-based excision).

    Returns: {project_name: {"edited": int, "deleted": [(filename, reason), ...]}}
    """
    try:
        import numpy as np
        from scipy.io import wavfile as _wavfile
    except ImportError:
        _emit(progress_cb, "numpy/scipy not available — cannot run cleanup")
        return {}

    output_dir = Path(output_dir)
    progress = _make_progress(progress_cb)
    results = {}

    FRAME_MS = 30  # analysis window length in ms
    FLATNESS_THRESH = (
        0.75  # above this fraction → noise  (speech ≈ 0.1–0.5, white noise ≈ 1.0)
    )
    MIN_RMS = 150  # ignore frames quieter than this (silence, not noise)
    CLIP_THRESH = 32000  # int16 samples above this count as clipped

    # Load voice reference fingerprint once
    from . import config

    ref_path = Path(voice_ref_path) if voice_ref_path else Path(config.VOICE_REF_PATH)
    ref_fp = None
    if ref_path.exists():
        try:
            sr_ref, samples_ref = _wavfile.read(str(ref_path))
            if samples_ref.ndim > 1:
                samples_ref = samples_ref.mean(axis=1)
            ref_fp = _mfcc_fingerprint(samples_ref, sr_ref)
            if ref_fp is None:
                progress(
                    "Voice reference fingerprint unavailable"
                    " (librosa missing or ref too short);"
                    " skipping voice-match check"
                )
        except Exception as e:
            logger.warning(f"Failed to load voice reference {ref_path}: {e}")
    else:
        progress(f"No voice reference at {ref_path}; skipping voice-match check")

    if project_names:
        dirs = [output_dir / n for n in project_names if (output_dir / n).is_dir()]
    else:
        dirs = (
            sorted([d for d in output_dir.iterdir() if d.is_dir()])
            if output_dir.exists()
            else []
        )

    if not dirs:
        progress("No project output directories found for cleanup")
        return results

    for project_dir in dirs:
        project_name = project_dir.name
        chunks_dir = project_dir / "chunks"
        if not chunks_dir.exists():
            progress(f"[{project_name}] no chunks directory, skipping")
            continue

        wav_files = sorted(chunks_dir.glob("chunk_*.wav"))
        if not wav_files:
            progress(f"[{project_name}] no chunk WAV files found")
            continue

        progress(
            f"[{project_name}] auditing {len(wav_files)} chunks (discriminator + volume + voice match)..."
        )
        edited = 0
        deleted: list[tuple[str, str]] = []
        already_removed: set[str] = set()

        for wav_path in wav_files:
            if wav_path.name in already_removed or not wav_path.exists():
                continue
            try:
                sample_rate, samples = _wavfile.read(str(wav_path))

                # Ensure mono int16 for consistent processing
                if samples.ndim > 1:
                    samples = samples.mean(axis=1).astype(np.int16)
                orig_dtype = samples.dtype
                samples_f = samples.astype(np.float32)

                frame_size = int(sample_rate * FRAME_MS / 1000)
                n_frames = len(samples_f) // frame_size
                min_noisy_frames = max(1, int(max_noise_ms / FRAME_MS))

                window = np.hanning(frame_size)
                frame_is_noise = []
                voiced_rms = []

                for fi in range(n_frames):
                    frame = samples_f[fi * frame_size : (fi + 1) * frame_size]
                    rms = np.sqrt(np.mean(frame**2))

                    if rms < MIN_RMS:
                        frame_is_noise.append(False)  # silence — leave it alone
                        continue

                    # Spectral flatness (Wiener entropy)
                    mag = np.abs(np.fft.rfft(frame * window)) + 1e-10
                    flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))
                    is_noise = flatness > FLATNESS_THRESH
                    frame_is_noise.append(is_noise)
                    if not is_noise:
                        voiced_rms.append(rms)

                # ---- Discriminator: speech ratio ----
                speech_ratio = (len(voiced_rms) / n_frames) if n_frames else 0.0
                avg_rms = float(np.mean(voiced_rms)) if voiced_rms else 0.0
                clip_ratio = float(np.mean(np.abs(samples) > CLIP_THRESH))

                fail_reason = None
                if n_frames < 5:
                    fail_reason = f"too short ({n_frames} frames)"
                elif speech_ratio < min_speech_ratio:
                    fail_reason = (
                        f"low speech ratio {speech_ratio:.0%} (<{min_speech_ratio:.0%})"
                    )
                elif avg_rms < min_avg_rms:
                    fail_reason = (
                        f"too quiet (avg RMS {avg_rms:.0f} < {min_avg_rms:.0f})"
                    )
                elif clip_ratio > max_clip_ratio:
                    fail_reason = f"clipping {clip_ratio:.1%} (>{max_clip_ratio:.1%})"
                else:
                    # ---- Voice-match check (only if reference loaded) ----
                    if ref_fp is not None:
                        chunk_fp = _mfcc_fingerprint(samples, sample_rate)
                        if chunk_fp is not None:
                            sim = _cosine(ref_fp, chunk_fp)
                            if sim < min_voice_similarity:
                                fail_reason = f"voice mismatch (sim {sim:.2f} < {min_voice_similarity:.2f})"

                if fail_reason:
                    # A failed sub-part means the whole chunk should be regenerated.
                    # chunk_NNNNNN.wav and chunk_NNNNNN_K.wav share the same NNNNNN id;
                    # glob them all and remove together.
                    import re as _re

                    m = _re.match(r"(chunk_\d+)", wav_path.stem)
                    base = m.group(1) if m else wav_path.stem
                    siblings = sorted(chunks_dir.glob(f"{base}*.wav"))
                    removed_names = []
                    for sib in siblings:
                        try:
                            sib.unlink()
                            removed_names.append(sib.name)
                            already_removed.add(sib.name)
                        except FileNotFoundError:
                            already_removed.add(sib.name)
                        except Exception as e:
                            logger.error(f"Failed to delete {sib.name}: {e}")
                    if not removed_names:
                        continue
                    sibling_note = (
                        f" (+{len(removed_names) - 1} sibling part(s))"
                        if len(removed_names) > 1
                        else ""
                    )
                    for nm in removed_names:
                        deleted.append((nm, fail_reason))
                    progress(
                        f"[{project_name}] {wav_path.name}: DELETED — {fail_reason}"
                        f"{sibling_note}"
                    )
                    continue

                # ---- Passed all checks: clean long noisy regions in place ----
                noisy_regions = []
                i = 0
                while i < len(frame_is_noise):
                    if frame_is_noise[i]:
                        j = i + 1
                        while j < len(frame_is_noise) and frame_is_noise[j]:
                            j += 1
                        if (j - i) >= min_noisy_frames:
                            noisy_regions.append((i * frame_size, j * frame_size))
                        i = j
                    else:
                        i += 1

                if not noisy_regions:
                    continue

                parts = []
                prev = 0
                for start, end in noisy_regions:
                    if prev < start:
                        parts.append(samples[prev:start])
                    prev = end
                if prev < len(samples):
                    parts.append(samples[prev:])

                result = np.concatenate(parts).astype(orig_dtype)
                original_s = len(samples) / sample_rate
                new_s = len(result) / sample_rate

                _wavfile.write(str(wav_path), sample_rate, result)
                progress(
                    f"[{project_name}] {wav_path.name}: removed {len(noisy_regions)} noise region(s), "
                    f"{original_s:.1f}s -> {new_s:.1f}s"
                )
                edited += 1

            except Exception as e:
                logger.error(f"Failed to process {wav_path.name}: {e}")

        results[project_name] = {"edited": edited, "deleted": deleted}
        progress(
            f"[{project_name}] cleanup done: {edited} cleaned, {len(deleted)} deleted"
        )

    return results


def assemble_final_for_project(
    project_name, segments, output_root, progress_cb=None, stop_event=None
):
    """Assemble final audio from existing chunk WAVs without re-running TTS synthesis.

    Groups segments by source_file and emits per-chapter MP3s if multiple chapters detected.
    Any chunks missing from disk are simply absent from the assembled output.
    """
    progress = _make_progress(progress_cb)
    output_root = Path(output_root)
    project_out = output_root / project_name
    chunks_dir = project_out / "chunks"

    if not chunks_dir.exists():
        progress(f"[{project_name}] no chunks directory; run Stage 2 first")
        return None

    wav_files = sorted(chunks_dir.glob("chunk_*.wav"))
    if not wav_files:
        progress(f"[{project_name}] no chunk WAVs found; run Stage 2 first")
        return None

    progress(f"[{project_name}] found {len(wav_files)} chunk(s) for assembly")

    if stop_event and stop_event.is_set():
        return None

    progress(f"[{project_name}] denoising {len(wav_files)} chunks...")
    for wav_path in wav_files:
        if stop_event and stop_event.is_set():
            progress(f"[{project_name}] stopped during denoising")
            return None
        denoise(wav_path)

    if stop_event and stop_event.is_set():
        return None

    progress(f"[{project_name}] assembling final audio...")
    final_path = project_out / "final"
    final_path.mkdir(parents=True, exist_ok=True)

    # Group segments by source_file for per-chapter assembly
    from collections import defaultdict

    by_source = defaultdict(list)
    for i, seg in enumerate(segments):
        source = seg.get("source_file", "default")
        by_source[source].append((i, seg))

    output_files = []

    if len(by_source) == 1 or not config.CHAPTER_BY_CHAPTER:
        # Single file: use classic "audiobook.mp3" name
        all_wavs = []
        all_segs = []
        for source_name in sorted(by_source.keys()):
            seg_indices = [i for i, _ in by_source[source_name]]
            chapter_wavs = _wavs_for_segments(wav_files, seg_indices)
            all_wavs.extend(chapter_wavs)
            all_segs.extend([seg for _, seg in by_source[source_name]])
        if all_wavs:
            assembled = assemble(all_wavs, all_segs, final_path / "audiobook.mp3")
            progress(f"[{project_name}] done -> {assembled}")
            output_files.append(assembled)
    else:
        # Multiple chapters: emit chapter_NN.mp3 per source file
        sorted_sources = sorted(by_source.keys())
        progress(f"[{project_name}] {len(sorted_sources)} chapter(s) detected")
        for chapter_num, source_name in enumerate(sorted_sources, start=1):
            seg_indices = [i for i, _ in by_source[source_name]]
            chapter_wavs = _wavs_for_segments(wav_files, seg_indices)
            if not chapter_wavs:
                progress(f"  chapter {chapter_num} ({source_name}): no WAVs found")
                continue
            out_file = final_path / f"chapter_{chapter_num:02d}.mp3"
            assembled = assemble(
                chapter_wavs, [seg for _, seg in by_source[source_name]], out_file
            )
            progress(
                f"  chapter {chapter_num} ({source_name}): {len(chapter_wavs)} chunks -> {assembled}"
            )
            output_files.append(assembled)

    return output_files[0] if output_files else None


def assemble_final(segments_by_project, output_dir, progress_cb=None, stop_event=None):
    """Assemble final audio for all projects from existing chunks — no TTS synthesis.

    segments_by_project provides metadata (ends_paragraph etc.) for silence insertion.
    """
    output_dir = Path(output_dir)
    progress = _make_progress(progress_cb)

    if not segments_by_project:
        progress("No segments loaded; run Stage 1 first to load segment metadata")
        return {}

    results = {}
    for name, segments in segments_by_project.items():
        if stop_event and stop_event.is_set():
            progress("Stopped by user")
            break
        results[name] = assemble_final_for_project(
            name, segments, output_dir, progress_cb, stop_event
        )

    progress(f"Assemble Final complete for {len(results)} project(s)")
    return results


def run(input_dir, output_dir, progress_cb=None):
    """Full pipeline (no confirmation gate). Processes every project under input_dir."""
    by_project = prepare_segments(input_dir, output_dir, progress_cb)
    if not by_project:
        return {}
    return generate_audio(by_project, output_dir, progress_cb)
