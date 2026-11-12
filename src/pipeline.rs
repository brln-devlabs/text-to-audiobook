// This file is part of text-to-audiobook
//
// text-to-audiobook is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// text-to-audiobook is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! The four stages:
//! 1. prepare segments (clean, chunk, LLM analysis; cached per project)
//! 2. generate audio (TTS per segment, denoise, assemble)
//! 3. cleanup (audit chunks, delete failures, cut noise)
//! 4. assemble final audio from existing chunks
//!
//! Every subfolder of the input directory is one project (book), every
//! loose `.md` file is a project of its own.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::Result;
use indexmap::IndexMap;
use regex::Regex;
use serde_json::{Value, json};

use crate::audio::assembler::{assemble, chunk_id};
use crate::audio::cleanup::{self, ProjectCleanup, Thresholds};
use crate::audio::filter::denoise;
use crate::cancel::{StopFlag, stopped};
use crate::chunker::{Chunk, split_to_chunks};
use crate::config::{Config, Settings};
use crate::llm::{FilePreview, LlmNormalizer, ProjectHints};
use crate::markdown_cleaner::{clean, clean_with, heading_text};
use crate::natsort::{natsorted, natural_cmp};
use crate::pyval;
use crate::segment::Segment;
use crate::storage::create_storage;
use crate::tts::TtsEngine;

/// Progress callback for human-readable messages.
pub type Progress<'a> = &'a (dyn Fn(&str) + Sync);

/// A progress callback that does nothing (messages still go to the log).
pub fn no_progress(_: &str) {}

fn emit(progress: Progress, msg: &str) {
    log::info!("{msg}");
    progress(msg);
}

/// Segments per project, in project order.
pub type SegmentsByProject = IndexMap<String, Vec<Segment>>;

#[derive(Debug, Clone, PartialEq)]
pub struct Project {
    pub name: String,
    pub files: Vec<PathBuf>,
    pub input_dir: PathBuf,
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn file_stem(p: &Path) -> String {
    p.file_stem()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default()
}

/// All `.md` files below `dir` (recursive).
fn md_files_recursive(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        let mut entries: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "md") {
                out.push(p);
            }
        }
    }
    out
}

/// Each subfolder of `input_dir` with `.md` files is one project; each loose
/// `.md` file is its own project.
pub fn discover_projects(input_dir: &Path) -> Vec<Project> {
    let mut projects = Vec::new();
    let Ok(entries) = std::fs::read_dir(input_dir) else {
        return projects;
    };
    let mut entries: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    entries.sort();
    for sub in entries.iter().filter(|p| p.is_dir()) {
        let files = md_files_recursive(sub);
        if !files.is_empty() {
            projects.push(Project {
                name: file_name(sub),
                files,
                input_dir: sub.clone(),
            });
        }
    }
    for f in entries
        .iter()
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "md"))
    {
        projects.push(Project {
            name: file_stem(f),
            files: vec![f.clone()],
            input_dir: input_dir.to_path_buf(),
        });
    }
    projects
}

/// Valid TTS-ready segments of a (complete or partial) stage-1 cache.
pub fn load_cached_segments(cache_dir: &Path, storage_backend: &str) -> Vec<Segment> {
    if !cache_dir.exists() {
        return Vec::new();
    }
    let storage = match create_storage(storage_backend, cache_dir) {
        Ok(s) => s,
        Err(e) => {
            log::warn!("Could not load cached segments: {e:#}");
            return Vec::new();
        }
    };
    if let Err(e) = storage.init_schema() {
        log::warn!("Could not load cached segments: {e:#}");
        return Vec::new();
    }
    storage
        .list_all_segments()
        .into_iter()
        .filter_map(Segment::from_value)
        .filter(|s| !s.normalized_text.is_empty())
        .collect()
}

static NUMERIC_PREFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\d").unwrap());

/// Read up to `n` characters from the start of a file.
fn read_head(path: &Path, n: usize) -> String {
    std::fs::read(path)
        .map(|b| pyval::prefix(&String::from_utf8_lossy(&b), n).to_string())
        .unwrap_or_default()
}

/// Narrative reading order of the `.md` files of a project.
///
/// If every name starts with a digit, natural sort decides (the 1B LLM cannot
/// collate numbers reliably). Otherwise the LLM sees content previews and
/// recognises ordinal words and structure (prologue, epilogue, ...). The
/// result is cached in `<cache_dir>/file_order.json`.
fn order_files(
    input_dir: &Path,
    llm: &mut LlmNormalizer,
    cache_dir: &Path,
    progress: Progress,
) -> Vec<PathBuf> {
    let md_paths = md_files_recursive(input_dir);
    if md_paths.is_empty() {
        return Vec::new();
    }
    let mut name_to_path: IndexMap<String, PathBuf> = IndexMap::new();
    for p in &md_paths {
        name_to_path.insert(file_name(p), p.clone());
    }
    let cache_file = cache_dir.join("file_order.json");
    if cache_file.exists() {
        let cached = std::fs::read_to_string(&cache_file)
            .ok()
            .and_then(|t| serde_json::from_str::<Value>(&t).ok());
        match cached {
            Some(v) => {
                let order: Vec<PathBuf> = v["order"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(Value::as_str)
                            .filter_map(|n| name_to_path.get(n).cloned())
                            .collect()
                    })
                    .unwrap_or_default();
                let extras = md_paths.iter().any(|p| !order.contains(p));
                if !extras && !order.is_empty() {
                    emit(
                        progress,
                        &format!("Using cached file order ({} files)", order.len()),
                    );
                    return order;
                }
            }
            None => log::warn!("Could not read cached file order: {}", cache_file.display()),
        }
    }
    if md_paths.len() == 1 {
        return md_paths;
    }

    let names: Vec<String> = md_paths.iter().map(|p| file_name(p)).collect();
    let ordered_names = if names.iter().all(|n| NUMERIC_PREFIX.is_match(n)) {
        let ordered = natsorted(&names);
        emit(
            progress,
            &format!(
                "Numeric-prefixed files: using natural sort ({} files)",
                ordered.len()
            ),
        );
        ordered
    } else if !llm.is_loaded() {
        let ordered = crate::llm::rule_order(&names);
        emit(
            progress,
            &format!(
                "Ordering {} files by name (number words, prologue/epilogue)",
                ordered.len()
            ),
        );
        ordered
    } else {
        let mut presorted = md_paths.clone();
        presorted.sort_by(|a, b| natural_cmp(&file_name(a), &file_name(b)));
        let previews: Vec<FilePreview> = presorted
            .iter()
            .map(|p| FilePreview {
                name: file_name(p),
                preview: pyval::prefix(&clean(&read_head(p, 400)), 200).to_string(),
            })
            .collect();
        emit(
            progress,
            &format!(
                "Asking LLM to order {} files by narrative sequence...",
                md_paths.len()
            ),
        );
        llm.sort_files(&previews)
    };

    let _ = std::fs::create_dir_all(cache_dir);
    let text = serde_json::to_string_pretty(&json!({ "order": ordered_names })).unwrap_or_default();
    if let Err(e) = std::fs::write(&cache_file, text) {
        log::warn!("Could not write {}: {e}", cache_file.display());
    }
    ordered_names
        .iter()
        .filter_map(|n| name_to_path.get(n).cloned())
        .collect()
}

/// Text of the first Markdown heading (`# ...`), formatting removed.
pub fn first_heading(md: &str) -> Option<String> {
    let mut in_fence = false;
    for line in md.lines() {
        let t = line.trim();
        if t.starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some(title) = heading_text(t) {
            return Some(title);
        }
    }
    None
}

fn build_chunks(
    md_files: &[PathBuf],
    settings: &Settings,
    progress: Progress,
    hints: &mut ProjectHints,
) -> Result<Vec<Chunk>> {
    emit(progress, &format!("Found {} file(s)", md_files.len()));
    emit(progress, "Cleaning and chunking text...");
    let mut chunks = Vec::new();
    for (n, f) in md_files.iter().enumerate() {
        let raw = std::fs::read_to_string(f)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", f.display()))?;
        // The first heading is the chapter name (VOCALIZE_CHAPTER_NAMES);
        // further headings are paragraphs of their own (SPEAK_HEADINGS).
        let cleaned = clean_with(&raw, settings.speak_headings);
        let source = file_stem(f);
        hints
            .chapters
            .insert(source.clone(), (n as i64 + 1, first_heading(&raw)));
        let file_chunks = split_to_chunks(&cleaned, settings.max_words_per_chunk, &source);
        emit(
            progress,
            &format!("  {source}: {} chunk(s)", file_chunks.len()),
        );
        chunks.extend(file_chunks);
    }
    emit(
        progress,
        &format!(
            "Total: {} chunks from {} file(s)",
            chunks.len(),
            md_files.len()
        ),
    );
    Ok(chunks)
}

/// Stage 1 for one project. Pass `llm` to share one loaded model between
/// projects; with None a model is loaded (and unloaded) for this project.
pub fn prepare_segments_for_project(
    cfg: &Config,
    project: &Project,
    progress: Progress,
    llm: Option<&mut LlmNormalizer>,
    stop: Option<&StopFlag>,
) -> Result<Vec<Segment>> {
    let name = &project.name;
    let cache_dir = cfg.paths.project_cache_dir(name);
    emit(
        progress,
        &format!("--- Project '{name}' ({} file(s)) ---", project.files.len()),
    );

    let mut own: Option<LlmNormalizer> = None;
    let llm: &mut LlmNormalizer = match llm {
        Some(l) => l,
        None => {
            if cfg.settings.use_llm {
                emit(progress, "Loading LLM...");
            }
            let mut l = LlmNormalizer::new(cfg.paths.llm_model_path(), &cfg.settings);
            l.load();
            own.insert(l)
        }
    };

    let result = (|| -> Result<Vec<Segment>> {
        if stopped(stop) {
            return Ok(Vec::new());
        }
        let ordered = if project.files.len() > 1 {
            order_files(&project.input_dir, llm, &cache_dir, progress)
        } else {
            project.files.clone()
        };
        emit(
            progress,
            &format!(
                "[{name}] file order: {}",
                ordered
                    .iter()
                    .map(|p| file_name(p))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        );
        let mut hints = ProjectHints {
            story_title: Some(name.clone()),
            ..Default::default()
        };
        let chunks = build_chunks(&ordered, &cfg.settings, progress, &mut hints)?;
        if chunks.is_empty() {
            return Ok(Vec::new());
        }
        emit(
            progress,
            &format!(
                "[{name}] {}...",
                if llm.is_loaded() {
                    "emotion analysis"
                } else {
                    "building segments"
                }
            ),
        );
        let segments = llm.process_all(
            &chunks,
            &cache_dir,
            &cfg.settings,
            &hints,
            Some(progress),
            stop,
        )?;
        emit(
            progress,
            &format!("[{name}] {} TTS segments ready", segments.len()),
        );
        Ok(segments)
    })();
    if let Some(mut l) = own {
        l.unload();
    }
    result
}

/// Stage 1 for the projects below `input_dir` (only `selected` ones when given).
pub fn prepare_segments(
    cfg: &Config,
    input_dir: &Path,
    output_dir: &Path,
    progress: Progress,
    stop: Option<&StopFlag>,
    selected: Option<&[String]>,
) -> Result<SegmentsByProject> {
    emit(progress, "Discovering projects in input directory...");
    let mut projects = discover_projects(input_dir);
    if projects.is_empty() {
        emit(
            progress,
            "No projects found (no folders with .md files and no loose .md files)",
        );
        return Ok(IndexMap::new());
    }
    if let Some(wanted) = selected {
        projects.retain(|p| wanted.contains(&p.name));
        if projects.is_empty() {
            emit(progress, "No selected projects to process");
            return Ok(IndexMap::new());
        }
    }
    emit(
        progress,
        &format!(
            "Found {} project(s): {}",
            projects.len(),
            projects
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    );

    if cfg.settings.use_llm {
        emit(progress, "Loading LLM...");
    }
    let mut llm = LlmNormalizer::new(cfg.paths.llm_model_path(), &cfg.settings);
    llm.load();

    let mut results = IndexMap::new();
    let outcome = (|| -> Result<()> {
        for project in &projects {
            if stopped(stop) {
                emit(progress, "Stopped by user");
                break;
            }
            let segs = prepare_segments_for_project(cfg, project, progress, Some(&mut llm), stop)?;
            if !segs.is_empty() {
                match write_texts(cfg, &project.name, &segs, output_dir) {
                    Ok(p) => emit(
                        progress,
                        &format!("[{}] text: {}", project.name, p.display()),
                    ),
                    Err(e) => {
                        log::warn!("[{}] could not write the text files: {e:#}", project.name)
                    }
                }
            }
            results.insert(project.name.clone(), segs);
        }
        Ok(())
    })();
    llm.unload();
    outcome?;
    emit(
        progress,
        &format!(
            "Stage 1 complete for {}/{} project(s)",
            results.len(),
            projects.len()
        ),
    );
    Ok(results)
}

/// All WAVs (including sub-parts `chunk_NNN_K.wav`) of the given segments.
/// WAVs with other names are matched by position.
pub fn wavs_for_segments(wav_paths: &[PathBuf], seg_indices: &[usize]) -> Vec<PathBuf> {
    let mut by_chunk: IndexMap<usize, Vec<PathBuf>> = IndexMap::new();
    let mut positional = Vec::new();
    for w in wav_paths {
        match chunk_id(w) {
            Some(id) => by_chunk.entry(id).or_default().push(w.clone()),
            None => positional.push(w.clone()),
        }
    }
    if by_chunk.is_empty() {
        return seg_indices
            .iter()
            .filter_map(|i| positional.get(*i).cloned())
            .collect();
    }
    seg_indices
        .iter()
        .flat_map(|i| by_chunk.get(i).cloned().unwrap_or_default())
        .collect()
}

/// Segment indices grouped by source file, in reading order (first
/// appearance), so chapter numbers follow the narrative order.
fn group_by_source(segments: &[Segment]) -> IndexMap<String, Vec<usize>> {
    let mut by_source: IndexMap<String, Vec<usize>> = IndexMap::new();
    for (i, s) in segments.iter().enumerate() {
        by_source.entry(s.source_file.clone()).or_default().push(i);
    }
    by_source
}

fn ends_paragraph(segments: &[Segment], indices: &[usize]) -> Vec<bool> {
    indices
        .iter()
        .map(|i| segments[*i].ends_paragraph)
        .collect()
}

/// One `audiobook.mp3`, or `chapter_NN.mp3` per source file when the
/// project has several and `chapter_by_chapter` is set.
fn assemble_outputs(
    cfg: &Config,
    wav_paths: &[PathBuf],
    segments: &[Segment],
    final_path: &Path,
    project_name: &str,
    chapter_by_chapter: bool,
    progress: Progress,
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(final_path)?;
    let by_source = group_by_source(segments);
    let mut outputs = Vec::new();
    if by_source.len() <= 1 || !chapter_by_chapter {
        let indices: Vec<usize> = by_source.values().flatten().copied().collect();
        let wavs = wavs_for_segments(wav_paths, &indices);
        if !wavs.is_empty() {
            let out = assemble(
                &wavs,
                &ends_paragraph(segments, &indices),
                &final_path.join("audiobook.mp3"),
                &cfg.settings,
            )?;
            emit(
                progress,
                &format!("[{project_name}] done -> {}", out.display()),
            );
            outputs.push(out);
        }
        return Ok(outputs);
    }
    emit(
        progress,
        &format!("[{project_name}] {} chapter(s) detected", by_source.len()),
    );
    for (n, (source, indices)) in by_source.iter().enumerate() {
        let chapter_num = n + 1;
        let wavs = wavs_for_segments(wav_paths, indices);
        if wavs.is_empty() {
            emit(
                progress,
                &format!("  chapter {chapter_num} ({source}): no WAVs found"),
            );
            continue;
        }
        let out = assemble(
            &wavs,
            &ends_paragraph(segments, indices),
            &final_path.join(format!("chapter_{chapter_num:02}.mp3")),
            &cfg.settings,
        )?;
        emit(
            progress,
            &format!(
                "  chapter {chapter_num} ({source}): {} chunks -> {}",
                wavs.len(),
                out.display()
            ),
        );
        outputs.push(out);
    }
    Ok(outputs)
}

/// Audit rounds of `AUTO_CLEANUP`: audit, delete failures, synthesize them
/// again with another seed. The chunks of the last round are kept, so no
/// text goes missing.
pub const CLEANUP_ROUNDS: usize = 2;

/// Write the prepared text of a project to `<output>/<project>/text/`:
/// `prepared.txt` (as cleaned and segmented) and, with `NORMALIZE_NUMBERS`,
/// `spoken.txt` (as it is spoken, review decisions applied) and, with
/// `REVIEW_NUMBERS`, `review.txt` (every uncertain spot with its decision).
/// Every line starts with the number of its audio chunk (`chunk_NNNNNN.wav`).
pub fn write_texts(
    cfg: &Config,
    project_name: &str,
    segments: &[Segment],
    output_dir: &Path,
) -> Result<PathBuf> {
    let dir = output_dir.join(project_name).join("text");
    std::fs::create_dir_all(&dir)?;
    let store = crate::review::load_synced(
        &cfg.paths.project_cache_dir(project_name),
        project_name,
        segments,
        &cfg.settings,
    );
    let render = |spoken: bool| -> String {
        let title = segments
            .first()
            .and_then(|s| s.story_title.clone())
            .unwrap_or_else(|| project_name.to_string());
        let mut out = format!("# {title}\n");
        let mut source: Option<&str> = None;
        let mut in_paragraph = false;
        for (i, seg) in segments.iter().enumerate() {
            if source != Some(seg.source_file.as_str()) {
                let chapter = seg.chapter_title.as_deref().unwrap_or(&seg.source_file);
                if chapter == seg.source_file {
                    out.push_str(&format!("\n## {chapter}\n\n"));
                } else {
                    out.push_str(&format!("\n## {chapter} ({})\n\n", seg.source_file));
                }
                source = Some(&seg.source_file);
            } else if !in_paragraph {
                out.push('\n');
            }
            let text = if spoken {
                let language = crate::tts::resolve_language(&cfg.settings, seg.language.as_deref());
                crate::tts::spoken_text(
                    &cfg.settings,
                    Some(&store),
                    i,
                    &seg.normalized_text,
                    &language,
                )
            } else {
                seg.normalized_text.clone()
            };
            out.push_str(&format!("{i:06} | {text}\n"));
            in_paragraph = !seg.ends_paragraph;
        }
        out
    };
    let prepared = dir.join("prepared.txt");
    std::fs::write(&prepared, render(false))?;
    let spoken = dir.join("spoken.txt");
    if cfg.settings.normalize_numbers {
        std::fs::write(&spoken, render(true))?;
    } else if spoken.exists() {
        std::fs::remove_file(&spoken)?;
    }
    // review.txt: every uncertain spot with its decision, in context.
    let report = dir.join(crate::review::REPORT_FILE);
    if cfg.settings.normalize_numbers && cfg.settings.review_numbers {
        std::fs::write(
            &report,
            crate::review::report(project_name, segments, &cfg.settings, &store),
        )?;
    } else if report.exists() {
        std::fs::remove_file(&report)?;
    }
    Ok(prepared)
}

/// Stage 2 for one project; returns the (first) assembled MP3.
#[allow(clippy::too_many_arguments)]
pub fn generate_audio_for_project(
    cfg: &Config,
    project_name: &str,
    segments: &[Segment],
    output_root: &Path,
    progress: Progress,
    tts: Option<&mut TtsEngine>,
    stop: Option<&StopFlag>,
    chunk_done: Option<&(dyn Fn() + Sync)>,
) -> Result<Option<PathBuf>> {
    let project_out = output_root.join(project_name);
    if segments.is_empty() {
        emit(progress, &format!("[{project_name}] no segments, skipping"));
        return Ok(None);
    }
    emit(
        progress,
        &format!(
            "--- Audio gen for project '{project_name}' ({} segments) ---",
            segments.len()
        ),
    );

    let mut own: Option<TtsEngine> = None;
    let tts: &mut TtsEngine = match tts {
        Some(t) => t,
        None => {
            emit(progress, "Loading TTS engine...");
            let mut t = TtsEngine::new(&cfg.settings, &cfg.paths);
            t.load()?;
            own.insert(t)
        }
    };

    let result = (|| -> Result<Option<PathBuf>> {
        let store = crate::review::load_synced(
            &cfg.paths.project_cache_dir(project_name),
            project_name,
            segments,
            &cfg.settings,
        );
        if cfg.settings.review_numbers && cfg.settings.normalize_numbers {
            let open = crate::review::pending(project_name, segments, &cfg.settings, &store).len();
            if open > 0 {
                log::warn!(
                    "[{project_name}] {open} number spot(s) not reviewed yet; the program defaults are used"
                );
            }
        }
        if let Err(e) = write_texts(cfg, project_name, segments, output_root) {
            log::warn!("[{project_name}] could not write the text files: {e:#}");
        }
        tts.set_review(Some(store));
        let chunks_dir = project_out.join("chunks");
        let wav_paths = (|| -> Result<Vec<PathBuf>> {
            let mut wavs = tts.process_all(segments, &chunks_dir, stop, chunk_done)?;
            if !cfg.settings.auto_cleanup {
                return Ok(wavs);
            }
            let report = |m: &str| emit(progress, m);
            for round in 1..=CLEANUP_ROUNDS {
                if stopped(stop) {
                    break;
                }
                emit(
                    progress,
                    &format!("[{project_name}] auditing chunks (round {round})..."),
                );
                let results = cleanup::cleanup_noisy_chunks(
                    output_root,
                    Some(&[project_name.to_string()]),
                    &report,
                    &cfg.paths.voice_ref_path(),
                    &Thresholds::default(),
                );
                let deleted = results.get(project_name).map_or(0, |r| r.deleted.len());
                if deleted == 0 {
                    emit(
                        progress,
                        &format!("[{project_name}] all chunks passed the audit"),
                    );
                    break;
                }
                emit(
                    progress,
                    &format!(
                        "[{project_name}] {deleted} chunk file(s) failed, synthesizing them again with a new seed"
                    ),
                );
                tts.set_seed_offset(round as u64);
                let again = tts.process_all(segments, &chunks_dir, stop, None);
                tts.set_seed_offset(0);
                wavs = again?;
                if round == CLEANUP_ROUNDS {
                    emit(
                        progress,
                        &format!(
                            "[{project_name}] kept the chunks of the last attempt after {CLEANUP_ROUNDS} rounds"
                        ),
                    );
                }
            }
            Ok(wavs)
        })();
        tts.set_review(None);
        let wav_paths = wav_paths?;
        if stopped(stop) {
            emit(
                progress,
                &format!(
                    "[{project_name}] stopped — {} chunk(s) generated so far",
                    wav_paths.len()
                ),
            );
            return Ok(None);
        }
        emit(
            progress,
            &format!("[{project_name}] denoising {} chunks...", wav_paths.len()),
        );
        for w in &wav_paths {
            if stopped(stop) {
                break;
            }
            denoise(w)?;
        }
        if stopped(stop) {
            emit(
                progress,
                &format!("[{project_name}] stopped during denoising"),
            );
            return Ok(None);
        }
        emit(
            progress,
            &format!("[{project_name}] assembling final audio..."),
        );
        let outputs = assemble_outputs(
            cfg,
            &wav_paths,
            segments,
            &project_out.join("final"),
            project_name,
            cfg.settings.chapter_by_chapter,
            progress,
        )?;
        if cfg.settings.chapter_by_chapter && outputs.len() > 1 {
            emit(
                progress,
                &format!("[{project_name}] done -> chapter files generated"),
            );
        }
        Ok(outputs.into_iter().next())
    })();
    if let Some(mut t) = own {
        t.unload();
    }
    result
}

/// Stage 2 for all projects, sharing one loaded TTS model.
pub fn generate_audio(
    cfg: &Config,
    segments_by_project: &SegmentsByProject,
    output_dir: &Path,
    progress: Progress,
    stop: Option<&StopFlag>,
    chunk_done: Option<&(dyn Fn() + Sync)>,
) -> Result<IndexMap<String, Option<PathBuf>>> {
    let mut results = IndexMap::new();
    if segments_by_project.is_empty() {
        emit(progress, "No segments to synthesize");
        return Ok(results);
    }
    emit(progress, "Loading TTS engine...");
    let mut tts = TtsEngine::new(&cfg.settings, &cfg.paths);
    tts.load()?;
    let outcome = (|| -> Result<()> {
        for (name, segments) in segments_by_project {
            if stopped(stop) {
                emit(progress, "Stopped by user");
                break;
            }
            let r = generate_audio_for_project(
                cfg,
                name,
                segments,
                output_dir,
                progress,
                Some(&mut tts),
                stop,
                chunk_done,
            )?;
            results.insert(name.clone(), r);
        }
        Ok(())
    })();
    tts.unload();
    outcome?;
    emit(
        progress,
        &format!("Stage 2 complete for {} project(s)", results.len()),
    );
    Ok(results)
}

/// Stage 3: audit chunk WAVs (see [`crate::audio::cleanup`]).
pub fn cleanup_noisy_chunks(
    cfg: &Config,
    output_dir: &Path,
    project_names: Option<&[String]>,
    progress: Progress,
) -> IndexMap<String, ProjectCleanup> {
    let report = |m: &str| emit(progress, m);
    cleanup::cleanup_noisy_chunks(
        output_dir,
        project_names,
        &report,
        &cfg.paths.voice_ref_path(),
        &Thresholds::default(),
    )
}

/// Stage 4 for one project: denoise and assemble the existing chunk WAVs
/// without running the TTS. Missing chunks are simply absent.
pub fn assemble_final_for_project(
    cfg: &Config,
    project_name: &str,
    segments: &[Segment],
    output_root: &Path,
    progress: Progress,
    stop: Option<&StopFlag>,
) -> Result<Option<PathBuf>> {
    let project_out = output_root.join(project_name);
    let chunks_dir = project_out.join("chunks");
    if !chunks_dir.exists() {
        emit(
            progress,
            &format!("[{project_name}] no chunks directory; run Stage 2 first"),
        );
        return Ok(None);
    }
    let wav_files = cleanup::chunk_wavs(&chunks_dir);
    if wav_files.is_empty() {
        emit(
            progress,
            &format!("[{project_name}] no chunk WAVs found; run Stage 2 first"),
        );
        return Ok(None);
    }
    emit(
        progress,
        &format!(
            "[{project_name}] found {} chunk(s) for assembly",
            wav_files.len()
        ),
    );
    if stopped(stop) {
        return Ok(None);
    }
    emit(
        progress,
        &format!("[{project_name}] denoising {} chunks...", wav_files.len()),
    );
    for w in &wav_files {
        if stopped(stop) {
            emit(
                progress,
                &format!("[{project_name}] stopped during denoising"),
            );
            return Ok(None);
        }
        denoise(w)?;
    }
    if stopped(stop) {
        return Ok(None);
    }
    emit(
        progress,
        &format!("[{project_name}] assembling final audio..."),
    );
    let outputs = assemble_outputs(
        cfg,
        &wav_files,
        segments,
        &project_out.join("final"),
        project_name,
        cfg.settings.chapter_by_chapter,
        progress,
    )?;
    Ok(outputs.into_iter().next())
}

/// Stage 4 for all projects.
pub fn assemble_final(
    cfg: &Config,
    segments_by_project: &SegmentsByProject,
    output_dir: &Path,
    progress: Progress,
    stop: Option<&StopFlag>,
) -> Result<IndexMap<String, Option<PathBuf>>> {
    let mut results = IndexMap::new();
    if segments_by_project.is_empty() {
        emit(
            progress,
            "No segments loaded; run Stage 1 first to load segment metadata",
        );
        return Ok(results);
    }
    for (name, segments) in segments_by_project {
        if stopped(stop) {
            emit(progress, "Stopped by user");
            break;
        }
        let r = assemble_final_for_project(cfg, name, segments, output_dir, progress, stop)?;
        results.insert(name.clone(), r);
    }
    emit(
        progress,
        &format!("Assemble Final complete for {} project(s)", results.len()),
    );
    Ok(results)
}

/// Cached stage-1 segments of the projects below `input_dir`.
pub fn cached_segments_by_project(cfg: &Config, input_dir: &Path) -> SegmentsByProject {
    let mut found = IndexMap::new();
    for p in discover_projects(input_dir) {
        let segs = load_cached_segments(
            &cfg.paths.project_cache_dir(&p.name),
            &cfg.settings.storage_backend,
        );
        if !segs.is_empty() {
            found.insert(p.name, segs);
        }
    }
    found
}

/// Re-check the language of segments that differ from the project's
/// majority language and store corrections.
pub fn fix_languages(cfg: &Config, selected: &[String]) -> Result<()> {
    if !cfg.settings.use_llm {
        anyhow::bail!("Fix Languages needs the LLM: set USE_LLM to true (Settings → LLM)");
    }
    let mut llm = LlmNormalizer::new(cfg.paths.llm_model_path(), &cfg.settings);
    llm.load();
    if !llm.is_loaded() {
        anyhow::bail!("LLM load failed");
    }
    let outcome = (|| -> Result<()> {
        for name in selected {
            let cache_dir = cfg.paths.project_cache_dir(name);
            if !cache_dir.exists() {
                log::info!("No cache for '{name}', skipping.");
                continue;
            }
            let storage = create_storage(&cfg.settings.storage_backend, &cache_dir)?;
            storage.init_schema()?;
            let data = storage.list_all_segments();
            if data.is_empty() {
                log::info!("No segments for '{name}', skipping.");
                continue;
            }
            let lang_of = |d: &Value| -> String {
                d.get("language")
                    .and_then(Value::as_str)
                    .unwrap_or("english")
                    .to_string()
            };
            let mut tally: IndexMap<String, usize> = IndexMap::new();
            for d in &data {
                *tally.entry(lang_of(d)).or_default() += 1;
            }
            let majority = tally
                .iter()
                .fold(None::<(&String, usize)>, |best, (l, n)| match best {
                    Some((_, b)) if b >= *n => best,
                    _ => Some((l, *n)),
                })
                .map(|(l, _)| l.clone())
                .unwrap_or_default();
            let outliers: Vec<(usize, Value)> = data
                .iter()
                .enumerate()
                .filter(|(_, d)| lang_of(d) != majority)
                .map(|(i, d)| (i, d.clone()))
                .collect();
            if outliers.is_empty() {
                log::info!("'{name}': all segments already '{majority}', nothing to fix.");
                continue;
            }
            log::info!(
                "'{name}': majority lang='{majority}', re-checking {} outlier(s)...",
                outliers.len()
            );
            let mut fixed = Vec::new();
            for (seg_id, mut d) in outliers {
                let text = d
                    .get("normalized_text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let old = d
                    .get("language")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string();
                let new = match llm.detect_language(&text) {
                    Ok(l) => l,
                    Err(e) => {
                        log::warn!("Language check failed for segment {seg_id}: {e:#}");
                        continue;
                    }
                };
                if new != old {
                    d["language"] = Value::from(new.clone());
                    storage.save_segment_meta(seg_id, &d);
                    fixed.push(format!("{old}→{new}"));
                }
            }
            if fixed.is_empty() {
                log::info!("'{name}': LLM confirmed all outliers are correct — no changes.");
            } else {
                log::info!(
                    "Fixed {} language detection(s) in '{name}': {}",
                    fixed.len(),
                    fixed.join(", ")
                );
            }
        }
        Ok(())
    })();
    llm.unload();
    outcome
}

/// Full pipeline (stage 1 + 2) for every project below `input_dir`.
pub fn run(
    cfg: &Config,
    input_dir: &Path,
    output_dir: &Path,
    progress: Progress,
) -> Result<IndexMap<String, Option<PathBuf>>> {
    let by_project = prepare_segments(cfg, input_dir, output_dir, progress, None, None)?;
    if by_project.is_empty() {
        return Ok(IndexMap::new());
    }
    generate_audio(cfg, &by_project, output_dir, progress, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Paths, Settings};

    fn cfg(root: &Path) -> Config {
        let paths = Paths::new(root);
        paths.ensure_dirs().unwrap();
        let settings = Settings {
            mock_models: true,
            ..Settings::default()
        };
        Config::new(paths, settings)
    }

    #[test]
    fn discover_projects_folders_and_loose_files() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path();
        std::fs::create_dir_all(input.join("Book/part")).unwrap();
        std::fs::write(input.join("Book/01.md"), "A.").unwrap();
        std::fs::write(input.join("Book/part/02.md"), "B.").unwrap();
        std::fs::create_dir_all(input.join("Empty")).unwrap();
        std::fs::write(input.join("story.md"), "C.").unwrap();
        std::fs::write(input.join("notes.txt"), "x").unwrap();
        let p = discover_projects(input);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].name, "Book");
        assert_eq!(p[0].files.len(), 2);
        assert_eq!(p[1].name, "story");
        assert_eq!(p[1].input_dir, input);
        assert!(discover_projects(&input.join("missing")).is_empty());
    }

    #[test]
    fn wavs_for_segments_handles_sub_parts() {
        let wavs: Vec<PathBuf> = [
            "chunk_000000.wav",
            "chunk_000001_0.wav",
            "chunk_000001_1.wav",
            "chunk_000002.wav",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        let got = wavs_for_segments(&wavs, &[1, 2]);
        assert_eq!(got, wavs[1..].to_vec());
        let other = vec![PathBuf::from("a.wav"), PathBuf::from("b.wav")];
        assert_eq!(
            wavs_for_segments(&other, &[1, 5]),
            vec![PathBuf::from("b.wav")]
        );
    }

    #[test]
    fn numeric_prefixed_files_use_natural_sort_and_cache() {
        let dir = tempfile::tempdir().unwrap();
        let c = cfg(dir.path());
        let book = c.paths.input_dir.join("Book");
        std::fs::create_dir_all(&book).unwrap();
        for n in ["10_Ende.md", "2_Mitte.md", "0_Prolog.md"] {
            std::fs::write(book.join(n), "Text.").unwrap();
        }
        let mut llm = LlmNormalizer::new("fake.gguf", &c.settings);
        let cache = c.paths.project_cache_dir("Book");
        let order = order_files(&book, &mut llm, &cache, &no_progress);
        let names: Vec<String> = order.iter().map(|p| file_name(p)).collect();
        assert_eq!(names, ["0_Prolog.md", "2_Mitte.md", "10_Ende.md"]);
        assert!(cache.join("file_order.json").exists());
        let again = order_files(&book, &mut llm, &cache, &no_progress);
        assert_eq!(again, order);
    }

    #[test]
    fn end_to_end_with_mock_models() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = cfg(dir.path());
        c.settings.chapter_by_chapter = true;
        let book = c.paths.input_dir.join("Book");
        std::fs::create_dir_all(&book).unwrap();
        std::fs::write(
            book.join("1_Anfang.md"),
            "# Titel\n\nErster Absatz. Mit **zwei** Sätzen.\n\nZweiter Absatz.",
        )
        .unwrap();
        std::fs::write(book.join("2_Ende.md"), "Das Ende.").unwrap();
        std::fs::write(c.paths.input_dir.join("Kurz.md"), "Eine kurze Geschichte.").unwrap();

        let segs = prepare_segments(
            &c,
            &c.paths.input_dir.clone(),
            &c.paths.output_dir.clone(),
            &no_progress,
            None,
            None,
        )
        .unwrap();
        assert_eq!(segs.keys().collect::<Vec<_>>(), ["Book", "Kurz"]);
        let book_segs = &segs["Book"];
        assert_eq!(book_segs.len(), 3);
        assert_eq!(
            book_segs[0].normalized_text,
            "Erster Absatz. Mit zwei Sätzen."
        );
        assert_eq!(book_segs[2].source_file, "2_Ende");
        assert!(book_segs.iter().all(|s| s.ends_paragraph));

        // Cached segments are found again.
        let cached = cached_segments_by_project(&c, &c.paths.input_dir.clone());
        assert_eq!(cached["Book"].len(), 3);

        let out = c.paths.output_dir.clone();
        let done = std::sync::atomic::AtomicUsize::new(0);
        let cb = || {
            done.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        };
        let res = generate_audio(&c, &segs, &out, &no_progress, None, Some(&cb)).unwrap();
        assert_eq!(done.load(std::sync::atomic::Ordering::SeqCst), 4);
        assert_eq!(
            res["Book"].as_deref(),
            Some(out.join("Book/final/chapter_01.mp3").as_path())
        );
        assert!(out.join("Book/final/chapter_02.wav").exists());
        assert!(out.join("Kurz/final/audiobook.mp3").exists());
        assert!(out.join("Book/chunks/chunk_000002.wav").exists());

        // Stage 4 without TTS, as one file.
        c.settings.chapter_by_chapter = false;
        let res = assemble_final(&c, &segs, &out, &no_progress, None).unwrap();
        assert_eq!(
            res["Book"].as_deref(),
            Some(out.join("Book/final/audiobook.mp3").as_path())
        );

        // Stage 3 on mock silence deletes everything.
        let report = cleanup_noisy_chunks(&c, &out, Some(&["Kurz".to_string()]), &no_progress);
        assert_eq!(report["Kurz"].deleted.len(), 1);

        // Prepared text with chunk numbers and chapter headings.
        let text = std::fs::read_to_string(out.join("Book/text/prepared.txt")).unwrap();
        assert!(text.starts_with("# Book\n"), "{text}");
        assert!(text.contains("## Titel (1_Anfang)"), "{text}");
        assert!(
            text.contains("000000 | Erster Absatz. Mit zwei Sätzen.\n\n000001 | Zweiter Absatz.")
        );
        assert!(text.contains("000002 | Das Ende."));
        assert!(!out.join("Book/text/spoken.txt").exists());
    }

    #[test]
    fn auto_cleanup_regenerates_and_keeps_last_attempt() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = cfg(dir.path());
        c.settings.auto_cleanup = true;
        c.settings.normalize_numbers = true;
        std::fs::write(
            c.paths.input_dir.join("Kurz.md"),
            "Am 3. Mai 1995 war es kalt.",
        )
        .unwrap();
        let input = c.paths.input_dir.clone();
        let out = c.paths.output_dir.clone();
        let segs = prepare_segments(&c, &input, &out, &no_progress, None, None).unwrap();
        let msgs = std::sync::Mutex::new(Vec::<String>::new());
        let progress = |m: &str| msgs.lock().unwrap().push(m.to_string());
        let res = generate_audio(&c, &segs, &out, &progress, None, None).unwrap();
        // Mock silence fails every audit: two rounds of regeneration, then kept.
        let msgs = msgs.into_inner().unwrap();
        assert_eq!(
            msgs.iter()
                .filter(|m| m.contains("synthesizing them again"))
                .count(),
            CLEANUP_ROUNDS
        );
        assert!(
            msgs.iter()
                .any(|m| m.contains("kept the chunks of the last attempt"))
        );
        assert!(out.join("Kurz/chunks/chunk_000000.wav").exists());
        assert!(res["Kurz"].is_some());
        let spoken = std::fs::read_to_string(out.join("Kurz/text/spoken.txt")).unwrap();
        assert!(
            spoken.contains("000000 | Am dritten Mai neunzehnhundertfünfundneunzig war es kalt."),
            "{spoken}"
        );
    }

    #[test]
    fn stop_before_start_returns_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let c = cfg(dir.path());
        std::fs::write(c.paths.input_dir.join("a.md"), "Text.").unwrap();
        let stop = StopFlag::new();
        stop.set();
        let r = prepare_segments(
            &c,
            &c.paths.input_dir.clone(),
            &c.paths.output_dir.clone(),
            &no_progress,
            Some(&stop),
            None,
        )
        .unwrap();
        assert!(r.is_empty());
    }
}
