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
//! 2. generate audio (TTS per segment, denoise, assemble); meanwhile a
//!    provisional MP3 of the chunks so far is kept up to date ([`crate::preview`])
//! 3. cleanup (audit chunks, delete failures, cut noise)
//! 4. assemble final audio from existing chunks
//!
//! Every subfolder of the input directory is one project (book), every
//! loose text file (`.md` or `.txt`) is a project of its own.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use anyhow::Result;
use indexmap::IndexMap;
use regex::Regex;
use serde_json::{Value, json};

use crate::audio::assembler::{assemble_with_progress, chunk_id};
use crate::audio::cleanup::{self, ProjectCleanup, Thresholds};
use crate::audio::filter::denoise;
use crate::cancel::{StopFlag, forced, stopped};
use crate::chunker::{Chunk, split_to_chunks};
use crate::config::{Config, Settings};
use crate::llm::{FilePreview, LlmNormalizer, ProjectHints};
use crate::markdown_cleaner::{clean, clean_with, heading_text};
use crate::natsort::{natsorted, natural_cmp};
use crate::preview::{self, PreviewQueue};
use crate::pyval;
use crate::segment::Segment;
use crate::storage::create_storage;
use crate::tts::TtsEngine;

/// Progress callback for human-readable messages.
pub type Progress<'a> = &'a (dyn Fn(&str) + Sync);

/// A progress callback that does nothing (messages still go to the log).
pub fn no_progress(_: &str) {}

/// What stage 2 reports to a progress display ([`generate_audio`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Stage2Event<'a> {
    /// The synthesis of `project` begins; its chunks follow.
    Synthesis { project: &'a str },
    /// Chunk `index` of `project` has its audio. `spoken`: how long its
    /// synthesis took; None when an existing WAV was reused.
    Chunk {
        project: &'a str,
        index: usize,
        spoken: Option<Duration>,
    },
    /// The post-processing of `project` after its synthesis (audit,
    /// denoising, assembly): the step running now and how much of the
    /// whole post-processing is done (0 to 1).
    Post {
        project: &'a str,
        step: &'a str,
        done: f32,
    },
}

/// Receiver of [`Stage2Event`]s.
pub type Stage2Progress<'a> = &'a (dyn Fn(Stage2Event) + Sync);

/// Shares of audit, denoising and assembly in the post-processing bar,
/// as measured (release build) on 88 chunks of 25 s: 9.7 s, 1.3 s and
/// 18 s; about the same on 40 chunks of 10 s. Without `AUTO_CLEANUP` the
/// other two fill the bar.
const POST_AUDIT: f32 = 0.33;
const POST_DENOISE: f32 = 0.05;
const POST_ASSEMBLE: f32 = 0.62;

#[derive(Clone, Copy)]
enum PostStep {
    Audit,
    Denoise,
    Assemble,
}

/// Turns the steps of one project's post-processing into
/// [`Stage2Event::Post`].
struct PostProgress<'a> {
    project: &'a str,
    on_event: Option<Stage2Progress<'a>>,
    audit: bool,
}

impl PostProgress<'_> {
    /// `fraction` (0 to 1) of `step` is done; `what` names it.
    fn report(&self, step: PostStep, fraction: f32, what: &str) {
        let Some(on_event) = self.on_event else {
            return;
        };
        let audit = if self.audit { POST_AUDIT } else { 0.0 };
        let (start, width) = match step {
            PostStep::Audit => (0.0, audit),
            PostStep::Denoise => (audit, POST_DENOISE),
            PostStep::Assemble => (audit + POST_DENOISE, POST_ASSEMBLE),
        };
        let done =
            (start + width * fraction.clamp(0.0, 1.0)) / (audit + POST_DENOISE + POST_ASSEMBLE);
        on_event(Stage2Event::Post {
            project: self.project,
            step: what,
            done,
        });
    }

    /// Audit round `round` (from 1): `fraction` of its audit (`redo` false)
    /// or of synthesizing its failed chunks again (`redo` true). Each round
    /// has an equal share, the audit the first half of it.
    fn audit_round(&self, round: usize, redo: bool, fraction: f32, what: &str) {
        let within = if redo {
            0.5 + fraction / 2.0
        } else {
            fraction / 2.0
        };
        let done = ((round - 1) as f32 + within) / CLEANUP_ROUNDS as f32;
        self.report(PostStep::Audit, done, what);
    }
}

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

/// Extensions of the text files the pipeline reads (any case): Markdown and
/// plain text. Plain text is valid Markdown, so both are cleaned alike.
const TEXT_EXTENSIONS: [&str; 2] = ["md", "txt"];

fn is_text_file(p: &Path) -> bool {
    p.is_file()
        && p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| TEXT_EXTENSIONS.iter().any(|t| e.eq_ignore_ascii_case(t)))
}

fn is_markdown(p: &Path) -> bool {
    p.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("md"))
}

/// The text files among `entries` (one folder). A `.txt` next to a `.md` of
/// the same name (in any case) is left out: both would be the same chapter
/// or project.
fn text_files_in(entries: &[PathBuf]) -> Vec<PathBuf> {
    let files: Vec<&PathBuf> = entries.iter().filter(|p| is_text_file(p)).collect();
    let stem = |p: &Path| file_stem(p).to_lowercase();
    files
        .iter()
        .filter(|p| {
            let twin =
                !is_markdown(p) && files.iter().any(|o| is_markdown(o) && stem(o) == stem(p));
            if twin {
                log::warn!(
                    "Ignoring {}: a .md file of the same name is used",
                    p.display()
                );
            }
            !twin
        })
        .map(|p| (*p).clone())
        .collect()
}

/// All text files (`.md`, `.txt`) below `dir` (recursive).
fn md_files_recursive(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        let mut entries: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        entries.sort();
        out.extend(text_files_in(&entries));
        stack.extend(entries.into_iter().filter(|p| p.is_dir()));
    }
    out
}

/// Each subfolder of `input_dir` with text files is one project; each loose
/// text file (`.md`, `.txt`) is its own project.
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
    for f in text_files_in(&entries) {
        projects.push(Project {
            name: file_stem(&f),
            files: vec![f],
            input_dir: input_dir.to_path_buf(),
        });
    }
    projects
}

/// Name of a story typed in the GUI; it is saved as `<name>.md`. A typed
/// `.md` or `.txt` ending is dropped; names a file system would refuse (or
/// that Windows reserves) are errors.
pub fn story_name(typed: &str) -> Result<String> {
    let mut name = typed.trim();
    if let Some((stem, ext)) = name.rsplit_once('.')
        && TEXT_EXTENSIONS.iter().any(|t| ext.eq_ignore_ascii_case(t))
    {
        name = stem.trim_end();
    }
    if name.is_empty() {
        anyhow::bail!("Please enter a file name.");
    }
    if let Some(c) = name
        .chars()
        .find(|c| c.is_control() || r#"<>:"/\|?*"#.contains(*c))
    {
        anyhow::bail!("The file name must not contain '{c}'.");
    }
    if name.starts_with('.') || name.ends_with('.') {
        anyhow::bail!("The file name must not start or end with a dot.");
    }
    let base = name.split('.').next().unwrap_or(name).trim_end();
    let base = base.to_ascii_uppercase();
    let reserved = matches!(base.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (base.len() == 4
            && (base.starts_with("COM") || base.starts_with("LPT"))
            && matches!(base.as_bytes()[3], b'1'..=b'9'));
    if reserved {
        anyhow::bail!("\"{name}\" is a reserved name on Windows.");
    }
    Ok(name.to_string())
}

/// Save `text` as a new Markdown story `<name>.md` in `input_dir` (see
/// [`story_name`]). An existing story of that name is never replaced.
pub fn create_story(input_dir: &Path, typed_name: &str, text: &str) -> Result<PathBuf> {
    let name = story_name(typed_name)?;
    if text.trim().is_empty() {
        anyhow::bail!("Please enter the text of the story.");
    }
    // Case-insensitive: on Windows and macOS "story" and "Story" are one file.
    let taken = |other: &str| other.to_lowercase() == name.to_lowercase();
    let folder_taken = std::fs::read_dir(input_dir).is_ok_and(|entries| {
        entries
            .filter_map(|e| e.ok())
            .any(|e| e.path().is_dir() && taken(&e.file_name().to_string_lossy()))
    });
    if folder_taken || discover_projects(input_dir).iter().any(|p| taken(&p.name)) {
        anyhow::bail!("A story named \"{name}\" already exists.");
    }
    std::fs::create_dir_all(input_dir)?;
    let path = input_dir.join(format!("{name}.md"));
    let mut content = text.to_string();
    if !content.ends_with('\n') {
        content.push('\n');
    }
    let written = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, content.as_bytes()));
    match written {
        Ok(()) => Ok(path),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            anyhow::bail!("A file named \"{name}.md\" already exists.")
        }
        Err(e) => anyhow::bail!("Could not write {}: {e}", path.display()),
    }
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

/// Windows-1252 characters for the bytes 0x80..0x9F (the rest is Latin-1);
/// unused bytes become U+FFFD.
const CP1252_HIGH: [char; 32] = [
    '€', '\u{FFFD}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{FFFD}', 'Ž',
    '\u{FFFD}', '\u{FFFD}', '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{FFFD}',
    'ž', 'Ÿ',
];

/// Text of an input file. UTF-8 (a BOM is dropped); anything else is read as
/// Windows-1252, the "ANSI" encoding older Windows editors save `.txt` in.
fn read_text(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let bytes = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
    if let Ok(s) = std::str::from_utf8(bytes) {
        return Ok(s.to_string());
    }
    log::info!(
        "{} is not UTF-8, reading it as Windows-1252",
        path.display()
    );
    Ok(bytes
        .iter()
        .map(|&b| match b {
            0x80..=0x9F => CP1252_HIGH[(b - 0x80) as usize],
            _ => b as char,
        })
        .collect())
}

/// Read up to `n` characters from the start of a file.
fn read_head(path: &Path, n: usize) -> String {
    read_text(path)
        .map(|t| pyval::prefix(&t, n).to_string())
        .unwrap_or_default()
}

/// Narrative reading order of the text files of a project.
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
        let raw = read_text(f).map_err(|e| anyhow::anyhow!("reading {}: {e}", f.display()))?;
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
            "No projects found (no folders with .md/.txt files and no loose .md/.txt files)",
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

/// `<name>.mp3` in `final_path`, named like the input file (`Kapitel 1.md`
/// gives `Kapitel 1.mp3`). `.mp3` is appended rather than set with
/// `with_extension`, so a dot in the name stays.
fn output_file(final_path: &Path, name: &str) -> PathBuf {
    final_path.join(format!("{name}.mp3"))
}

/// One MP3 named like the input file (several files merged: like the
/// project folder), or one per source file, named like it, when the
/// project has several and `chapter_by_chapter` is set. The provisional
/// audiobook of stage 2 is deleted once they are written. `assembled`
/// hears how much is done (0 to 1) and what is being written.
#[allow(clippy::too_many_arguments)]
fn assemble_outputs(
    cfg: &Config,
    wav_paths: &[PathBuf],
    segments: &[Segment],
    final_path: &Path,
    project_name: &str,
    chapter_by_chapter: bool,
    progress: Progress,
    assembled: &dyn Fn(f32, &str),
) -> Result<Vec<PathBuf>> {
    let outputs = assemble_files(
        cfg,
        wav_paths,
        segments,
        final_path,
        project_name,
        chapter_by_chapter,
        progress,
        assembled,
    )?;
    if !outputs.is_empty() {
        preview::remove(final_path);
    }
    Ok(outputs)
}

#[allow(clippy::too_many_arguments)]
fn assemble_files(
    cfg: &Config,
    wav_paths: &[PathBuf],
    segments: &[Segment],
    final_path: &Path,
    project_name: &str,
    chapter_by_chapter: bool,
    progress: Progress,
    assembled: &dyn Fn(f32, &str),
) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(final_path)?;
    let by_source = group_by_source(segments);
    let mut outputs = Vec::new();
    if by_source.len() <= 1 || !chapter_by_chapter {
        let indices: Vec<usize> = by_source.values().flatten().copied().collect();
        let wavs = wavs_for_segments(wav_paths, &indices);
        // One input file: its name; several merged: the project's.
        let name = match by_source.keys().next() {
            Some(source) if by_source.len() == 1 && !source.is_empty() => source,
            _ => project_name,
        };
        if !wavs.is_empty() {
            let out = assemble_with_progress(
                &wavs,
                &ends_paragraph(segments, &indices),
                &output_file(final_path, name),
                &cfg.settings,
                &|f| assembled(f, "assembling the audiobook"),
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
    // Progress over the chapters, weighted by their number of WAVs.
    let chapters: Vec<Vec<PathBuf>> = by_source
        .values()
        .map(|indices| wavs_for_segments(wav_paths, indices))
        .collect();
    let all = chapters.iter().map(Vec::len).sum::<usize>().max(1) as f32;
    let mut before = 0;
    for (n, ((source, indices), wavs)) in by_source.iter().zip(&chapters).enumerate() {
        let chapter_num = n + 1;
        if wavs.is_empty() {
            emit(
                progress,
                &format!("  chapter {chapter_num} ({source}): no WAVs found"),
            );
            continue;
        }
        let what = format!("assembling chapter {chapter_num} / {}", by_source.len());
        let out = assemble_with_progress(
            wavs,
            &ends_paragraph(segments, indices),
            &if source.is_empty() {
                output_file(final_path, &format!("chapter_{chapter_num:02}"))
            } else {
                output_file(final_path, source)
            },
            &cfg.settings,
            &|f| assembled((before as f32 + f * wavs.len() as f32) / all, &what),
        )?;
        before += wavs.len();
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

/// [`TtsEngine::process_all`] while a second thread rebuilds the
/// provisional audiobook `preview_path` after every newly spoken chunk
/// (see [`crate::preview`]). The TTS does not wait for it. Every chunk is
/// reported as [`Stage2Event::Chunk`], with the time it took when spoken.
#[allow(clippy::too_many_arguments)]
fn synthesize_with_preview(
    cfg: &Config,
    project_name: &str,
    tts: &mut TtsEngine,
    segments: &[Segment],
    chunks_dir: &Path,
    preview_path: &Path,
    progress: Progress,
    stop: Option<&StopFlag>,
    on_event: Option<Stage2Progress>,
) -> Result<Vec<PathBuf>> {
    let queue = PreviewQueue::default();
    let report = |m: &str| emit(progress, &format!("[{project_name}] {m}"));
    std::thread::scope(|s| {
        let _end = preview::EndOnDrop(&queue);
        let worker =
            s.spawn(|| preview::run(&queue, &cfg.settings, segments, preview_path, stop, &report));
        if let Some(on_event) = on_event {
            on_event(Stage2Event::Synthesis {
                project: project_name,
            });
        }
        // A spoken chunk took the time since the report before it.
        let mut last = Instant::now();
        let result = tts.process_all_with(segments, chunks_dir, stop, &mut |c| {
            let took = last.elapsed();
            last = Instant::now();
            if let Some(on_event) = on_event {
                on_event(Stage2Event::Chunk {
                    project: project_name,
                    index: c.index,
                    spoken: c.synthesized.then_some(took),
                });
            }
            if c.synthesized {
                queue.push(c.wavs.to_vec());
            }
        });
        // Complete: the final assembly follows and replaces the preview.
        // Stopped or failed: bring it up to date, unless force-stopped.
        let keep = (result.is_err() || stopped(stop)) && !forced(stop);
        if keep {
            if queue.busy() {
                report("finishing the provisional audiobook...");
            }
            queue.close();
        } else {
            queue.cancel();
        }
        match worker.join() {
            Ok(Some(last)) if keep => report(&last),
            Ok(_) => {}
            Err(_) => log::warn!("[{project_name}] the provisional audiobook thread crashed"),
        }
        result
    })
}

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
    on_event: Option<Stage2Progress>,
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
        let preview_path = project_out.join("final").join(preview::PREVIEW_FILE);
        let post = PostProgress {
            project: project_name,
            on_event,
            audit: cfg.settings.auto_cleanup,
        };
        let wav_paths = (|| -> Result<Vec<PathBuf>> {
            let mut wavs = synthesize_with_preview(
                cfg,
                project_name,
                tts,
                segments,
                &chunks_dir,
                &preview_path,
                progress,
                stop,
                on_event,
            )?;
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
                let what = |done, total| {
                    let again = if round > 1 { " again" } else { "" };
                    format!("auditing the chunk files{again}: {done} / {total}")
                };
                post.audit_round(round, false, 0.0, &what(0, wavs.len()));
                let results = cleanup::cleanup_noisy_chunks_with(
                    output_root,
                    Some(&[project_name.to_string()]),
                    &report,
                    &cfg.paths.voice_ref_path(),
                    &Thresholds::default(),
                    &|done, total| {
                        post.audit_round(
                            round,
                            false,
                            done as f32 / total as f32,
                            &what(done, total),
                        )
                    },
                );
                let failed = results
                    .get(project_name)
                    .map_or(&[][..], |r| &r.deleted[..]);
                if failed.is_empty() {
                    emit(
                        progress,
                        &format!("[{project_name}] all chunks passed the audit"),
                    );
                    post.report(PostStep::Audit, 1.0, "all chunks passed the audit");
                    break;
                }
                emit(
                    progress,
                    &format!(
                        "[{project_name}] {} chunk file(s) failed, synthesizing them again with a new seed",
                        failed.len()
                    ),
                );
                // Chunks to speak again (a failed file takes its sub-parts along).
                let mut redo: Vec<usize> = failed
                    .iter()
                    .filter_map(|(name, _)| chunk_id(Path::new(name)))
                    .collect();
                redo.sort_unstable();
                redo.dedup();
                let what = |n: usize| {
                    format!(
                        "synthesizing {} failed chunk(s) again: {n} / {}",
                        redo.len(),
                        redo.len()
                    )
                };
                post.audit_round(round, true, 0.0, &what(0));
                let mut spoken = 0;
                tts.set_seed_offset(round as u64);
                let again = tts.process_all_with(segments, &chunks_dir, stop, &mut |c| {
                    if c.synthesized {
                        spoken += 1;
                        let fraction = spoken as f32 / redo.len().max(1) as f32;
                        post.audit_round(round, true, fraction, &what(spoken));
                    }
                });
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
        let n = wav_paths.len();
        for (i, w) in wav_paths.iter().enumerate() {
            if stopped(stop) {
                break;
            }
            post.report(
                PostStep::Denoise,
                i as f32 / n as f32,
                &format!("denoising {} / {n}", i + 1),
            );
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
            &|f, what| post.report(PostStep::Assemble, f, what),
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
    on_event: Option<Stage2Progress>,
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
                on_event,
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
        &|_, _| {},
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
        std::fs::write(input.join("notes.json"), "{}").unwrap();
        let p = discover_projects(input);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].name, "Book");
        assert_eq!(p[0].files.len(), 2);
        assert_eq!(p[1].name, "story");
        assert_eq!(p[1].input_dir, input);
        assert!(discover_projects(&input.join("missing")).is_empty());
    }

    #[test]
    fn txt_files_are_projects_and_chapters_too() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path();
        std::fs::create_dir_all(input.join("Book")).unwrap();
        std::fs::write(input.join("Book/01.md"), "A.").unwrap();
        std::fs::write(input.join("Book/02.txt"), "B.").unwrap();
        std::fs::write(input.join("Notiz.TXT"), "C.").unwrap();
        // The .md wins over a .txt of the same name.
        std::fs::write(input.join("story.md"), "D.").unwrap();
        std::fs::write(input.join("Story.txt"), "E.").unwrap();
        let p = discover_projects(input);
        let names: Vec<&str> = p.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["Book", "Notiz", "story"]);
        let book: Vec<String> = p[0].files.iter().map(|f| file_name(f)).collect();
        assert_eq!(book, ["01.md", "02.txt"]);
        assert_eq!(p[2].files, [input.join("story.md")]);
    }

    #[test]
    fn text_is_read_as_utf8_or_windows_1252() {
        let dir = tempfile::tempdir().unwrap();
        let utf8 = dir.path().join("utf8.txt");
        std::fs::write(&utf8, "\u{FEFF}# Größe\n„Text“ – 5 €").unwrap();
        assert_eq!(read_text(&utf8).unwrap(), "# Größe\n„Text“ – 5 €");
        // "Größe „Text“ – 5 €" saved by an old Windows editor ("ANSI").
        let ansi = dir.path().join("ansi.txt");
        std::fs::write(&ansi, b"Gr\xF6\xDFe \x84Text\x93 \x96 5 \x80").unwrap();
        assert_eq!(read_text(&ansi).unwrap(), "Größe „Text“ – 5 €");
    }

    #[test]
    fn story_names_are_checked() {
        assert_eq!(story_name("  Kapitel 1 ").unwrap(), "Kapitel 1");
        assert_eq!(story_name("Probe.md").unwrap(), "Probe");
        assert_eq!(story_name("Probe.TXT").unwrap(), "Probe");
        assert_eq!(story_name("Teil 1.2").unwrap(), "Teil 1.2");
        assert_eq!(story_name("Märchen").unwrap(), "Märchen");
        assert_eq!(story_name("Comet").unwrap(), "Comet");
        for bad in [
            "", "  ", ".md", "a/b", "a\\b", "Wer?", "a:b", ".hidden", "Ende.", "CON", "com1",
            "Lpt9.x",
        ] {
            assert!(story_name(bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn create_story_writes_markdown_once() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input");
        let path = create_story(&input, "Neue Geschichte.md", "Es war einmal.").unwrap();
        assert_eq!(path, input.join("Neue Geschichte.md"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "Es war einmal.\n");
        let p = discover_projects(&input);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].name, "Neue Geschichte");
        // Never replaced, whatever the case or text extension.
        assert!(create_story(&input, "neue geschichte", "Anders.").is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "Es war einmal.\n");
        std::fs::write(input.join("Alt.txt"), "Text.").unwrap();
        assert!(create_story(&input, "Alt", "Neu.").is_err());
        std::fs::create_dir_all(input.join("Buch")).unwrap();
        assert!(create_story(&input, "Buch", "Neu.").is_err());
        assert!(create_story(&input, "Leer", " \n ").is_err());
        assert!(!input.join("Leer.md").exists());
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

    /// A [`Stage2Event`] with owned strings; `Chunk`: spoken or reused.
    #[derive(Debug, Clone, PartialEq)]
    enum Seen {
        Chunk(String, usize, bool),
        Post(String, String, f32),
    }

    #[derive(Default)]
    struct Recorder(std::sync::Mutex<Vec<Seen>>);

    impl Recorder {
        fn cb(&self) -> impl Fn(Stage2Event) + Sync + '_ {
            |e| {
                let e = match e {
                    Stage2Event::Synthesis { .. } => return,
                    Stage2Event::Chunk {
                        project,
                        index,
                        spoken,
                    } => Seen::Chunk(project.into(), index, spoken.is_some()),
                    Stage2Event::Post {
                        project,
                        step,
                        done,
                    } => Seen::Post(project.into(), step.into(), done),
                };
                self.0.lock().unwrap().push(e);
            }
        }

        fn take(self) -> Vec<Seen> {
            self.0.into_inner().unwrap()
        }
    }

    /// The post-processing steps of `project`: (step, done).
    fn post_of(seen: &[Seen], project: &str) -> Vec<(String, f32)> {
        seen.iter()
            .filter_map(|e| match e {
                Seen::Post(p, step, done) if p == project => Some((step.clone(), *done)),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn audit_rounds_show_in_the_post_processing() {
        let dir = tempfile::tempdir().unwrap();
        let mut c = cfg(dir.path());
        // The mock TTS speaks silence, so every chunk fails both rounds.
        c.settings.auto_cleanup = true;
        std::fs::write(
            c.paths.input_dir.join("Kurz.md"),
            "Erster Absatz.\n\nZweiter Absatz.",
        )
        .unwrap();
        let input = c.paths.input_dir.clone();
        let out = c.paths.output_dir.clone();
        let segs = prepare_segments(&c, &input, &out, &no_progress, None, None).unwrap();
        assert_eq!(segs["Kurz"].len(), 2);
        let seen = Recorder::default();
        generate_audio(&c, &segs, &out, &no_progress, None, Some(&seen.cb())).unwrap();
        let seen = seen.take();
        // Speaking chunks again in the audit is post-processing, not new chunks.
        let chunks = seen.iter().filter(|e| matches!(e, Seen::Chunk(..))).count();
        assert_eq!(chunks, 2, "{seen:?}");
        let post = post_of(&seen, "Kurz");
        assert!(post.windows(2).all(|w| w[0].1 <= w[1].1), "{post:?}");
        // Where on the bar `step` was shown (the last time).
        let at = |step: &str| {
            post.iter()
                .rev()
                .find(|(s, _)| s == step)
                .unwrap_or_else(|| panic!("{step} missing: {post:?}"))
                .1
        };
        let near = |a: f32, b: f32| assert!((a - b).abs() < 1e-6, "{a} != {b}: {post:?}");
        // Two rounds, each an audit and then speaking the failures again.
        let audit = POST_AUDIT / (POST_AUDIT + POST_DENOISE + POST_ASSEMBLE);
        near(at("auditing the chunk files: 0 / 2"), 0.0);
        near(at("auditing the chunk files: 2 / 2"), audit * 0.25);
        near(at("auditing the chunk files again: 0 / 2"), audit * 0.5);
        near(at("auditing the chunk files again: 2 / 2"), audit * 0.75);
        near(
            at("synthesizing 2 failed chunk(s) again: 1 / 2"),
            audit * 0.875,
        );
        near(at("synthesizing 2 failed chunk(s) again: 2 / 2"), audit);
        near(at("denoising 1 / 2"), audit);
        assert_eq!(
            post.last().unwrap(),
            &("assembling the audiobook".to_string(), 1.0)
        );
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
        let seen = Recorder::default();
        let res = generate_audio(&c, &segs, &out, &no_progress, None, Some(&seen.cb())).unwrap();
        let seen = seen.take();
        let chunks: Vec<(String, usize, bool)> = seen
            .iter()
            .filter_map(|e| match e {
                Seen::Chunk(p, i, spoken) => Some((p.clone(), *i, *spoken)),
                Seen::Post(..) => None,
            })
            .collect();
        let chunk = |p: &str, i| (p.to_string(), i, true);
        assert_eq!(
            chunks,
            [
                chunk("Book", 0),
                chunk("Book", 1),
                chunk("Book", 2),
                chunk("Kurz", 0)
            ]
        );
        // Each project: its chunks, then its post-processing up to 1.
        let kinds: Vec<&str> = seen
            .iter()
            .map(|e| match e {
                Seen::Chunk(p, ..) | Seen::Post(p, ..) => p.as_str(),
            })
            .collect();
        let book = kinds.iter().filter(|p| **p == "Book").count();
        assert!(kinds[..book].iter().all(|p| *p == "Book"), "{seen:?}");
        for project in ["Book", "Kurz"] {
            let post = post_of(&seen, project);
            assert!(post.windows(2).all(|w| w[0].1 <= w[1].1), "{post:?}");
            assert_eq!(post.last().unwrap().1, 1.0, "{post:?}");
        }
        assert!(
            post_of(&seen, "Book")
                .iter()
                .any(|(s, _)| s == "assembling chapter 2 / 2")
        );
        assert!(
            post_of(&seen, "Kurz")
                .iter()
                .any(|(s, _)| s == "denoising 1 / 1")
        );

        // Run again: every chunk is reused, none reported as spoken.
        let seen = Recorder::default();
        generate_audio(&c, &segs, &out, &no_progress, None, Some(&seen.cb())).unwrap();
        let seen = seen.take();
        let reused = seen
            .iter()
            .filter(|e| matches!(e, Seen::Chunk(_, _, false)))
            .count();
        assert_eq!(reused, 4, "{seen:?}");
        assert!(!seen.iter().any(|e| matches!(e, Seen::Chunk(_, _, true))));
        assert_eq!(
            res["Book"].as_deref(),
            Some(out.join("Book/final/1_Anfang.mp3").as_path())
        );
        assert!(out.join("Book/final/2_Ende.wav").exists());
        assert!(out.join("Kurz/final/Kurz.mp3").exists());
        assert!(out.join("Book/chunks/chunk_000002.wav").exists());

        // Stage 4 without TTS, as one file.
        c.settings.chapter_by_chapter = false;
        let res = assemble_final(&c, &segs, &out, &no_progress, None).unwrap();
        assert_eq!(
            res["Book"].as_deref(),
            Some(out.join("Book/final/Book.mp3").as_path())
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
    fn provisional_audiobook_follows_the_synthesis() {
        let dir = tempfile::tempdir().unwrap();
        let c = cfg(dir.path());
        std::fs::write(
            c.paths.input_dir.join("Buch.md"),
            "Erster Absatz.\n\nZweiter Absatz.\n\nDritter Absatz.",
        )
        .unwrap();
        let input = c.paths.input_dir.clone();
        let out = c.paths.output_dir.clone();
        let segs = prepare_segments(&c, &input, &out, &no_progress, None, None).unwrap();
        assert_eq!(segs["Buch"].len(), 3);
        let final_dir = out.join("Buch/final");
        let preview_mp3 = final_dir.join(preview::PREVIEW_FILE);

        // Stopped after two chunks: the preview holds exactly those two.
        let stop = StopFlag::new();
        let done = std::sync::atomic::AtomicUsize::new(0);
        let cb = |e: Stage2Event| {
            if matches!(e, Stage2Event::Chunk { .. })
                && done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 1
            {
                stop.set();
            }
        };
        let msgs = std::sync::Mutex::new(Vec::<String>::new());
        let progress = |m: &str| msgs.lock().unwrap().push(m.to_string());
        let res = generate_audio(&c, &segs, &out, &progress, Some(&stop), Some(&cb)).unwrap();
        assert!(res.is_empty() || res["Buch"].is_none());
        assert!(preview_mp3.exists());
        assert!(!crate::audio::assembler::part_path(&preview_mp3).exists());
        assert!(!final_dir.join("Buch.mp3").exists());
        let chunks = out.join("Buch/chunks");
        let two = vec![
            chunks.join("chunk_000000.wav"),
            chunks.join("chunk_000001.wav"),
        ];
        let reference = dir.path().join("two.mp3");
        crate::audio::assembler::assemble_preview(
            &two,
            &[true, true],
            &reference,
            &c.settings,
            &|| false,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            std::fs::read(&preview_mp3).unwrap(),
            std::fs::read(&reference).unwrap()
        );
        let msgs = msgs.into_inner().unwrap();
        assert!(
            msgs.iter()
                .any(|m| m.contains("provisional audiobook (2 of 3 chunks")),
            "{msgs:?}"
        );

        // Resumed to the end: the final file replaces the preview.
        let res = generate_audio(&c, &segs, &out, &no_progress, None, None).unwrap();
        assert!(res["Buch"].is_some());
        assert!(final_dir.join("Buch.mp3").exists());
        assert!(!preview_mp3.exists());
    }

    #[test]
    fn output_is_named_like_the_input_file() {
        let dir = tempfile::tempdir().unwrap();
        let c = cfg(dir.path());
        // A dot in the name stays; a folder with one file takes its name.
        std::fs::write(c.paths.input_dir.join("Teil 1.2.md"), "Text.").unwrap();
        let folder = c.paths.input_dir.join("Ordner");
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("Geschichte.md"), "Text.").unwrap();
        let input = c.paths.input_dir.clone();
        let out = c.paths.output_dir.clone();
        let segs = prepare_segments(&c, &input, &out, &no_progress, None, None).unwrap();
        let res = generate_audio(&c, &segs, &out, &no_progress, None, None).unwrap();
        assert_eq!(
            res["Teil 1.2"].as_deref(),
            Some(out.join("Teil 1.2/final/Teil 1.2.mp3").as_path())
        );
        assert!(out.join("Teil 1.2/final/Teil 1.2.wav").exists());
        assert_eq!(
            res["Ordner"].as_deref(),
            Some(out.join("Ordner/final/Geschichte.mp3").as_path())
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
