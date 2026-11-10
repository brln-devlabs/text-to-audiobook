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

//! Desktop GUI (egui): project list, voice description, quick config,
//! the four stage buttons, stop controls, settings and the live log.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use eframe::egui::{self, Color32, RichText};
use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::audio::cleanup::ProjectCleanup;
use crate::cancel::StopFlag;
use crate::config::{self, Config, Settings};
use crate::logging;
use crate::pipeline::{self, SegmentsByProject};
use crate::tts::TtsEngine;

const MAX_LOG_CHARS: usize = 400_000;

const RED: Color32 = Color32::from_rgb(0xa8, 0x32, 0x32);
const ORANGE: Color32 = Color32::from_rgb(0xa8, 0x78, 0x32);
const PURPLE: Color32 = Color32::from_rgb(0x5a, 0x32, 0x80);
const BLUE: Color32 = Color32::from_rgb(0x2a, 0x5a, 0x8a);
const GREEN: Color32 = Color32::from_rgb(0x2a, 0x6a, 0x2a);

/// Start the GUI (blocks until the window is closed).
pub fn launch(cfg: Config, input_dir: PathBuf, output_dir: PathBuf) -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("AI TTS Pipeline")
            .with_inner_size([1000.0, 860.0])
            .with_min_inner_size([780.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native(
        "AI TTS Pipeline",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, cfg, input_dir, output_dir)))),
    )
    .map_err(|e| anyhow::anyhow!("GUI failed: {e}"))
}

/// Results that worker threads hand back to the UI thread.
enum Event {
    Stage1Done(SegmentsByProject),
    Stage2Done,
    CleanupDone(IndexMap<String, ProjectCleanup>),
    RefreshProjects,
}

/// State shared with worker threads.
#[derive(Default)]
struct Shared {
    busy: AtomicBool,
    progress: Mutex<f32>,
    label: Mutex<String>,
    events: Mutex<Vec<Event>>,
}

impl Shared {
    fn set_progress(&self, v: f32) {
        *self.progress.lock().unwrap() = v.clamp(0.0, 1.0);
    }

    fn bump_progress(&self, step: f32, cap: f32) {
        let mut p = self.progress.lock().unwrap();
        *p = (*p + step).min(cap);
    }

    fn set_label(&self, s: impl Into<String>) {
        *self.label.lock().unwrap() = s.into();
    }

    fn push(&self, e: Event) {
        self.events.lock().unwrap().push(e);
    }
}

struct ProjectRow {
    name: String,
    label: String,
    selected: bool,
}

/// One editable settings row.
struct Field {
    key: &'static str,
    label: &'static str,
    value: String,
    choices: Option<Vec<String>>,
}

struct SettingsForm {
    sections: Vec<(&'static str, Vec<Field>)>,
}

fn tts_model_choices(current_id: &str) -> Vec<String> {
    let mut labels: Vec<String> = config::TTS_MODEL_CHOICES
        .iter()
        .map(|(l, _)| l.to_string())
        .collect();
    let current = config::tts_model_label(current_id);
    if !labels.contains(&current) {
        labels.push(current);
    }
    labels
}

fn value_text(v: Option<Value>) -> String {
    match v {
        Some(Value::String(s)) => s,
        Some(Value::Bool(b)) => b.to_string(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| s.to_string()).collect()
}

impl SettingsForm {
    fn new(s: &Settings) -> Self {
        let field = |key: &'static str, label: &'static str, choices: Option<Vec<String>>| {
            let value = if key == "TTS_MODEL_NAME" {
                config::tts_model_label(&s.tts_model_name)
            } else {
                value_text(s.get_value(key))
            };
            Field {
                key,
                label,
                value,
                choices,
            }
        };
        let bools = || Some(strings(&["true", "false"]));
        let levels = || Some(strings(config::LOG_LEVEL_CHOICES));
        Self {
            sections: vec![
                (
                    "Text Processing",
                    vec![
                        field("MAX_WORDS_PER_CHUNK", "Max words per chunk", None),
                        field("NORMALIZE_NUMBERS", "Spell out numbers (German)", bools()),
                    ],
                ),
                (
                    "LLM",
                    vec![
                        field("USE_LLM", "Use LLM (emotions, titles, order)", bools()),
                        field("LLM_MAX_RETRIES", "LLM retry attempts (per call)", None),
                        field("LLM_GPU_LAYERS", "LLM GPU layers (0 = CPU, -1 = all)", None),
                    ],
                ),
                (
                    "Storage",
                    vec![field(
                        "STORAGE_BACKEND",
                        "Storage backend",
                        Some(strings(config::STORAGE_BACKEND_CHOICES)),
                    )],
                ),
                (
                    "TTS",
                    vec![
                        field(
                            "TTS_MODEL_NAME",
                            "TTS model",
                            Some(tts_model_choices(&s.tts_model_name)),
                        ),
                        field(
                            "TTS_LANGUAGE",
                            "Language",
                            Some(strings(config::TTS_LANGUAGE_CHOICES)),
                        ),
                        field(
                            "TTS_DTYPE",
                            "Precision (GPU)",
                            Some(strings(config::TTS_DTYPE_CHOICES)),
                        ),
                        field("TTS_MAX_TOKENS", "Max tokens per chunk", None),
                        field(
                            "TTS_MAX_WORDS_FALLBACK",
                            "Max words per chunk (no tokenizer)",
                            None,
                        ),
                    ],
                ),
                (
                    "Audio",
                    vec![
                        field("VOICE_SEED", "Voice seed (RNG)", None),
                        field("SAMPLE_RATE", "Sample rate (Hz)", None),
                        field("CROSSFADE_MS", "Crossfade (ms)", None),
                        field(
                            "SILENCE_AFTER_PERIOD_MS",
                            "Silence after sentence (ms)",
                            None,
                        ),
                        field(
                            "SILENCE_AFTER_PARAGRAPH_MS",
                            "Silence after paragraph (ms)",
                            None,
                        ),
                    ],
                ),
                (
                    "Output Format",
                    vec![
                        field("CHAPTER_BY_CHAPTER", "Chapter-by-chapter files", bools()),
                        field("VOCALIZE_CHAPTER_NAMES", "Vocalize chapter names", bools()),
                    ],
                ),
                (
                    "Logging",
                    vec![
                        field("LOG_LEVEL_CONSOLE", "Console log level", levels()),
                        field("LOG_LEVEL_FILE", "File log level", levels()),
                    ],
                ),
            ],
        }
    }

    fn fields(&self) -> impl Iterator<Item = &Field> {
        self.sections.iter().flat_map(|(_, f)| f.iter())
    }
}

/// Parse GUI field values into settings. Invalid numbers are skipped with a
/// warning; returns the applied values for `settings.json`.
fn apply_fields<'a>(
    settings: &mut Settings,
    fields: impl Iterator<Item = (&'a str, &'a str)>,
    origin: &str,
) -> Map<String, Value> {
    let int_keys = [
        "MAX_WORDS_PER_CHUNK",
        "VOICE_SEED",
        "SAMPLE_RATE",
        "CROSSFADE_MS",
        "SILENCE_AFTER_PERIOD_MS",
        "SILENCE_AFTER_PARAGRAPH_MS",
        "LLM_MAX_RETRIES",
        "TTS_MAX_TOKENS",
        "TTS_MAX_WORDS_FALLBACK",
        "LLM_GPU_LAYERS",
    ];
    let bool_keys = [
        "CHAPTER_BY_CHAPTER",
        "VOCALIZE_CHAPTER_NAMES",
        "USE_LLM",
        "NORMALIZE_NUMBERS",
    ];
    let mut applied = Map::new();
    for (key, raw) in fields {
        let raw = raw.trim();
        let value = if key == "TTS_MODEL_NAME" {
            Value::from(config::tts_model_id(raw))
        } else if int_keys.contains(&key) {
            match raw.parse::<i64>() {
                Ok(i) => Value::from(i),
                Err(_) => {
                    log::warn!("{origin}: invalid integer for {key}: {raw:?} — skipped");
                    continue;
                }
            }
        } else if bool_keys.contains(&key) {
            Value::from(matches!(raw.to_lowercase().as_str(), "true" | "1" | "yes"))
        } else {
            Value::from(raw)
        };
        if settings.apply_value(key, &value) {
            applied.insert(key.to_string(), value);
        } else {
            log::warn!("{origin}: invalid value for {key}: {raw:?} — skipped");
        }
    }
    applied
}

struct QuickConfig {
    model_label: String,
    fields: Vec<(&'static str, &'static str, String)>,
}

impl QuickConfig {
    fn new(s: &Settings) -> Self {
        let defs: [(&str, &str); 6] = [
            ("MAX_WORDS_PER_CHUNK", "Max words / chunk"),
            ("TTS_MAX_TOKENS", "TTS max tokens / chunk"),
            ("VOICE_SEED", "Voice seed"),
            ("CROSSFADE_MS", "Crossfade (ms)"),
            ("SILENCE_AFTER_PERIOD_MS", "Silence / sentence (ms)"),
            ("SILENCE_AFTER_PARAGRAPH_MS", "Silence / paragraph (ms)"),
        ];
        Self {
            model_label: config::tts_model_label(&s.tts_model_name),
            fields: defs
                .iter()
                .map(|(k, l)| (*k, *l, value_text(s.get_value(k))))
                .collect(),
        }
    }
}

pub struct App {
    cfg: Config,
    input_dir: PathBuf,
    output_dir: PathBuf,
    projects: Vec<ProjectRow>,
    segments: SegmentsByProject,
    voice_prompt: String,
    quick: QuickConfig,
    settings_form: Option<SettingsForm>,
    cleanup_report: Option<IndexMap<String, ProjectCleanup>>,
    confirm_close: bool,
    allow_close: bool,
    shared: Arc<Shared>,
    stop: StopFlag,
    stop_clicked: bool,
    force_clicked: bool,
    log_rx: Receiver<String>,
    log_text: String,
    ctx: egui::Context,
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        cfg: Config,
        input_dir: PathBuf,
        output_dir: PathBuf,
    ) -> Self {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        let (tx, rx) = channel::<String>();
        let ctx = cc.egui_ctx.clone();
        let repaint = ctx.clone();
        logging::attach_gui_handler(move |line| {
            let _ = tx.send(line);
            repaint.request_repaint();
        });
        let mut app = Self {
            voice_prompt: cfg.settings.narrator_prompt.clone(),
            quick: QuickConfig::new(&cfg.settings),
            cfg,
            input_dir,
            output_dir,
            projects: Vec::new(),
            segments: IndexMap::new(),
            settings_form: None,
            cleanup_report: None,
            confirm_close: false,
            allow_close: false,
            shared: Arc::new(Shared::default()),
            stop: StopFlag::new(),
            stop_clicked: false,
            force_clicked: false,
            log_rx: rx,
            log_text: String::new(),
            ctx,
        };
        app.refresh_project_list();
        app.check_existing_segments();
        app
    }

    fn busy(&self) -> bool {
        self.shared.busy.load(Ordering::SeqCst)
    }

    fn selected_names(&self) -> Vec<String> {
        self.projects
            .iter()
            .filter(|p| p.selected)
            .map(|p| p.name.clone())
            .collect()
    }

    // ---------------------------------------------------------------
    // Projects and cache
    // ---------------------------------------------------------------

    fn refresh_project_list(&mut self) {
        let previous: IndexMap<String, bool> = self
            .projects
            .iter()
            .map(|p| (p.name.clone(), p.selected))
            .collect();
        self.projects = pipeline::discover_projects(&self.input_dir)
            .into_iter()
            .map(|p| {
                let cached = pipeline::load_cached_segments(
                    &self.cfg.paths.project_cache_dir(&p.name),
                    &self.cfg.settings.storage_backend,
                );
                let suffix = if cached.is_empty() {
                    format!("  ({} file(s))", p.files.len())
                } else {
                    format!("  [cached: {} segments]", cached.len())
                };
                ProjectRow {
                    label: format!("{}{suffix}", p.name),
                    selected: previous.get(&p.name).copied().unwrap_or(true),
                    name: p.name,
                }
            })
            .collect();
    }

    fn check_existing_segments(&mut self) {
        let found = pipeline::cached_segments_by_project(&self.cfg, &self.input_dir);
        if !found.is_empty() {
            let total: usize = found.values().map(Vec::len).sum();
            log::info!(
                "Found cached segments: {} project(s), {total} total — Stage 2 is ready.",
                found.len()
            );
        }
        self.segments = found;
    }

    // ---------------------------------------------------------------
    // Worker plumbing
    // ---------------------------------------------------------------

    fn reset_progress(&self) {
        self.shared.set_progress(0.0);
        self.shared.set_label("");
    }

    /// Run `work` on a background thread with the busy state set.
    fn spawn(&mut self, work: impl FnOnce(Config, StopFlag, Arc<Shared>) + Send + 'static) {
        self.shared.busy.store(true, Ordering::SeqCst);
        self.stop_clicked = false;
        self.force_clicked = false;
        let cfg = self.cfg.clone();
        let stop = self.stop.clone();
        let shared = self.shared.clone();
        let ctx = self.ctx.clone();
        std::thread::spawn(move || {
            let s2 = shared.clone();
            work(cfg, stop, shared);
            s2.busy.store(false, Ordering::SeqCst);
            ctx.request_repaint();
        });
    }

    fn start_stage1(&mut self) {
        let selected = self.selected_names();
        if selected.is_empty() {
            log::warn!("No stories selected. Tick at least one project to process.");
            return;
        }
        self.stop.clear();
        self.log_text.clear();
        self.reset_progress();
        log::info!(
            "Stage 1: processing {} project(s): {}",
            selected.len(),
            selected.join(", ")
        );
        let input = self.input_dir.clone();
        self.spawn(move |cfg, stop, shared| {
            let progress = |_: &str| shared.bump_progress(0.02, 0.95);
            match pipeline::prepare_segments(&cfg, &input, &progress, Some(&stop), Some(&selected))
            {
                Ok(segs) => {
                    shared.push(Event::Stage1Done(segs));
                    shared.push(Event::RefreshProjects);
                    shared.set_progress(1.0);
                }
                Err(e) => log::error!("Stage 1 failed: {e:#}"),
            }
        });
    }

    fn on_stage1_done(&mut self, segs: SegmentsByProject) {
        let total: usize = segs.values().map(Vec::len).sum();
        if self.stop.is_set() {
            log::info!("Stage 1 stopped: {total} segment(s) cached so far.");
        }
        if segs.is_empty() || total == 0 {
            log::info!("Stage 1 produced no segments");
            return;
        }
        for (name, s) in &segs {
            log::info!("  - {name}: {} segments", s.len());
        }
        if !self.stop.is_set() {
            log::info!(
                "Stage 1 done: {} project(s), {total} segments. Click '2. Generate Audio' when ready.",
                segs.len()
            );
        }
        for (name, s) in segs {
            self.segments.insert(name, s);
        }
    }

    fn selected_segments(&self) -> SegmentsByProject {
        let selected = self.selected_names();
        self.segments
            .iter()
            .filter(|(n, _)| selected.is_empty() || selected.contains(n))
            .map(|(n, s)| (n.clone(), s.clone()))
            .collect()
    }

    fn start_stage2(&mut self) {
        if self.segments.is_empty() {
            log::warn!("No segments to generate; run Stage 1 first.");
            return;
        }
        let to_process = self.selected_segments();
        if to_process.is_empty() {
            log::warn!("No selected projects have cached segments to synthesize.");
            return;
        }
        let skipped: Vec<&String> = self
            .segments
            .keys()
            .filter(|n| !to_process.contains_key(*n))
            .collect();
        if !skipped.is_empty() {
            log::info!(
                "Skipping {} unselected project(s): {}",
                skipped.len(),
                skipped
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        self.stop.clear();
        self.reset_progress();
        let total: usize = to_process.values().map(Vec::len).sum();
        let output = self.output_dir.clone();
        self.spawn(move |cfg, stop, shared| {
            let start = Instant::now();
            let done = AtomicUsize::new(0);
            let on_chunk = || {
                let d = done.fetch_add(1, Ordering::SeqCst) + 1;
                shared.set_progress(0.85 * d as f32 / total.max(1) as f32);
                shared.set_label(progress_label(d, total, Some(start)));
            };
            let result = pipeline::generate_audio(
                &cfg,
                &to_process,
                &output,
                &pipeline::no_progress,
                Some(&stop),
                Some(&on_chunk),
            );
            match result {
                Ok(_) => {
                    shared.set_progress(1.0);
                    shared.set_label(progress_label(total, total, None));
                    if stop.is_forced() {
                        log::warn!("Stage 2 force-stopped.");
                    } else if stop.is_set() {
                        log::info!("Stage 2 stopped by user — partial audio saved");
                    } else {
                        log::info!("Pipeline completed successfully");
                    }
                }
                Err(e) if e.is::<crate::cancel::ForceStopped>() => {
                    log::warn!("Stage 2 force-stopped. The partial chunk was removed.");
                }
                Err(e) => log::error!("Stage 2 failed: {e:#}"),
            }
            shared.push(Event::Stage2Done);
        });
    }

    fn start_cleanup(&mut self) {
        let selected = self.selected_names();
        if selected.is_empty() {
            log::warn!("No stories selected. Tick at least one project to clean.");
            return;
        }
        self.stop.clear();
        self.reset_progress();
        log::info!(
            "Stage 3 (Cleanup): scanning {} project(s) for noisy chunks...",
            selected.len()
        );
        let output = self.output_dir.clone();
        self.spawn(move |cfg, _stop, shared| {
            let progress = |_: &str| shared.bump_progress(0.05, 0.95);
            let results = pipeline::cleanup_noisy_chunks(&cfg, &output, Some(&selected), &progress);
            let edited: usize = results.values().map(|r| r.edited).sum();
            let deleted: usize = results.values().map(|r| r.deleted.len()).sum();
            if deleted > 0 {
                log::info!(
                    "Cleanup done: {edited} cleaned, {deleted} deleted. Re-run Stage 2 to re-synthesize the deleted chunks, then Stage 4 to assemble."
                );
            } else {
                log::info!("Cleanup done: {edited} chunk(s) cleaned, none deleted.");
            }
            shared.set_progress(1.0);
            shared.push(Event::CleanupDone(results));
        });
    }

    fn start_assemble_final(&mut self) {
        if self.segments.is_empty() {
            log::warn!("No segments loaded; run Stage 1 first.");
            return;
        }
        let to_process = self.selected_segments();
        if to_process.is_empty() {
            log::warn!("No selected projects have cached segments.");
            return;
        }
        self.stop.clear();
        self.reset_progress();
        log::info!(
            "Stage 4 (Assemble Final): assembling {} project(s)...",
            to_process.len()
        );
        let output = self.output_dir.clone();
        self.spawn(move |cfg, stop, shared| {
            let progress = |_: &str| shared.bump_progress(0.05, 0.95);
            match pipeline::assemble_final(&cfg, &to_process, &output, &progress, Some(&stop)) {
                Ok(_) => {
                    shared.set_progress(1.0);
                    if stop.is_set() {
                        log::info!("Assemble Final stopped by user.");
                    } else {
                        log::info!("Assemble Final completed successfully.");
                    }
                }
                Err(e) => log::error!("Assemble Final failed: {e:#}"),
            }
        });
    }

    fn fix_languages(&mut self) {
        let selected = self.selected_names();
        if selected.is_empty() {
            log::warn!("No stories selected for language fix.");
            return;
        }
        self.spawn(move |cfg, _stop, shared| {
            if let Err(e) = pipeline::fix_languages(&cfg, &selected) {
                log::error!("Language fix failed: {e:#}");
            }
            shared.push(Event::RefreshProjects);
        });
    }

    fn regen_voice_reference(&mut self) {
        self.reset_progress();
        log::info!("Regenerating voice reference...");
        self.spawn(move |cfg, _stop, shared| {
            match TtsEngine::regenerate_voice_reference(&cfg.settings, &cfg.paths) {
                Ok(p) => {
                    shared.set_progress(1.0);
                    log::info!(
                        "Voice reference saved to {}. Delete existing chunks and re-run Stage 2 to apply.",
                        p.display()
                    );
                }
                Err(e) => log::error!("Voice reference generation failed: {e:#}"),
            }
        });
    }

    fn request_stop(&mut self) {
        self.stop.set();
        self.stop_clicked = true;
        log::info!("Stop requested — finishing current segment then halting...");
    }

    fn force_stop(&mut self) {
        if !self.busy() {
            return;
        }
        log::warn!(
            "Force stop requested — abandoning the current segment at the next possible point \
             (a model call already running finishes first)..."
        );
        self.stop.force();
        self.force_clicked = true;
        self.stop_clicked = true;
    }

    fn clear_cache(&mut self) {
        let selected = self.selected_names();
        let projects: Vec<_> = pipeline::discover_projects(&self.input_dir)
            .into_iter()
            .filter(|p| selected.is_empty() || selected.contains(&p.name))
            .collect();
        if projects.is_empty() {
            log::info!("No projects selected — nothing to clear.");
            return;
        }
        let mut cleared = 0;
        for p in &projects {
            let dir = self.cfg.paths.project_cache_dir(&p.name);
            if dir.exists() {
                match std::fs::remove_dir_all(&dir) {
                    Ok(()) => {
                        log::info!("Cleared cache: {}", dir.display());
                        cleared += 1;
                    }
                    Err(e) => log::error!("Failed to clear cache for {}: {e}", p.name),
                }
            }
        }
        log::info!(
            "Cleared cache for {cleared}/{} selected project(s)",
            projects.len()
        );
        for name in &selected {
            self.segments.shift_remove(name);
        }
        self.reset_progress();
        self.refresh_project_list();
    }

    fn clean_output(&mut self) {
        let selected = self.selected_names();
        let projects: Vec<_> = pipeline::discover_projects(&self.input_dir)
            .into_iter()
            .filter(|p| selected.is_empty() || selected.contains(&p.name))
            .collect();
        if projects.is_empty() {
            log::info!("No projects selected — nothing to clean.");
            return;
        }
        let mut cleaned = 0;
        for p in &projects {
            let dir = self.output_dir.join(&p.name);
            if dir.exists() {
                match std::fs::remove_dir_all(&dir) {
                    Ok(()) => {
                        log::info!("Cleaned output: {}", dir.display());
                        cleaned += 1;
                    }
                    Err(e) => log::error!("Failed to clean output for {}: {e}", p.name),
                }
            }
        }
        log::info!(
            "Cleaned output for {cleaned}/{} selected project(s)",
            projects.len()
        );
        self.reset_progress();
    }

    fn save_log(&self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Text files", &["txt"])
            .add_filter("All files", &["*"])
            .set_file_name("tts_log.txt")
            .save_file()
        else {
            return;
        };
        match std::fs::write(&path, &self.log_text) {
            Ok(()) => log::info!("Log saved to: {}", path.display()),
            Err(e) => log::error!("Failed to save log: {e}"),
        }
    }

    fn open_logs_folder(&self) {
        if let Err(e) = open_in_file_manager(&self.cfg.paths.logs_dir) {
            log::warn!("Could not open logs folder: {e}");
        }
    }

    fn apply_quick_config(&mut self) {
        let mut fields: Vec<(&str, &str)> =
            vec![("TTS_MODEL_NAME", self.quick.model_label.as_str())];
        fields.extend(self.quick.fields.iter().map(|(k, _, v)| (*k, v.as_str())));
        let applied = apply_fields(&mut self.cfg.settings, fields.into_iter(), "Quick config");
        if let Err(e) = config::save_settings_to_file(&applied, &self.cfg.paths.settings_file()) {
            log::error!("Could not save settings: {e:#}");
        }
        log::info!("Quick config applied.");
    }

    fn apply_settings_form(&mut self) {
        let Some(form) = self.settings_form.take() else {
            return;
        };
        let applied = apply_fields(
            &mut self.cfg.settings,
            form.fields().map(|f| (f.key, f.value.as_str())),
            "Settings",
        );
        logging::set_levels(
            &self.cfg.settings.log_level_console,
            &self.cfg.settings.log_level_file,
        );
        if let Err(e) = config::save_settings_to_file(&applied, &self.cfg.paths.settings_file()) {
            log::error!("Could not save settings: {e:#}");
        }
        self.quick = QuickConfig::new(&self.cfg.settings);
        log::info!("Settings saved and applied.");
    }

    fn reset_settings_to_defaults(&mut self) {
        let mock = self.cfg.settings.mock_models;
        let mut defaults = config::reset_to_defaults(&self.cfg.paths.settings_file());
        defaults.mock_models = mock;
        self.cfg.settings = defaults;
        self.settings_form = Some(SettingsForm::new(&self.cfg.settings));
        self.quick = QuickConfig::new(&self.cfg.settings);
        logging::set_levels(
            &self.cfg.settings.log_level_console,
            &self.cfg.settings.log_level_file,
        );
        log::info!("Settings reset to in-code defaults.");
    }

    // ---------------------------------------------------------------
    // Per-frame
    // ---------------------------------------------------------------

    fn drain(&mut self) {
        let mut added = false;
        while let Ok(line) = self.log_rx.try_recv() {
            self.log_text.push_str(&line);
            self.log_text.push('\n');
            added = true;
        }
        if added && self.log_text.len() > MAX_LOG_CHARS {
            let cut = self.log_text.len() - MAX_LOG_CHARS;
            let cut = self.log_text[cut..]
                .find('\n')
                .map(|i| cut + i + 1)
                .unwrap_or(cut);
            self.log_text.drain(..cut);
        }
        let events: Vec<Event> = std::mem::take(&mut *self.shared.events.lock().unwrap());
        for e in events {
            match e {
                Event::Stage1Done(segs) => self.on_stage1_done(segs),
                Event::Stage2Done => {}
                Event::CleanupDone(r) => self.cleanup_report = Some(r),
                Event::RefreshProjects => self.refresh_project_list(),
            }
        }
    }

    fn top_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        ui.vertical_centered(|ui| {
            ui.label(
                RichText::new("AI Text-to-Speech Pipeline")
                    .size(18.0)
                    .strong(),
            );
        });
        ui.add_space(4.0);
        for (label, is_input) in [("Input Directory:", true), ("Output Directory:", false)] {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(label);
                    let dir = if is_input {
                        &self.input_dir
                    } else {
                        &self.output_dir
                    };
                    ui.label(RichText::new(dir.display().to_string()).color(Color32::GRAY));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add_enabled(!busy, egui::Button::new("Browse")).clicked() {
                            let start = if is_input {
                                &self.input_dir
                            } else {
                                &self.output_dir
                            };
                            if let Some(p) =
                                rfd::FileDialog::new().set_directory(start).pick_folder()
                            {
                                if is_input {
                                    self.input_dir = p;
                                    self.refresh_project_list();
                                    self.check_existing_segments();
                                } else {
                                    self.output_dir = p;
                                }
                            }
                        }
                    });
                });
            });
        }
    }

    fn projects_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        ui.label(RichText::new("Stories to process:").strong());
        ui.horizontal(|ui| {
            if ui.button("Select All").clicked() {
                self.projects.iter_mut().for_each(|p| p.selected = true);
            }
            if ui.button("Select None").clicked() {
                self.projects.iter_mut().for_each(|p| p.selected = false);
            }
            if ui.button("Refresh").clicked() {
                self.refresh_project_list();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_enabled(!busy, egui::Button::new("Fix Languages").fill(GREEN))
                    .clicked()
                {
                    self.fix_languages();
                }
            });
        });
        egui::Frame::group(ui.style()).show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("projects")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if self.projects.is_empty() {
                        ui.label(
                            RichText::new("(no projects found in this directory)")
                                .color(Color32::GRAY),
                        );
                    }
                    for p in &mut self.projects {
                        ui.add_enabled(
                            !busy,
                            egui::Checkbox::new(&mut p.selected, p.label.as_str()),
                        );
                    }
                });
        });
    }

    fn side_panels_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        egui::CollapsingHeader::new(RichText::new("Voice Description").strong())
            .default_open(false)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Reset Default").clicked() {
                        self.voice_prompt = config::DEFAULT_NARRATOR_PROMPT.to_string();
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("Regen Voice Ref"))
                        .clicked()
                    {
                        self.regen_voice_reference();
                    }
                });
                ui.add(
                    egui::TextEdit::multiline(&mut self.voice_prompt)
                        .desired_rows(5)
                        .desired_width(f32::INFINITY),
                );
                ui.label(
                    RichText::new(
                        "The Base models clone cache/voice_reference.wav; replace that file \
                         to change the narrator.",
                    )
                    .small()
                    .color(Color32::GRAY),
                );
            });
        egui::CollapsingHeader::new(RichText::new("Quick Config").strong())
            .default_open(false)
            .show(ui, |ui| {
                egui::Grid::new("quick_config")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("TTS model");
                        egui::ComboBox::from_id_salt("quick_model")
                            .width(170.0)
                            .selected_text(self.quick.model_label.clone())
                            .show_ui(ui, |ui| {
                                for label in tts_model_choices(&self.cfg.settings.tts_model_name) {
                                    ui.selectable_value(
                                        &mut self.quick.model_label,
                                        label.clone(),
                                        label,
                                    );
                                }
                            });
                        ui.end_row();
                        for (_, label, value) in &mut self.quick.fields {
                            ui.label(*label);
                            ui.add(egui::TextEdit::singleline(value).desired_width(80.0));
                            ui.end_row();
                        }
                    });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                    if ui.button("Apply Config").clicked() {
                        self.apply_quick_config();
                    }
                });
            });
    }

    fn buttons_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.columns(4, |cols| {
                let full = |ui: &mut egui::Ui, enabled: bool, b: egui::Button| {
                    ui.add_enabled(enabled, b.min_size(egui::vec2(ui.available_width(), 28.0)))
                        .clicked()
                };
                if full(
                    &mut cols[0],
                    !busy,
                    egui::Button::new("1. Prepare Segments"),
                ) {
                    self.start_stage1();
                }
                if full(
                    &mut cols[1],
                    !busy && !self.segments.is_empty(),
                    egui::Button::new("2. Generate Audio"),
                ) {
                    self.start_stage2();
                }
                if full(
                    &mut cols[2],
                    !busy,
                    egui::Button::new("3. Cleanup Noisy").fill(BLUE),
                ) {
                    self.start_cleanup();
                }
                if full(
                    &mut cols[3],
                    !busy,
                    egui::Button::new("4. Assemble Final").fill(GREEN),
                ) {
                    self.start_assemble_final();
                }
            });
        });
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(
                        busy && !self.stop_clicked,
                        egui::Button::new("Stop").fill(ORANGE),
                    )
                    .clicked()
                {
                    self.request_stop();
                }
                if ui
                    .add_enabled(
                        busy && !self.force_clicked,
                        egui::Button::new("Force Stop").fill(RED),
                    )
                    .clicked()
                {
                    self.force_stop();
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add_enabled(!busy, egui::Button::new("Clear Cache").fill(RED))
                        .clicked()
                    {
                        self.clear_cache();
                    }
                    if ui
                        .add_enabled(!busy, egui::Button::new("Clean Output").fill(PURPLE))
                        .clicked()
                    {
                        self.clean_output();
                    }
                    if ui.button("⚙ Settings").clicked() {
                        self.settings_form = Some(SettingsForm::new(&self.cfg.settings));
                    }
                    if ui.button("💾 Save Log").clicked() {
                        self.save_log();
                    }
                    if ui.button("📂 Logs").clicked() {
                        self.open_logs_folder();
                    }
                });
            });
        });
        ui.horizontal(|ui| {
            let p = *self.shared.progress.lock().unwrap();
            let label = self.shared.label.lock().unwrap().clone();
            let w = (ui.available_width() - 200.0).max(100.0);
            ui.add(egui::ProgressBar::new(p).desired_width(w));
            ui.label(label);
        });
    }

    fn log_ui(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Log:").strong());
        egui::Frame::group(ui.style()).show(ui, |ui| {
            egui::ScrollArea::both()
                .id_salt("log")
                .auto_shrink([false, false])
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut self.log_text.as_str())
                            .font(egui::TextStyle::Monospace)
                            .desired_width(f32::INFINITY),
                    );
                });
        });
    }

    fn settings_window(&mut self, ctx: &egui::Context) {
        if self.settings_form.is_none() {
            return;
        }
        let mut open = true;
        let mut action: Option<&str> = None;
        egui::Window::new("Settings")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_size([480.0, 520.0])
            .show(ctx, |ui| {
                let form = self.settings_form.as_mut().unwrap();
                egui::ScrollArea::vertical()
                    .max_height(440.0)
                    .show(ui, |ui| {
                        for (title, fields) in &mut form.sections {
                            ui.add_space(8.0);
                            ui.label(RichText::new(*title).strong().size(13.0));
                            egui::Grid::new(*title)
                                .num_columns(2)
                                .min_col_width(220.0)
                                .show(ui, |ui| {
                                    for f in fields.iter_mut() {
                                        ui.label(f.label);
                                        match &f.choices {
                                            Some(choices) => {
                                                egui::ComboBox::from_id_salt(f.key)
                                                    .width(if f.key == "TTS_MODEL_NAME" {
                                                        170.0
                                                    } else {
                                                        120.0
                                                    })
                                                    .selected_text(f.value.clone())
                                                    .show_ui(ui, |ui| {
                                                        for c in choices {
                                                            ui.selectable_value(
                                                                &mut f.value,
                                                                c.clone(),
                                                                c,
                                                            );
                                                        }
                                                    });
                                            }
                                            None => {
                                                ui.add(
                                                    egui::TextEdit::singleline(&mut f.value)
                                                        .desired_width(120.0),
                                                );
                                            }
                                        }
                                        ui.end_row();
                                    }
                                });
                        }
                    });
                ui.separator();
                ui.horizontal(|ui| {
                    if ui.button("Apply & Save").clicked() {
                        action = Some("apply");
                    }
                    if ui
                        .add(egui::Button::new("Reset to Defaults").fill(PURPLE))
                        .clicked()
                    {
                        action = Some("reset");
                    }
                    if ui
                        .add(egui::Button::new("Cancel").fill(Color32::from_gray(0x66)))
                        .clicked()
                    {
                        action = Some("cancel");
                    }
                });
            });
        match action {
            Some("apply") => self.apply_settings_form(),
            Some("reset") => self.reset_settings_to_defaults(),
            Some(_) => self.settings_form = None,
            None if !open => self.settings_form = None,
            None => {}
        }
    }

    fn cleanup_report_window(&mut self, ctx: &egui::Context) {
        let Some(results) = &self.cleanup_report else {
            return;
        };
        let edited: usize = results.values().map(|r| r.edited).sum();
        let deleted: usize = results.values().map(|r| r.deleted.len()).sum();
        let mut header = format!(
            "Cleaned in place: {edited} chunk(s)\nDeleted (failed audit): {deleted} chunk(s)\n"
        );
        header.push_str(if deleted > 0 {
            "Re-run Stage 2 to regenerate deleted chunks, then Stage 4 to assemble."
        } else {
            "All chunks passed discriminator + volume + voice-match checks."
        });
        let mut lines = Vec::new();
        for (name, r) in results {
            lines.push(format!(
                "── {name} — cleaned {}, deleted {} ──",
                r.edited,
                r.deleted.len()
            ));
            if r.deleted.is_empty() {
                lines.push("  (none deleted)".into());
            }
            for (file, reason) in &r.deleted {
                lines.push(format!("  ✗ {file}    {reason}"));
            }
            lines.push(String::new());
        }
        let body = if lines.is_empty() {
            "No projects scanned.".to_string()
        } else {
            lines.join("\n")
        };
        let mut open = true;
        let mut close = false;
        egui::Window::new("Cleanup Report")
            .open(&mut open)
            .default_size([680.0, 460.0])
            .show(ctx, |ui| {
                ui.label(header);
                egui::ScrollArea::both().max_height(340.0).show(ui, |ui| {
                    ui.add(
                        egui::TextEdit::multiline(&mut body.as_str())
                            .font(egui::TextStyle::Monospace),
                    );
                });
                if ui.button("Close").clicked() {
                    close = true;
                }
            });
        if close || !open {
            self.cleanup_report = None;
        }
    }

    fn close_handling(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close && self.busy() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.confirm_close = true;
        }
        if !self.confirm_close {
            return;
        }
        let modal = egui::Modal::new(egui::Id::unique("confirm_close")).show(ctx, |ui| {
            ui.set_width(320.0);
            ui.label("A model is still running.\nQuit anyway?");
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new("Quit Anyway").fill(RED)).clicked() {
                    self.allow_close = true;
                    self.confirm_close = false;
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                }
                if ui.button("Cancel").clicked() {
                    self.confirm_close = false;
                }
            });
        });
        if modal.should_close() {
            self.confirm_close = false;
        }
    }
}

/// Progress label "Chunk 3 / 10  (ETA ~1m 5s)".
fn progress_label(done: usize, total: usize, start: Option<Instant>) -> String {
    if total == 0 {
        return String::new();
    }
    match start {
        Some(t) if done > 0 => {
            let avg = t.elapsed().as_secs_f64() / done as f64;
            let remaining = ((total.saturating_sub(done)) as f64 * avg) as u64;
            let (m, s) = (remaining / 60, remaining % 60);
            let eta = if m > 0 {
                format!("{m}m {s}s")
            } else {
                format!("{s}s")
            };
            format!("Chunk {done} / {total}  (ETA ~{eta})")
        }
        _ => format!("Chunk {done} / {total}"),
    }
}

fn open_in_file_manager(dir: &Path) -> std::io::Result<()> {
    let program = if cfg!(target_os = "windows") {
        "explorer"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    std::process::Command::new(program)
        .arg(dir)
        .spawn()
        .map(|_| ())
}

impl eframe::App for App {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.drain();
        if self.busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }

        egui::Panel::top("top").show(ui, |ui| self.top_ui(ui));
        egui::Panel::bottom("log")
            .resizable(true)
            .default_size(320.0)
            .min_size(120.0)
            .show(ui, |ui| self.log_ui(ui));
        egui::Panel::bottom("controls").show(ui, |ui| self.buttons_ui(ui));
        let side_width = (ui.available_width() * 0.42).max(300.0);
        egui::Panel::right("side")
            .resizable(true)
            .default_size(side_width)
            .min_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("side")
                    .show(ui, |ui| self.side_panels_ui(ui));
            });
        egui::CentralPanel::default().show(ui, |ui| self.projects_ui(ui));

        self.settings_window(&ctx);
        self.cleanup_report_window(&ctx);
        self.close_handling(&ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eta_label() {
        assert_eq!(progress_label(0, 0, None), "");
        assert_eq!(progress_label(2, 10, None), "Chunk 2 / 10");
        assert!(progress_label(2, 10, Some(Instant::now())).starts_with("Chunk 2 / 10  (ETA ~"));
    }

    #[test]
    fn fields_are_parsed_like_the_python_gui() {
        let mut s = Settings::default();
        let applied = apply_fields(
            &mut s,
            [
                ("TTS_MODEL_NAME", "1.7B (6 GB+ VRAM)"),
                ("VOICE_SEED", "7"),
                ("CROSSFADE_MS", "abc"),
                ("CHAPTER_BY_CHAPTER", "true"),
                ("TTS_LANGUAGE", "english"),
            ]
            .into_iter(),
            "test",
        );
        assert_eq!(s.tts_model_name, "Qwen/Qwen3-TTS-12Hz-1.7B-Base");
        assert_eq!(s.voice_seed, 7);
        assert_eq!(s.crossfade_ms, 80);
        assert!(s.chapter_by_chapter);
        assert_eq!(s.tts_language, "english");
        assert!(!applied.contains_key("CROSSFADE_MS"));
        assert_eq!(applied["VOICE_SEED"], 7);
    }

    #[test]
    fn settings_form_covers_all_keys() {
        let form = SettingsForm::new(&Settings::default());
        let keys: Vec<&str> = form.fields().map(|f| f.key).collect();
        for k in config::SETTINGS_KEYS {
            if *k != "NARRATOR_PROMPT" {
                assert!(keys.contains(k), "{k} missing");
            }
        }
        let model = form.fields().find(|f| f.key == "TTS_MODEL_NAME").unwrap();
        assert_eq!(model.value, "0.6B (4 GB+ VRAM)");
    }
}
