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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use eframe::egui::{self, Color32, RichText};
use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::cancel::StopFlag;
use crate::config::{self, Config, Settings};
use crate::logging;
use crate::pipeline::{self, SegmentsByProject, Stage2Event};
use crate::review::{self, ReviewStore};
use crate::selection::{self, ProjectSelection};
use crate::tts::{self, TtsEngine, voice};

const MAX_LOG_CHARS: usize = 400_000;

const RED: Color32 = Color32::from_rgb(0xa8, 0x32, 0x32);
const ORANGE: Color32 = Color32::from_rgb(0xa8, 0x78, 0x32);
const PURPLE: Color32 = Color32::from_rgb(0x5a, 0x32, 0x80);
const BLUE: Color32 = Color32::from_rgb(0x2a, 0x5a, 0x8a);
const GREEN: Color32 = Color32::from_rgb(0x2a, 0x6a, 0x2a);

/// App icon for the window, task bar and Dock (rendered from `assets/icon.svg`).
const ICON_PNG: &[u8] = include_bytes!("../assets/icon-256.png");
/// The same icon, small, for the title row.
const LOGO_PNG: &[u8] = include_bytes!("../assets/icon-64.png");
/// Window app ID (Wayland) / WM_CLASS (X11). Linux desktops take the icon
/// from the `.desktop` file of the same name (see `desktop_entry`).
pub const APP_ID: &str = "text-to-audiobook";

/// Start the GUI (blocks until the window is closed).
pub fn launch(cfg: Config, input_dir: PathBuf, output_dir: PathBuf) -> Result<()> {
    // Before the window opens, so the shell already knows the entry.
    #[cfg(all(unix, not(target_os = "macos")))]
    crate::desktop_entry::install_for_current_user(&cfg.paths.root, ICON_PNG);
    let mut viewport = egui::ViewportBuilder::default()
        .with_title("AI TTS Pipeline")
        .with_app_id(APP_ID)
        .with_inner_size([1000.0, 860.0])
        .with_min_inner_size([780.0, 620.0]);
    match eframe::icon_data::from_png_bytes(ICON_PNG) {
        Ok(icon) => viewport = viewport.with_icon(icon),
        Err(e) => log::warn!("app icon not loaded: {e}"),
    }
    let options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    eframe::run_native(
        "AI TTS Pipeline",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, cfg, input_dir, output_dir)))),
    )
    .map_err(|e| anyhow::anyhow!("GUI failed: {e}"))
}

fn load_logo(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let icon = eframe::icon_data::from_png_bytes(LOGO_PNG)
        .inspect_err(|e| log::warn!("logo not loaded: {e}"))
        .ok()?;
    let image = egui::ColorImage::from_rgba_unmultiplied(
        [icon.width as usize, icon.height as usize],
        &icon.rgba,
    );
    Some(ctx.load_texture("app-logo", image, egui::TextureOptions::LINEAR))
}

/// Results that worker threads hand back to the UI thread.
enum Event {
    Stage1Done(SegmentsByProject),
    Stage2Done,
    RefreshProjects,
    /// Model download finished (`Err` holds the message for the dialog).
    DownloadDone(std::result::Result<(), String>),
}

/// What to start once the models are downloaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterDownload {
    Stage2,
    Voice,
}

/// Dialog that downloads missing models: shown at the first start and
/// before a step that needs a model that is not on disk yet.
struct DownloadDialog {
    models: Vec<String>,
    then: Option<AfterDownload>,
    running: bool,
    /// Closed while the download goes on (the main progress bar shows it).
    hidden: bool,
    error: Option<String>,
}

impl DownloadDialog {
    fn new(models: Vec<String>, then: Option<AfterDownload>) -> Self {
        Self {
            models,
            then,
            running: false,
            hidden: false,
            error: None,
        }
    }
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
                        field("REVIEW_NUMBERS", "Review uncertain numbers", bools()),
                        field("SPEAK_HEADINGS", "Speak headings", bools()),
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
                        field("VOICE_DESIGN_MODEL", "Voice design model", None),
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
                        field("LOUDNESS_TARGET_LUFS", "Loudness (LUFS, 0 = off)", None),
                        field("AUTO_CLEANUP", "Auto cleanup (audit + regenerate)", bools()),
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
        "LOUDNESS_TARGET_LUFS",
        "LLM_MAX_RETRIES",
        "TTS_MAX_TOKENS",
        "TTS_MAX_WORDS_FALLBACK",
        "LLM_GPU_LAYERS",
    ];
    let bool_keys = [
        "CHAPTER_BY_CHAPTER",
        "VOCALIZE_CHAPTER_NAMES",
        "SPEAK_HEADINGS",
        "USE_LLM",
        "NORMALIZE_NUMBERS",
        "REVIEW_NUMBERS",
        "AUTO_CLEANUP",
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

/// Guided review of uncertain number spots.
struct ReviewSession {
    items: Vec<review::Item>,
    pos: usize,
    stores: IndexMap<String, ReviewStore>,
    apply_same: bool,
    /// Opened before stage 2: offer to start it when done.
    then_generate: bool,
    /// All spots were already decided when the window opened.
    revisiting: bool,
    /// Segment texts per project, for "more context".
    texts: IndexMap<String, Vec<String>>,
    /// Spot (position) whose surrounding segments are shown; resets as soon
    /// as another spot is shown.
    expanded_at: Option<usize>,
}

impl ReviewSession {
    fn decided(&self, i: usize) -> Option<&review::Decision> {
        let it = &self.items[i];
        self.stores.get(&it.project)?.get(it.segment, &it.spot)
    }

    fn open_count(&self) -> usize {
        (0..self.items.len())
            .filter(|i| self.decided(*i).is_none())
            .count()
    }

    /// Store a decision (also for identical spots if wanted) and move on.
    fn decide(&mut self, replacement: Option<String>) {
        let current = self.items[self.pos].clone();
        let targets: Vec<usize> = if self.apply_same {
            (0..self.items.len())
                .filter(|i| {
                    *i == self.pos
                        || (self.items[*i].spot.key == current.spot.key
                            && self.decided(*i).is_none())
                })
                .collect()
        } else {
            vec![self.pos]
        };
        // Identical spots take the choice at the same position ("Erstens:" for
        // "I.", "Zweitens:" for "II." ...), or the same text as fallback.
        let choice_index = replacement.as_ref().and_then(|r| {
            current
                .spot
                .choices
                .iter()
                .position(|c| &c.replacement == r)
        });
        // "Keep as written" means each spot's own text.
        let keep = replacement.as_deref() == Some(current.spot.original.as_str());
        // "Remove" only where the spot offers it (something else is left
        // to speak); the others stay open.
        let remove = replacement.as_deref().is_some_and(review::is_removal);
        let targets: Vec<usize> = targets
            .into_iter()
            .filter(|i| {
                !remove
                    || self.items[*i]
                        .spot
                        .choices
                        .iter()
                        .any(|c| review::is_removal(&c.replacement))
            })
            .collect();
        let mut touched: Vec<String> = Vec::new();
        for i in targets {
            let it = &self.items[i];
            let rep = match (choice_index, &replacement) {
                _ if keep => Some(it.spot.original.clone()),
                _ if remove => Some(String::new()),
                (Some(idx), _) if i != self.pos => it
                    .spot
                    .choices
                    .get(idx)
                    .map(|c| c.replacement.clone())
                    .or_else(|| replacement.clone()),
                _ => replacement.clone(),
            };
            if let Some(store) = self.stores.get_mut(&it.project) {
                store.decide(it, rep);
                if !touched.contains(&it.project) {
                    touched.push(it.project.clone());
                }
            }
        }
        for p in touched {
            if let Some(store) = self.stores.get(&p)
                && let Err(e) = store.save()
            {
                log::error!("Could not save review decisions: {e:#}");
            }
        }
        self.advance();
    }

    /// Next spot without a decision after the current one (or stay at the end).
    fn advance(&mut self) {
        let n = self.items.len();
        if let Some(next) = (self.pos + 1..n)
            .chain(0..self.pos)
            .find(|i| self.decided(*i).is_none())
        {
            self.pos = next;
        } else if self.revisiting && self.pos + 1 < n {
            self.pos += 1;
        }
    }
}

/// The "Voice" panel: where the narrator voice comes from.
struct VoiceForm {
    /// "file", "design" or "random" (see `config::VOICE_SOURCE_CHOICES`).
    source: String,
    file: String,
    description: String,
    /// What `cache/voice_reference.wav` currently is.
    status: String,
}

impl VoiceForm {
    fn new(cfg: &Config) -> Self {
        Self {
            source: voice::source(&cfg.settings).to_string(),
            file: cfg.settings.voice_reference_file.clone(),
            description: cfg.settings.narrator_prompt.clone(),
            status: voice::describe(&cfg.paths),
        }
    }

    /// The form differs from the saved settings.
    fn changed(&self, s: &Settings) -> bool {
        self.source != voice::source(s)
            || self.file != s.voice_reference_file
            || self.description != s.narrator_prompt
    }
}

pub struct App {
    /// App icon shown next to the title (None if the PNG could not be read).
    logo: Option<egui::TextureHandle>,
    review: Option<ReviewSession>,
    download: Option<DownloadDialog>,
    /// Stage 2 was requested while spots are open: number of open spots.
    stage2_guard: Option<usize>,
    skip_stage2_guard: bool,
    cfg: Config,
    input_dir: PathBuf,
    output_dir: PathBuf,
    projects: Vec<ProjectRow>,
    /// Ticked projects, remembered in `cache/project_selection.json`.
    selection: ProjectSelection,
    segments: SegmentsByProject,
    voice: VoiceForm,
    quick: QuickConfig,
    settings_form: Option<SettingsForm>,
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
        let logo = load_logo(&ctx);
        let mut app = Self {
            logo,
            review: None,
            download: None,
            stage2_guard: None,
            skip_stage2_guard: false,
            voice: VoiceForm::new(&cfg),
            quick: QuickConfig::new(&cfg.settings),
            selection: ProjectSelection::load(&cfg.paths.cache_dir),
            cfg,
            input_dir,
            output_dir,
            projects: Vec::new(),
            segments: IndexMap::new(),
            settings_form: None,
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
        let missing = tts::missing_models(&app.cfg.settings, &app.cfg.paths);
        if !missing.is_empty() {
            app.download = Some(DownloadDialog::new(missing, None));
        }
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

    /// Ticked states come from the saved selection; new projects are ticked.
    fn refresh_project_list(&mut self) {
        let key = self.selection_key();
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
                    selected: self.selection.is_selected(&key, &p.name),
                    name: p.name,
                }
            })
            .collect();
        self.save_selection();
    }

    fn selection_key(&self) -> String {
        selection::dir_key(&self.cfg.paths.root, &self.input_dir)
    }

    /// Remember the ticked projects of the current input directory.
    fn save_selection(&mut self) {
        let key = self.selection_key();
        let rows = self.projects.iter().map(|p| (p.name.as_str(), p.selected));
        if self.selection.set(&key, rows)
            && let Err(e) = self.selection.save()
        {
            log::warn!("Could not save project selection: {e:#}");
        }
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
        let output = self.output_dir.clone();
        self.spawn(move |cfg, stop, shared| {
            let progress = |_: &str| shared.bump_progress(0.02, 0.95);
            match pipeline::prepare_segments(
                &cfg,
                &input,
                &output,
                &progress,
                Some(&stop),
                Some(&selected),
            ) {
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
        let names: Vec<String> = segs.keys().cloned().collect();
        for (name, s) in segs {
            self.segments.insert(name, s);
        }
        if !self.stop.is_set() && self.review_enabled() {
            let open = self.count_open(&names);
            if open > 0 {
                log::info!("{open} uncertain number spot(s) found — please review them.");
                self.open_review(Some(names), false);
            }
        }
    }

    fn review_enabled(&self) -> bool {
        self.cfg.settings.normalize_numbers && self.cfg.settings.review_numbers
    }

    /// Open review spots of the given projects.
    fn count_open(&self, names: &[String]) -> usize {
        names
            .iter()
            .filter_map(|n| self.segments.get(n).map(|s| (n, s)))
            .map(|(n, s)| {
                let store = review::load_synced(
                    &self.cfg.paths.project_cache_dir(n),
                    n,
                    s,
                    &self.cfg.settings,
                );
                review::pending(n, s, &self.cfg.settings, &store).len()
            })
            .sum()
    }

    /// Open the review window for `names` (default: selected projects).
    fn open_review(&mut self, names: Option<Vec<String>>, then_generate: bool) {
        if !self.review_enabled() {
            log::warn!(
                "Number review is off: enable 'Spell out numbers' and 'Review uncertain numbers' in Settings."
            );
            return;
        }
        let names = names.unwrap_or_else(|| {
            let sel = self.selected_names();
            self.segments
                .keys()
                .filter(|n| sel.is_empty() || sel.contains(n))
                .cloned()
                .collect()
        });
        if names.is_empty() {
            log::warn!("No projects with segments to review; run Stage 1 first.");
            return;
        }
        let mut stores = IndexMap::new();
        let mut texts = IndexMap::new();
        let mut all = Vec::new();
        for n in &names {
            let Some(segs) = self.segments.get(n) else {
                continue;
            };
            let store = review::load_synced(
                &self.cfg.paths.project_cache_dir(n),
                n,
                segs,
                &self.cfg.settings,
            );
            all.extend(review::items(n, segs, &self.cfg.settings));
            stores.insert(n.clone(), store);
            texts.insert(
                n.clone(),
                segs.iter()
                    .map(|s| s.normalized_text.clone())
                    .collect::<Vec<_>>(),
            );
        }
        let mut session = ReviewSession {
            items: all,
            pos: 0,
            stores,
            apply_same: false,
            then_generate,
            revisiting: false,
            texts,
            expanded_at: None,
        };
        match (0..session.items.len()).find(|i| session.decided(*i).is_none()) {
            Some(first) => session.pos = first,
            None => session.revisiting = true,
        }
        self.review = Some(session);
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
        if self.review_enabled() && !std::mem::take(&mut self.skip_stage2_guard) {
            let names: Vec<String> = to_process.keys().cloned().collect();
            let open = self.count_open(&names);
            if open > 0 {
                self.stage2_guard = Some(open);
                return;
            }
        }
        if self.offer_download(AfterDownload::Stage2) {
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
        let output = self.output_dir.clone();
        self.spawn(move |cfg, stop, shared| {
            // The bar shows the synthesis, then the post-processing of each
            // project from 0 again.
            let synth = Mutex::new(SynthProgress::new(&to_process));
            let several = to_process.len() > 1;
            let show_synthesis = |p: &SynthProgress| {
                shared.set_progress(p.fraction());
                shared.set_label(p.label());
            };
            let on_event = |e: Stage2Event| match e {
                Stage2Event::Synthesis { .. } => show_synthesis(&synth.lock().unwrap()),
                Stage2Event::Chunk {
                    project,
                    index,
                    spoken,
                } => {
                    let mut p = synth.lock().unwrap();
                    p.chunk(project, index, spoken);
                    show_synthesis(&p);
                }
                Stage2Event::Post {
                    project,
                    step,
                    done,
                } => {
                    shared.set_progress(done);
                    shared.set_label(post_label(several.then_some(project), step));
                }
            };
            let result = pipeline::generate_audio(
                &cfg,
                &to_process,
                &output,
                &pipeline::no_progress,
                Some(&stop),
                Some(&on_event),
            );
            let synth = synth.into_inner().unwrap();
            let failed = result
                .as_ref()
                .is_err_and(|e| !e.is::<crate::cancel::ForceStopped>());
            if result.is_ok() && !stop.is_set() {
                shared.set_progress(1.0);
                shared.set_label(format!("Done: {} chunks", synth.total_chunks));
            } else {
                shared.set_progress(synth.fraction());
                shared.set_label(synth.end_label(if failed { "Failed" } else { "Stopped" }));
            }
            match result {
                Ok(_) => {
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

    /// Save the Voice panel and make the voice reference in the background.
    /// `next_seed`: a new random voice (voice seed + 1).
    fn apply_voice(&mut self, next_seed: bool) {
        let seed = (self.cfg.settings.voice_seed + u64::from(next_seed)).to_string();
        let fields = [
            ("VOICE_SOURCE", self.voice.source.as_str()),
            ("VOICE_REFERENCE_FILE", self.voice.file.as_str()),
            ("NARRATOR_PROMPT", self.voice.description.as_str()),
            ("VOICE_SEED", seed.as_str()),
        ];
        let applied = apply_fields(&mut self.cfg.settings, fields.into_iter(), "Voice");
        if let Err(e) = config::save_settings_to_file(&applied, &self.cfg.paths.settings_file()) {
            log::error!("Could not save settings: {e:#}");
        }
        self.quick = QuickConfig::new(&self.cfg.settings);
        if self.offer_download(AfterDownload::Voice) {
            return;
        }
        self.prepare_voice();
    }

    /// Make the voice reference in the background.
    fn prepare_voice(&mut self) {
        self.reset_progress();
        log::info!("Preparing the narrator voice...");
        self.spawn(move |cfg, _stop, shared| {
            match TtsEngine::prepare_voice(&cfg.settings, &cfg.paths) {
                Ok(_) => {
                    shared.set_progress(1.0);
                    log::info!("Narrator voice ready: {}", voice::describe(&cfg.paths));
                }
                Err(e) => log::error!("Could not prepare the voice: {e:#}"),
            }
            shared.push(Event::RefreshProjects);
        });
    }

    // ---------------------------------------------------------------
    // Model download
    // ---------------------------------------------------------------

    /// Open the download dialog if `then` needs models that are not on disk
    /// yet. True: `then` starts after the download instead of now.
    fn offer_download(&mut self, then: AfterDownload) -> bool {
        let missing = tts::missing_models(&self.cfg.settings, &self.cfg.paths);
        if missing.is_empty() {
            return false;
        }
        log::info!("Model(s) to download first: {}", missing.join(", "));
        self.download = Some(DownloadDialog::new(missing, Some(then)));
        true
    }

    fn start_download(&mut self) {
        let Some(dialog) = self.download.as_mut() else {
            return;
        };
        if self.shared.busy.load(Ordering::SeqCst) {
            return;
        }
        dialog.running = true;
        dialog.hidden = false;
        dialog.error = None;
        let models = dialog.models.clone();
        self.reset_progress();
        self.spawn(move |_cfg, _stop, shared| {
            let n = models.len();
            let mut result = Ok(());
            for (i, model) in models.iter().enumerate() {
                let which = if n > 1 {
                    format!("Model {}/{n}", i + 1)
                } else {
                    "Model".to_string()
                };
                shared.set_label(format!("{which}: connecting to Hugging Face ..."));
                let on_progress = |p: &crate::hub::DownloadProgress| {
                    shared.set_progress(p.fraction());
                    if p.total > 0 {
                        shared.set_label(format!(
                            "{which}: {} of {} ({:.0} %)",
                            crate::hub::human_size(p.done),
                            crate::hub::human_size(p.total),
                            p.fraction() * 100.0
                        ));
                    }
                };
                if let Err(e) = crate::hub::download_tts_model(model, Some(&on_progress)) {
                    log::error!("Download of {model} failed: {e:#}");
                    result = Err(format!("{e:#}"));
                    break;
                }
            }
            if result.is_ok() {
                shared.set_progress(1.0);
                shared.set_label("Models downloaded");
                log::info!("All models downloaded.");
            }
            shared.push(Event::DownloadDone(result));
        });
    }

    fn on_download_done(&mut self, result: std::result::Result<(), String>) {
        let Some(dialog) = self.download.as_mut() else {
            return;
        };
        dialog.running = false;
        match result {
            Ok(()) => {
                let then = dialog.then;
                self.download = None;
                match then {
                    Some(AfterDownload::Stage2) => {
                        // The review check already ran before the download.
                        self.skip_stage2_guard = self.review_enabled();
                        self.start_stage2();
                    }
                    Some(AfterDownload::Voice) => self.prepare_voice(),
                    None => {}
                }
            }
            Err(msg) => {
                dialog.error = Some(msg);
                dialog.hidden = false;
            }
        }
    }

    fn download_dialog(&mut self, ctx: &egui::Context) {
        let Some(dialog) = self.download.as_ref() else {
            return;
        };
        if dialog.hidden {
            return;
        }
        let running = dialog.running;
        let models = dialog.models.join("\n");
        let error = dialog.error.clone();
        let then = dialog.then;
        let progress = *self.shared.progress.lock().unwrap();
        let label = self.shared.label.lock().unwrap().clone();
        let busy_elsewhere = !running && self.busy();
        let mut action = None;
        let modal = egui::Modal::new(egui::Id::unique("model_download")).show(ctx, |ui| {
            ui.set_width(480.0);
            ui.heading("Download the speech model");
            ui.add_space(6.0);
            ui.label(match then {
                None => "Before the first audiobook the program needs this speech model:",
                Some(_) => "This step needs a speech model that is not on this computer yet:",
            });
            ui.label(RichText::new(models).strong());
            ui.add_space(6.0);
            ui.label(format!(
                "It is downloaded once from Hugging Face (several GB) and stored in\n{}\n\
                 Depending on the connection this takes a while. If the download is \
                 interrupted (or the program is closed), it continues where it stopped \
                 the next time.",
                crate::hub::model_cache_dir().display()
            ));
            ui.add_space(10.0);
            if running {
                ui.add(
                    egui::ProgressBar::new(progress)
                        .show_percentage()
                        .animate(true),
                );
                ui.label(RichText::new(label).color(Color32::GRAY));
                ui.add_space(8.0);
                if ui
                    .button("Hide")
                    .on_hover_text("The download goes on; the progress bar at the bottom shows it")
                    .clicked()
                {
                    action = Some("hide");
                }
                return;
            }
            if let Some(err) = &error {
                ui.label(
                    RichText::new(format!(
                        "The download failed:\n{err}\n\nCheck the internet connection and \
                         try again."
                    ))
                    .color(Color32::from_rgb(0xe0, 0x60, 0x60)),
                );
                ui.add_space(8.0);
            }
            ui.horizontal(|ui| {
                let start = if error.is_some() {
                    "Try again"
                } else {
                    "Download now"
                };
                if ui
                    .add_enabled(!busy_elsewhere, egui::Button::new(start).fill(GREEN))
                    .clicked()
                {
                    action = Some("start");
                }
                let close = if then.is_none() { "Later" } else { "Cancel" };
                if ui.button(close).clicked() {
                    action = Some("close");
                }
            });
        });
        if modal.should_close() && action.is_none() {
            action = Some(if running { "hide" } else { "close" });
        }
        match action {
            Some("start") => self.start_download(),
            Some("hide") => {
                if let Some(d) = self.download.as_mut() {
                    d.hidden = true;
                }
            }
            Some("close") => {
                self.download = None;
                if then.is_none() {
                    log::info!(
                        "Model download postponed; it is offered again before the first \
                         audio is made."
                    );
                }
            }
            _ => {}
        }
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
        self.voice = VoiceForm::new(&self.cfg);
        log::info!("Settings saved and applied.");
    }

    fn reset_settings_to_defaults(&mut self) {
        let mock = self.cfg.settings.mock_models;
        let mut defaults = config::reset_to_defaults(&self.cfg.paths.settings_file());
        defaults.mock_models = mock;
        self.cfg.settings = defaults;
        self.settings_form = Some(SettingsForm::new(&self.cfg.settings));
        self.quick = QuickConfig::new(&self.cfg.settings);
        self.voice = VoiceForm::new(&self.cfg);
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
        if added {
            trim_log(&mut self.log_text, MAX_LOG_CHARS);
        }
        let events: Vec<Event> = std::mem::take(&mut *self.shared.events.lock().unwrap());
        for e in events {
            match e {
                Event::Stage1Done(segs) => self.on_stage1_done(segs),
                Event::Stage2Done => {}
                Event::RefreshProjects => {
                    self.refresh_project_list();
                    self.voice.status = voice::describe(&self.cfg.paths);
                }
                Event::DownloadDone(result) => self.on_download_done(result),
            }
        }
    }

    /// Title row: icon and heading, centred together.
    fn title_ui(&self, ui: &mut egui::Ui) {
        const LOGO_SIZE: f32 = 32.0;
        let title = egui::WidgetText::from(
            RichText::new("AI Text-to-Speech Pipeline")
                .size(18.0)
                .strong(),
        )
        .into_galley(
            ui,
            Some(egui::TextWrapMode::Extend),
            f32::INFINITY,
            egui::TextStyle::Body,
        );
        let logo_width = match self.logo {
            Some(_) => LOGO_SIZE + ui.spacing().item_spacing.x,
            None => 0.0,
        };
        ui.horizontal(|ui| {
            let indent = (ui.available_width() - logo_width - title.size().x) / 2.0;
            ui.add_space(indent.max(0.0));
            if let Some(logo) = &self.logo {
                ui.add(egui::Image::new(logo).fit_to_exact_size(egui::vec2(LOGO_SIZE, LOGO_SIZE)));
            }
            ui.label(title);
        });
    }

    fn top_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        self.title_ui(ui);
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
                        if ui
                            .button("Open")
                            .on_hover_text("Show this folder in the file manager")
                            .clicked()
                        {
                            let dir = if is_input {
                                &self.input_dir
                            } else {
                                &self.output_dir
                            };
                            if let Err(e) = std::fs::create_dir_all(dir)
                                .and_then(|()| open_in_file_manager(dir))
                            {
                                log::warn!("Could not open {}: {e}", dir.display());
                            }
                        }
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
                self.save_selection();
            }
            if ui.button("Select None").clicked() {
                self.projects.iter_mut().for_each(|p| p.selected = false);
                self.save_selection();
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
                if ui
                    .add_enabled(!busy, egui::Button::new("Review Numbers").fill(BLUE))
                    .on_hover_text("Decide uncertain ordinals and Roman numerals")
                    .clicked()
                {
                    self.open_review(None, false);
                }
            });
        });
        let mut toggled = false;
        egui::Frame::group(ui.style()).show(ui, |ui| {
            egui::ScrollArea::vertical()
                .id_salt("projects")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if self.projects.is_empty() {
                        ui.label(
                            RichText::new(
                                "(no projects found in this directory)\n\n\
                                 Put your texts as Markdown (.md) files into the input \
                                 directory: one file per book, or one folder per book \
                                 with a file per chapter. \"Open\" next to \"Input \
                                 Directory\" shows the folder, \"Refresh\" reloads the \
                                 list.",
                            )
                            .color(Color32::GRAY),
                        );
                    }
                    for p in &mut self.projects {
                        toggled |= ui
                            .add_enabled(
                                !busy,
                                egui::Checkbox::new(&mut p.selected, p.label.as_str()),
                            )
                            .changed();
                    }
                });
        });
        if toggled {
            self.save_selection();
        }
    }

    fn side_panels_ui(&mut self, ui: &mut egui::Ui) {
        let busy = self.busy();
        egui::CollapsingHeader::new(RichText::new("Voice").strong())
            .default_open(true)
            .show(ui, |ui| {
                let hint = |ui: &mut egui::Ui, text: &str| {
                    ui.label(RichText::new(text).small().color(Color32::GRAY));
                };
                ui.add_enabled_ui(!busy, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.radio_value(&mut self.voice.source, "file".to_string(), "Recording");
                        ui.radio_value(
                            &mut self.voice.source,
                            "design".to_string(),
                            "From description",
                        );
                        ui.radio_value(&mut self.voice.source, "random".to_string(), "Random");
                    });
                    match self.voice.source.as_str() {
                        "file" => {
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.voice.file)
                                        .hint_text("path to a WAV file")
                                        .desired_width(ui.available_width() - 80.0),
                                );
                                if ui.button("Browse").clicked()
                                    && let Some(p) = rfd::FileDialog::new()
                                        .add_filter("WAV", &["wav", "WAV"])
                                        .pick_file()
                                {
                                    self.voice.file = p.to_string_lossy().to_string();
                                }
                            });
                            hint(
                                ui,
                                "10–15 s of clear speech by one speaker, no music (WAV). \
                                 Works with both models; longer files are cut to 20 s.",
                            );
                        }
                        "design" => {
                            ui.add(
                                egui::TextEdit::multiline(&mut self.voice.description)
                                    .desired_rows(4)
                                    .desired_width(f32::INFINITY),
                            );
                            if ui.small_button("Reset Default").clicked() {
                                self.voice.description =
                                    config::DEFAULT_NARRATOR_PROMPT.to_string();
                            }
                            hint(
                                ui,
                                "Made once by the extra model Qwen3-TTS 1.7B VoiceDesign \
                                 (about 4 GB download, loaded only for this). Its sample is \
                                 then cloned by the selected model (0.6B or 1.7B).",
                            );
                        }
                        _ => {
                            hint(
                                ui,
                                &format!(
                                    "The selected model invents a voice; the voice seed ({}) \
                                     decides which.",
                                    self.cfg.settings.voice_seed
                                ),
                            );
                        }
                    }
                    ui.label(format!("Current voice: {}", self.voice.status));
                    ui.horizontal(|ui| {
                        let changed = self.voice.changed(&self.cfg.settings);
                        let apply = egui::Button::new("Apply Voice");
                        let apply = if changed { apply.fill(GREEN) } else { apply };
                        if ui
                            .add(apply)
                            .on_hover_text("Save the choice and make the voice reference now")
                            .clicked()
                        {
                            self.apply_voice(false);
                        }
                        if self.voice.source == "random"
                            && ui
                                .button("New Random Voice")
                                .on_hover_text("Next voice seed")
                                .clicked()
                        {
                            self.apply_voice(true);
                        }
                    });
                    hint(
                        ui,
                        "With a new voice, '2. Generate Audio' speaks the chunks again.",
                    );
                });
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
            ui.columns(2, |cols| {
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
                let generate = if self.cfg.settings.auto_cleanup {
                    "2. Generate Audio (with cleanup)"
                } else {
                    "2. Generate Audio"
                };
                if full(
                    &mut cols[1],
                    !busy && !self.segments.is_empty(),
                    egui::Button::new(generate).fill(GREEN),
                ) {
                    self.start_stage2();
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
        // Full-width bar in a frame; chunk count / ETA and percentage inside.
        let p = *self.shared.progress.lock().unwrap();
        let label = self.shared.label.lock().unwrap().clone();
        let pct = (p * 100.0).round() as u32;
        let text = if label.is_empty() {
            format!("{pct} %")
        } else {
            format!("{label}    {pct} %")
        };
        egui::Frame::new()
            .stroke(ui.visuals().widgets.noninteractive.bg_stroke)
            .corner_radius(4.0)
            .inner_margin(2.0)
            .show(ui, |ui| {
                ui.add(
                    egui::ProgressBar::new(p)
                        .desired_width(ui.available_width())
                        .desired_height(18.0)
                        .corner_radius(3.0)
                        .text(text),
                );
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

    fn review_window(&mut self, ctx: &egui::Context) {
        let Some(session) = self.review.as_mut() else {
            return;
        };
        let mut open = true;
        let mut close = false;
        let mut generate = false;
        egui::Window::new("Review Numbers")
            .open(&mut open)
            .collapsible(false)
            .default_size([620.0, 420.0])
            .show(ctx, |ui| {
                let n = session.items.len();
                if n == 0 {
                    ui.label("No uncertain number spots in these projects.");
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                    return;
                }
                let open_count = session.open_count();
                let pos = session.pos.min(n - 1);
                let item = session.items[pos].clone();
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&item.project).strong());
                    ui.label(format!("— spot {} of {n}", pos + 1));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(RichText::new(format!("{open_count} open")).color(
                            if open_count > 0 {
                                Color32::from_rgb(0xe0, 0xb0, 0x40)
                            } else {
                                Color32::from_rgb(0x60, 0xc0, 0x60)
                            },
                        ));
                    });
                });
                ui.separator();
                let expanded = session.expanded_at == Some(pos);
                let full = session
                    .texts
                    .get(&item.project)
                    .and_then(|t| t.get(item.segment))
                    .cloned();
                let neighbour_label = |ui: &mut egui::Ui, n: usize, text: &str| {
                    ui.label(
                        RichText::new(format!("{n:06} | {text}"))
                            .size(13.0)
                            .color(Color32::GRAY),
                    );
                };
                let texts = session
                    .texts
                    .get(&item.project)
                    .cloned()
                    .unwrap_or_default();
                let (above, below) =
                    review::neighbours(&texts, item.segment, review::CONTEXT_SEGMENTS);
                let body = |ui: &mut egui::Ui| {
                    if expanded {
                        for (n, t) in &above {
                            neighbour_label(ui, *n, t);
                        }
                    }
                    egui::Frame::group(ui.style()).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        // Expanded: the whole segment instead of the excerpt.
                        let (before, after) = match (&full, expanded) {
                            (Some(t), true) => (
                                t[..item.spot.start].to_string(),
                                t[item.spot.end..].to_string(),
                            ),
                            _ => (item.before.clone(), item.after.clone()),
                        };
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing.x = 0.0;
                            ui.label(RichText::new(before).size(15.0));
                            ui.label(
                                RichText::new(&item.spot.original)
                                    .size(15.0)
                                    .strong()
                                    .color(Color32::BLACK)
                                    .background_color(Color32::from_rgb(0xe0, 0xb0, 0x40)),
                            );
                            ui.label(RichText::new(after).size(15.0));
                        });
                    });
                    if expanded {
                        for (n, t) in &below {
                            neighbour_label(ui, *n, t);
                        }
                    }
                };
                if expanded {
                    egui::ScrollArea::vertical()
                        .id_salt("review_context")
                        .max_height(320.0)
                        .show(ui, body);
                } else {
                    body(ui);
                }
                let toggle = if expanded {
                    "▲ Less context"
                } else {
                    "▼ More context"
                };
                if ui
                    .small_button(toggle)
                    .on_hover_text("Show the 3 segments above and below")
                    .clicked()
                {
                    session.expanded_at = if expanded { None } else { Some(pos) };
                }
                let current = session.decided(pos).cloned();
                if let Some(d) = &current {
                    ui.label(
                        RichText::new(format!(
                            "Current choice: {}",
                            match d.replacement.as_deref() {
                                None => "program default".to_string(),
                                Some(r) if r == item.spot.original => {
                                    format!("{r} (keep as written)")
                                }
                                Some(r) if review::is_removal(r) => {
                                    "removed (not spoken)".to_string()
                                }
                                Some(r) => r.to_string(),
                            }
                        ))
                        .color(Color32::GRAY),
                    );
                }
                ui.add_space(6.0);
                ui.label("How should it be read?");
                let mut chosen: Option<Option<String>> = None;
                egui::Grid::new("review_choices")
                    .num_columns(2)
                    .show(ui, |ui| {
                        for c in &item.spot.choices {
                            let preview = review::with_choice(
                                &item.before,
                                &c.replacement,
                                &item.after,
                                &item.spot.original,
                            );
                            let selected = current.as_ref().and_then(|d| d.replacement.as_ref())
                                == Some(&c.replacement);
                            // "Remove": the spot struck through.
                            let label = if review::is_removal(&c.replacement) {
                                RichText::new(&item.spot.original).strikethrough()
                            } else {
                                RichText::new(&c.replacement)
                            };
                            let mut b = egui::Button::new(label.size(15.0));
                            if selected {
                                b = b.fill(GREEN);
                            }
                            if ui.add(b).on_hover_text(preview).clicked() {
                                chosen = Some(Some(c.replacement.clone()));
                            }
                            ui.label(RichText::new(&c.hint).color(Color32::GRAY));
                            ui.end_row();
                        }
                    });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if ui
                        .button("Program default")
                        .on_hover_text("Let the rules decide (digits are read by the TTS)")
                        .clicked()
                    {
                        chosen = Some(None);
                    }
                    if ui.button("◀ Back").clicked() && pos > 0 {
                        session.pos = pos - 1;
                    }
                    if ui.button("Skip ▶").clicked() {
                        session.pos = (pos + 1) % n;
                    }
                    ui.checkbox(&mut session.apply_same, "Apply to all identical spots");
                });
                if let Some(rep) = chosen {
                    session.pos = pos;
                    session.decide(rep);
                }
                ui.separator();
                ui.horizontal(|ui| {
                    if session.open_count() == 0 {
                        ui.label(
                            RichText::new("All spots decided.")
                                .color(Color32::from_rgb(0x60, 0xc0, 0x60)),
                        );
                        if session.then_generate
                            && ui
                                .add(egui::Button::new("2. Generate Audio").fill(GREEN))
                                .clicked()
                        {
                            generate = true;
                        }
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Close").clicked() {
                            close = true;
                        }
                    });
                });
            });
        if close || !open || generate {
            // Update spoken.txt with the decisions.
            if let Some(session) = self.review.take() {
                for name in session.stores.keys() {
                    if let Some(segs) = self.segments.get(name)
                        && let Err(e) =
                            pipeline::write_texts(&self.cfg, name, segs, &self.output_dir)
                    {
                        log::warn!("[{name}] could not write the text files: {e:#}");
                    }
                }
            }
        }
        if generate {
            self.skip_stage2_guard = true;
            self.start_stage2();
        }
    }

    fn stage2_guard_dialog(&mut self, ctx: &egui::Context) {
        let Some(open) = self.stage2_guard else {
            return;
        };
        let mut action = None;
        let modal = egui::Modal::new(egui::Id::unique("stage2_guard")).show(ctx, |ui| {
            ui.set_width(360.0);
            ui.label(format!(
                "{open} uncertain number spot(s) have not been reviewed yet.\n\
                 Without a decision the program default is used."
            ));
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new("Review now").fill(BLUE)).clicked() {
                    action = Some("review");
                }
                if ui.button("Generate anyway").clicked() {
                    action = Some("generate");
                }
                if ui.button("Cancel").clicked() {
                    action = Some("cancel");
                }
            });
        });
        if modal.should_close() && action.is_none() {
            action = Some("cancel");
        }
        match action {
            Some("review") => {
                self.stage2_guard = None;
                let sel = self.selected_segments().keys().cloned().collect();
                self.open_review(Some(sel), true);
            }
            Some("generate") => {
                self.stage2_guard = None;
                self.skip_stage2_guard = true;
                self.start_stage2();
            }
            Some(_) => self.stage2_guard = None,
            None => {}
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

/// Drop whole lines from the front of the log until at most `max` bytes are
/// left. The cut is moved to a char boundary first: slicing inside an umlaut
/// (e.g. in a project name that every chunk path repeats) used to panic and
/// take the whole GUI down.
fn trim_log(text: &mut String, max: usize) {
    if text.len() <= max {
        return;
    }
    let cut = text.ceil_char_boundary(text.len() - max);
    let cut = text[cut..].find('\n').map_or(cut, |i| cut + i + 1);
    text.drain(..cut);
}

/// Progress of the synthesis in stage 2. Chunks are weighted by their
/// length: a long chunk (split into several TTS calls) takes longer and
/// moves the bar further. Only chunks spoken in this run count for the
/// speed; reused ones are done at once and would make the ETA too short.
struct SynthProgress {
    /// Weight of every chunk, per project.
    weights: IndexMap<String, Vec<u64>>,
    total_chunks: usize,
    total_weight: u64,
    done_chunks: usize,
    done_weight: u64,
    /// Weight spoken in this run and the time it took.
    spoken_weight: u64,
    spoken_time: Duration,
}

impl SynthProgress {
    fn new(segments: &SegmentsByProject) -> Self {
        let weights: IndexMap<String, Vec<u64>> = segments
            .iter()
            .map(|(name, segs)| {
                let w = segs
                    .iter()
                    .map(|s| s.normalized_text.chars().count().max(1) as u64)
                    .collect();
                (name.clone(), w)
            })
            .collect();
        Self {
            total_chunks: weights.values().map(Vec::len).sum(),
            total_weight: weights.values().flatten().sum(),
            weights,
            done_chunks: 0,
            done_weight: 0,
            spoken_weight: 0,
            spoken_time: Duration::ZERO,
        }
    }

    /// Chunk `index` of `project` is done: spoken in `spoken`, or reused.
    fn chunk(&mut self, project: &str, index: usize, spoken: Option<Duration>) {
        let w = self
            .weights
            .get(project)
            .and_then(|w| w.get(index))
            .copied()
            .unwrap_or(1);
        self.done_chunks += 1;
        self.done_weight += w;
        if let Some(t) = spoken {
            self.spoken_weight += w;
            self.spoken_time += t;
        }
    }

    fn fraction(&self) -> f32 {
        if self.total_weight == 0 {
            return 0.0;
        }
        (self.done_weight as f64 / self.total_weight as f64) as f32
    }

    /// Time left at the speed of the chunks spoken so far.
    fn eta(&self) -> Option<Duration> {
        if self.spoken_weight == 0 {
            return None;
        }
        let left = self.total_weight.saturating_sub(self.done_weight);
        Some(
            self.spoken_time
                .mul_f64(left as f64 / self.spoken_weight as f64),
        )
    }

    /// "Chunk 3 / 10  (ETA ~1m 5s)".
    fn label(&self) -> String {
        if self.total_chunks == 0 {
            return String::new();
        }
        let count = format!("Chunk {} / {}", self.done_chunks, self.total_chunks);
        match self.eta() {
            Some(eta) if self.done_chunks < self.total_chunks => {
                format!("{count}  (ETA ~{})", short_duration(eta))
            }
            _ => count,
        }
    }

    /// "Stopped at chunk 3 / 10" (`what`: Stopped, Failed).
    fn end_label(&self, what: &str) -> String {
        format!(
            "{what} at chunk {} / {}",
            self.done_chunks, self.total_chunks
        )
    }
}

/// "1h 5m", "4m 10s" or "33s".
fn short_duration(d: Duration) -> String {
    let secs = d.as_secs_f64().round() as u64;
    let (h, m, s) = (secs / 3600, secs / 60 % 60, secs % 60);
    if h > 0 {
        format!("{h}h {m}m")
    } else if m > 0 {
        format!("{m}m {s}s")
    } else {
        format!("{s}s")
    }
}

/// "Post-processing: denoising 3 / 10", with the project when there are
/// several.
fn post_label(project: Option<&str>, step: &str) -> String {
    match project {
        Some(p) => format!("Post-processing '{p}': {step}"),
        None => format!("Post-processing: {step}"),
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
        self.review_window(&ctx);
        self.stage2_guard_dialog(&ctx);
        self.download_dialog(&ctx);
        self.close_handling(&ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_icons_decode() {
        for (png, size) in [(ICON_PNG, 256), (LOGO_PNG, 64)] {
            let icon = eframe::icon_data::from_png_bytes(png).unwrap();
            assert_eq!((icon.width, icon.height), (size, size));
            assert_eq!(icon.rgba.len(), (size * size * 4) as usize);
        }
    }

    #[test]
    fn log_trim_never_cuts_inside_an_umlaut() {
        let log = "INFO | Denoising: Übersetzung/chunk_1.wav\n\
                   DEBUG | Post-processing complete: Übersetzung wörtlicher/chunk_1.wav\n\
                   INFO | ✓ fertig\n";
        // Every cut position, including the bytes inside 'Ü', 'ö' and '✓'.
        for max in 0..=log.len() {
            let mut t = log.to_string();
            trim_log(&mut t, max);
            assert!(t.len() <= max, "max {max}: {t:?}");
            assert!(log.ends_with(&t));
            // Only whole lines are kept.
            assert!(
                t.is_empty() || t.len() == log.len() || log[..log.len() - t.len()].ends_with('\n')
            );
        }
        let mut t = log.to_string();
        trim_log(&mut t, 20);
        assert_eq!(t, "INFO | ✓ fertig\n");
    }

    fn segments(projects: &[(&str, &[&str])]) -> SegmentsByProject {
        projects
            .iter()
            .map(|(name, texts)| {
                let segs = texts
                    .iter()
                    .map(|t| crate::segment::Segment::with_text(*t))
                    .collect();
                (name.to_string(), segs)
            })
            .collect()
    }

    #[test]
    fn synthesis_progress_is_weighted_by_text_length() {
        let long = "x".repeat(300);
        let segs = segments(&[("A", &["x".repeat(100).as_str(), &long]), ("B", &[&long])]);
        let mut p = SynthProgress::new(&segs);
        assert_eq!(p.label(), "Chunk 0 / 3");
        assert_eq!(p.fraction(), 0.0);
        // 100 characters in 20 s: the 600 left take 120 s.
        p.chunk("A", 0, Some(Duration::from_secs(20)));
        assert!((p.fraction() - 1.0 / 7.0).abs() < 1e-6);
        assert_eq!(p.eta(), Some(Duration::from_secs(120)));
        assert_eq!(p.label(), "Chunk 1 / 3  (ETA ~2m 0s)");
        // The long chunk weighs three times as much: 400 characters in
        // 90 s, the 300 left take 67.5 s.
        p.chunk("A", 1, Some(Duration::from_secs(70)));
        assert_eq!(p.eta(), Some(Duration::from_millis(67_500)));
        assert_eq!(p.label(), "Chunk 2 / 3  (ETA ~1m 8s)");
        p.chunk("B", 0, Some(Duration::from_secs(60)));
        assert_eq!(p.fraction(), 1.0);
        assert_eq!(p.label(), "Chunk 3 / 3");
        assert_eq!(p.end_label("Stopped"), "Stopped at chunk 3 / 3");
    }

    #[test]
    fn reused_chunks_do_not_shorten_the_eta() {
        let segs = segments(&[("A", &["Satz."; 10])]);
        let mut p = SynthProgress::new(&segs);
        // A resumed run: eight chunks exist already.
        for i in 0..8 {
            p.chunk("A", i, None);
        }
        assert!((p.fraction() - 0.8).abs() < 1e-6);
        assert_eq!(p.eta(), None);
        assert_eq!(p.label(), "Chunk 8 / 10");
        p.chunk("A", 8, Some(Duration::from_secs(150)));
        assert_eq!(p.eta(), Some(Duration::from_secs(150)));
        assert_eq!(p.label(), "Chunk 9 / 10  (ETA ~2m 30s)");
    }

    #[test]
    fn durations_and_post_labels() {
        assert_eq!(short_duration(Duration::from_secs(33)), "33s");
        assert_eq!(short_duration(Duration::from_secs(250)), "4m 10s");
        assert_eq!(
            short_duration(Duration::from_secs(3 * 3600 + 25 * 60 + 9)),
            "3h 25m"
        );
        assert_eq!(
            post_label(None, "denoising 3 / 10"),
            "Post-processing: denoising 3 / 10"
        );
        assert_eq!(
            post_label(Some("Buch"), "assembling the audiobook"),
            "Post-processing 'Buch': assembling the audiobook"
        );
        assert_eq!(SynthProgress::new(&IndexMap::new()).label(), "");
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
            // Set in the Voice panel.
            if !["NARRATOR_PROMPT", "VOICE_SOURCE", "VOICE_REFERENCE_FILE"].contains(k) {
                assert!(keys.contains(k), "{k} missing");
            }
        }
        let model = form.fields().find(|f| f.key == "TTS_MODEL_NAME").unwrap();
        assert_eq!(model.value, "0.6B (4 GB+ VRAM)");
    }
}
