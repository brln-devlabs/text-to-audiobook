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

//! Logging to the console, a rotating file (`logs/pipeline.log`, 5 MB x 5)
//! and optionally the GUI.
//!
//! Format: `YYYY-MM-DD HH:MM:SS,mmm | LEVEL    | module | message`

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, TryLockError};

use log::{Level, LevelFilter, Log, Metadata, Record};

use crate::config::{LOG_BACKUP_COUNT, LOG_MAX_BYTES};

const CRATE_PREFIX: &str = "text_to_audiobook";

type GuiSink = Box<dyn Fn(String) + Send + Sync>;

struct RotatingFile {
    path: PathBuf,
    file: Option<File>,
    size: u64,
    max_bytes: u64,
    backups: usize,
}

impl RotatingFile {
    fn open(path: &Path, max_bytes: u64, backups: usize) -> Self {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let file = OpenOptions::new().create(true).append(true).open(path).ok();
        let size = file
            .as_ref()
            .and_then(|f| f.metadata().ok())
            .map(|m| m.len())
            .unwrap_or(0);
        Self {
            path: path.to_path_buf(),
            file,
            size,
            max_bytes,
            backups,
        }
    }

    fn rotate(&mut self) {
        self.file = None;
        let name = |i: usize| PathBuf::from(format!("{}.{i}", self.path.display()));
        if self.backups > 0 {
            let _ = std::fs::remove_file(name(self.backups));
            for i in (1..self.backups).rev() {
                let _ = std::fs::rename(name(i), name(i + 1));
            }
            let _ = std::fs::rename(&self.path, name(1));
        } else {
            let _ = std::fs::remove_file(&self.path);
        }
        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .ok();
        self.size = 0;
    }

    fn write_line(&mut self, line: &str) {
        let len = line.len() as u64 + 1;
        if self.max_bytes > 0 && self.size > 0 && self.size + len >= self.max_bytes {
            self.rotate();
        }
        if let Some(f) = self.file.as_mut()
            && writeln!(f, "{line}").is_ok()
        {
            self.size += len;
        }
    }
}

struct State {
    console_level: LevelFilter,
    file_level: LevelFilter,
    file: Option<RotatingFile>,
    gui: Option<GuiSink>,
}

impl State {
    /// Send one record to the console (if `console`), the log file and the GUI.
    fn write(&mut self, record: &Record, console: bool) {
        let line = format_record(record);
        if console && record.level() <= self.console_level {
            eprintln!("{line}");
        }
        if record.level() <= self.file_level
            && let Some(f) = self.file.as_mut()
        {
            f.write_line(&line);
        }
        // The GUI shows everything down to DEBUG, like the Python version.
        if let Some(gui) = self.gui.as_ref().filter(|_| record.level() <= Level::Debug) {
            gui(line);
        }
    }
}

struct Logger {
    state: Mutex<State>,
}

static LOGGER: OnceLock<Logger> = OnceLock::new();

/// Python-style level name ("WARNING" instead of "WARN").
pub fn level_name(level: Level) -> &'static str {
    match level {
        Level::Error => "ERROR",
        Level::Warn => "WARNING",
        Level::Info => "INFO",
        Level::Debug => "DEBUG",
        Level::Trace => "TRACE",
    }
}

/// Parse "DEBUG"/"INFO"/"WARNING"/"ERROR" (case-insensitive); unknown -> default.
pub fn parse_level(name: &str, default: LevelFilter) -> LevelFilter {
    match name.trim().to_ascii_uppercase().as_str() {
        "TRACE" => LevelFilter::Trace,
        "DEBUG" => LevelFilter::Debug,
        "INFO" => LevelFilter::Info,
        "WARN" | "WARNING" => LevelFilter::Warn,
        "ERROR" | "CRITICAL" => LevelFilter::Error,
        "OFF" => LevelFilter::Off,
        _ => default,
    }
}

/// Format one record like Python's `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`.
pub fn format_record(record: &Record) -> String {
    let now = chrono::Local::now();
    let target = record.target();
    let name = target
        .strip_prefix(CRATE_PREFIX)
        .map(|t| t.trim_start_matches("::"))
        .filter(|t| !t.is_empty())
        .unwrap_or(target);
    format!(
        "{} | {:<8} | {} | {}",
        now.format("%Y-%m-%d %H:%M:%S,%3f"),
        level_name(record.level()),
        name,
        record.args()
    )
}

/// Third-party warnings that are known to be harmless and only confuse.
fn is_known_noise(record: &Record) -> bool {
    // qwen_tts warns about every sampled code_0 >= 2048, but its sampler
    // suppresses all of them except the end-of-speech token (2150): the
    // warning fires once at the normal end of every generated part.
    record.target().starts_with("qwen_tts::nn::generation")
        && record.args().to_string().contains("invalid for decoder")
}

impl Log for Logger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        // Our own messages at every level; third-party crates only from WARNING up.
        metadata.target().starts_with(CRATE_PREFIX) || metadata.level() <= Level::Warn
    }

    fn log(&self, record: &Record) {
        if !self.enabled(record.metadata()) || is_known_noise(record) {
            return;
        }
        let mut st = match self.state.lock() {
            Ok(st) => st,
            Err(poisoned) => poisoned.into_inner(),
        };
        st.write(record, true);
    }

    fn flush(&self) {
        if let Ok(mut st) = self.state.lock()
            && let Some(f) = st.file.as_mut().and_then(|f| f.file.as_mut())
        {
            let _ = f.flush();
        }
    }
}

fn logger() -> &'static Logger {
    LOGGER.get_or_init(|| Logger {
        state: Mutex::new(State {
            console_level: LevelFilter::Info,
            file_level: LevelFilter::Debug,
            file: None,
            gui: None,
        }),
    })
}

/// Install the logger: console at INFO, `log_file` at DEBUG. Safe to call
/// more than once (later calls only replace the log file).
pub fn bootstrap(log_file: Option<&Path>) {
    let l = logger();
    if let Ok(mut st) = l.state.lock() {
        st.file = log_file.map(|p| RotatingFile::open(p, LOG_MAX_BYTES, LOG_BACKUP_COUNT));
    }
    if log::set_logger(l).is_ok() {
        log::set_max_level(LevelFilter::Trace);
        install_panic_hook();
    }
}

/// Also write panics to the log file and the GUI. The default hook only
/// prints to stderr, which nobody sees when the GUI was started from a
/// desktop launcher, so a crash left no trace in `pipeline.log`.
fn install_panic_hook() {
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // try_lock: the panic may come from inside the logger itself.
        let st = match logger().state.try_lock() {
            Ok(st) => Some(st),
            Err(TryLockError::Poisoned(p)) => Some(p.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        };
        if let Some(mut st) = st {
            let thread = std::thread::current();
            let name = thread.name().unwrap_or("<unnamed>");
            st.write(
                &Record::builder()
                    .level(Level::Error)
                    .target("text_to_audiobook::panic")
                    .args(format_args!("Crash in thread '{name}': {info}"))
                    .build(),
                // The default hook prints it to the console.
                false,
            );
        }
        default_hook(info);
    }));
}

/// Send every formatted log line to `callback` as well (used by the GUI).
pub fn attach_gui_handler(callback: impl Fn(String) + Send + Sync + 'static) {
    if let Ok(mut st) = logger().state.lock() {
        st.gui = Some(Box::new(callback));
    }
}

/// Reconfigure the console and file levels (called from the settings window).
pub fn set_levels(console_level: &str, file_level: &str) {
    if let Ok(mut st) = logger().state.lock() {
        st.console_level = parse_level(console_level, LevelFilter::Info);
        st.file_level = parse_level(file_level, LevelFilter::Debug);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_python_level_names() {
        assert_eq!(parse_level("warning", LevelFilter::Info), LevelFilter::Warn);
        assert_eq!(parse_level("DEBUG", LevelFilter::Info), LevelFilter::Debug);
        assert_eq!(parse_level("bogus", LevelFilter::Info), LevelFilter::Info);
    }

    #[test]
    fn end_of_speech_warning_is_dropped() {
        let noise = |target: &str, msg: &str| {
            is_known_noise(
                &Record::builder()
                    .level(Level::Warn)
                    .target(target)
                    .args(format_args!("{msg}"))
                    .build(),
            )
        };
        assert!(noise(
            "qwen_tts::nn::generation",
            "code_0 >= 2048 (invalid for decoder) step=257 token=2150"
        ));
        assert!(!noise(
            "qwen_tts::nn::generation",
            "something else went wrong"
        ));
        assert!(!noise(
            "text_to_audiobook::tts",
            "code_0 >= 2048 (invalid for decoder)"
        ));
    }

    #[test]
    fn rotating_file_keeps_backups() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pipeline.log");
        let mut f = RotatingFile::open(&path, 100, 2);
        for i in 0..20 {
            f.write_line(&format!("line {i:03} ........................"));
        }
        assert!(path.exists());
        assert!(dir.path().join("pipeline.log.1").exists());
        assert!(dir.path().join("pipeline.log.2").exists());
        assert!(!dir.path().join("pipeline.log.3").exists());
        assert!(std::fs::metadata(&path).unwrap().len() < 100);
    }
}
