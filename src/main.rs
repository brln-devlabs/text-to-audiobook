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

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Parser, Subcommand};
use text_to_audiobook::config::{Config, Paths, Settings};
use text_to_audiobook::pipeline::{self, SegmentsByProject};
use text_to_audiobook::tts::TtsEngine;
use text_to_audiobook::{logging, pipeline::no_progress};

/// Markdown-Texte in Hörbücher verwandeln (Llama 3.2 + Qwen3-TTS).
///
/// Without a command the desktop GUI starts.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    /// Project root with settings.json, input/, output/, cache/, logs/, models/
    /// (default: $TTA_ROOT or the current directory)
    #[arg(long, global = true)]
    root: Option<PathBuf>,

    /// Input directory (default: <root>/input)
    #[arg(long, short, global = true)]
    input: Option<PathBuf>,

    /// Output directory (default: <root>/output)
    #[arg(long, short, global = true)]
    output: Option<PathBuf>,

    /// Only these projects (repeatable; default: all)
    #[arg(long = "project", short = 'p', global = true)]
    projects: Vec<String>,

    /// Run without models: neutral LLM answers and silent audio (dry run)
    #[arg(long, global = true)]
    mock_models: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the desktop GUI (default)
    Gui,
    /// List projects and their cache state
    Projects,
    /// Stage 1: clean, chunk and analyse texts (cached per project)
    Prepare,
    /// Stage 2: synthesize audio for cached segments and assemble it
    Generate,
    /// Stage 3: audit chunk WAVs, delete failed ones, cut noise
    Cleanup,
    /// Stage 4: assemble final audio from existing chunks (no TTS)
    Assemble,
    /// Stage 1 + 2 for all (or the selected) projects
    Run,
    /// Re-check languages of segments that differ from the project majority
    FixLanguages,
    /// Regenerate the voice reference cache/voice_reference.wav
    VoiceRef,
    /// Show how NORMALIZE_NUMBERS rewrites German text (argument or stdin)
    Normalize {
        /// Text to normalize; reads standard input when omitted
        text: Option<String>,
    },
}

fn selected(cli: &Cli) -> Option<&[String]> {
    (!cli.projects.is_empty()).then_some(cli.projects.as_slice())
}

fn cached(cfg: &Config, input: &std::path::Path, cli: &Cli) -> SegmentsByProject {
    let mut segs = pipeline::cached_segments_by_project(cfg, input);
    if let Some(sel) = selected(cli) {
        segs.retain(|name, _| sel.contains(name));
    }
    segs
}

fn run(cli: Cli) -> Result<()> {
    let paths = match &cli.root {
        Some(r) => Paths::new(r),
        None => Paths::from_env(),
    };
    paths.ensure_dirs()?;
    logging::bootstrap(Some(&paths.log_file()));
    let mut settings = Settings::load(&paths.settings_file());
    settings.mock_models = cli.mock_models;
    logging::set_levels(&settings.log_level_console, &settings.log_level_file);
    let cfg = Config::new(paths, settings);
    let input = cli
        .input
        .clone()
        .unwrap_or_else(|| cfg.paths.input_dir.clone());
    let output = cli
        .output
        .clone()
        .unwrap_or_else(|| cfg.paths.output_dir.clone());

    let command = cli.command.as_ref().unwrap_or(&Command::Gui);
    match command {
        Command::Gui => {
            #[cfg(feature = "gui")]
            {
                text_to_audiobook::gui::launch(cfg, input, output)?;
            }
            #[cfg(not(feature = "gui"))]
            {
                anyhow::bail!("built without the GUI (feature \"gui\"); see --help for commands");
            }
        }
        Command::Projects => {
            for p in pipeline::discover_projects(&input) {
                let segs = pipeline::load_cached_segments(
                    &cfg.paths.project_cache_dir(&p.name),
                    &cfg.settings.storage_backend,
                );
                if segs.is_empty() {
                    println!("{}  ({} file(s))", p.name, p.files.len());
                } else {
                    println!("{}  [cached: {} segments]", p.name, segs.len());
                }
            }
        }
        Command::Prepare => {
            pipeline::prepare_segments(&cfg, &input, &no_progress, None, selected(&cli))?;
        }
        Command::Generate => {
            let segs = cached(&cfg, &input, &cli);
            if segs.is_empty() {
                anyhow::bail!("No segments to generate; run `prepare` first.");
            }
            pipeline::generate_audio(&cfg, &segs, &output, &no_progress, None, None)?;
        }
        Command::Cleanup => {
            let names: Vec<String> = match selected(&cli) {
                Some(s) => s.to_vec(),
                None => pipeline::discover_projects(&input)
                    .into_iter()
                    .map(|p| p.name)
                    .collect(),
            };
            let report = pipeline::cleanup_noisy_chunks(&cfg, &output, Some(&names), &no_progress);
            for (name, r) in &report {
                println!(
                    "── {name} — cleaned {}, deleted {} ──",
                    r.edited,
                    r.deleted.len()
                );
                for (file, reason) in &r.deleted {
                    println!("  ✗ {file}    {reason}");
                }
            }
        }
        Command::Assemble => {
            let segs = cached(&cfg, &input, &cli);
            pipeline::assemble_final(&cfg, &segs, &output, &no_progress, None)?;
        }
        Command::Run => {
            let segs =
                pipeline::prepare_segments(&cfg, &input, &no_progress, None, selected(&cli))?;
            if !segs.is_empty() {
                pipeline::generate_audio(&cfg, &segs, &output, &no_progress, None, None)?;
            }
        }
        Command::FixLanguages => {
            let names: Vec<String> = match selected(&cli) {
                Some(s) => s.to_vec(),
                None => pipeline::discover_projects(&input)
                    .into_iter()
                    .map(|p| p.name)
                    .collect(),
            };
            pipeline::fix_languages(&cfg, &names)?;
        }
        Command::Normalize { text } => {
            let input = match text {
                Some(t) => t.clone(),
                None => {
                    let mut s = String::new();
                    std::io::Read::read_to_string(&mut std::io::stdin(), &mut s)?;
                    s
                }
            };
            println!("{}", text_to_audiobook::normalize_de::normalize(&input));
        }
        Command::VoiceRef => {
            let p = TtsEngine::regenerate_voice_reference(&cfg.settings, &cfg.paths)?;
            log::info!(
                "Voice reference saved to {}. Delete existing chunks and re-run Stage 2 to apply.",
                p.display()
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if log::max_level() == log::LevelFilter::Off {
                eprintln!("Error: {e:#}");
            } else {
                log::error!("{e:#}");
            }
            ExitCode::FAILURE
        }
    }
}
