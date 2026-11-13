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

// The GUI build is a Windows program, not a console program, so starting it
// from Explorer or the start menu opens no console window. Commands typed in
// a console go through `text-to-audiobook.com` (src/bin/console.rs).
#![cfg_attr(all(windows, feature = "gui"), windows_subsystem = "windows")]

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
    /// (default: $TTA_ROOT, else the current directory if it has settings.json
    /// or input/, else Documents/text-to-audiobook)
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
    /// Decide uncertain ordinals and Roman numerals interactively
    Review,
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

/// Interactive number review in the terminal.
fn review_cli(
    cfg: &Config,
    input: &std::path::Path,
    output: &std::path::Path,
    cli: &Cli,
) -> Result<()> {
    use std::io::{BufRead, Write};
    use text_to_audiobook::review::pending;
    if !cfg.settings.normalize_numbers {
        anyhow::bail!("set NORMALIZE_NUMBERS to true first (settings.json)");
    }
    let segs = cached(cfg, input, cli);
    if segs.is_empty() {
        anyhow::bail!("No segments to review; run `prepare` first.");
    }
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    // spoken.txt and review.txt follow the decisions.
    let write =
        |name: &str, segments: &[text_to_audiobook::segment::Segment]| match pipeline::write_texts(
            cfg, name, segments, output,
        ) {
            Ok(p) => println!(
                "{name}: texts updated in {}",
                p.parent().unwrap_or(&p).display()
            ),
            Err(e) => eprintln!("{name}: could not write texts: {e:#}"),
        };
    for (name, segments) in &segs {
        let mut store = text_to_audiobook::review::load_synced(
            &cfg.paths.project_cache_dir(name),
            name,
            segments,
            &cfg.settings,
        );
        let items = pending(name, segments, &cfg.settings, &store);
        if items.is_empty() {
            println!("{name}: nothing to review.");
            continue;
        }
        for (k, it) in items.iter().enumerate() {
            if store.is_decided(it.segment, &it.spot) {
                continue;
            }
            println!("\n── {name} — spot {} of {} ──", k + 1, items.len());
            println!("{}[[{}]]{}", it.before, it.spot.original, it.after);
            for (i, c) in it.spot.choices.iter().enumerate() {
                let label = if text_to_audiobook::review::is_removal(&c.replacement) {
                    "(remove)"
                } else {
                    &c.replacement
                };
                println!("  {:>2}) {label:<22} {}", i + 1, c.hint);
            }
            println!("   d) program default   s) skip   m) more context   q) quit");
            let choice = loop {
                print!("> ");
                std::io::stdout().flush()?;
                let Some(line) = lines.next() else {
                    write(name, segments);
                    return Ok(());
                };
                let line = line?;
                match line.trim() {
                    "q" => {
                        store.save()?;
                        write(name, segments);
                        return Ok(());
                    }
                    "s" => break None,
                    "m" => {
                        let texts: Vec<String> =
                            segments.iter().map(|s| s.normalized_text.clone()).collect();
                        let (above, below) = text_to_audiobook::review::neighbours(
                            &texts,
                            it.segment,
                            text_to_audiobook::review::CONTEXT_SEGMENTS,
                        );
                        for (n, t) in above {
                            println!("  {n:06} | {t}");
                        }
                        let t = &texts[it.segment];
                        println!(
                            "> {:06} | {}[[{}]]{}",
                            it.segment,
                            &t[..it.spot.start],
                            it.spot.original,
                            &t[it.spot.end..]
                        );
                        for (n, t) in below {
                            println!("  {n:06} | {t}");
                        }
                    }
                    "d" => break Some(None),
                    n => match n.parse::<usize>() {
                        Ok(i) if (1..=it.spot.choices.len()).contains(&i) => {
                            break Some(Some(it.spot.choices[i - 1].replacement.clone()));
                        }
                        _ => println!("Please enter a number, d, s, m or q."),
                    },
                }
            };
            if let Some(rep) = choice {
                store.decide(it, rep);
                store.save()?;
            }
        }
        write(name, segments);
    }
    Ok(())
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
            pipeline::prepare_segments(&cfg, &input, &output, &no_progress, None, selected(&cli))?;
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
            let segs = pipeline::prepare_segments(
                &cfg,
                &input,
                &output,
                &no_progress,
                None,
                selected(&cli),
            )?;
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
        Command::Review => review_cli(&cfg, &input, &output, &cli)?,
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

/// Commands that load the speech model and so profit from the GPU.
#[cfg(not(feature = "cuda"))]
fn speaks(cli: &Cli) -> bool {
    !cli.mock_models
        && matches!(
            cli.command,
            None | Some(Command::Gui | Command::Generate | Command::Run | Command::VoiceRef)
        )
}

fn main() -> ExitCode {
    #[cfg(all(windows, feature = "gui"))]
    let has_console = windows_console::attach_to_parent();
    let cli = Cli::parse();
    // CUDA packages: the CUDA build next to this one takes over if there is
    // an NVIDIA GPU (src/gpu.rs).
    #[cfg(not(feature = "cuda"))]
    if speaks(&cli)
        && let Some(code) = text_to_audiobook::gpu::run_cuda_build()
    {
        return code;
    }
    #[cfg(all(windows, feature = "gui"))]
    let is_gui = matches!(cli.command, None | Some(Command::Gui));
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            if log::max_level() == log::LevelFilter::Off {
                eprintln!("Error: {e:#}");
            } else {
                log::error!("{e:#}");
            }
            // Started by a double click: nobody sees stderr.
            #[cfg(all(windows, feature = "gui"))]
            if is_gui && !has_console {
                rfd::MessageDialog::new()
                    .set_level(rfd::MessageLevel::Error)
                    .set_title("text-to-audiobook")
                    .set_description(format!("The program could not start:\n\n{e:#}"))
                    .show();
            }
            ExitCode::FAILURE
        }
    }
}

/// Output for commands when the Windows GUI build is started from a console
/// directly (`text-to-audiobook.exe run`) instead of through the
/// `text-to-audiobook.com` front end.
#[cfg(all(windows, feature = "gui"))]
mod windows_console {
    use windows_sys::Win32::System::Console::{
        ATTACH_PARENT_PROCESS, AttachConsole, GetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE,
    };

    /// True if output can be seen: it is redirected (pipe, file, the `.com`
    /// front end) or the console of the parent process could be attached.
    pub fn attach_to_parent() -> bool {
        unsafe {
            let redirected = |h| {
                let h = GetStdHandle(h);
                !h.is_null() && h != windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE
            };
            if redirected(STD_OUTPUT_HANDLE) || redirected(STD_ERROR_HANDLE) {
                return true;
            }
            AttachConsole(ATTACH_PARENT_PROCESS) != 0
        }
    }
}
