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

import ctypes
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import filedialog

    import customtkinter as ctk
except ImportError:
    ctk: Any = None  # type: ignore[no-redef]
    tk: Any = None  # type: ignore[no-redef]
    filedialog: Any = None  # type: ignore[no-redef]

from . import config, logging_setup
from .pipeline import (
    assemble_final,
    cleanup_noisy_chunks,
    discover_projects,
    generate_audio,
    load_cached_segments,
    prepare_segments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _async_raise(tid: int, exctype: type) -> None:
    """Raise an exception in the target thread by its native thread ID."""
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(exctype)
    )
    if res == 0:
        raise ValueError("invalid thread id")
    if res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


# ---------------------------------------------------------------------------
# Collapsible frame widget
# ---------------------------------------------------------------------------


class CollapsibleFrame(ctk.CTkFrame):
    """A frame whose content can be shown/hidden via an arrow-button header."""

    def __init__(self, parent, title: str, start_expanded: bool = False, **kwargs):
        super().__init__(parent, **kwargs)

        self._expanded = start_expanded
        self._extra_header_widgets: list = []

        # Header row
        self._header = ctk.CTkFrame(self, fg_color="transparent")
        self._header.pack(fill="x", padx=4, pady=(4, 0))

        self._toggle_btn = ctk.CTkButton(
            self._header,
            text=("▼ " if start_expanded else "▶ ") + title,
            anchor="w",
            fg_color="transparent",
            hover_color=("gray80", "gray30"),
            text_color=("gray10", "gray90"),
            font=("Arial", 12, "bold"),
            command=self.toggle,
            width=10,
        )
        self._toggle_btn.pack(side="left", fill="x", expand=True)

        # Content area — hidden until expanded
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        if start_expanded:
            self.content.pack(fill="both", expand=True, padx=4, pady=(2, 4))

    def add_header_widget(self, widget_factory):
        """Call with a callable(parent) → widget; places widget on the right of the header."""
        w = widget_factory(self._header)
        w.pack(side="right", padx=4)
        self._extra_header_widgets.append(w)
        return w

    def toggle(self):
        self._expanded = not self._expanded
        title = self._toggle_btn.cget("text")[2:]  # strip old arrow
        if self._expanded:
            self._toggle_btn.configure(text="▼ " + title)
            self.content.pack(fill="both", expand=True, padx=4, pady=(2, 4))
        else:
            self._toggle_btn.configure(text="▶ " + title)
            self.content.pack_forget()

    def expand(self):
        if not self._expanded:
            self.toggle()

    def collapse(self):
        if self._expanded:
            self.toggle()


# ---------------------------------------------------------------------------
# Settings popup
# ---------------------------------------------------------------------------


def _tts_model_choices() -> list:
    """Labels for the TTS model menu, plus the current model if it is custom."""
    labels = list(config.TTS_MODEL_CHOICES)
    current = config.tts_model_label(config.TTS_MODEL_NAME)
    if current not in labels:
        labels.append(current)
    return labels


class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("480x520")
        self.resizable(False, False)
        self.grab_set()

        self._fields: dict[str, ctk.CTkEntry | ctk.CTkOptionMenu] = {}

        frame = ctk.CTkScrollableFrame(self)
        frame.pack(fill="both", expand=True, padx=16, pady=16)

        def _section(label: str):
            ctk.CTkLabel(frame, text=label, font=("Arial", 13, "bold")).pack(
                anchor="w", pady=(12, 2)
            )

        def _row(
            parent,
            key: str,
            label: str,
            default,
            entry_type="int",
            choices=None,
            width=120,
        ):
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", pady=3)
            ctk.CTkLabel(row, text=label, width=220, anchor="w").pack(side="left")
            if entry_type == "choice":
                var = ctk.StringVar(value=str(default))
                if choices is None:
                    choices = ["DEBUG", "INFO", "WARNING", "ERROR"]
                w = ctk.CTkOptionMenu(row, values=choices, variable=var, width=width)
                w.pack(side="right")
                self._fields[key] = var
            else:
                var = ctk.StringVar(value=str(default))
                w = ctk.CTkEntry(row, textvariable=var, width=120)
                w.pack(side="right")
                self._fields[key] = var

        # Helper: prefer current live config value; fall back to in-code default.
        def _cur(key: str):
            return getattr(config, key, config.get_default(key))

        _section("Text Processing")
        _row(
            frame,
            "MAX_WORDS_PER_CHUNK",
            "Max words per chunk",
            _cur("MAX_WORDS_PER_CHUNK"),
        )

        _section("LLM")
        _row(
            frame,
            "LLM_MAX_RETRIES",
            "LLM retry attempts (per call)",
            _cur("LLM_MAX_RETRIES"),
        )
        _row(
            frame,
            "LLM_GPU_LAYERS",
            "LLM GPU layers (0 = CPU, -1 = all)",
            _cur("LLM_GPU_LAYERS"),
        )

        _section("Storage")
        _row(
            frame,
            "STORAGE_BACKEND",
            "Storage backend",
            _cur("STORAGE_BACKEND"),
            entry_type="choice",
            choices=["json", "sqlite"],
        )

        _section("TTS")
        _row(
            frame,
            "TTS_MODEL_NAME",
            "TTS model",
            config.tts_model_label(_cur("TTS_MODEL_NAME")),
            entry_type="choice",
            choices=_tts_model_choices(),
            width=170,
        )
        _row(
            frame,
            "TTS_LANGUAGE",
            "Language",
            _cur("TTS_LANGUAGE"),
            entry_type="choice",
            choices=list(config.TTS_LANGUAGE_CHOICES),
        )
        _row(
            frame,
            "TTS_DTYPE",
            "Precision (GPU)",
            _cur("TTS_DTYPE"),
            entry_type="choice",
            choices=list(config.TTS_DTYPE_CHOICES),
        )
        _row(frame, "TTS_MAX_TOKENS", "Max tokens per chunk", _cur("TTS_MAX_TOKENS"))
        _row(
            frame,
            "TTS_MAX_WORDS_FALLBACK",
            "Max words per chunk (no tokenizer)",
            _cur("TTS_MAX_WORDS_FALLBACK"),
        )

        _section("Audio")
        _row(frame, "VOICE_SEED", "Voice seed (RNG)", _cur("VOICE_SEED"))
        _row(frame, "SAMPLE_RATE", "Sample rate (Hz)", _cur("SAMPLE_RATE"))
        _row(frame, "CROSSFADE_MS", "Crossfade (ms)", _cur("CROSSFADE_MS"))
        _row(
            frame,
            "SILENCE_AFTER_PERIOD_MS",
            "Silence after sentence (ms)",
            _cur("SILENCE_AFTER_PERIOD_MS"),
        )
        _row(
            frame,
            "SILENCE_AFTER_PARAGRAPH_MS",
            "Silence after paragraph (ms)",
            _cur("SILENCE_AFTER_PARAGRAPH_MS"),
        )

        _section("Output Format")
        chapter_var = ctk.StringVar(value=str(_cur("CHAPTER_BY_CHAPTER")).lower())
        self._fields["CHAPTER_BY_CHAPTER"] = chapter_var
        _row(
            frame,
            "CHAPTER_BY_CHAPTER",
            "Chapter-by-chapter files",
            _cur("CHAPTER_BY_CHAPTER"),
            entry_type="choice",
            choices=["true", "false"],
        )
        _row(
            frame,
            "VOCALIZE_CHAPTER_NAMES",
            "Vocalize chapter names",
            _cur("VOCALIZE_CHAPTER_NAMES"),
            entry_type="choice",
            choices=["true", "false"],
        )

        _section("Logging")
        _row(
            frame,
            "LOG_LEVEL_CONSOLE",
            "Console log level",
            _cur("LOG_LEVEL_CONSOLE"),
            entry_type="choice",
        )
        _row(
            frame,
            "LOG_LEVEL_FILE",
            "File log level",
            _cur("LOG_LEVEL_FILE"),
            entry_type="choice",
        )

        # Buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkButton(btn_row, text="Apply & Save", command=self._apply).pack(
            side="left", padx=(0, 8)
        )
        ctk.CTkButton(
            btn_row,
            text="Reset to Defaults",
            command=self._reset_to_defaults,
            fg_color="#5a3280",
            hover_color="#3d2258",
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row,
            text="Cancel",
            command=self.destroy,
            fg_color="gray40",
            hover_color="gray30",
        ).pack(side="left")

    def _apply(self):
        int_keys = {
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
        }
        bool_keys = {"CHAPTER_BY_CHAPTER", "VOCALIZE_CHAPTER_NAMES"}
        settings = {}
        for key, var in self._fields.items():
            raw = var.get().strip()
            if key == "TTS_MODEL_NAME":
                val = config.tts_model_id(raw)
            elif key in int_keys:
                try:
                    val = int(raw)
                except ValueError:
                    logger.warning(
                        f"Settings: invalid integer for {key}: {raw!r} — skipped"
                    )
                    continue
            elif key in bool_keys:
                val = raw.lower() in ("true", "1", "yes")
            else:
                val = raw
            settings[key] = val
            setattr(config, key, val)

        # Apply log levels immediately.
        console_lvl = settings.get("LOG_LEVEL_CONSOLE", "INFO")
        file_lvl = settings.get("LOG_LEVEL_FILE", "DEBUG")
        logging_setup.set_levels(console_lvl, file_lvl)

        config.save_settings_to_file(settings)
        logger.info("Settings saved and applied.")
        self.destroy()

    def _reset_to_defaults(self):
        """Restore in-code defaults: update fields, config attrs, and settings.json."""
        defaults = config.reset_to_defaults()
        for key, var in self._fields.items():
            if key in defaults:
                val = defaults[key]
                if key in ("CHAPTER_BY_CHAPTER", "VOCALIZE_CHAPTER_NAMES"):
                    var.set(str(val).lower())
                elif key == "TTS_MODEL_NAME":
                    var.set(config.tts_model_label(val))
                else:
                    var.set(str(val))
        # Apply log levels right away so the running app reflects the reset.
        logging_setup.set_levels(
            defaults.get("LOG_LEVEL_CONSOLE", "INFO"),
            defaults.get("LOG_LEVEL_FILE", "DEBUG"),
        )
        logger.info("Settings reset to in-code defaults.")


# ---------------------------------------------------------------------------
# Close-confirmation dialog
# ---------------------------------------------------------------------------


class _ConfirmDialog(ctk.CTkToplevel):
    def __init__(self, parent, message: str):
        super().__init__(parent)
        self.title("Confirm")
        self.geometry("360x140")
        self.resizable(False, False)
        self.grab_set()
        self.result = False

        ctk.CTkLabel(self, text=message, wraplength=320).pack(pady=20, padx=16)
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack()
        ctk.CTkButton(
            row,
            text="Quit Anyway",
            fg_color="#a83232",
            hover_color="#7a2424",
            command=self._yes,
        ).pack(side="left", padx=8)
        ctk.CTkButton(row, text="Cancel", command=self._no).pack(side="left", padx=8)
        self.protocol("WM_DELETE_WINDOW", self._no)

    def _yes(self):
        self.result = True
        self.destroy()

    def _no(self):
        self.result = False
        self.destroy()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class TTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI TTS Pipeline")
        self.root.geometry("1000x860")
        self.root.minsize(780, 620)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Load persisted settings before building the GUI.
        config.load_settings_from_file()

        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._busy = False
        self._stage2_start_time: float | None = None
        self._stage2_total_chunks = 0
        self._stage2_done_count = 0
        self._project_checkboxes: dict[str, ctk.CTkCheckBox] = {}
        self._ready_projects: set[str] = set()

        # Outer frame fills the window
        frame = ctk.CTkFrame(root)
        frame.pack(padx=20, pady=20, fill="both", expand=True)

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=0)  # title
        frame.rowconfigure(1, weight=0)  # input dir
        frame.rowconfigure(2, weight=0)  # output dir
        frame.rowconfigure(3, weight=30)  # middle section (stories + right panels)
        frame.rowconfigure(4, weight=0)  # row A: 4 numbered stage buttons
        frame.rowconfigure(5, weight=0)  # row B: control buttons
        frame.rowconfigure(7, weight=0)  # progress row
        frame.rowconfigure(8, weight=0)  # log label
        frame.rowconfigure(9, weight=70)  # log textbox

        _PAD = {"padx": 5, "pady": 4}

        # ── Row 0: title ───────────────────────────────────────────────────
        ctk.CTkLabel(
            frame, text="AI Text-to-Speech Pipeline", font=("Arial", 18, "bold")
        ).grid(row=0, column=0, pady=(10, 4), sticky="ew")

        # ── Row 1: input directory ─────────────────────────────────────────
        input_frame = ctk.CTkFrame(frame)
        input_frame.grid(row=1, column=0, sticky="ew", **_PAD)
        ctk.CTkLabel(input_frame, text="Input Directory:").pack(side="left", padx=5)
        self.input_path = ctk.CTkLabel(
            input_frame, text=str(config.INPUT_DIR), text_color="gray"
        )
        self.input_path.pack(side="left", padx=5)
        ctk.CTkButton(
            input_frame, text="Browse", command=self.select_input_dir, width=100
        ).pack(side="right", padx=5)

        # ── Row 2: output directory ────────────────────────────────────────
        output_frame = ctk.CTkFrame(frame)
        output_frame.grid(row=2, column=0, sticky="ew", **_PAD)
        ctk.CTkLabel(output_frame, text="Output Directory:").pack(side="left", padx=5)
        self.output_path = ctk.CTkLabel(
            output_frame, text=str(config.OUTPUT_DIR), text_color="gray"
        )
        self.output_path.pack(side="left", padx=5)
        ctk.CTkButton(
            output_frame, text="Browse", command=self.select_output_dir, width=100
        ).pack(side="right", padx=5)

        # ── Row 3: two-column middle section ───────────────────────────────
        middle = ctk.CTkFrame(frame, fg_color="transparent")
        middle.grid(row=3, column=0, sticky="nsew", **_PAD)
        middle.columnconfigure(0, weight=55)  # left: stories
        middle.columnconfigure(1, weight=45)  # right: collapsible panels
        middle.rowconfigure(0, weight=1)

        # Left: stories list ------------------------------------------------
        projects_outer = ctk.CTkFrame(middle)
        projects_outer.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        projects_outer.rowconfigure(2, weight=1)
        projects_outer.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            projects_outer, text="Stories to process:", font=("Arial", 12, "bold")
        ).grid(row=0, column=0, sticky="w", padx=5, pady=(5, 0))

        select_row = ctk.CTkFrame(projects_outer, fg_color="transparent")
        select_row.grid(row=1, column=0, sticky="ew", padx=5, pady=2)
        ctk.CTkButton(
            select_row, text="Select All", command=self._select_all_projects, width=90
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            select_row, text="Select None", command=self._select_no_projects, width=90
        ).pack(side="left", padx=2)
        ctk.CTkButton(
            select_row, text="Refresh", command=self._refresh_project_list, width=80
        ).pack(side="left", padx=2)
        self._fix_lang_btn = ctk.CTkButton(
            select_row,
            text="Fix Languages",
            command=self._fix_languages,
            width=110,
            fg_color="#2a6a2a",
            hover_color="#1e4e1e",
        )
        self._fix_lang_btn.pack(side="right", padx=2)

        self.projects_list = ctk.CTkScrollableFrame(projects_outer)
        self.projects_list.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self._project_vars: dict[str, tk.BooleanVar] = {}

        # Right: collapsible panels -----------------------------------------
        right_col = ctk.CTkFrame(middle, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_col.columnconfigure(0, weight=1)
        right_col.rowconfigure(0, weight=0)
        right_col.rowconfigure(1, weight=0)
        right_col.rowconfigure(2, weight=1)  # spacer

        # Panel 2: Voice Description
        self._voice_panel = CollapsibleFrame(
            right_col, "Voice Description", start_expanded=False
        )
        self._voice_panel.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        def _make_reset_btn(parent):
            return ctk.CTkButton(
                parent,
                text="Reset Default",
                width=110,
                command=self._reset_voice_prompt,
            )

        self._voice_panel.add_header_widget(_make_reset_btn)

        def _make_regen_ref_btn(parent):
            return ctk.CTkButton(
                parent,
                text="Regen Voice Ref",
                width=120,
                command=self._regen_voice_reference,
            )

        self._voice_panel.add_header_widget(_make_regen_ref_btn)

        self.voice_prompt = ctk.CTkTextbox(
            self._voice_panel.content, wrap="word", height=120
        )
        self.voice_prompt.pack(fill="both", expand=True, padx=4, pady=4)
        self.voice_prompt.insert("1.0", config.NARRATOR_PROMPT)

        # Panel 3: Quick Config
        self._config_panel = CollapsibleFrame(
            right_col, "Quick Config", start_expanded=False
        )
        self._config_panel.grid(row=1, column=0, sticky="ew")

        self._quick_config_vars: dict[str, tk.StringVar] = {}

        model_row = ctk.CTkFrame(self._config_panel.content, fg_color="transparent")
        model_row.pack(fill="x", padx=4, pady=2)
        ctk.CTkLabel(model_row, text="TTS model", anchor="w", width=180).pack(
            side="left"
        )
        model_var = tk.StringVar(value=config.tts_model_label(config.TTS_MODEL_NAME))
        self._quick_config_vars["TTS_MODEL_NAME"] = model_var
        ctk.CTkOptionMenu(
            model_row, values=_tts_model_choices(), variable=model_var, width=170
        ).pack(side="right")

        _qc_defs = [
            ("MAX_WORDS_PER_CHUNK", "Max words / chunk"),
            ("TTS_MAX_TOKENS", "TTS max tokens / chunk"),
            ("VOICE_SEED", "Voice seed"),
            ("CROSSFADE_MS", "Crossfade (ms)"),
            ("SILENCE_AFTER_PERIOD_MS", "Silence / sentence (ms)"),
            ("SILENCE_AFTER_PARAGRAPH_MS", "Silence / paragraph (ms)"),
        ]
        for attr, label in _qc_defs:
            row_f = ctk.CTkFrame(self._config_panel.content, fg_color="transparent")
            row_f.pack(fill="x", padx=4, pady=2)
            ctk.CTkLabel(row_f, text=label, anchor="w", width=180).pack(side="left")
            var = tk.StringVar(value=str(getattr(config, attr)))
            self._quick_config_vars[attr] = var
            entry = ctk.CTkEntry(row_f, textvariable=var, width=80)
            entry.pack(side="right")

        apply_btn = ctk.CTkButton(
            self._config_panel.content,
            text="Apply Config",
            command=self._apply_quick_config,
            width=120,
        )
        apply_btn.pack(anchor="e", padx=4, pady=(4, 4))

        # ── Row 4: stage buttons (numbered pipeline steps) ────────────────
        stages_frame = ctk.CTkFrame(frame)
        stages_frame.grid(row=4, column=0, sticky="ew", **_PAD)

        self.start_btn = ctk.CTkButton(
            stages_frame,
            text="1. Prepare Segments",
            command=self.start_stage1,
            font=("Arial", 13),
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=2, pady=4)

        self.confirm_btn = ctk.CTkButton(
            stages_frame,
            text="2. Generate Audio",
            command=self.start_stage2,
            font=("Arial", 13),
            state="disabled",
        )
        self.confirm_btn.pack(side="left", expand=True, fill="x", padx=2, pady=4)

        self._cleanup_btn = ctk.CTkButton(
            stages_frame,
            text="3. Cleanup Noisy",
            command=self.start_cleanup,
            font=("Arial", 13),
            fg_color="#2a5a8a",
            hover_color="#1e3f60",
        )
        self._cleanup_btn.pack(side="left", expand=True, fill="x", padx=2, pady=4)

        self._assemble_btn = ctk.CTkButton(
            stages_frame,
            text="4. Assemble Final",
            command=self.start_assemble_final,
            font=("Arial", 13),
            fg_color="#2a6a2a",
            hover_color="#1e4e1e",
        )
        self._assemble_btn.pack(side="left", expand=True, fill="x", padx=2, pady=4)

        # ── Row 5: control buttons (stop / cleanup / utilities) ───────────
        control_frame = ctk.CTkFrame(frame)
        control_frame.grid(row=5, column=0, sticky="ew", **_PAD)

        self.stop_btn = ctk.CTkButton(
            control_frame,
            text="Stop",
            command=self._request_stop,
            fg_color="#a87832",
            hover_color="#7a5a24",
            width=80,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=4, pady=4)

        self.force_stop_btn = ctk.CTkButton(
            control_frame,
            text="Force Stop",
            command=self._force_stop,
            fg_color="#a83232",
            hover_color="#7a2424",
            width=95,
            state="disabled",
        )
        self.force_stop_btn.pack(side="left", padx=4, pady=4)

        self.clear_cache_btn = ctk.CTkButton(
            control_frame,
            text="Clear Cache",
            command=self.clear_cache,
            fg_color="#a83232",
            hover_color="#7a2424",
            width=110,
        )
        self.clear_cache_btn.pack(side="right", padx=4, pady=4)

        self.clean_output_btn = ctk.CTkButton(
            control_frame,
            text="Clean Output",
            command=self.clean_output,
            fg_color="#5a3280",
            hover_color="#3d2258",
            width=115,
        )
        self.clean_output_btn.pack(side="right", padx=4, pady=4)

        ctk.CTkButton(
            control_frame,
            text="⚙ Settings",
            command=self._open_settings,
            width=95,
        ).pack(side="right", padx=4, pady=4)

        ctk.CTkButton(
            control_frame,
            text="💾 Save Log",
            command=self._save_log,
            width=95,
        ).pack(side="right", padx=4, pady=4)

        ctk.CTkButton(
            control_frame,
            text="📂 Logs",
            command=self._open_logs_folder,
            width=80,
        ).pack(side="right", padx=4, pady=4)

        # ── Row 7: progress row ────────────────────────────────────────────
        progress_row = ctk.CTkFrame(frame, fg_color="transparent")
        progress_row.grid(row=7, column=0, sticky="ew", padx=5, pady=2)
        progress_row.columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(progress_row, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.progress.set(0)

        self._progress_label = ctk.CTkLabel(
            progress_row, text="", width=180, anchor="w"
        )
        self._progress_label.grid(row=0, column=1, sticky="w")

        # ── Row 8: log label ───────────────────────────────────────────────
        ctk.CTkLabel(frame, text="Log:", font=("Arial", 12, "bold")).grid(
            row=8, column=0, sticky="w", padx=5, pady=(6, 2)
        )

        # ── Row 9: log textbox ─────────────────────────────────────────────
        self.log_textbox = ctk.CTkTextbox(frame)
        self.log_textbox.grid(row=9, column=0, sticky="nsew", padx=5, pady=(0, 5))
        self.log_textbox.configure(state="disabled")

        # ── State ──────────────────────────────────────────────────────────
        self.input_dir = config.INPUT_DIR
        self.output_dir = config.OUTPUT_DIR
        self.segments: dict = {}

        self._attach_log_handler()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._refresh_project_list)
        self.root.after(200, self._check_existing_segments)

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------

    def _attach_log_handler(self):
        textbox = self.log_textbox
        root = self.root

        def append(msg):
            textbox.configure(state="normal")
            textbox.insert("end", msg + "\n")
            textbox.see("end")
            textbox.configure(state="disabled")

        class GUILogHandler(logging.Handler):
            def emit(self, record):
                msg = self.format(record)
                root.after(0, lambda m=msg: append(m))

        handler = GUILogHandler()
        handler.setFormatter(logging.Formatter(config.LOG_FORMAT))
        logging.getLogger().addHandler(handler)

    def _clear_log(self):
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def _open_logs_folder(self):
        try:
            os.startfile(config.LOGS_DIR)
        except Exception as e:
            logger.warning(f"Could not open logs folder: {e}")

    def _save_log(self):
        if filedialog is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile="tts_log.txt",
        )
        if not path:
            return
        content = self.log_textbox.get("1.0", "end")
        try:
            Path(path).write_text(content, encoding="utf-8")
            logger.info(f"Log saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save log: {e}")

    def _show_cleanup_report(self, results: dict):
        """Pop a dialog summarizing the cleanup audit (deleted chunks + reasons)."""
        if ctk is None:
            return
        total_edited = sum(r.get("edited", 0) for r in results.values())
        total_deleted = sum(len(r.get("deleted", [])) for r in results.values())

        win = ctk.CTkToplevel(self.root)
        win.title("Cleanup Report")
        win.geometry("680x460")
        win.transient(self.root)

        header = (
            f"Cleaned in place: {total_edited} chunk(s)\n"
            f"Deleted (failed audit): {total_deleted} chunk(s)\n"
        )
        if total_deleted:
            header += "Re-run Stage 2 to regenerate deleted chunks, then Stage 4 to assemble.\n"
        else:
            header += "All chunks passed discriminator + volume + voice-match checks.\n"

        ctk.CTkLabel(
            win, text=header, justify="left", anchor="w", font=("Arial", 12)
        ).pack(fill="x", padx=12, pady=(12, 4))

        body = ctk.CTkTextbox(win, wrap="none")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        lines = []
        for project_name, r in results.items():
            deleted = r.get("deleted", [])
            edited = r.get("edited", 0)
            lines.append(
                f"── {project_name} — cleaned {edited}, deleted {len(deleted)} ──"
            )
            if deleted:
                for fname, reason in deleted:
                    lines.append(f"  ✗ {fname}    {reason}")
            else:
                lines.append("  (none deleted)")
            lines.append("")

        body.insert("1.0", "\n".join(lines) if lines else "No projects scanned.")
        body.configure(state="disabled")

        ctk.CTkButton(win, text="Close", command=win.destroy, width=100).pack(
            pady=(0, 12)
        )

    # -----------------------------------------------------------------------
    # Settings & config
    # -----------------------------------------------------------------------

    def _open_settings(self):
        SettingsWindow(self.root)

    def _apply_quick_config(self):
        int_keys = {
            "MAX_WORDS_PER_CHUNK",
            "VOICE_SEED",
            "CROSSFADE_MS",
            "SILENCE_AFTER_PERIOD_MS",
            "SILENCE_AFTER_PARAGRAPH_MS",
            "TTS_MAX_TOKENS",
        }
        settings = {}
        for key, var in self._quick_config_vars.items():
            try:
                if key == "TTS_MODEL_NAME":
                    val = config.tts_model_id(var.get())
                else:
                    val = int(var.get()) if key in int_keys else var.get()
                setattr(config, key, val)
                settings[key] = val
            except ValueError:
                logger.warning(f"Quick config: invalid value for {key}: {var.get()!r}")
        config.save_settings_to_file(settings)
        logger.info("Quick config applied.")

    # -----------------------------------------------------------------------
    # Close confirmation
    # -----------------------------------------------------------------------

    def _on_close(self):
        if self._busy:
            dlg = _ConfirmDialog(self.root, "A model is still running.\nQuit anyway?")
            self.root.wait_window(dlg)
            if not dlg.result:
                return
        self.root.destroy()

    # -----------------------------------------------------------------------
    # Stop controls
    # -----------------------------------------------------------------------

    def _request_stop(self):
        self._stop_event.set()
        logger.info("Stop requested — finishing current segment then halting...")
        self.stop_btn.configure(state="disabled")

    def _force_stop(self):
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return
        logger.warning("Force stop requested — terminating synthesis thread...")
        try:
            _async_raise(self._worker_thread.ident, KeyboardInterrupt)
        except Exception as e:
            logger.error(f"Force stop failed: {e}")
        self.force_stop_btn.configure(state="disabled")
        self.stop_btn.configure(state="disabled")

    # -----------------------------------------------------------------------
    # Busy state (button locking)
    # -----------------------------------------------------------------------

    def _set_busy(self, busy: bool):
        self._busy = busy

        def _apply():
            state_cb = "disabled" if busy else "normal"
            if busy:
                self.start_btn.configure(state="disabled")
                self.confirm_btn.configure(state="disabled")
                self._cleanup_btn.configure(state="disabled")
                self._assemble_btn.configure(state="disabled")
                self.stop_btn.configure(state="normal")
                self.force_stop_btn.configure(state="normal")
                self.clear_cache_btn.configure(state="disabled")
                self.clean_output_btn.configure(state="disabled")
                self._fix_lang_btn.configure(state="disabled")
            else:
                self.start_btn.configure(state="normal")
                self._cleanup_btn.configure(state="normal")
                self._assemble_btn.configure(state="normal")
                self.stop_btn.configure(state="disabled")
                self.force_stop_btn.configure(state="disabled")
                self.clear_cache_btn.configure(state="normal")
                self.clean_output_btn.configure(state="normal")
                self._fix_lang_btn.configure(state="normal")

            # Lock / unlock story checkboxes
            for cb in self._project_checkboxes.values():
                try:
                    cb.configure(state=state_cb)
                except Exception:
                    pass

        self.root.after(0, _apply)

    # -----------------------------------------------------------------------
    # Directory selection
    # -----------------------------------------------------------------------

    def select_input_dir(self):
        if filedialog is None:
            return
        path = filedialog.askdirectory(initialdir=self.input_dir)
        if path:
            self.input_dir = Path(path)
            self.input_path.configure(text=str(self.input_dir))
            self.root.after(0, self._refresh_project_list)
            self.root.after(0, self._check_existing_segments)

    def select_output_dir(self):
        if filedialog is None:
            return
        path = filedialog.askdirectory(initialdir=self.output_dir)
        if path:
            self.output_dir = Path(path)
            self.output_path.configure(text=str(self.output_dir))

    # -----------------------------------------------------------------------
    # Project list
    # -----------------------------------------------------------------------

    def _refresh_project_list(self):
        if ctk is None:
            return
        previous = {n: v.get() for n, v in self._project_vars.items()}
        for child in self.projects_list.winfo_children():
            child.destroy()
        self._project_vars.clear()
        self._project_checkboxes.clear()

        projects = discover_projects(self.input_dir)
        if not projects:
            ctk.CTkLabel(
                self.projects_list,
                text="(no projects found in this directory)",
                text_color="gray",
            ).pack(anchor="w", padx=5, pady=2)
            return

        for project in projects:
            name = project["name"]
            cached = load_cached_segments(config.CACHE_DIR / name)
            if cached:
                suffix = f"  [cached: {len(cached)} segments]"
            else:
                suffix = f"  ({len(project['files'])} file(s))"
            var = tk.BooleanVar(value=previous.get(name, True))
            cb = ctk.CTkCheckBox(
                self.projects_list,
                text=name + suffix,
                variable=var,
            )
            cb.pack(anchor="w", padx=5, pady=2)
            self._project_vars[name] = var
            self._project_checkboxes[name] = cb

    def _selected_project_names(self):
        return [n for n, v in self._project_vars.items() if v.get()]

    def _select_all_projects(self):
        for v in self._project_vars.values():
            v.set(True)

    def _select_no_projects(self):
        for v in self._project_vars.values():
            v.set(False)

    # -----------------------------------------------------------------------
    # Voice prompt
    # -----------------------------------------------------------------------

    def _reset_voice_prompt(self):
        self.voice_prompt.delete("1.0", "end")
        self.voice_prompt.insert("1.0", config.get_default("NARRATOR_PROMPT"))

    def _regen_voice_reference(self):
        from pathlib import Path

        from .tts_engine import TTSEngine

        narrator_prompt = self._get_voice_prompt()
        ref_path = Path(config.VOICE_REF_PATH)
        self._set_busy(True)
        self._set_progress(0.0)
        logger.info("Regenerating voice reference...")

        def work():
            try:
                if ref_path.exists():
                    ref_path.unlink()
                engine = TTSEngine()
                engine._generate_voice_reference(ref_path, narrator_prompt)
                self._set_progress(1.0)
                logger.info(
                    f"Voice reference saved to {ref_path}. Delete existing chunks and re-run Stage 2 to apply."
                )
            except Exception as e:
                logger.exception(f"Voice reference generation failed: {e}")
            finally:
                self._set_busy(False)

        threading.Thread(target=work, daemon=True).start()

    def _get_voice_prompt(self) -> str:
        text = self.voice_prompt.get("1.0", "end").strip()
        return text or config.NARRATOR_PROMPT

    # -----------------------------------------------------------------------
    # Progress
    # -----------------------------------------------------------------------

    def _set_progress(self, value: float):
        self.root.after(0, lambda: self.progress.set(value))

    def _update_progress_label(self, done: int, total: int):
        if total == 0:
            label = ""
        elif self._stage2_start_time is not None and done > 0:
            elapsed = time.monotonic() - self._stage2_start_time
            avg_s = elapsed / done
            remaining = (total - done) * avg_s
            mins, secs = divmod(int(remaining), 60)
            eta = f"{mins}m {secs}s" if mins else f"{secs}s"
            label = f"Chunk {done} / {total}  (ETA ~{eta})"
        else:
            label = f"Chunk {done} / {total}"
        self.root.after(0, lambda: self._progress_label.configure(text=label))

    # -----------------------------------------------------------------------
    # Segment cache
    # -----------------------------------------------------------------------

    def _check_existing_segments(self):
        projects = discover_projects(self.input_dir)
        found = {}
        for project in projects:
            cache_dir = config.CACHE_DIR / project["name"]
            segs = load_cached_segments(cache_dir)
            if segs:
                found[project["name"]] = segs

        if found:
            self.segments = found
            total = sum(len(s) for s in found.values())
            logger.info(
                f"Found cached segments: {len(found)} project(s), {total} total — "
                "Stage 2 is ready."
            )
            self.confirm_btn.configure(state="normal")
        else:
            self.segments = {}
            self.confirm_btn.configure(state="disabled")

    # -----------------------------------------------------------------------
    # Stage 1
    # -----------------------------------------------------------------------

    def start_stage1(self):
        selected = self._selected_project_names()
        if not selected:
            logger.warning("No stories selected. Tick at least one project to process.")
            return
        self._stop_event.clear()
        self._clear_log()
        self.progress.set(0)
        self._progress_label.configure(text="")
        self._set_busy(True)
        logger.info(
            f"Stage 1: processing {len(selected)} project(s): " + ", ".join(selected)
        )

        def work():
            try:

                def progress_cb(msg):
                    cur = self.progress.get()
                    self._set_progress(min(cur + 0.02, 0.95))

                segments = prepare_segments(
                    self.input_dir,
                    self.output_dir,
                    progress_cb,
                    stop_event=self._stop_event,
                    selected_names=selected,
                )
                self.segments = segments
                self.root.after(0, self._on_stage1_done)
                self.root.after(0, self._refresh_project_list)
                self._set_progress(1.0)
            except Exception as e:
                logger.exception(f"Stage 1 failed: {e}")
            finally:
                self._set_busy(False)

        t = threading.Thread(target=work, daemon=True)
        self._worker_thread = t
        t.start()

    def _on_stage1_done(self):
        total = sum(len(segs) for segs in self.segments.values())
        if self._stop_event.is_set():
            logger.info(f"Stage 1 stopped: {total} segment(s) cached so far.")
        if not self.segments or total == 0:
            logger.info("Stage 1 produced no segments")
            return
        self.confirm_btn.configure(state="normal")
        for name, segs in self.segments.items():
            logger.info(f"  - {name}: {len(segs)} segments")
        if not self._stop_event.is_set():
            logger.info(
                f"Stage 1 done: {len(self.segments)} project(s), {total} segments. "
                "Click 'Confirm and Generate Audio' when ready."
            )

    # -----------------------------------------------------------------------
    # Stage 2
    # -----------------------------------------------------------------------

    def start_stage2(self):
        if not self.segments:
            logger.warning("No segments to generate; run Stage 1 first.")
            return
        selected = set(self._selected_project_names())
        to_process = {
            n: s for n, s in self.segments.items() if not selected or n in selected
        }
        if not to_process:
            logger.warning("No selected projects have cached segments to synthesize.")
            return
        skipped = [n for n in self.segments if n not in to_process]
        if skipped:
            logger.info(
                f"Skipping {len(skipped)} unselected project(s): " + ", ".join(skipped)
            )
        narrator_prompt = self._get_voice_prompt()
        self._stop_event.clear()
        self.progress.set(0)
        self._progress_label.configure(text="")
        self._set_busy(True)

        total_chunks = sum(len(s) for s in to_process.values())
        self._stage2_total_chunks = total_chunks
        self._stage2_done_count = 0
        self._stage2_start_time = time.monotonic()

        def work():
            try:
                done_count = [0]

                def on_chunk_done():
                    done_count[0] += 1
                    self._stage2_done_count = done_count[0]
                    frac = 0.85 * done_count[0] / max(total_chunks, 1)
                    self._set_progress(frac)
                    self._update_progress_label(done_count[0], total_chunks)

                generate_audio(
                    to_process,
                    self.output_dir,
                    None,
                    stop_event=self._stop_event,
                    narrator_prompt=narrator_prompt,
                    chunk_done_cb=on_chunk_done,
                )
                self._set_progress(1.0)
                self._update_progress_label(total_chunks, total_chunks)
                if self._stop_event.is_set():
                    logger.info("Stage 2 stopped by user — partial audio saved")
                else:
                    logger.info("Pipeline completed successfully")
            except KeyboardInterrupt:
                logger.warning(
                    "Stage 2 force-stopped. If the app behaves oddly, restart it to "
                    "clear GPU state."
                )
            except Exception as e:
                logger.exception(f"Stage 2 failed: {e}")
            finally:
                self._set_busy(False)
                self.root.after(0, lambda: self.confirm_btn.configure(state="normal"))

        t = threading.Thread(target=work, daemon=True)
        self._worker_thread = t
        t.start()

    # -----------------------------------------------------------------------
    # Stage 3: Cleanup noisy chunks
    # -----------------------------------------------------------------------

    def start_cleanup(self):
        selected = self._selected_project_names()
        if not selected:
            logger.warning("No stories selected. Tick at least one project to clean.")
            return
        self._stop_event.clear()
        self.progress.set(0)
        self._progress_label.configure(text="")
        self._set_busy(True)
        logger.info(
            f"Stage 3 (Cleanup): scanning {len(selected)} project(s) for noisy chunks..."
        )

        def work():
            try:

                def progress_cb(msg):
                    cur = self.progress.get()
                    self._set_progress(min(cur + 0.05, 0.95))

                results = cleanup_noisy_chunks(
                    self.output_dir,
                    project_names=selected,
                    progress_cb=progress_cb,
                )
                total_edited = sum(r.get("edited", 0) for r in results.values())
                total_deleted = sum(len(r.get("deleted", [])) for r in results.values())

                if total_deleted:
                    logger.info(
                        f"Cleanup done: {total_edited} cleaned, {total_deleted} deleted. "
                        "Re-run Stage 2 to re-synthesize the deleted chunks, then Stage 4 to assemble."
                    )
                else:
                    logger.info(
                        f"Cleanup done: {total_edited} chunk(s) cleaned, none deleted."
                    )
                self._set_progress(1.0)
                self.root.after(0, lambda: self._show_cleanup_report(results))
            except Exception as e:
                logger.exception(f"Cleanup failed: {e}")
            finally:
                self._set_busy(False)

        t = threading.Thread(target=work, daemon=True)
        self._worker_thread = t
        t.start()

    # -----------------------------------------------------------------------
    # Stage 4: Assemble final audio from existing chunks
    # -----------------------------------------------------------------------

    def start_assemble_final(self):
        if not self.segments:
            logger.warning("No segments loaded; run Stage 1 first.")
            return
        selected = set(self._selected_project_names())
        to_process = {
            n: s for n, s in self.segments.items() if not selected or n in selected
        }
        if not to_process:
            logger.warning("No selected projects have cached segments.")
            return
        self._stop_event.clear()
        self.progress.set(0)
        self._progress_label.configure(text="")
        self._set_busy(True)
        logger.info(
            f"Stage 4 (Assemble Final): assembling {len(to_process)} project(s)..."
        )

        def work():
            try:

                def progress_cb(msg):
                    cur = self.progress.get()
                    self._set_progress(min(cur + 0.05, 0.95))

                assemble_final(
                    to_process,
                    self.output_dir,
                    progress_cb,
                    stop_event=self._stop_event,
                )
                self._set_progress(1.0)
                if self._stop_event.is_set():
                    logger.info("Assemble Final stopped by user.")
                else:
                    logger.info("Assemble Final completed successfully.")
            except KeyboardInterrupt:
                logger.warning("Assemble Final force-stopped.")
            except Exception as e:
                logger.exception(f"Assemble Final failed: {e}")
            finally:
                self._set_busy(False)

        t = threading.Thread(target=work, daemon=True)
        self._worker_thread = t
        t.start()

    # -----------------------------------------------------------------------
    # Cache / output management
    # -----------------------------------------------------------------------

    def clear_cache(self):
        selected = set(self._selected_project_names())
        projects = discover_projects(self.input_dir)
        if selected:
            projects = [p for p in projects if p["name"] in selected]
        if not projects:
            logger.info("No projects selected — nothing to clear.")
            return
        cleared = 0
        for project in projects:
            cache_dir = config.CACHE_DIR / project["name"]
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    logger.info(f"Cleared cache: {cache_dir}")
                    cleared += 1
                except Exception as e:
                    logger.error(f"Failed to clear cache for {project['name']}: {e}")
        logger.info(f"Cleared cache for {cleared}/{len(projects)} selected project(s)")
        for name in selected:
            self.segments.pop(name, None)
        if not self.segments:
            self.confirm_btn.configure(state="disabled")
        self.progress.set(0)
        self._progress_label.configure(text="")
        self._refresh_project_list()

    def clean_output(self):
        selected = set(self._selected_project_names())
        projects = discover_projects(self.input_dir)
        if selected:
            projects = [p for p in projects if p["name"] in selected]
        if not projects:
            logger.info("No projects selected — nothing to clean.")
            return
        cleaned = 0
        for project in projects:
            out_dir = self.output_dir / project["name"]
            if out_dir.exists():
                try:
                    shutil.rmtree(out_dir)
                    logger.info(f"Cleaned output: {out_dir}")
                    cleaned += 1
                except Exception as e:
                    logger.error(f"Failed to clean output for {project['name']}: {e}")
        logger.info(f"Cleaned output for {cleaned}/{len(projects)} selected project(s)")
        self.progress.set(0)
        self._progress_label.configure(text="")

    # -----------------------------------------------------------------------
    # Fix Languages
    # -----------------------------------------------------------------------

    def _fix_languages(self):
        selected = self._selected_project_names()
        if not selected:
            logger.warning("No stories selected for language fix.")
            return
        self._set_busy(True)
        self._fix_lang_btn.configure(state="disabled")

        def work():
            from collections import Counter

            from .storage_factory import create_storage

            try:
                from .llm_normalizer import LLMNormalizer
            except Exception as e:
                logger.error(f"Could not load LLM for language fix: {e}")
                self._set_busy(False)
                return

            try:
                normalizer = LLMNormalizer(str(config.LLM_MODEL_PATH))
                normalizer.load()
            except Exception as e:
                logger.error(f"LLM load failed: {e}")
                self._set_busy(False)
                return

            try:
                for project_name in selected:
                    project_cache_dir = config.CACHE_DIR / project_name
                    if not project_cache_dir.exists():
                        logger.info(f"No cache for '{project_name}', skipping.")
                        continue

                    storage = create_storage(project_cache_dir)
                    segments_data = storage.list_all_segments()

                    if not segments_data:
                        logger.info(f"No segments for '{project_name}', skipping.")
                        continue

                    # Determine majority language.
                    langs = [d.get("language", "english") for d in segments_data]
                    majority_lang = Counter(langs).most_common(1)[0][0]
                    outliers = [
                        (i, d)
                        for i, d in enumerate(segments_data)
                        if d.get("language", "english") != majority_lang
                    ]

                    if not outliers:
                        logger.info(
                            f"'{project_name}': all segments already '{majority_lang}', nothing to fix."
                        )
                        continue

                    logger.info(
                        f"'{project_name}': majority lang='{majority_lang}', "
                        f"re-checking {len(outliers)} outlier(s)..."
                    )
                    fixed = []
                    for seg_id, d in outliers:
                        text = d.get("normalized_text", "")
                        old_lang = d.get("language", "unknown")
                        new_lang = normalizer.detect_language(text)
                        if new_lang != old_lang:
                            d["language"] = new_lang
                            storage.save_segment_meta(seg_id, d)
                            fixed.append(f"{old_lang}→{new_lang}")

                    if fixed:
                        logger.info(
                            f"Fixed {len(fixed)} language detection(s) in '{project_name}': "
                            + ", ".join(fixed)
                        )
                    else:
                        logger.info(
                            f"'{project_name}': LLM confirmed all outliers are correct — no changes."
                        )
            finally:
                try:
                    normalizer.unload()
                except Exception:
                    pass
                self._set_busy(False)

        t = threading.Thread(target=work, daemon=True)
        self._worker_thread = t
        t.start()


def launch():
    if ctk is None:
        logger.error("customtkinter not installed")
        return

    root = ctk.CTk()
    TTSApp(root)
    root.mainloop()
