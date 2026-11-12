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

//! Linux desktop entry, so GNOME and KDE show the app icon.
//!
//! Wayland has no window icon: the shell looks up the `.desktop` file whose
//! name matches the window's app ID and takes the icon from there (GNOME does
//! the same for X11 windows via `StartupWMClass`). The GUI therefore writes
//! `~/.local/share/applications/text-to-audiobook.desktop` and the icon at
//! start. The entry starts the running binary in the current project folder.
//! An entry without our marker key was written by the user and is left alone.

use std::io;
use std::path::{Path, PathBuf};

use crate::gui::APP_ID;

/// Marks entries this module wrote (and may therefore rewrite).
const MARKER: &str = "X-TextToAudiobook-Managed=true";

/// What [`install`] did.
#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Written,
    Unchanged,
    /// The user's own entry exists; nothing was touched.
    UserManaged,
}

/// Install entry and icon for the current user; failures are only logged.
pub fn install_for_current_user(project_root: &Path, icon_png: &[u8]) {
    let Some(data_home) = data_home() else {
        log::debug!("desktop entry skipped: neither XDG_DATA_HOME nor HOME set");
        return;
    };
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(e) => {
            log::debug!("desktop entry skipped: {e}");
            return;
        }
    };
    let root = std::fs::canonicalize(project_root).unwrap_or_else(|_| project_root.to_path_buf());
    let (Some(exe), Some(root)) = (exe.to_str(), root.to_str()) else {
        log::debug!("desktop entry skipped: path is not UTF-8");
        return;
    };
    match install(&data_home, exe, root, icon_png) {
        Ok(Outcome::Written) => log::info!(
            "desktop entry for the app icon written: {}",
            desktop_file(&data_home).display()
        ),
        Ok(Outcome::Unchanged) => {}
        Ok(Outcome::UserManaged) => log::debug!(
            "desktop entry {} is the user's own, left as is",
            desktop_file(&data_home).display()
        ),
        Err(e) => log::warn!("desktop entry not written: {e}"),
    }
}

/// `$XDG_DATA_HOME`, else `~/.local/share`.
fn data_home() -> Option<PathBuf> {
    let xdg = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from);
    if let Some(dir) = xdg.filter(|d| d.is_absolute()) {
        return Some(dir);
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    home.is_absolute().then(|| home.join(".local/share"))
}

fn desktop_file(data_home: &Path) -> PathBuf {
    data_home
        .join("applications")
        .join(format!("{APP_ID}.desktop"))
}

fn icon_file(data_home: &Path) -> PathBuf {
    data_home
        .join("icons/hicolor/256x256/apps")
        .join(format!("{APP_ID}.png"))
}

/// Write icon and entry below `data_home` unless they are up to date.
pub fn install(data_home: &Path, exe: &str, root: &str, icon_png: &[u8]) -> io::Result<Outcome> {
    let desktop = desktop_file(data_home);
    let existing = std::fs::read_to_string(&desktop).ok();
    if let Some(text) = &existing
        && !text.lines().any(|l| l.trim() == MARKER)
    {
        return Ok(Outcome::UserManaged);
    }
    let icon = icon_file(data_home);
    let icon_written = write_if_changed(&icon, icon_png)?;
    let entry = entry_text(exe, root, icon.to_str().unwrap_or(APP_ID));
    let entry_written = write_if_changed(&desktop, entry.as_bytes())?;
    Ok(if icon_written || entry_written {
        Outcome::Written
    } else {
        Outcome::Unchanged
    })
}

fn write_if_changed(path: &Path, content: &[u8]) -> io::Result<bool> {
    if std::fs::read(path).is_ok_and(|old| old == content) {
        return Ok(false);
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, content)?;
    Ok(true)
}

fn entry_text(exe: &str, root: &str, icon: &str) -> String {
    format!(
        "[Desktop Entry]
Type=Application
Name=text-to-audiobook
GenericName=Audiobook Maker
GenericName[de]=Hörbuch-Erzeugung
Comment=Turn Markdown texts into audiobooks
Comment[de]=Markdown-Texte in Hörbücher verwandeln
Keywords=TTS;Text-to-Speech;Audiobook;Hörbuch;
Exec={exec}
Path={path}
Icon={icon}
Terminal=false
Categories=AudioVideo;Audio;
StartupWMClass={APP_ID}
{MARKER}
",
        exec = exec_arg(exe),
        path = escape(root),
        icon = escape(icon),
    )
}

/// One argument of the `Exec` key: quoted if needed, `%` doubled, then
/// escaped like every other string value.
fn exec_arg(arg: &str) -> String {
    const RESERVED: &[char] = &[
        ' ', '\t', '\n', '"', '\'', '\\', '>', '<', '~', '|', '&', ';', '$', '*', '?', '#', '(',
        ')', '`',
    ];
    let arg = arg.replace('%', "%%");
    if !arg.contains(RESERVED) {
        return escape(&arg);
    }
    let mut quoted = String::from("\"");
    for c in arg.chars() {
        if matches!(c, '"' | '`' | '$' | '\\') {
            quoted.push('\\');
        }
        quoted.push(c);
    }
    quoted.push('"');
    escape(&quoted)
}

/// Escapes of the desktop entry `string` type.
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('\t', "\\t")
        .replace('\r', "\\r")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exec_quoting() {
        assert_eq!(
            exec_arg("/opt/tta/text-to-audiobook"),
            "/opt/tta/text-to-audiobook"
        );
        assert_eq!(exec_arg("/home/a b/tta"), "\"/home/a b/tta\"");
        assert_eq!(exec_arg("/x/100%"), "/x/100%%");
        // `$` is escaped inside the quotes, the backslash once more as a string
        assert_eq!(exec_arg("/a $b"), "\"/a \\\\$b\"");
    }

    #[test]
    fn writes_once_then_leaves_alone() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let icon = b"png bytes";
        let out = install(
            data,
            "/opt/tta/text-to-audiobook",
            "/home/u/Hörbücher",
            icon,
        )
        .unwrap();
        assert_eq!(out, Outcome::Written);
        let text = std::fs::read_to_string(desktop_file(data)).unwrap();
        assert!(text.starts_with("[Desktop Entry]\n"));
        assert!(text.contains("\nExec=/opt/tta/text-to-audiobook\n"));
        assert!(text.contains("\nPath=/home/u/Hörbücher\n"));
        assert!(text.contains(&format!("\nIcon={}\n", icon_file(data).display())));
        assert!(text.contains("\nStartupWMClass=text-to-audiobook\n"));
        assert_eq!(std::fs::read(icon_file(data)).unwrap(), icon);

        let again = install(
            data,
            "/opt/tta/text-to-audiobook",
            "/home/u/Hörbücher",
            icon,
        )
        .unwrap();
        assert_eq!(again, Outcome::Unchanged);

        // another project folder: the entry follows
        let moved = install(data, "/opt/tta/text-to-audiobook", "/home/u/Neu", icon).unwrap();
        assert_eq!(moved, Outcome::Written);
        let text = std::fs::read_to_string(desktop_file(data)).unwrap();
        assert!(text.contains("\nPath=/home/u/Neu\n"));
    }

    #[test]
    fn users_own_entry_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path();
        let own = "[Desktop Entry]\nType=Application\nName=Mein TTA\nExec=tta\n";
        std::fs::create_dir_all(data.join("applications")).unwrap();
        std::fs::write(desktop_file(data), own).unwrap();
        let out = install(data, "/opt/tta/text-to-audiobook", "/home/u", b"png").unwrap();
        assert_eq!(out, Outcome::UserManaged);
        assert_eq!(std::fs::read_to_string(desktop_file(data)).unwrap(), own);
        assert!(!icon_file(data).exists());
    }
}
