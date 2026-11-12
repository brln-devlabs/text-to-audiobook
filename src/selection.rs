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

//! Which input projects are ticked in the GUI, remembered across runs in
//! `cache/project_selection.json`. Projects the file does not know yet are
//! ticked.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub const SELECTION_FILE: &str = "project_selection.json";

/// Ticked state per input directory ([`dir_key`]) and project name.
#[derive(Debug, Default)]
pub struct ProjectSelection {
    dirs: BTreeMap<String, BTreeMap<String, bool>>,
    path: PathBuf,
}

impl ProjectSelection {
    /// Read `cache_dir/project_selection.json`; a missing or broken file
    /// means nothing is known yet (everything ticked).
    pub fn load(cache_dir: &Path) -> Self {
        let path = cache_dir.join(SELECTION_FILE);
        let dirs = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        Self { dirs, path }
    }

    /// Stored state of `project` in the input directory `dir`; projects
    /// that are not stored yet are ticked.
    pub fn is_selected(&self, dir: &str, project: &str) -> bool {
        self.dirs
            .get(dir)
            .and_then(|m| m.get(project))
            .copied()
            .unwrap_or(true)
    }

    /// Replace the stored state of `dir` with `projects`. Projects that are
    /// no longer listed are forgotten, so they come back ticked when they
    /// reappear. An empty list (directory missing or empty) keeps the stored
    /// state. Returns whether anything changed.
    pub fn set<'a>(
        &mut self,
        dir: &str,
        projects: impl IntoIterator<Item = (&'a str, bool)>,
    ) -> bool {
        let new: BTreeMap<String, bool> = projects
            .into_iter()
            .map(|(name, selected)| (name.to_string(), selected))
            .collect();
        if new.is_empty() || self.dirs.get(dir) == Some(&new) {
            return false;
        }
        self.dirs.insert(dir.to_string(), new);
        true
    }

    pub fn save(&self) -> Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&self.path, serde_json::to_string_pretty(&self.dirs)?)
            .with_context(|| format!("writing {}", self.path.display()))
    }
}

/// Key of an input directory: relative to the project root when it lies
/// inside it (so the selection survives moving the whole folder), otherwise
/// the absolute path.
pub fn dir_key(root: &Path, input_dir: &Path) -> String {
    let abs = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let input = abs(input_dir);
    match input.strip_prefix(abs(root)) {
        Ok(rel) if rel.as_os_str().is_empty() => ".".to_string(),
        Ok(rel) => rel
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/"),
        Err(_) => input.to_string_lossy().into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_projects_are_ticked_and_state_survives_reload() {
        let dir = tempfile::tempdir().unwrap();
        let mut sel = ProjectSelection::load(dir.path());
        assert!(sel.is_selected("input", "A"));
        assert!(sel.set("input", [("A", true), ("B", false)]));
        assert!(!sel.set("input", [("A", true), ("B", false)]));
        sel.save().unwrap();

        let sel = ProjectSelection::load(dir.path());
        assert!(sel.is_selected("input", "A"));
        assert!(!sel.is_selected("input", "B"));
        assert!(sel.is_selected("input", "New"));
        // Same name in another input directory is independent.
        assert!(sel.is_selected("/elsewhere", "B"));
    }

    #[test]
    fn vanished_projects_are_forgotten_but_empty_lists_keep_state() {
        let dir = tempfile::tempdir().unwrap();
        let mut sel = ProjectSelection::load(dir.path());
        sel.set("input", [("A", false), ("B", false)]);
        assert!(!sel.set("input", std::iter::empty()));
        assert!(!sel.is_selected("input", "A"));
        assert!(sel.set("input", [("B", false)]));
        assert!(sel.is_selected("input", "A"));
        assert!(!sel.is_selected("input", "B"));
    }

    #[test]
    fn broken_file_ticks_everything() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join(SELECTION_FILE), "{not json").unwrap();
        assert!(ProjectSelection::load(dir.path()).is_selected("input", "A"));
    }

    #[test]
    fn dir_keys() {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("input");
        std::fs::create_dir_all(input.join("sub")).unwrap();
        assert_eq!(dir_key(root.path(), &input), "input");
        assert_eq!(dir_key(root.path(), &input.join("sub")), "input/sub");
        assert_eq!(dir_key(root.path(), root.path()), ".");
        let other = tempfile::tempdir().unwrap();
        let key = dir_key(root.path(), other.path());
        assert!(Path::new(&key).is_absolute(), "{key}");
    }
}
