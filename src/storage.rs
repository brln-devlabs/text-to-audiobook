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

//! Persistence of stage-1 chunk metadata and TTS segments.
//!
//! Two backends with the same layout as the Python version:
//! - JSON: `<cache>/analyze/chunk_NNN.json` and `<cache>/segments/seg_GGGG.json`
//! - SQLite: `<cache>/storage.db` with tables `chunks` and `segments`

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

/// Interface for persisting chunk metadata and segments. Records are JSON
/// objects; errors on individual records are logged and reported as `None`
/// or `false`, like in the Python version.
pub trait StorageBackend: Send {
    /// Initialize the storage (create directories or tables).
    fn init_schema(&self) -> Result<()>;
    /// Persist a TTS-ready segment record.
    fn save_segment_meta(&self, segment_id: usize, data: &Value);
    /// Load a segment record by id (None if missing or unreadable).
    fn load_segment_meta(&self, segment_id: usize) -> Option<Value>;
    /// Persist analysis-stage chunk metadata.
    fn save_chunk_meta(&self, chunk_id: usize, data: &Value);
    /// Load chunk metadata by id (None if missing or unreadable).
    fn load_chunk_meta(&self, chunk_id: usize) -> Option<Value>;
    /// Delete a segment record; true if it existed.
    fn delete_segment(&self, segment_id: usize) -> bool;
    /// Delete chunk metadata; true if it existed.
    fn delete_chunk(&self, chunk_id: usize) -> bool;
    /// All segments ordered by id.
    fn list_all_segments(&self) -> Vec<Value>;
    /// All chunks ordered by id.
    fn list_all_chunks(&self) -> Vec<Value>;
    /// Remove all data.
    fn clear(&self);
}

/// Create the backend named by `STORAGE_BACKEND` ("json" or "sqlite").
pub fn create_storage(backend: &str, cache_dir: &Path) -> Result<Box<dyn StorageBackend>> {
    match backend.trim().to_lowercase().as_str() {
        "json" => {
            log::trace!("Using JSON storage backend");
            Ok(Box::new(JsonStorage::new(cache_dir)))
        }
        "sqlite" => {
            log::trace!("Using SQLite storage backend");
            Ok(Box::new(SqliteStorage::new(cache_dir)))
        }
        other => bail!("Unknown STORAGE_BACKEND: {other:?}. Must be 'json' or 'sqlite'."),
    }
}

// ---------------------------------------------------------------------------
// JSON files
// ---------------------------------------------------------------------------

pub struct JsonStorage {
    pub cache_dir: PathBuf,
    pub analyze_dir: PathBuf,
    pub segments_dir: PathBuf,
}

impl JsonStorage {
    pub fn new(cache_dir: &Path) -> Self {
        Self {
            cache_dir: cache_dir.to_path_buf(),
            analyze_dir: cache_dir.join("analyze"),
            segments_dir: cache_dir.join("segments"),
        }
    }

    fn chunk_path(&self, id: usize) -> PathBuf {
        self.analyze_dir.join(format!("chunk_{id:03}.json"))
    }

    fn segment_path(&self, id: usize) -> PathBuf {
        self.segments_dir.join(format!("seg_{id:04}.json"))
    }

    fn write(&self, path: &Path, data: &Value, what: &str) {
        if let Err(e) = self.init_schema() {
            log::error!("Failed to save {what}: {e}");
            return;
        }
        let result = serde_json::to_string_pretty(data)
            .map_err(anyhow::Error::from)
            .and_then(|text| std::fs::write(path, text).map_err(anyhow::Error::from));
        if let Err(e) = result {
            log::error!("Failed to save {what}: {e}");
        }
    }

    fn read(path: &Path, what: &str) -> Option<Value> {
        if !path.exists() {
            return None;
        }
        match std::fs::read_to_string(path)
            .map_err(anyhow::Error::from)
            .and_then(|t| serde_json::from_str(&t).map_err(anyhow::Error::from))
        {
            Ok(v) => Some(v),
            Err(e) => {
                log::error!("Failed to load {what}: {e}");
                None
            }
        }
    }

    fn delete(path: &Path, what: &str) -> bool {
        if !path.exists() {
            return false;
        }
        match std::fs::remove_file(path) {
            Ok(()) => true,
            Err(e) => {
                log::error!("Failed to delete {what}: {e}");
                false
            }
        }
    }

    /// Files `<prefix><number>.json` in `dir`, ordered by number.
    fn list(dir: &Path, prefix: &str) -> Vec<Value> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Vec::new();
        };
        let mut files: Vec<(u64, PathBuf)> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let num = name.strip_prefix(prefix)?.strip_suffix(".json")?;
                Some((num.parse().ok()?, e.path()))
            })
            .collect();
        files.sort();
        files
            .into_iter()
            .filter_map(|(_, p)| {
                match std::fs::read_to_string(&p)
                    .map_err(anyhow::Error::from)
                    .and_then(|t| serde_json::from_str(&t).map_err(anyhow::Error::from))
                {
                    Ok(v) => Some(v),
                    Err(e) => {
                        log::warn!("Failed to load {}: {e}", p.display());
                        None
                    }
                }
            })
            .collect()
    }
}

impl StorageBackend for JsonStorage {
    fn init_schema(&self) -> Result<()> {
        for d in [&self.cache_dir, &self.analyze_dir, &self.segments_dir] {
            std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
        }
        Ok(())
    }

    fn save_chunk_meta(&self, chunk_id: usize, data: &Value) {
        self.write(
            &self.chunk_path(chunk_id),
            data,
            &format!("chunk {chunk_id}"),
        );
    }

    fn load_chunk_meta(&self, chunk_id: usize) -> Option<Value> {
        Self::read(&self.chunk_path(chunk_id), &format!("chunk {chunk_id}"))
    }

    fn save_segment_meta(&self, segment_id: usize, data: &Value) {
        self.write(
            &self.segment_path(segment_id),
            data,
            &format!("segment {segment_id}"),
        );
    }

    fn load_segment_meta(&self, segment_id: usize) -> Option<Value> {
        Self::read(
            &self.segment_path(segment_id),
            &format!("segment {segment_id}"),
        )
    }

    fn delete_chunk(&self, chunk_id: usize) -> bool {
        Self::delete(&self.chunk_path(chunk_id), &format!("chunk {chunk_id}"))
    }

    fn delete_segment(&self, segment_id: usize) -> bool {
        Self::delete(
            &self.segment_path(segment_id),
            &format!("segment {segment_id}"),
        )
    }

    fn list_all_segments(&self) -> Vec<Value> {
        Self::list(&self.segments_dir, "seg_")
    }

    fn list_all_chunks(&self) -> Vec<Value> {
        Self::list(&self.analyze_dir, "chunk_")
    }

    fn clear(&self) {
        for d in [&self.analyze_dir, &self.segments_dir] {
            if d.exists()
                && let Err(e) = std::fs::remove_dir_all(d)
            {
                log::error!("Failed to clear {}: {e}", d.display());
            }
        }
        if let Err(e) = self.init_schema() {
            log::error!("Failed to clear storage: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// SQLite
// ---------------------------------------------------------------------------

/// Tables `chunks(chunk_id INTEGER PRIMARY KEY, data TEXT)` and
/// `segments(segment_id INTEGER PRIMARY KEY, data TEXT)` with JSON payloads.
pub struct SqliteStorage {
    pub cache_dir: PathBuf,
    pub db_path: PathBuf,
    conn: Mutex<Option<Connection>>,
}

impl SqliteStorage {
    pub fn new(cache_dir: &Path) -> Self {
        Self {
            cache_dir: cache_dir.to_path_buf(),
            db_path: cache_dir.join("storage.db"),
            conn: Mutex::new(None),
        }
    }

    fn with_conn<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        let mut guard = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("lock poisoned"))?;
        if guard.is_none() {
            std::fs::create_dir_all(&self.cache_dir)?;
            let conn = Connection::open(&self.db_path)
                .with_context(|| format!("opening {}", self.db_path.display()))?;
            *guard = Some(conn);
        }
        f(guard.as_ref().expect("connection just opened"))
    }

    fn save(&self, table: &str, key: &str, id: usize, data: &Value, what: &str) {
        let result = self.with_conn(|c| {
            let json = serde_json::to_string(data)?;
            c.execute(
                &format!("INSERT OR REPLACE INTO {table} ({key}, data) VALUES (?1, ?2)"),
                params![id as i64, json],
            )?;
            Ok(())
        });
        if let Err(e) = result {
            log::error!("Failed to save {what}: {e}");
        }
    }

    fn load(&self, table: &str, key: &str, id: usize, what: &str) -> Option<Value> {
        let result = self.with_conn(|c| {
            let row: Option<String> = c
                .query_row(
                    &format!("SELECT data FROM {table} WHERE {key} = ?1"),
                    params![id as i64],
                    |r| r.get(0),
                )
                .optional()?;
            Ok(match row {
                Some(text) => Some(serde_json::from_str(&text)?),
                None => None,
            })
        });
        match result {
            Ok(v) => v,
            Err(e) => {
                log::error!("Failed to load {what}: {e}");
                None
            }
        }
    }

    fn delete(&self, table: &str, key: &str, id: usize, what: &str) -> bool {
        let result = self.with_conn(|c| {
            Ok(c.execute(
                &format!("DELETE FROM {table} WHERE {key} = ?1"),
                params![id as i64],
            )?)
        });
        match result {
            Ok(n) => n > 0,
            Err(e) => {
                log::error!("Failed to delete {what}: {e}");
                false
            }
        }
    }

    /// All rows ordered by id, each with the id injected under `key`.
    fn list(&self, table: &str, key: &str) -> Vec<Value> {
        let result = self.with_conn(|c| {
            let mut stmt = c.prepare(&format!("SELECT {key}, data FROM {table} ORDER BY {key}"))?;
            let rows = stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?;
            let mut out = Vec::new();
            for row in rows {
                let (id, text) = row?;
                let mut v: Value = serde_json::from_str(&text)?;
                if let Value::Object(m) = &mut v {
                    m.insert(key.to_string(), Value::from(id));
                }
                out.push(v);
            }
            Ok(out)
        });
        result.unwrap_or_else(|e| {
            log::error!("Failed to list {table}: {e}");
            Vec::new()
        })
    }

    /// Close the database connection.
    pub fn close(&self) {
        if let Ok(mut g) = self.conn.lock() {
            *g = None;
        }
    }
}

impl StorageBackend for SqliteStorage {
    fn init_schema(&self) -> Result<()> {
        self.with_conn(|c| {
            c.execute_batch(
                "CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id INTEGER PRIMARY KEY,
                    data TEXT NOT NULL
                );",
            )?;
            Ok(())
        })
        .context("Failed to initialize schema")
    }

    fn save_chunk_meta(&self, chunk_id: usize, data: &Value) {
        self.save(
            "chunks",
            "chunk_id",
            chunk_id,
            data,
            &format!("chunk {chunk_id}"),
        );
    }

    fn load_chunk_meta(&self, chunk_id: usize) -> Option<Value> {
        self.load("chunks", "chunk_id", chunk_id, &format!("chunk {chunk_id}"))
    }

    fn save_segment_meta(&self, segment_id: usize, data: &Value) {
        self.save(
            "segments",
            "segment_id",
            segment_id,
            data,
            &format!("segment {segment_id}"),
        );
    }

    fn load_segment_meta(&self, segment_id: usize) -> Option<Value> {
        self.load(
            "segments",
            "segment_id",
            segment_id,
            &format!("segment {segment_id}"),
        )
    }

    fn delete_chunk(&self, chunk_id: usize) -> bool {
        self.delete("chunks", "chunk_id", chunk_id, &format!("chunk {chunk_id}"))
    }

    fn delete_segment(&self, segment_id: usize) -> bool {
        self.delete(
            "segments",
            "segment_id",
            segment_id,
            &format!("segment {segment_id}"),
        )
    }

    fn list_all_chunks(&self) -> Vec<Value> {
        self.list("chunks", "chunk_id")
    }

    fn list_all_segments(&self) -> Vec<Value> {
        self.list("segments", "segment_id")
    }

    fn clear(&self) {
        let result = self.with_conn(|c| {
            c.execute_batch("DELETE FROM chunks; DELETE FROM segments;")?;
            Ok(())
        });
        if let Err(e) = result {
            log::error!("Failed to clear storage: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn backends(dir: &Path) -> Vec<Box<dyn StorageBackend>> {
        vec![
            create_storage("json", &dir.join("j")).unwrap(),
            create_storage("SQLite ", &dir.join("s")).unwrap(),
        ]
    }

    #[test]
    fn json_init_creates_directories() {
        let dir = tempfile::tempdir().unwrap();
        let s = JsonStorage::new(&dir.path().join("cache"));
        s.init_schema().unwrap();
        assert!(dir.path().join("cache/analyze").is_dir());
        assert!(dir.path().join("cache/segments").is_dir());
    }

    #[test]
    fn sqlite_init_creates_database() {
        let dir = tempfile::tempdir().unwrap();
        let s = SqliteStorage::new(&dir.path().join("cache"));
        s.init_schema().unwrap();
        assert!(dir.path().join("cache/storage.db").exists());
    }

    #[test]
    fn save_load_delete_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        for s in backends(dir.path()) {
            s.init_schema().unwrap();
            let data = json!({"segments": [{"text": "Hi.", "emotion": "happy"}], "ends_paragraph": true, "language": "english"});
            s.save_chunk_meta(5, &data);
            assert_eq!(s.load_chunk_meta(5), Some(data));
            let seg = json!({"normalized_text": "Ünïcödé text", "emotion": "calm"});
            s.save_segment_meta(3, &seg);
            assert_eq!(s.load_segment_meta(3), Some(seg));
            assert_eq!(s.load_chunk_meta(999), None);
            assert_eq!(s.load_segment_meta(999), None);
            assert!(s.delete_chunk(5));
            assert!(!s.delete_chunk(5));
            assert!(s.delete_segment(3));
            assert_eq!(s.load_segment_meta(3), None);
        }
    }

    #[test]
    fn list_is_ordered_by_id() {
        let dir = tempfile::tempdir().unwrap();
        for s in backends(dir.path()) {
            s.init_schema().unwrap();
            for i in [2usize, 0, 10_000, 1] {
                s.save_segment_meta(i, &json!({"id": i}));
                s.save_chunk_meta(i, &json!({"id": i}));
            }
            let ids: Vec<u64> = s
                .list_all_segments()
                .iter()
                .map(|v| v["id"].as_u64().unwrap())
                .collect();
            assert_eq!(ids, [0, 1, 2, 10_000]);
            let ids: Vec<u64> = s
                .list_all_chunks()
                .iter()
                .map(|v| v["id"].as_u64().unwrap())
                .collect();
            assert_eq!(ids, [0, 1, 2, 10_000]);
            s.clear();
            assert!(s.list_all_segments().is_empty());
            assert!(s.list_all_chunks().is_empty());
            assert_eq!(s.load_chunk_meta(0), None);
        }
    }

    #[test]
    fn sqlite_listing_injects_id() {
        let dir = tempfile::tempdir().unwrap();
        let s = SqliteStorage::new(dir.path());
        s.init_schema().unwrap();
        s.save_segment_meta(4, &json!({"normalized_text": "x"}));
        assert_eq!(s.list_all_segments()[0]["segment_id"], 4);
    }

    #[test]
    fn unknown_backend_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = create_storage("postgres", dir.path()).err().unwrap();
        assert!(err.to_string().contains("Unknown STORAGE_BACKEND"));
    }
}
