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

//! Stop requests from the GUI to worker threads.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// "Stop" finishes the current segment and then halts. "Force stop" also
/// abandons the segment in progress at the next possible point (between
/// sub-parts and model calls) and removes its partial output.
#[derive(Debug, Clone, Default)]
pub struct StopFlag {
    stop: Arc<AtomicBool>,
    force: Arc<AtomicBool>,
}

impl StopFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn force(&self) {
        self.force.store(true, Ordering::SeqCst);
        self.stop.store(true, Ordering::SeqCst);
    }

    pub fn clear(&self) {
        self.stop.store(false, Ordering::SeqCst);
        self.force.store(false, Ordering::SeqCst);
    }

    pub fn is_set(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    pub fn is_forced(&self) -> bool {
        self.force.load(Ordering::SeqCst)
    }
}

/// `Some(flag)` is set?
pub fn stopped(flag: Option<&StopFlag>) -> bool {
    flag.is_some_and(StopFlag::is_set)
}

/// `Some(flag)` force-stopped?
pub fn forced(flag: Option<&StopFlag>) -> bool {
    flag.is_some_and(StopFlag::is_forced)
}

/// Error returned when a force stop interrupted work.
#[derive(Debug, thiserror::Error)]
#[error("force-stopped by user")]
pub struct ForceStopped;
