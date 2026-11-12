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

//! Console front end for Windows, shipped as `text-to-audiobook.com`.
//!
//! The GUI build of `text-to-audiobook.exe` is a Windows (not console)
//! program, so a double click opens no black console window. A console that
//! starts it does not wait for it, though, and its output only shows by
//! luck. This small console program sits next to it: `cmd` and PowerShell
//! prefer `.com` over `.exe` when the command is typed without extension,
//! so `text-to-audiobook run` lands here. It starts the `.exe` with the same
//! arguments, passes input and output through pipes, waits and returns the
//! exit code. If the console is closed or Ctrl+C is pressed, a job object
//! takes the `.exe` down with it.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::{Command, ExitCode, Stdio};

/// The GUI program next to this one.
fn target_exe() -> Option<PathBuf> {
    let me = std::env::current_exe().ok()?;
    let exe = me.with_file_name(if cfg!(windows) {
        "text-to-audiobook.exe"
    } else {
        "text-to-audiobook"
    });
    (exe != me && exe.is_file()).then_some(exe)
}

/// Copy `from` to `to` until `from` ends, flushing after every read so
/// prompts appear before the program waits for input. After a write error
/// the rest is read and dropped, so the other side never blocks on a full
/// pipe.
fn pump(mut from: impl Read, mut to: impl Write) {
    let mut buf = [0u8; 8192];
    let mut writable = true;
    loop {
        match from.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                if writable {
                    writable = to.write_all(&buf[..n]).and_then(|()| to.flush()).is_ok();
                }
            }
        }
    }
}

fn main() -> ExitCode {
    let Some(exe) = target_exe() else {
        eprintln!("text-to-audiobook.exe not found next to this program.");
        return ExitCode::FAILURE;
    };
    let mut child = match Command::new(&exe)
        .args(std::env::args_os().skip(1))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Could not start {}: {e}", exe.display());
            return ExitCode::FAILURE;
        }
    };
    #[cfg(windows)]
    let _job = job::kill_with_this_process(&child);

    let stdin = child.stdin.take();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    // Not joined: it blocks on our console input until we exit.
    if let Some(input) = stdin {
        std::thread::spawn(move || pump(std::io::stdin(), input));
    }
    let out = stdout.map(|o| std::thread::spawn(move || pump(o, std::io::stdout())));
    let err = stderr.map(|e| std::thread::spawn(move || pump(e, std::io::stderr())));
    let status = child.wait();
    for t in [out, err].into_iter().flatten() {
        let _ = t.join();
    }
    match status {
        Ok(s) => match s.code() {
            Some(0) => ExitCode::SUCCESS,
            Some(code) => ExitCode::from(u8::try_from(code).unwrap_or(1)),
            None => ExitCode::FAILURE,
        },
        Err(e) => {
            eprintln!("Waiting for {} failed: {e}", exe.display());
            ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
mod job {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    /// Closes the job (and so ends the child) when dropped or when this
    /// process ends in any way.
    pub struct Job(HANDLE);

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe { CloseHandle(self.0) };
        }
    }

    /// Put `child` into a job that ends it together with this process.
    /// Best effort: without the job the program still works, it just keeps
    /// running if the console window is closed.
    pub fn kill_with_this_process(child: &Child) -> Option<Job> {
        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() {
                return None;
            }
            let job = Job(handle);
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job.0,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 || AssignProcessToJobObject(job.0, child.as_raw_handle() as HANDLE) == 0 {
                return None;
            }
            Some(job)
        }
    }
}
