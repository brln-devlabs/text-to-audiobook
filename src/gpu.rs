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

//! NVIDIA GPU: detection and the hand-over to the CUDA build.
//!
//! The CUDA installation packages carry two builds of the program: the
//! normal one, which computes on the CPU, and one built with
//! `--features cuda`, together with the CUDA libraries it needs (Windows:
//! `cuda\` next to the program, Linux: `../lib/text-to-audiobook/cuda/`).
//! The CUDA build is linked against the NVIDIA driver and does not even
//! start on a computer without one. So the normal build is the one that
//! starts. For the commands that speak, it asks the driver for a GPU
//! ([`detect`]) and, if there is one, runs the CUDA build in its place
//! ([`run_cuda_build`]); otherwise it carries on itself on the CPU. The
//! CUDA build checks the GPU once more against what its kernels were
//! compiled for and falls back to the CPU as well ([`cuda_device`]).
//!
//! `TTA_DEVICE=cpu` keeps everything on the CPU.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// File name of the CUDA build in the packages.
pub const CUDA_BUILD_NAME: &str = if cfg!(windows) {
    "text-to-audiobook-cuda.exe"
} else {
    "text-to-audiobook-cuda"
};

/// Set for the CUDA build: the program that started it (for the Linux
/// desktop entry, which has to start the normal build, not the CUDA one).
const LAUNCHER_ENV: &str = "TTA_LAUNCHER";

/// Why the CUDA build that came with the package is not used.
static CPU_NOTE: OnceLock<String> = OnceLock::new();

/// What speech synthesis computes on (shown in the GUI).
static DEVICE: Mutex<Option<ComputeDevice>> = Mutex::new(None);

/// What speech synthesis computes on, for the GUI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeDevice {
    /// Name of the GPU; `None`: the CPU.
    pub gpu: Option<String>,
    /// Details, or why the GPU is not used.
    pub detail: String,
    /// A speech model is loaded on it. Before that it is what loading the
    /// model will most likely pick.
    pub in_use: bool,
}

impl ComputeDevice {
    fn cpu(detail: impl Into<String>, in_use: bool) -> Self {
        Self {
            gpu: None,
            detail: detail.into(),
            in_use,
        }
    }
}

/// The current state, `None` until [`check_device`] or a model load.
pub fn current_device() -> Option<ComputeDevice> {
    DEVICE.lock().ok()?.clone()
}

/// Work out what the next model load will compute on, unless a model was
/// loaded already. Asks the driver, which can take a moment: call it from
/// a worker thread.
pub fn check_device() {
    let guess = expected_device();
    if let Ok(mut device) = DEVICE.lock()
        && device.is_none()
    {
        *device = Some(guess);
    }
}

fn set_device(device: ComputeDevice) {
    if let Ok(mut d) = DEVICE.lock() {
        *d = Some(device);
    }
}

/// A speech model was loaded on the GPU.
pub fn report_gpu() {
    #[cfg(feature = "cuda")]
    let (name, detail) = match detect() {
        Ok(gpu) => (gpu.name.clone(), gpu.to_string()),
        Err(_) => ("NVIDIA GPU".to_string(), "CUDA".to_string()),
    };
    #[cfg(not(feature = "cuda"))]
    let (name, detail) = ("Apple GPU".to_string(), "Metal".to_string());
    set_device(ComputeDevice {
        gpu: Some(name),
        detail,
        in_use: true,
    });
}

/// A speech model was loaded on the CPU; `why`: why not on the GPU, if
/// there was a reason beyond this build having no GPU support.
pub fn report_cpu(why: Option<String>) {
    let detail = match why {
        Some(why) => format!("The GPU is not used: {why}."),
        None => match current_device() {
            Some(d) if d.gpu.is_none() => d.detail,
            _ => expected_device().detail,
        },
    };
    set_device(ComputeDevice::cpu(detail, true));
}

/// What loading the model will pick, without loading anything.
fn expected_device() -> ComputeDevice {
    let cpu = |detail: String| ComputeDevice::cpu(detail, false);
    if cpu_forced() {
        return cpu("TTA_DEVICE=cpu is set: computing on the CPU.".into());
    }
    #[cfg(feature = "cuda")]
    {
        let (needs_cuda, needs_cc) = built_for();
        match detect() {
            Err(why) => cpu(format!("The GPU is not used: {why}.")),
            Ok(gpu) => match unsuitable(&gpu, needs_cuda, needs_cc) {
                Some(why) => cpu(format!("The GPU is not used: {why}.")),
                None => ComputeDevice {
                    gpu: Some(gpu.name.clone()),
                    detail: gpu.to_string(),
                    in_use: false,
                },
            },
        }
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        ComputeDevice {
            gpu: Some("Apple GPU".into()),
            detail: "Metal".into(),
            in_use: false,
        }
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        if let Some(why) = cpu_note() {
            return cpu(format!("The GPU is not used: {why}."));
        }
        match detect() {
            Ok(gpu) => cpu(format!(
                "This package computes on the CPU. The NVIDIA GPU ({}) is used by the \
                 CUDA package (file name with \"cuda\") or a build with --features cuda.",
                gpu.name
            )),
            Err(_) => cpu("This package computes on the CPU.".into()),
        }
    }
}

/// The first NVIDIA GPU as the driver describes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NvidiaGpu {
    pub name: String,
    /// (major, minor), e.g. (8, 6) for an RTX 30xx
    pub compute_capability: (i32, i32),
    pub memory_mb: u64,
    /// Newest CUDA version the driver supports, as the driver reports it
    /// (12080 = CUDA 12.8)
    pub driver_cuda: i32,
}

impl fmt::Display for NvidiaGpu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (major, minor) = self.compute_capability;
        write!(
            f,
            "{}, {} MB, compute capability {major}.{minor}, driver for CUDA {}",
            self.name,
            self.memory_mb,
            cuda_version_text(self.driver_cuda)
        )
    }
}

/// 12080 → "12.8"
pub fn cuda_version_text(version: i32) -> String {
    format!("{}.{}", version / 1000, (version % 1000) / 10)
}

/// `TTA_DEVICE=cpu`: never use the GPU.
pub fn cpu_forced() -> bool {
    std::env::var("TTA_DEVICE").is_ok_and(|v| v.trim().eq_ignore_ascii_case("cpu"))
}

/// Why the GPU is not used although the package has a CUDA build. `None`
/// if there is no CUDA build or it is running.
pub fn cpu_note() -> Option<&'static str> {
    CPU_NOTE.get().map(String::as_str)
}

/// The normal build that started this CUDA build (see [`LAUNCHER_ENV`]).
pub fn launcher_exe() -> Option<PathBuf> {
    std::env::var_os(LAUNCHER_ENV)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute() && p.is_file())
}

/// Where the packages put the CUDA build, relative to the normal one.
fn cuda_build_candidates(exe: &Path) -> Vec<PathBuf> {
    let Some(dir) = exe.parent() else {
        return Vec::new();
    };
    vec![
        // Windows installer and portable ZIP
        dir.join("cuda").join(CUDA_BUILD_NAME),
        // Linux: /usr/bin → /usr/lib/text-to-audiobook (.deb, AppImage)
        dir.join("../lib/text-to-audiobook/cuda")
            .join(CUDA_BUILD_NAME),
    ]
}

/// The CUDA build that came with this program, if any.
pub fn cuda_build() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    cuda_build_candidates(&exe)
        .into_iter()
        .find(|p| p.is_file())
}

/// Why `gpu` cannot run kernels compiled for compute capability `needs_cc`
/// with CUDA `needs_cuda` (PTX from a newer CUDA cannot be loaded by an
/// older driver). `None`: it can.
pub fn unsuitable(
    gpu: &NvidiaGpu,
    needs_cuda: Option<i32>,
    needs_cc: Option<(i32, i32)>,
) -> Option<String> {
    if let Some(cc) = needs_cc
        && gpu.compute_capability < cc
    {
        let (major, minor) = gpu.compute_capability;
        return Some(format!(
            "the GPU is too old (compute capability {major}.{minor}, this build needs {}.{} \
             or newer, e.g. RTX 30xx)",
            cc.0, cc.1
        ));
    }
    if let Some(version) = needs_cuda
        && gpu.driver_cuda < version
    {
        return Some(format!(
            "the NVIDIA driver is too old (supports CUDA {}, this build needs CUDA {}); \
             please update the NVIDIA driver",
            cuda_version_text(gpu.driver_cuda),
            cuda_version_text(version)
        ));
    }
    None
}

/// Text for a `CUresult` from `cuInit`.
fn init_error(code: i32) -> String {
    match code {
        100 => "no NVIDIA GPU found".into(),
        // CUDA_ERROR_INSUFFICIENT_DRIVER
        35 => "the NVIDIA driver does not match its kernel module (restart the computer)".into(),
        // CUDA_ERROR_UNKNOWN, typically after standby
        999 => "the NVIDIA driver reports an unknown error (after standby: restart the \
                computer or reload the nvidia_uvm module)"
            .into(),
        _ => format!("the NVIDIA driver cannot use the GPU (cuInit error {code})"),
    }
}

/// Ask the NVIDIA driver for the first GPU (once; later calls return the
/// same answer). `Err`: why there is none, for the log.
pub fn detect() -> Result<NvidiaGpu, String> {
    static ANSWER: OnceLock<Result<NvidiaGpu, String>> = OnceLock::new();
    ANSWER.get_or_init(ask_driver).clone()
}

#[cfg(any(target_os = "linux", windows))]
fn ask_driver() -> Result<NvidiaGpu, String> {
    use std::ffi::{CStr, c_char, c_int, c_uint};

    type Init = unsafe extern "system" fn(c_uint) -> c_int;
    type GetInt = unsafe extern "system" fn(*mut c_int) -> c_int;
    type DeviceGet = unsafe extern "system" fn(*mut c_int, c_int) -> c_int;
    type Attribute = unsafe extern "system" fn(*mut c_int, c_int, c_int) -> c_int;
    type Name = unsafe extern "system" fn(*mut c_char, c_int, c_int) -> c_int;
    type TotalMem = unsafe extern "system" fn(*mut usize, c_int) -> c_int;
    const DRIVER: &str = if cfg!(windows) {
        "nvcuda.dll"
    } else {
        "libcuda.so.1"
    };
    // CUdevice_attribute
    const COMPUTE_CAPABILITY_MAJOR: c_int = 75;
    const COMPUTE_CAPABILITY_MINOR: c_int = 76;

    // SAFETY: the NVIDIA driver library, called with the signatures from
    // cuda.h and with valid pointers into this stack frame.
    unsafe {
        let lib = libloading::Library::new(DRIVER)
            .map_err(|_| "no NVIDIA driver installed".to_string())?;
        macro_rules! func {
            ($name:literal, $ty:ty) => {
                *lib.get::<$ty>(concat!($name, "\0").as_bytes())
                    .map_err(|e| format!("the NVIDIA driver is too old ({e})"))?
            };
        }
        let cu_init = func!("cuInit", Init);
        let driver_get_version = func!("cuDriverGetVersion", GetInt);
        let device_get_count = func!("cuDeviceGetCount", GetInt);
        let device_get = func!("cuDeviceGet", DeviceGet);
        let device_get_attribute = func!("cuDeviceGetAttribute", Attribute);
        let device_get_name = func!("cuDeviceGetName", Name);
        let device_total_mem = func!("cuDeviceTotalMem_v2", TotalMem);

        let check = |what: &str, code: c_int| {
            if code == 0 {
                Ok(())
            } else {
                Err(format!("the NVIDIA driver failed at {what} (error {code})"))
            }
        };
        let code = cu_init(0);
        if code != 0 {
            return Err(init_error(code));
        }
        let mut driver_cuda = 0;
        check("cuDriverGetVersion", driver_get_version(&mut driver_cuda))?;
        let mut count = 0;
        check("cuDeviceGetCount", device_get_count(&mut count))?;
        if count < 1 {
            return Err(init_error(100));
        }
        let mut device = 0;
        check("cuDeviceGet", device_get(&mut device, 0))?;
        let (mut major, mut minor) = (0, 0);
        check(
            "cuDeviceGetAttribute",
            device_get_attribute(&mut major, COMPUTE_CAPABILITY_MAJOR, device),
        )?;
        check(
            "cuDeviceGetAttribute",
            device_get_attribute(&mut minor, COMPUTE_CAPABILITY_MINOR, device),
        )?;
        let mut name = [0 as c_char; 256];
        let name = if device_get_name(name.as_mut_ptr(), name.len() as c_int, device) == 0 {
            CStr::from_ptr(name.as_ptr())
                .to_string_lossy()
                .trim()
                .to_string()
        } else {
            "NVIDIA GPU".to_string()
        };
        let mut memory = 0usize;
        let _ = device_total_mem(&mut memory, device);
        // The driver stays loaded: unloading it after cuInit is not safe.
        std::mem::forget(lib);
        Ok(NvidiaGpu {
            name,
            compute_capability: (major, minor),
            memory_mb: memory as u64 / (1024 * 1024),
            driver_cuda,
        })
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
fn ask_driver() -> Result<NvidiaGpu, String> {
    Err("CUDA is only supported on Linux and Windows".into())
}

/// Normal build: run the CUDA build in place of this process if the
/// package has one and the computer has an NVIDIA GPU. `Some`: the CUDA
/// build ran, with this exit code. `None`: carry on here on the CPU
/// ([`cpu_note`] says why, if there is a CUDA build).
#[cfg(not(feature = "cuda"))]
pub fn run_cuda_build() -> Option<std::process::ExitCode> {
    let exe = cuda_build()?;
    let reason = if cpu_forced() {
        "TTA_DEVICE=cpu is set".to_string()
    } else {
        match detect() {
            Err(why) => why,
            Ok(_) => match launch::run(&exe) {
                Ok(code) => return Some(code),
                Err(why) => format!("the CUDA build could not start: {why}"),
            },
        }
    };
    let _ = CPU_NOTE.set(reason);
    None
}

/// Start the CUDA build with our arguments, wait for it and pass on its
/// exit code. `Err` if it could not start at all (missing libraries), so
/// the caller can carry on on the CPU.
#[cfg(not(feature = "cuda"))]
mod launch {
    use std::path::Path;
    use std::process::{Command, ExitCode, ExitStatus};

    pub fn run(exe: &Path) -> Result<ExitCode, String> {
        let mut cmd = Command::new(exe);
        cmd.args(std::env::args_os().skip(1));
        if let Ok(me) = std::env::current_exe() {
            cmd.env(super::LAUNCHER_ENV, me);
        }
        let status = platform::spawn_and_wait(&mut cmd).map_err(|e| e.to_string())?;
        if let Some(why) = platform::did_not_load(&status) {
            return Err(why.into());
        }
        Ok(exit_code(&status))
    }

    fn exit_code(status: &ExitStatus) -> ExitCode {
        if let Some(code) = status.code() {
            return ExitCode::from(u8::try_from(code).unwrap_or(1));
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            // Killed by a signal: report it like a shell does.
            if let Some(signal) = status.signal() {
                return ExitCode::from(u8::try_from(128 + signal).unwrap_or(1));
            }
        }
        ExitCode::FAILURE
    }

    #[cfg(unix)]
    mod platform {
        use std::process::{Command, ExitStatus};

        pub fn spawn_and_wait(cmd: &mut Command) -> std::io::Result<ExitStatus> {
            #[cfg(target_os = "linux")]
            {
                use std::os::unix::process::CommandExt;
                // SAFETY: prctl is async-signal-safe. If this process is
                // killed, the CUDA build gets SIGTERM instead of running on.
                unsafe {
                    cmd.pre_exec(|| {
                        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                        Ok(())
                    });
                }
            }
            let mut child = cmd.spawn()?;
            // Ctrl+C in the terminal reaches both processes; the CUDA build
            // decides what happens, this one just waits for it.
            // SAFETY: plain signal dispositions, restored below.
            unsafe {
                libc::signal(libc::SIGINT, libc::SIG_IGN);
                libc::signal(libc::SIGQUIT, libc::SIG_IGN);
            }
            let status = child.wait();
            unsafe {
                libc::signal(libc::SIGINT, libc::SIG_DFL);
                libc::signal(libc::SIGQUIT, libc::SIG_DFL);
            }
            status
        }

        /// The dynamic loader exits with 127 when a library is missing;
        /// the program itself never uses that code.
        pub fn did_not_load(status: &ExitStatus) -> Option<&'static str> {
            (status.code() == Some(127)).then_some("CUDA libraries missing or unreadable")
        }
    }

    #[cfg(windows)]
    mod platform {
        use std::process::{Command, ExitStatus};

        use windows_sys::Win32::System::Diagnostics::Debug::{
            SEM_FAILCRITICALERRORS, SetErrorMode,
        };

        pub fn spawn_and_wait(cmd: &mut Command) -> std::io::Result<ExitStatus> {
            // The child inherits the error mode: a missing DLL then ends it
            // with an exit code (see did_not_load) instead of a message box.
            // SAFETY: plain process flag, restored right after the start.
            let old = unsafe { SetErrorMode(SEM_FAILCRITICALERRORS) };
            let child = cmd.spawn();
            unsafe { SetErrorMode(old) };
            child?.wait()
        }

        /// NTSTATUS codes of a program that could not be loaded.
        pub fn did_not_load(status: &ExitStatus) -> Option<&'static str> {
            match status.code().map(|c| c as u32) {
                Some(0xC000_0135) => Some("a CUDA DLL is missing"),
                Some(0xC000_0139) => Some("the NVIDIA driver is too old (entry point not found)"),
                Some(0xC000_007B) => Some("a CUDA DLL is damaged or not 64-bit"),
                Some(0xC000_0142) => Some("a CUDA DLL failed to initialise"),
                _ => None,
            }
        }
    }

    #[cfg(not(any(unix, windows)))]
    mod platform {
        use std::process::{Command, ExitStatus};

        pub fn spawn_and_wait(cmd: &mut Command) -> std::io::Result<ExitStatus> {
            cmd.status()
        }

        pub fn did_not_load(_: &ExitStatus) -> Option<&'static str> {
            None
        }
    }
}

/// CUDA build: CUDA version and compute capability the kernels were
/// compiled for (set by build.rs when known).
#[cfg(feature = "cuda")]
fn built_for() -> (Option<i32>, Option<(i32, i32)>) {
    let cuda = option_env!("TTA_CUDA_VERSION").and_then(|v| v.parse().ok());
    let cc = option_env!("TTA_CUDA_COMPUTE_CAP")
        .and_then(|v| v.trim().parse::<i32>().ok())
        .map(|c| (c / 10, c % 10));
    (cuda, cc)
}

/// CUDA build: the GPU to compute on. `Err`: why the CPU is used instead.
#[cfg(feature = "cuda")]
pub fn cuda_device() -> Result<candle_core::Device, String> {
    if cpu_forced() {
        return Err("TTA_DEVICE=cpu is set".into());
    }
    let gpu = detect()?;
    log::info!("NVIDIA GPU: {gpu}");
    let (needs_cuda, needs_cc) = built_for();
    if let Some(why) = unsuitable(&gpu, needs_cuda, needs_cc) {
        return Err(why);
    }
    let device = std::panic::catch_unwind(|| candle_core::Device::new_cuda(0))
        .map_err(|_| "CUDA could not be started".to_string())?
        .map_err(|e| format!("CUDA could not be started: {e}"))?;
    probe(&device).map_err(|e| format!("the GPU failed a test calculation: {e}"))?;
    Ok(device)
}

/// A small calculation with cuBLAS and Candle's own kernels: kernels the
/// GPU or driver cannot load fail here and not in the middle of a book.
#[cfg(feature = "cuda")]
fn probe(device: &candle_core::Device) -> candle_core::Result<()> {
    use candle_core::{DType, Tensor};
    let t = Tensor::ones((2, 2), DType::F32, device)?;
    let sum = (t.matmul(&t)? + 1.0)?.sum_all()?.to_scalar::<f32>()?;
    if (sum - 12.0).abs() > 1e-3 {
        return Err(candle_core::Error::Msg(format!(
            "2×2 test gave {sum} instead of 12"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(cc: (i32, i32), driver: i32) -> NvidiaGpu {
        NvidiaGpu {
            name: "Test GPU".into(),
            compute_capability: cc,
            memory_mb: 4096,
            driver_cuda: driver,
        }
    }

    #[test]
    fn version_text() {
        assert_eq!(cuda_version_text(12080), "12.8");
        assert_eq!(cuda_version_text(12040), "12.4");
        assert_eq!(cuda_version_text(13000), "13.0");
    }

    #[test]
    fn suitable_gpus() {
        // RTX A500 (Ampere) with a new driver
        assert_eq!(
            unsuitable(&gpu((8, 6), 12080), Some(12080), Some((8, 0))),
            None
        );
        // RTX 50xx
        assert_eq!(
            unsuitable(&gpu((12, 0), 13000), Some(12080), Some((8, 0))),
            None
        );
        // nothing known about the build: no objection
        assert_eq!(unsuitable(&gpu((6, 1), 11040), None, None), None);
    }

    #[test]
    fn too_old_gpu() {
        let why = unsuitable(&gpu((7, 5), 12080), Some(12080), Some((8, 0))).unwrap();
        assert!(why.contains("compute capability 7.5"), "{why}");
        assert!(why.contains("8.0"), "{why}");
    }

    #[test]
    fn too_old_driver() {
        let why = unsuitable(&gpu((8, 6), 12040), Some(12080), Some((8, 0))).unwrap();
        assert!(why.contains("CUDA 12.4"), "{why}");
        assert!(why.contains("CUDA 12.8"), "{why}");
        assert!(why.contains("update"), "{why}");
    }

    #[test]
    fn init_errors_are_explained() {
        assert_eq!(init_error(100), "no NVIDIA GPU found");
        assert!(init_error(999).contains("standby"));
        assert!(init_error(3).contains("error 3"));
    }

    #[test]
    fn cuda_build_locations() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("usr/bin");
        std::fs::create_dir_all(&bin).unwrap();
        let exe = bin.join("text-to-audiobook");
        assert!(
            cuda_build_candidates(&exe)
                .into_iter()
                .all(|p| !p.is_file())
        );

        // Linux packages: /usr/lib/text-to-audiobook/cuda/
        let lib = dir.path().join("usr/lib/text-to-audiobook/cuda");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join(CUDA_BUILD_NAME), b"").unwrap();
        let found: Vec<_> = cuda_build_candidates(&exe)
            .into_iter()
            .filter(|p| p.is_file())
            .collect();
        assert_eq!(found.len(), 1);
        assert_eq!(
            std::fs::canonicalize(&found[0]).unwrap(),
            std::fs::canonicalize(lib.join(CUDA_BUILD_NAME)).unwrap()
        );

        // Windows: cuda\ next to the program
        let win = dir.path().join("app");
        std::fs::create_dir_all(win.join("cuda")).unwrap();
        std::fs::write(win.join("cuda").join(CUDA_BUILD_NAME), b"").unwrap();
        let first = cuda_build_candidates(&win.join("text-to-audiobook.exe"))
            .into_iter()
            .find(|p| p.is_file())
            .unwrap();
        assert_eq!(first, win.join("cuda").join(CUDA_BUILD_NAME));
    }
}
