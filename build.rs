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

//! Windows: embed the app icon (`assets/icon.ico`) and version info into the
//! .exe, so Explorer and shortcuts show it. Other platforms get the icon at
//! runtime from the GUI (`src/gui.rs`).
//!
//! CUDA build (`--features cuda`): note what the kernels are compiled for
//! and, on Linux, let the binary find the CUDA libraries next to it.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon.ico");
    #[cfg(windows)]
    windows_resources();
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        cuda();
    }
}

/// The CUDA version (nvcc) and compute capability (`CUDA_COMPUTE_CAP`,
/// read by Candle's kernel build) end up in the program, so it can tell a
/// GPU or driver that is too old for the kernels (src/gpu.rs).
///
/// Linux: the packages put the CUDA libraries next to the CUDA build
/// (`/usr/lib/text-to-audiobook/cuda/`), so the binary looks there first
/// (DT_RPATH, which also covers the libraries' own dependencies).
fn cuda() {
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    if let Ok(cap) = std::env::var("CUDA_COMPUTE_CAP") {
        println!("cargo:rustc-env=TTA_CUDA_COMPUTE_CAP={}", cap.trim());
    }
    if let Some(version) = nvcc_version() {
        println!("cargo:rustc-env=TTA_CUDA_VERSION={version}");
    }
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg-bins=-Wl,--disable-new-dtags,-rpath,$ORIGIN");
    }
}

/// "Cuda compilation tools, release 12.8, V12.8.93" → 12080 (the format
/// of `cuDriverGetVersion`).
fn nvcc_version() -> Option<u32> {
    let out = std::process::Command::new("nvcc")
        .arg("--version")
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let release = text.split("release ").nth(1)?;
    let version = release.split([',', ' ', '\n']).next()?;
    let (major, minor) = version.split_once('.')?;
    Some(major.trim().parse::<u32>().ok()? * 1000 + minor.trim().parse::<u32>().ok()? * 10)
}

#[cfg(windows)]
fn windows_resources() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let mut res = winresource::WindowsResource::new();
    res.set_icon("assets/icon.ico")
        .set("FileDescription", "text-to-audiobook")
        .set("ProductName", "text-to-audiobook");
    // A missing resource compiler must not break the build: the exe then
    // just has no icon in Explorer (the window still gets one at runtime).
    if let Err(e) = res.compile_for("text-to-audiobook") {
        println!("cargo:warning=Windows icon not embedded: {e}");
    }
}
