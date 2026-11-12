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

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/icon.ico");
    #[cfg(windows)]
    windows_resources();
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
