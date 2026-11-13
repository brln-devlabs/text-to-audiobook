#!/usr/bin/env bash
# Build text-to-audiobook-<version>-x86_64.AppImage from a release binary.
#
#   packaging/linux/build-appimage.sh <binary> <version> <output dir> [<cuda dir>]
#
# With <cuda dir> (made by bundle-cuda.sh) the CUDA build and its libraries
# go to usr/lib/text-to-audiobook/cuda/, where the normal build looks for
# them (src/gpu.rs), and the file is called
# text-to-audiobook-<version>-cuda-x86_64.AppImage.
#
# Needs appimagetool (APPIMAGETOOL, default: appimagetool in PATH) and, to
# bundle the fallback copy of libxkbcommon-x11, the packages
# libxkbcommon-x11-0 and libxcb-xkb1 on the build machine.
set -euo pipefail

bin=${1:?binary}
version=${2:?version}
out=${3:?output dir}
cuda=${4:-}
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"
tool=${APPIMAGETOOL:-appimagetool}

appdir="$(mktemp -d)/text-to-audiobook.AppDir"
mkdir -p "$appdir/usr/bin" "$appdir/usr/lib/fallback" \
    "$appdir/usr/share/applications" \
    "$appdir/usr/share/icons/hicolor/256x256/apps" \
    "$appdir/usr/share/doc/text-to-audiobook"

install -m 755 "$bin" "$appdir/usr/bin/text-to-audiobook"
install -m 755 "$here/AppRun" "$appdir/AppRun"
install -m 644 "$here/text-to-audiobook.desktop" "$appdir/text-to-audiobook.desktop"
install -m 644 "$here/text-to-audiobook.desktop" "$appdir/usr/share/applications/"
install -m 644 "$root/assets/icon-256.png" "$appdir/text-to-audiobook.png"
install -m 644 "$root/assets/icon-256.png" \
    "$appdir/usr/share/icons/hicolor/256x256/apps/text-to-audiobook.png"
ln -s text-to-audiobook.png "$appdir/.DirIcon"
for f in LICENSE README.md FORK_DE.md docs/LICENSES_NOTICES.md; do
    install -m 644 "$root/$f" "$appdir/usr/share/doc/text-to-audiobook/"
done

# Fallback libraries (see AppRun)
for lib in libxkbcommon-x11.so.0 libxcb-xkb.so.1; do
    path=$(ldconfig -p | awk -v l="$lib" '$1 == l && /x86-64/ { print $NF; exit }')
    if [ -n "$path" ]; then
        cp -L "$path" "$appdir/usr/lib/fallback/$lib"
    else
        echo "warning: $lib not found, not bundled" >&2
    fi
done

name="text-to-audiobook-$version-x86_64.AppImage"
if [ -n "$cuda" ]; then
    if [ ! -x "$cuda/text-to-audiobook-cuda" ]; then
        echo "error: $cuda/text-to-audiobook-cuda missing (run bundle-cuda.sh)" >&2
        exit 1
    fi
    mkdir -p "$appdir/usr/lib/text-to-audiobook"
    cp -a "$cuda" "$appdir/usr/lib/text-to-audiobook/cuda"
    name="text-to-audiobook-$version-cuda-x86_64.AppImage"
fi

mkdir -p "$out"
target="$out/$name"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$tool" --no-appstream "$appdir" "$target"
echo "$target"
