#!/usr/bin/env bash
# Build text-to-audiobook-<version>-x86_64.AppImage from a release binary.
#
#   packaging/linux/build-appimage.sh <binary> <version> <output dir>
#
# Needs appimagetool (APPIMAGETOOL, default: appimagetool in PATH) and, to
# bundle the fallback copy of libxkbcommon-x11, the packages
# libxkbcommon-x11-0 and libxcb-xkb1 on the build machine.
set -euo pipefail

bin=${1:?binary}
version=${2:?version}
out=${3:?output dir}
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

mkdir -p "$out"
target="$out/text-to-audiobook-$version-x86_64.AppImage"
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$tool" --no-appstream "$appdir" "$target"
echo "$target"
