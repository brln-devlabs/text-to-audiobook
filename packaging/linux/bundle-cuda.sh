#!/usr/bin/env bash
# Collect the CUDA build and the CUDA libraries it needs into one directory,
# which the CUDA packages install as /usr/lib/text-to-audiobook/cuda/.
#
#   packaging/linux/bundle-cuda.sh <cuda binary> <CUDA lib dir> <output dir>
#
# The binary (cargo build --features cuda) finds the libraries next to it
# through its RPATH $ORIGIN (build.rs). Copied are all libraries it needs,
# directly or through another CUDA library, except the NVIDIA driver
# (libcuda.so.1, comes with the driver) and the system libraries.
set -euo pipefail

bin=${1:?cuda binary}
libdir=${2:?CUDA lib dir}
out=${3:?output dir}

needed() {
    readelf -d "$1" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p'
}

is_system_lib() {
    case "$1" in
        libc.so.* | libm.so.* | libdl.so.* | librt.so.* | libpthread.so.* | \
            libgcc_s.so.* | libstdc++.so.* | ld-linux-x86-64.so.*) return 0 ;;
        libcuda.so.1) return 0 ;; # NVIDIA driver
    esac
    return 1
}

if ! readelf -d "$bin" | grep -q 'R.*PATH.*\$ORIGIN'; then
    echo "error: $bin has no RPATH \$ORIGIN (not built with --features cuda?)" >&2
    exit 1
fi
if ! needed "$bin" | grep -qx 'libcuda.so.1'; then
    echo "error: $bin does not link the NVIDIA driver (not a CUDA build?)" >&2
    exit 1
fi

rm -rf "$out"
mkdir -p "$out"
install -m 755 "$bin" "$out/text-to-audiobook-cuda"

queue=$(needed "$bin")
while [ -n "$queue" ]; do
    next=""
    for lib in $queue; do
        if is_system_lib "$lib" || [ -e "$out/$lib" ]; then
            continue
        fi
        if [ ! -e "$libdir/$lib" ]; then
            echo "error: $lib (needed by the CUDA build) not found in $libdir" >&2
            exit 1
        fi
        cp -L "$libdir/$lib" "$out/$lib"
        chmod 644 "$out/$lib"
        next="$next $(needed "$out/$lib")"
    done
    queue=$next
done

# NVRTC loads its builtins library at run time (dlopen), not through NEEDED.
if [ -e "$out/libnvrtc.so.12" ] || ls "$out"/libnvrtc.so.* >/dev/null 2>&1; then
    builtins=$(ls "$libdir"/libnvrtc-builtins.so.* 2>/dev/null | grep -v '\.alt\.' | head -1 || true)
    if [ -n "$builtins" ]; then
        soname=$(readelf -d "$builtins" | sed -n 's/.*(SONAME).*\[\(.*\)\]/\1/p')
        cp -L "$builtins" "$out/${soname:-$(basename "$builtins")}"
        chmod 644 "$out/${soname:-$(basename "$builtins")}"
    else
        echo "warning: libnvrtc-builtins not found in $libdir" >&2
    fi
fi

echo "CUDA bundle in $out:"
ls -l "$out"
du -sh "$out"
