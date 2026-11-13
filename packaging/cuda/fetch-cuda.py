#!/usr/bin/env python3
# This file is part of text-to-audiobook (AGPL-3.0-or-later).
"""Unpack parts of the CUDA toolkit into one directory usable as CUDA_PATH.

Downloads NVIDIA's redistributable archives
(https://developer.download.nvidia.com/compute/cuda/redist/), checks their
SHA-256 against the release manifest and merges them, so the release
workflow needs no installer or package repository and gets the same layout
on Linux and Windows (bin/, include/, lib/ or lib/x64/, nvvm/).

    fetch-cuda.py --version 12.8.1 --dest DIR [--drop-static] COMPONENT...

A component ending in "?" is optional: it is skipped if the release does
not have it (some releases split nvcc into cuda_crt and libnvvm).
"""

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"


def default_platform():
    system = {"Linux": "linux", "Windows": "windows"}.get(platform.system())
    if system is None or platform.machine().lower() not in ("x86_64", "amd64"):
        sys.exit(f"unsupported platform {platform.system()} {platform.machine()}")
    return f"{system}-x86_64"


def download(url, target):
    with urllib.request.urlopen(url) as response, open(target, "wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unpack(archive, into):
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(into)
    else:
        with tarfile.open(archive) as t:
            if hasattr(tarfile, "data_filter"):
                t.extractall(into, filter="data")
            else:
                t.extractall(into)


def merge(src, dest):
    """Copy the tree src into dest (symlinks stay symlinks)."""
    for item in src.iterdir():
        target = dest / item.name
        if item.is_dir() and not item.is_symlink():
            target.mkdir(parents=True, exist_ok=True)
            merge(item, target)
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.move(str(item), str(target))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", required=True, help="CUDA release, e.g. 12.8.1")
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--platform", default=None, help="linux-x86_64 or windows-x86_64")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument(
        "--drop-static",
        action="store_true",
        help="delete the static libraries afterwards (large, not needed here)",
    )
    ap.add_argument("components", nargs="+")
    args = ap.parse_args()
    plat = args.platform or default_platform()
    base = args.base_url.rstrip("/") + "/"

    with urllib.request.urlopen(f"{base}redistrib_{args.version}.json") as r:
        manifest = json.load(r)
    available = sorted(k for k, v in manifest.items() if isinstance(v, dict))
    args.dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for spec in args.components:
            name = spec.rstrip("?")
            entry = manifest.get(name, {}).get(plat) if name in available else None
            if entry is None:
                if spec.endswith("?"):
                    print(f"{name}: not in CUDA {args.version} for {plat}, skipped")
                    continue
                sys.exit(
                    f"{name}: not in CUDA {args.version} for {plat}; "
                    f"available: {', '.join(available)}"
                )
            rel = entry["relative_path"]
            archive = tmp / Path(rel).name
            print(f"{name} {manifest[name].get('version', '?')}: {rel}", flush=True)
            download(base + rel, archive)
            if sha256(archive) != entry["sha256"]:
                sys.exit(f"{name}: SHA-256 mismatch for {rel}")
            unpacked = tmp / f"unpacked-{name}"
            unpack(archive, unpacked)
            archive.unlink()
            # Every archive holds one directory <name>-<platform>-<version>-archive
            tops = [p for p in unpacked.iterdir() if p.is_dir()]
            if len(tops) != 1:
                sys.exit(f"{name}: unexpected archive layout {[p.name for p in tops]}")
            merge(tops[0], args.dest)
            shutil.rmtree(unpacked)

    if args.drop_static:
        for pattern in ("*_static.a", "*_static_*.a", "*_static.lib"):
            for f in args.dest.rglob(pattern):
                f.unlink()
    print(f"CUDA {args.version} ({plat}) in {args.dest}")


if __name__ == "__main__":
    main()
