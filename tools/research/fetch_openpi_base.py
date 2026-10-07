# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Download and verify one official OpenPI base checkpoint inside policy Docker."""

import base64
import datetime
import hashlib
import json
import pathlib
import sys

import fsspec
import google_crc32c
from openpi.shared import download

MODEL_NAMES = {"pi0": "pi0_base", "pi05": "pi05_base"}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in MODEL_NAMES:
        raise SystemExit("Usage: fetch_openpi_base.py pi0|pi05")

    name = MODEL_NAMES[sys.argv[1]]
    uri = f"gs://openpi-assets/checkpoints/{name}"
    print(f"Fetching {uri}", flush=True)
    root = download.maybe_download(uri)
    print(f"Cached at {root}", flush=True)

    fs, _ = fsspec.core.url_to_fs(uri)
    remote_prefix = f"openpi-assets/checkpoints/{name}/"
    remote_files = {
        object_name.removeprefix(remote_prefix): info
        for object_name, info in fs.find(uri, detail=True).items()
        if info.get("type") == "file" and object_name.startswith(remote_prefix)
    }
    if not remote_files:
        raise RuntimeError(f"No remote files found at {uri}")

    local_files = {str(path.relative_to(root)) for path in pathlib.Path(root).rglob("*") if path.is_file()}
    if local_files != remote_files.keys():
        raise RuntimeError(
            f"File set mismatch: missing={sorted(remote_files.keys() - local_files)}, "
            f"unexpected={sorted(local_files - remote_files.keys())}"
        )

    total_bytes = 0
    for relative_path, info in sorted(remote_files.items()):
        path = root / relative_path
        size = int(info["size"])
        if path.stat().st_size != size:
            raise RuntimeError(f"Size mismatch for {relative_path}")
        crc = google_crc32c.Checksum()
        md5 = hashlib.md5(usedforsecurity=False) if info.get("md5Hash") else None
        with path.open("rb") as stream:
            while block := stream.read(16 * 1024 * 1024):
                crc.update(block)
                if md5 is not None:
                    md5.update(block)
        if crc.digest() != base64.b64decode(info["crc32c"]):
            raise RuntimeError(f"CRC32C mismatch for {relative_path}")
        if md5 is not None and md5.digest() != base64.b64decode(info["md5Hash"]):
            raise RuntimeError(f"MD5 mismatch for {relative_path}")
        total_bytes += size
        print(f"Verified {relative_path}: {size} bytes", flush=True)

    report = {
        "model": sys.argv[1],
        "source": uri,
        "cache_path": str(root),
        "files": len(remote_files),
        "bytes": total_bytes,
        "checks": "GCS CRC32C and MD5 where available",
        "objects": {
            relative_path: {
                "size": int(info["size"]),
                "generation": info["generation"],
                "crc32c": info["crc32c"],
                "md5": info.get("md5Hash"),
            }
            for relative_path, info in sorted(remote_files.items())
        },
        "verified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    report_path = pathlib.Path(f"/data/outputs/openpi-{sys.argv[1]}-base-verify.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {total_bytes} bytes across {len(remote_files)} files; report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
