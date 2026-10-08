# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Embed recorded experiment metadata into the offline and online video gallery."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATA_START = "      // BEGIN MANIFEST DATA\n"
DATA_END = "      // END MANIFEST DATA"
LARGE_PROVENANCE_FIELDS = {
    "source_frame_indices",
    "source_timestamps_s",
    "selected_image_sha256",
    "mp4_source_frame_indices",
    "mp4_source_timestamps_s",
    "mp4_selected_image_sha256",
    "sample_durations_ms",
    "session_root_value",
}


def gallery_records(manifest: list[dict]) -> list[dict]:
    """Keep conditions and whole-file hashes while omitting per-frame hash arrays."""
    records = []
    names = set()
    for row in manifest:
        name = row["name"]
        experiment_id = row.get("experiment_id", name)
        if not re.fullmatch(r"[a-z0-9_-]+", name) or not re.fullmatch(r"[a-z0-9_-]+", experiment_id):
            raise ValueError(f"Invalid record or experiment identifier: {name!r}, {experiment_id!r}")
        if name in names:
            raise ValueError(f"Duplicate record name: {name}")
        names.add(name)
        record = {key: value for key, value in row.items() if key not in LARGE_PROVENANCE_FIELDS}
        record["experiment_id"] = experiment_id
        if "category" not in record:
            if experiment_id.startswith("policy_"):
                record["category"] = "models"
            elif experiment_id.startswith("physical_") or experiment_id == "expert_base_joint":
                record["category"] = "aircraft"
            elif experiment_id.startswith("expert_"):
                record["category"] = "tasks"
            else:
                record["category"] = "checks"
        if record["category"] not in {"models", "tasks", "aircraft", "checks"}:
            raise ValueError(f"Unsupported experiment category: {record['category']!r}")
        for extension in ("gif", "mp4"):
            filename = record.get(extension, f"{name}.{extension}")
            if not re.fullmatch(rf"[a-z0-9_-]+\.{extension}", filename):
                raise ValueError(f"Invalid {extension} filename: {filename!r}")
            record[extension] = filename
        records.append(record)
    if not records:
        raise ValueError("The gallery manifest contains no records")
    return records


def main() -> int:
    """Update the marked JSON region without fetching data in the browser."""
    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "usage_assets/playback.html"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo_root / "usage_assets/animations/manifest.json")
    parser.add_argument("--output", type=Path, default=template_path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        parser.error("--manifest must contain a JSON list of recorded previews")
    records = gallery_records(manifest)
    template = (args.output if args.output.is_file() else template_path).read_text(encoding="utf-8")
    if template.count(DATA_START) != 1 or template.count(DATA_END) != 1:
        raise ValueError("The gallery template must contain exactly one marked manifest region")
    before, remainder = template.split(DATA_START, maxsplit=1)
    _, after = remainder.split(DATA_END, maxsplit=1)
    embedded = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    embedded = embedded.replace("<", r"\u003c").replace("\u2028", r"\u2028").replace("\u2029", r"\u2029")
    output = f"{before}{DATA_START}      const records = {embedded};\n{DATA_END}{after}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    group_count = len({record["experiment_id"] for record in records})
    print(f"Embedded {len(records)} records in {group_count} experiment groups: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
