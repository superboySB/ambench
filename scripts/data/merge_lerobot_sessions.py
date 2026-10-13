# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Materialize selected successful sessions with official LeRobot editing APIs.

The sources JSON contains an ordered sources list; each entry declares a
canonical dataset_root and increasing episode_indices. All sources must share
one robot, task, FPS, feature schema and AM-Bench action/state contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info


def file_hashes(root: Path) -> dict[str, str]:
    """Stream source bytes for an immutable before/after receipt."""
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result[path.relative_to(root).as_posix()] = digest.hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    entries = json.loads(args.sources_json.read_text())["sources"]
    if not entries:
        raise ValueError("Declare at least one source session.")
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Merge requires a new canonical output root.")
    for path in (output.parent / "episode_outcomes.json", output.parent / "merge_report.json"):
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing merge evidence: {path}")
    sources = []
    common = None
    seen_roots = set()
    for entry in entries:
        root = Path(entry["dataset_root"]).expanduser()
        if not root.is_absolute():
            root = args.sources_json.resolve().parent / root
        root = root.resolve()
        if (root / "lerobot/meta/info.json").exists():
            root /= "lerobot"
        if root in seen_roots or output.is_relative_to(root):
            raise ValueError("Source sessions must be distinct and outside the new output.")
        seen_roots.add(root)
        info = json.loads((root / "meta/info.json").read_text())
        outcomes_path = root.parent / "episode_outcomes.json"
        outcomes = json.loads(outcomes_path.read_text())
        ids = entry["episode_indices"]
        if not ids or any(not isinstance(index, int) or isinstance(index, bool) for index in ids):
            raise ValueError("Select at least one integer source episode per session.")
        if len(set(ids)) != len(ids) or ids != sorted(ids) or min(ids) < 0 or max(ids) >= info["total_episodes"]:
            raise ValueError("Source episode IDs must be unique, increasing, and within the canonical dataset.")
        exported = {
            int(row["exported_episode_index"]): row
            for row in outcomes["episodes"]
            if row.get("exported") and row.get("exported_episode_index") is not None
        }
        if any(index not in exported or exported[index]["termination_reason"] != "success" for index in ids):
            raise ValueError("Every selected source episode must have a completed successful native outcome.")
        contract = {
            "robot_type": info["robot_type"],
            "fps": info["fps"],
            "features": info["features"],
            "ambench": info["ambench"],
            "task": outcomes["task"],
        }
        if common is None:
            common = contract
        elif contract != common:
            raise ValueError("Source robot/task/FPS/feature/action/state contracts differ.")
        sources.append({
            "root": root,
            "ids": ids,
            "info": info,
            "outcomes": outcomes,
            "exported": exported,
            "source_files_sha256": file_hashes(root),
            "episode_outcomes_sha256": hashlib.sha256(outcomes_path.read_bytes()).hexdigest(),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping = []
    merged_outcomes = []
    with tempfile.TemporaryDirectory(prefix=".selected-lerobot-", dir=output.parent) as temp:
        datasets = []
        for source_index, source in enumerate(sources):
            raw = LeRobotDataset(f"ambench/local_source_{source_index}", root=source["root"])
            deleted = sorted(set(range(raw.meta.total_episodes)) - set(source["ids"]))
            if deleted:
                raw = delete_episodes(
                    raw,
                    episode_indices=deleted,
                    output_dir=Path(temp) / f"source_{source_index}",
                    repo_id=f"ambench/selected_source_{source_index}",
                )
            raw.meta.info["ambench"] = common["ambench"]
            # A complete source is already canonical; do not rewrite its metadata.
            if deleted:
                write_info(raw.meta.info, raw.root)
            datasets.append(raw)
            for old_index in source["ids"]:
                new_index = len(mapping)
                original = source["exported"][old_index]
                mapping.append({
                    "source_dataset_root": str(source["root"]),
                    "source_session_index": source_index,
                    "source_seed": source["outcomes"].get("seed"),
                    "source_episode_index": old_index,
                    "output_episode_index": new_index,
                    "frames": int(original["step_count"]),
                })
                merged_outcomes.append({
                    **original,
                    "episode_index": new_index,
                    "exported_episode_index": new_index,
                    "source_session_index": source_index,
                    "source_seed": source["outcomes"].get("seed"),
                    "source_episode_index": old_index,
                })
        merged = merge_datasets(datasets, output_repo_id=args.repo_id, output_dir=output)
    merged.meta.info["ambench"] = common["ambench"]
    write_info(merged.meta.info, output)
    if merged.meta.total_episodes != len(mapping) or merged.meta.total_frames != sum(row["frames"] for row in mapping):
        raise ValueError("Official merged episode/frame totals differ from successful source outcomes.")
    for source in sources:
        if file_hashes(source["root"]) != source["source_files_sha256"]:
            raise ValueError("Source canonical bytes changed during merge.")
        path = source["root"].parent / "episode_outcomes.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["episode_outcomes_sha256"]:
            raise ValueError("Source native outcome bytes changed during merge.")
    outcome_report = {
        "status": "merged_selected_successes",
        "task": common["task"],
        "seed": None,
        "source_seeds": [source["outcomes"].get("seed") for source in sources],
        "episodes": merged_outcomes,
        "successful_episodes": len(mapping),
        "claim_limit": "Storage merge of existing successful episodes; no new physics runs.",
    }
    (output.parent / "episode_outcomes.json").write_text(json.dumps(outcome_report, indent=2) + "\n")
    report = {
        "status": "passed",
        "repo_id": args.repo_id,
        "output_root": str(output),
        "robot_type": common["robot_type"],
        "task_id": common["task"],
        "fps": common["fps"],
        "ambench": common["ambench"],
        "total_episodes": len(mapping),
        "total_frames": merged.meta.total_frames,
        "source_bytes_unchanged": True,
        "sources": [
            {
                "dataset_root": str(source["root"]),
                "seed": source["outcomes"].get("seed"),
                "selected_episode_indices": source["ids"],
                "source_files_sha256": source["source_files_sha256"],
                "episode_outcomes_sha256": source["episode_outcomes_sha256"],
            }
            for source in sources
        ],
        "episode_mapping": mapping,
        "output_files_sha256": file_hashes(output),
        "official_api": "lerobot.datasets.dataset_tools.delete_episodes + merge_datasets",
        "no_new_physics_episodes": True,
    }
    (output.parent / "merge_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "episodes": len(mapping), "frames": merged.meta.total_frames}))


if __name__ == "__main__":
    main()
