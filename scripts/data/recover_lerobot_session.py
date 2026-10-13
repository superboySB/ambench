# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Recover explicitly selected successful episodes from closed Parquet shards.

This creates a new canonical dataset through the official LeRobot writer. It
never opens damaged episode metadata, guesses missing outcomes, or edits sources.
The selection JSON contains source_root and episodes with episode_index/data_file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info
from PIL import Image

RESERVED_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def file_hashes(root: Path) -> dict[str, str]:
    """Hash original bytes including damaged/unselected files without decoding."""
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
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_json.read_text())
    source = Path(selection["source_root"]).expanduser()
    if not source.is_absolute():
        source = args.selection_json.resolve().parent / source
    source = source.resolve()
    output = args.output_root.expanduser().resolve()
    if output.exists() or output.is_relative_to(source):
        raise FileExistsError("Recovery requires a new output outside the original dataset.")
    outcome_path = source.parent / "episode_outcomes.json"
    info = json.loads((source / "meta/info.json").read_text())
    outcomes = json.loads(outcome_path.read_text())
    exported = {
        int(row["exported_episode_index"]): row
        for row in outcomes["episodes"]
        if row.get("exported") and row.get("exported_episode_index") is not None
    }
    selected = selection["episodes"]
    ids = [row["episode_index"] for row in selected]
    if not ids or any(not isinstance(index, int) or isinstance(index, bool) for index in ids):
        raise ValueError("Select at least one integer source episode index.")
    if len(set(ids)) != len(ids) or ids != sorted(ids) or min(ids) < 0:
        raise ValueError("Source episode IDs must be unique, nonnegative and increasing.")
    shards = []
    for row in selected:
        index = row["episode_index"]
        if index not in exported or exported[index]["termination_reason"] != "success":
            raise ValueError(f"Source episode {index} lacks a completed successful native outcome.")
        path = (source / row["data_file"]).resolve()
        if not path.is_relative_to(source / "data") or not path.is_file():
            raise ValueError("Selected data_file must identify an existing canonical data shard.")
        with path.open("rb") as stream:
            header = stream.read(4)
            stream.seek(-4, 2)
            footer = stream.read(4)
        if header != b"PAR1" or footer != b"PAR1":
            raise ValueError(f"Selected shard has no valid Parquet header/footer: {path}")
        shards.append(path)
    for path in (output.parent / "episode_outcomes.json", output.parent / "recovery_report.json"):
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing recovery evidence: {path}")
    before = file_hashes(source)
    outcome_sha256 = hashlib.sha256(outcome_path.read_bytes()).hexdigest()
    tasks = pd.read_parquet(source / "meta/tasks.parquet")
    features = {
        key: {**value, "shape": tuple(value["shape"])}
        for key, value in info["features"].items()
        if key not in RESERVED_FEATURES
    }
    if any(value["dtype"] == "video" for value in features.values()):
        raise ValueError("Recovery currently requires inline canonical RGB images, not video features.")
    writer = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output,
        robot_type=info["robot_type"],
        fps=info["fps"],
        features=features,
        use_videos=False,
        image_writer_threads=1,
        image_writer_processes=0,
    )
    writer.meta.info["ambench"] = info["ambench"]
    write_info(writer.meta.info, output)
    mapping = []
    recovered_outcomes = []
    for new_index, (source_index, path) in enumerate(zip(ids, shards, strict=True)):
        frame_count = 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                if int(row["episode_index"]) != source_index:
                    continue
                if int(row["frame_index"]) != frame_count:
                    raise ValueError(f"Source episode {source_index} has missing or reordered frames.")
                if not np.isclose(float(row["timestamp"]), frame_count / info["fps"], atol=1e-5):
                    raise ValueError(f"Source episode {source_index} timestamps do not match canonical FPS.")
                frame = {"task": str(tasks.iloc[int(row["task_index"])].name)}
                for key, feature in features.items():
                    value = row[key]
                    if feature["dtype"] == "image":
                        if value.get("bytes") is None:
                            raise ValueError("Recovery requires image bytes embedded in the selected shard.")
                        frame[key] = np.asarray(Image.open(BytesIO(value["bytes"])).convert("RGB"))
                    else:
                        frame[key] = np.asarray(value, dtype=feature["dtype"])
                        if not np.isfinite(frame[key]).all():
                            raise ValueError(f"Nonfinite source feature {key}.")
                writer.add_frame(frame)
                frame_count += 1
        if frame_count != int(exported[source_index]["step_count"]):
            raise ValueError(f"Source episode {source_index} frame count disagrees with its native outcome.")
        writer.save_episode()
        mapping.append({
            "source_episode_index": source_index,
            "source_seed": outcomes.get("seed"),
            "source_data_file": path.relative_to(source).as_posix(),
            "output_episode_index": new_index,
            "frames": frame_count,
        })
        recovered_outcomes.append({
            **exported[source_index],
            "episode_index": new_index,
            "exported_episode_index": new_index,
            "source_episode_index": source_index,
            "source_seed": outcomes.get("seed"),
        })
    writer.finalize()
    if file_hashes(source) != before or hashlib.sha256(outcome_path.read_bytes()).hexdigest() != outcome_sha256:
        raise ValueError("Original source bytes changed during recovery.")
    if (source / "env_cfg.yaml").exists():
        shutil.copy2(source / "env_cfg.yaml", output / "env_cfg.yaml")
    recovered = {
        "status": "recovered_selected_successes",
        "task": outcomes["task"],
        "seed": outcomes.get("seed"),
        "episodes": recovered_outcomes,
        "successful_episodes": len(mapping),
        "source_session_status": outcomes.get("status"),
        "claim_limit": "Rewritten canonical storage for preserved successful native episodes; no new simulation.",
    }
    (output.parent / "episode_outcomes.json").write_text(json.dumps(recovered, indent=2) + "\n")
    report = {
        "status": "passed",
        "source_root": str(source),
        "source_file_sha256_before_and_after": before,
        "source_episode_outcomes_sha256": outcome_sha256,
        "source_bytes_unchanged": True,
        "selected_source_episode_indices": ids,
        "episode_mapping": mapping,
        "output_root": str(output),
        "output_repo_id": args.repo_id,
        "output_files_sha256": file_hashes(output),
        "official_writer_recomputed_episode_statistics": True,
        "no_new_physics_episodes": True,
    }
    (output.parent / "recovery_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "recovered_episodes": len(mapping), "output_root": str(output)}))


if __name__ == "__main__":
    main()
