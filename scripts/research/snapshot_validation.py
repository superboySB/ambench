# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Publish a compact, source-hashed record of completed research validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAMES = ("act", "dp", "pi0", "pi05")


def load_report(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Load a JSON report and record its location and content digest."""
    data = path.read_bytes()
    return json.loads(data), {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def main() -> int:
    """Check expected coverage before writing a small tracked snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True, help="Merged 106-ID matrix JSON.")
    parser.add_argument("--scripted", type=Path, required=True, help="Merged scripted family JSON.")
    parser.add_argument("--policy-root", type=Path, required=True, help="Directory with four full20s eval directories.")
    parser.add_argument("--require-base-joint", action="store_true", help="Require all four BaseJoint short rollouts.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix, matrix_source = load_report(args.matrix)
    scripted, scripted_source = load_report(args.scripted)
    if (matrix["registry_count"], matrix["task_count"], matrix["passed_count"], matrix["failed_count"]) != (
        106,
        106,
        106,
        0,
    ):
        parser.error("Matrix must contain 106 passed registered IDs and no failures.")
    if (scripted["registry_count"], scripted["family_count"], scripted["passed_count"], scripted["failed_count"]) != (
        106,
        12,
        12,
        0,
    ):
        parser.error("Scripted report must contain 12 passed families and no failures.")

    policies = []
    for name in POLICY_NAMES:
        report, source = load_report(args.policy_root / f"{name}_isaac_full20s" / "eval_summary.json")
        rollouts = report["rollouts"]
        summary = report["summary"]
        if report["status"] != "completed" or summary["num_rollouts"] != 1 or len(rollouts) != 1:
            parser.error(f"{name} did not complete exactly one rollout.")
        metadata = report["metadata"]
        if metadata.get("Episode length override (s)") != 20 or metadata.get("Action semantics") != "ee_absolute":
            parser.error(f"{name} full episode must use 20 seconds and EE absolute actions.")
        policies.append({
            "policy": name,
            "source": source,
            "status": report["status"],
            "rollouts": summary["num_rollouts"],
            "successes": summary["num_successes"],
            "steps": rollouts[0]["executed_steps"],
            "termination_reason": rollouts[0]["termination_reason"],
            "wall_time_s": round(summary["eval_wall_time_s"], 3),
            "seed": metadata.get("Seed"),
            "task_id": metadata["Task"],
        })

    base_joint_policies = []
    for name in POLICY_NAMES:
        path = args.policy_root / f"{name}_base_joint_isaac_step1" / "eval_summary.json"
        if not path.is_file():
            if args.require_base_joint:
                parser.error(f"Missing {name} BaseJoint rollout: {path}")
            continue
        report, source = load_report(path)
        summary = report["summary"]
        metadata = report["metadata"]
        if report["status"] != "completed" or summary["num_rollouts"] != 1 or len(report["rollouts"]) != 1:
            parser.error(f"{name} BaseJoint did not complete exactly one rollout.")
        if metadata.get("Action semantics") != "base_joint_absolute":
            parser.error(f"{name} BaseJoint rollout has incompatible action semantics.")
        rollout = report["rollouts"][0]
        base_joint_policies.append({
            "policy": name,
            "source": source,
            "status": report["status"],
            "rollouts": summary["num_rollouts"],
            "successes": summary["num_successes"],
            "steps": rollout["executed_steps"],
            "termination_reason": rollout["termination_reason"],
            "wall_time_s": round(summary["eval_wall_time_s"], 3),
            "episode_length_s": metadata.get("Episode length override (s)"),
            "seed": metadata.get("Seed"),
            "task_id": metadata["Task"],
        })

    snapshot = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix": {
            "source": matrix_source,
            "registry_count": matrix["registry_count"],
            "families": matrix["family_count"],
            "passed": matrix["passed_count"],
            "tasks": [
                {
                    "id": row["task_id"],
                    "family": row["family"],
                    "robot": row["robot"],
                    "steps": row["steps_completed"],
                    "seed": row.get("seed"),
                }
                for row in matrix["results"]
            ],
        },
        "scripted": {
            "source": scripted_source,
            "passed_families": scripted["passed_count"],
            "families": [
                {
                    "family": row["family"],
                    "task_id": row["selected"]["task_id"],
                    "episodes": row["selected"]["episodes"],
                    "frames": row["selected"]["total_frames"],
                    "action_semantics": row["selected"]["action_semantics"],
                    "seed": row["selected"].get("seed"),
                    "episode_length_s": row["selected"].get("episode_length_s"),
                }
                for row in scripted["results"]
            ],
        },
        "policy_full_20s": policies,
        "policy_base_joint": base_joint_policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Wrote {args.output}: 106 environments, 12 experts, 4 EE and {len(base_joint_policies)} BaseJoint episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
