# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Embed only real OmniHexa topic recordings into the standalone interactive page."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATA_START = "      // BEGIN TILTING RECORDS\n"
DATA_END = "      // END TILTING RECORDS"
METRICS_START = "      // BEGIN TILTING METRICS\n"
METRICS_END = "      // END TILTING METRICS"
EXCLUDED_FIELDS = {
    "source_frame_indices",
    "source_timestamps_s",
    "selected_image_sha256",
    "mp4_source_frame_indices",
    "mp4_selected_image_sha256",
    "sample_durations_ms",
    "session_root_value",
}


def select_records(manifest: list[dict]) -> list[dict]:
    """Select the topic prefix, preserving actual conditions and source time maps."""
    records = []
    names = set()
    trials = {}
    for row in manifest:
        experiment_id = row.get("experiment_id", "")
        if not isinstance(experiment_id, str) or not experiment_id.startswith("tilting_"):
            continue
        name = row["name"]
        if not re.fullmatch(r"[a-z0-9_-]+", name) or not re.fullmatch(r"[a-z0-9_-]+", experiment_id):
            raise ValueError(f"Invalid topic record identifier: {name!r}, {experiment_id!r}")
        if name in names:
            raise ValueError(f"Duplicate OmniHexa record name: {name}")
        names.add(name)
        reference = experiment_id == "tilting_reference_fahexa" and "-Am-FAHexa-" in row.get("task_id", "")
        if row.get("task_id") and "-Am-OmniHexa-" not in row["task_id"] and not reference:
            raise ValueError(f"OmniHexa topic record uses a different embodiment: {name}")
        for media_type in ("gif", "mp4"):
            if not re.fullmatch(rf"[a-z0-9_-]+\.{media_type}", row.get(media_type, "")):
                raise ValueError(f"Invalid {media_type} filename for {name}")
        if row.get("trial_id"):
            conditions = (
                row.get("experiment_id"),
                row.get("task_id"),
                row.get("policy_family"),
                row.get("model_variant"),
                row.get("env_seed"),
                row.get("inference_seed"),
                row.get("inference_device"),
                row.get("training_seed"),
                row.get("training_steps"),
                row.get("training_robot"),
                row.get("source_dataset"),
                row.get("training_initialized_pretrained"),
                row.get("training_regime"),
                row.get("action_semantics"),
                row.get("steps"),
                row.get("experiment_conditions"),
                row.get("outcome"),
                row.get("outcome_value"),
            )
            previous = trials.setdefault(row["trial_id"], conditions)
            if previous != conditions:
                raise ValueError(f"Same-trial camera records disagree on conditions: {row['trial_id']}")
        records.append({key: value for key, value in row.items() if key not in EXCLUDED_FIELDS})
    return records


def verify_policy_records(records: list[dict], summary: dict | None, expected_count: int) -> None:
    """Match each model view and its tilt telemetry to the actual verified rollout."""
    if not summary or summary.get("verification_status") != "passed":
        raise ValueError("Strict publication requires a passed model rollout and tilt summary")
    verified = {trial["name"]: trial for trial in summary["trials"] if trial.get("status") == "verified"}
    if len(verified) != expected_count or len(verified) != len(summary["trials"]):
        raise ValueError("Strict publication requires all declared model trials to be verified")
    observed = set()
    for record in records:
        if record.get("policy_family") == "scripted":
            continue
        trial = verified.get(record.get("trial_id"))
        if trial is None:
            raise ValueError(f"Model view has no verified rollout: {record['name']}")
        if (record["task_id"], record["env_seed"], record["steps"], record["outcome_value"]) != (
            trial["task_id"],
            trial["env_seed"],
            trial["steps"],
            trial["outcome"],
        ):
            raise ValueError(f"Model view and rollout conditions differ: {record['name']}")
        tilt = trial.get("motor_tilt")
        if not tilt or tilt.get("verification_status") != "passed" or not tilt.get("sampled", {}).get("points"):
            raise ValueError(f"Model rollout has no verified measured tilt trajectory: {trial['name']}")
        if record["source_tracking_sha256"] != trial["sources"]["tracking_jsonl"]["sha256"]:
            raise ValueError(f"Model view and tilt summary use different tracking: {record['name']}")
        observed.add(trial["name"])
    if observed != set(verified):
        raise ValueError("Strict publication requires media for every verified model trial")


def main() -> int:
    """Refresh embedded metadata without requiring a web server or dependencies."""
    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "usage_assets/tilting/index.html"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=repo_root / "usage_assets/animations/manifest.json")
    parser.add_argument("--output", type=Path, default=template_path)
    parser.add_argument("--summary", type=Path, default=repo_root / "usage_assets/tilting/trials.json")
    parser.add_argument("--trajectories", type=Path, default=repo_root / "usage_assets/tilting/trajectories.json")
    parser.add_argument("--policy-summary", type=Path, default=repo_root / "usage_assets/tilting/policy_trials.json")
    parser.add_argument("--animation-specs", type=Path, default=repo_root / "usage_assets/tilting/animation_specs.json")
    parser.add_argument(
        "--expected-model-trials",
        type=int,
        default=12,
        help="Expected independent learned-policy trials at the strict publication gate.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require a passed summary, linked telemetry and three views per new trial.",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        parser.error("--manifest must contain a JSON list")
    records = select_records(manifest)
    summary = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary.is_file() else None
    trajectories = json.loads(args.trajectories.read_text(encoding="utf-8")) if args.trajectories.is_file() else None
    policy_summary = (
        json.loads(args.policy_summary.read_text(encoding="utf-8")) if args.policy_summary.is_file() else None
    )
    for label, payload in (("summary", summary), ("trajectories", trajectories)):
        if payload is not None and (not isinstance(payload, dict) or not isinstance(payload.get("trials"), list)):
            raise ValueError(f"The {label} must be an object with a trials list")
    if args.strict:
        verify_policy_records(records, policy_summary, args.expected_model_trials)
        if args.expected_model_trials < 0:
            raise ValueError("--expected-model-trials must be nonnegative")
        if not args.animation_specs.is_file():
            raise ValueError("Strict publication requires the declared animation source specifications")
        source_specs = json.loads(args.animation_specs.read_text(encoding="utf-8"))
        if not isinstance(source_specs, list) or {spec["name"] for spec in source_specs} != {
            record["name"] for record in records
        }:
            raise ValueError("Strict publication requires a preview for every declared animation source")
        if not summary or summary.get("verification_status") != "passed" or not trajectories:
            raise ValueError("Strict publication requires actual summary/trajectories and a passed verification status")
        verified = {trial["name"]: trial for trial in summary["trials"] if trial.get("status") == "verified"}
        sampled = {trial["name"]: trial for trial in trajectories["trials"]}
        if not verified or len(verified) != len(summary["trials"]) or set(verified) != set(sampled):
            raise ValueError("Strict publication requires matching nonempty verified trials and trajectories")
        coverage = summary.get("coverage", {})
        if coverage.get("expected_count") != len(verified) or coverage.get("verified_count") != len(verified):
            raise ValueError("Strict publication requires all expected trials to be verified")
        trial_views = {}
        model_views = {}
        for record in records:
            trial_id = record.get("trial_id")
            if record.get("policy_family") in {"act", "dp", "pi", "pi0", "pi05"}:
                if not trial_id or not record.get("source_report_sha256") or not record.get("task_id"):
                    raise ValueError(
                        f"Transferred policy preview is missing actual evaluation provenance: {record['name']}"
                    )
                if (
                    not record.get("training_robot")
                    or not record.get("source_dataset")
                    or record.get("training_steps") is None
                ):
                    raise ValueError(f"Learned-policy preview is missing actual training conditions: {record['name']}")
                if not isinstance(record.get("training_initialized_pretrained"), bool):
                    raise ValueError(f"Learned-policy preview is missing initialization provenance: {record['name']}")
                model_views.setdefault(trial_id, []).append(record.get("camera"))
                continue  # Learned-policy tracking has no scripted controller telemetry.
            if trial_id not in verified:
                raise ValueError(f"Preview has no verified summary: {trial_id}")
            trial = verified[trial_id]
            if (record.get("task_id"), record.get("env_seed"), record.get("steps"), record.get("outcome_value")) != (
                trial["task_id"],
                trial["seed"],
                trial["steps"],
                trial["termination_reason"],
            ):
                raise ValueError(f"Preview and summary conditions disagree: {trial_id}")
            telemetry_sha = trial["source"]["telemetry"]["sha256"]
            if (
                record.get("source_telemetry_sha256") != telemetry_sha
                or sampled[trial_id].get("source_telemetry_sha256") != telemetry_sha
            ):
                raise ValueError(f"Preview and sampled trajectory use different telemetry: {trial_id}")
            trial_views.setdefault(trial_id, []).append(record.get("camera"))
        if set(trial_views) != set(verified) or any(
            sorted(views) != ["base_camera", "ee_camera", "scene_camera"] for views in trial_views.values()
        ):
            raise ValueError("Strict publication requires exactly three camera views for each verified new trial")
        if any(sorted(views) != ["base_camera", "ee_camera", "scene_camera"] for views in model_views.values()):
            raise ValueError("Strict publication requires exactly three views per learned-policy evaluation trial")
        if len(model_views) != args.expected_model_trials:
            raise ValueError(
                f"Strict publication requires {args.expected_model_trials} independent model trials, got"
                f" {len(model_views)}"
            )
    template = (args.output if args.output.is_file() else template_path).read_text(encoding="utf-8")
    if template.count(DATA_START) != 1 or template.count(DATA_END) != 1:
        raise ValueError("The OmniHexa template must contain exactly one marked data region")
    before, remainder = template.split(DATA_START, maxsplit=1)
    _, after = remainder.split(DATA_END, maxsplit=1)
    embedded = json.dumps(records, ensure_ascii=False, separators=(",", ":"))
    embedded = embedded.replace("<", r"\u003c").replace("\u2028", r"\u2028").replace("\u2029", r"\u2029")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = f"{before}{DATA_START}      const RECORDS = {embedded};\n{DATA_END}{after}"
    if rendered.count(METRICS_START) != 1 or rendered.count(METRICS_END) != 1:
        raise ValueError("The OmniHexa template must contain exactly one marked metrics region")
    before_metrics, metrics_remainder = rendered.split(METRICS_START, maxsplit=1)
    _, after_metrics = metrics_remainder.split(METRICS_END, maxsplit=1)
    metrics = json.dumps(
        {"summary": summary, "trajectories": trajectories, "policy_summary": policy_summary},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    metrics = metrics.replace("<", r"\u003c").replace("\u2028", r"\u2028").replace("\u2029", r"\u2029")
    rendered = f"{before_metrics}{METRICS_START}      const METRICS = {metrics};\n{METRICS_END}{after_metrics}"
    args.output.write_text(rendered, encoding="utf-8")
    count = len({record["trial_id"] for record in records if record.get("trial_id")})
    print(f"Embedded {len(records)} OmniHexa camera records with {count} identified trials: {args.output}")
    if not records:
        print("No OmniHexa topic records yet; the page displays the documented workflow and a pending evidence state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
