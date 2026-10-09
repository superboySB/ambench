# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Publish verified UAQuad learned-policy trials from completed evaluator outputs.

This standard-library helper reads the existing twelve-trial driver report. It
does not launch Isaac, load a policy, or synthesize an initial observation. An
incomplete or inconsistent run leaves the public output untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_ROOT = REPO_ROOT / "outputs"
CAMERAS = ("scene_camera", "base_camera", "ee_camera")
TASK_ID = "PressButton-Am-UAQuad-Abs-PID-Direct-v0"
INITIALIZATION = {
    "ResNet18 IMAGENET1K_V1": True,
    "random ResNet18": False,
    "OpenPI base checkpoint": True,
}


def sha256_file(path: Path) -> str:
    """Hash a saved artifact without reading a whole video into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def saved_file(value: str, *, parent: Path | None = None) -> Path:
    """Resolve a driver or container output path to this repository's outputs."""
    path = Path(value)
    if path.is_absolute():
        if "outputs" not in path.parts:
            raise ValueError(f"Saved artifact has no outputs/ anchor: {value}")
        path = REPO_ROOT / Path(*path.parts[path.parts.index("outputs") :])
    elif parent is not None and (parent / path).is_file():
        path = parent / path
    else:
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.is_relative_to(OUTPUTS_ROOT) or not path.is_file():
        raise ValueError(f"Saved artifact is missing or outside ignored outputs/: {value}")
    return path


def source(path: Path) -> dict[str, Any]:
    """Describe one preserved source file using a repository-relative path."""
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}


def training_source(spec: dict[str, Any]) -> dict[str, Any]:
    """Link the served checkpoint to its actual dataset and training record."""
    reference_path = (REPO_ROOT / spec["training_reference"]).resolve()
    if not reference_path.is_relative_to(REPO_ROOT / "usage_assets") or not reference_path.is_file():
        raise ValueError(f"Missing training provenance for {spec['name']}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if spec["training_robot"] == "EE":
        matches = [row for row in reference["rows"] if row["mode"] == "ee" and row["policy"] == spec["model_variant"]]
        if len(matches) != 1:
            raise ValueError(f"No unique EE training job for {spec['name']}")
        job = matches[0]
        if (
            job["checkpoint"] != spec["checkpoint"]
            or job["training_steps"] != spec["training_steps"]
            or job["seed"] != spec["training_seed"]
        ):
            raise ValueError(f"EE training provenance differs from {spec['name']}")
        dataset = job["canonical_info"]
        dataset_path = dataset["path"].removesuffix("/meta/info.json")
        dataset_sha256 = dataset["sha256"]
        dataset_hash_basis = "canonical meta/info.json"
        regime = "ee_transfer"
    elif spec["training_robot"] == "UAQuad":
        job = reference["act" if spec["family"] == "act" else "diffusion_policy"]
        if (
            job["checkpoint_container_path"] != spec["checkpoint"]
            or job["optimizer_steps"] != spec["training_steps"]
            or job["training_seed"] != spec["training_seed"]
        ):
            raise ValueError(f"UAQuad training provenance differs from {spec['name']}")
        dataset = reference["canonical_dataset"]
        dataset_path = dataset["source_relative_path"]
        dataset_sha256 = dataset["sha256"]["data/chunk-000/file-000.parquet"]
        dataset_hash_basis = "canonical data/chunk-000/file-000.parquet"
        if dataset_sha256 != spec["source_dataset_sha256"]:
            raise ValueError(f"UAQuad training dataset hash differs from {spec['name']}")
        local_dataset = (REPO_ROOT / dataset_path).resolve()
        if not local_dataset.is_relative_to(OUTPUTS_ROOT) or not local_dataset.is_dir():
            raise ValueError(f"UAQuad training dataset is unavailable for {spec['name']}")
        if sha256_file(local_dataset / "data/chunk-000/file-000.parquet") != dataset_sha256:
            raise ValueError(f"UAQuad canonical data changed after training: {spec['name']}")
        regime = "uaquad_one_step"
    else:
        raise ValueError(f"Unsupported training robot for {spec['name']}")
    return {
        "regime": regime,
        "robot": spec["training_robot"],
        "seed": spec["training_seed"],
        "optimizer_steps": spec["training_steps"],
        "pretrained_initialization": spec["pretrained_initialization"],
        "initialized_pretrained": INITIALIZATION[spec["pretrained_initialization"]],
        "dataset": dataset_path,
        "dataset_sha256": dataset_sha256,
        "dataset_hash_basis": dataset_hash_basis,
        "checkpoint": spec["checkpoint"],
        "reference": {**source(reference_path), "job": job},
    }


def verify_identity(identity: dict[str, Any], spec: dict[str, Any]) -> None:
    """Require the saved identity to describe this CPU policy listener."""
    if identity["environment"] != {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONHASHSEED": "42"}:
        raise ValueError(f"CPU inference environment differs for {spec['name']}")
    port = 8000 if spec["family"] == "pi" else 8001
    if port not in identity["listening_ports"]:
        raise ValueError(f"Server did not own port {port} for {spec['name']}")
    arguments = identity["argv"]
    if spec["family"] == "pi":
        expected = [f"--policy.config={spec['openpi_config']}", f"--policy.dir={spec['checkpoint']}"]
        if any(argument not in arguments for argument in expected):
            raise ValueError(f"OpenPI service arguments differ for {spec['name']}")
    else:
        for option, expected in (
            ("--policy", spec["family"]),
            ("--checkpoint", spec["checkpoint"]),
            ("--device", "cpu"),
        ):
            if option not in arguments or arguments[arguments.index(option) + 1] != expected:
                raise ValueError(f"ACT/DP service arguments differ for {spec['name']}")


def verify_tracking(
    row: dict[str, Any], eval_summary: dict[str, Any], rollout: dict[str, Any], run_dir: Path, name: str
) -> tuple[Path, Path, dict[str, Any]]:
    """Check preserved per-step records against evaluator tracking metrics."""
    tracking_metrics = rollout["tracking"]
    if not isinstance(tracking_metrics, dict) or not isinstance(tracking_metrics.get("metrics"), dict):
        raise ValueError(f"Missing actual tracking metrics for {name}")
    tracking_path = saved_file(
        row["tracking"] if isinstance(row.get("tracking"), str) else eval_summary["tracking"]["tracking_jsonl"]
    )
    if tracking_path != run_dir / "tracking/tracking.jsonl" or sha256_file(tracking_path) != row["tracking_sha256"]:
        raise ValueError(f"Tracking source path or hash differs for {name}")
    tracked_count = 0
    first_step = last_step = None
    dt = None
    with tracking_path.open(encoding="utf-8") as tracking_file:
        first_event = json.loads(tracking_file.readline())
        for line in tracking_file:
            event = json.loads(line)
            if event["event"] != "timestep":
                continue
            if event["rollout_idx"] != 0:
                raise ValueError(f"Tracking log contains another rollout: {name}")
            record = event["record"]
            timestep = int(record["timestep"])
            current_dt = float(record["dt"])
            if last_step is not None and timestep != last_step + 1:
                raise ValueError(f"Tracking timesteps are discontinuous: {name}")
            if dt is not None and not math.isclose(current_dt, dt, rel_tol=1e-9):
                raise ValueError(f"Tracking timestep duration changed: {name}")
            first_step = timestep if first_step is None else first_step
            last_step = timestep
            dt = current_dt
            tracked_count += 1
    if first_event["event"] != "run_metadata" or first_event["metadata"]["Policy ID"] != name:
        raise ValueError(f"Tracking log belongs to a different policy trial: {name}")
    if any(first_event["metadata"].get(key) != value for key, value in eval_summary["metadata"].items()):
        raise ValueError(f"Tracking conditions differ from evaluator summary: {name}")
    if tracked_count == 0 or tracked_count != tracking_metrics["num_tracking_steps"] or first_step != 1:
        raise ValueError(f"Tracking sample count differs from evaluator summary: {name}")
    stats_count = tracked_count - 1 if tracked_count > 1 else tracked_count
    if not math.isclose(float(tracking_metrics["execution_time"]), stats_count * dt, abs_tol=1e-6):
        raise ValueError(f"Tracking execution time differs from recorded samples: {name}")
    analysis_path = saved_file(eval_summary["tracking"]["analysis_json"])
    if analysis_path != run_dir / "tracking/analysis.json":
        raise ValueError(f"Tracking analysis path differs for {name}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if (
        analysis["status"] != "completed"
        or analysis["num_rollouts"] != 1
        or analysis["rollouts"][0]["summary"] != tracking_metrics
        or analysis["rollouts"][0]["executed_steps"] != rollout["executed_steps"]
    ):
        raise ValueError(f"Tracking analysis differs from evaluator summary: {name}")
    sample_basis = {
        "post_step_records": tracked_count,
        "metric_input_records": stats_count,
        "first_recorded_timestep": first_step,
        "last_recorded_timestep": last_step,
        "step_dt_s": dt,
    }
    return tracking_path, analysis_path, sample_basis


def verify_server(
    row: dict[str, Any], spec: dict[str, Any], run_dir: Path, name: str
) -> tuple[dict[str, Any], Path, Path, Path]:
    """Check that the same identified CPU server handled the whole trial."""
    before_path = saved_file(str(run_dir / "server-identity-before.json"))
    after_path = saved_file(str(run_dir / "server-identity-after.json"))
    identity_before = json.loads(before_path.read_text(encoding="utf-8"))
    identity_after = json.loads(after_path.read_text(encoding="utf-8"))
    if row["server_identity"] != identity_before:
        raise ValueError(f"Driver's CPU server identity differs from saved evidence: {name}")
    for identity in (identity_before, identity_after):
        verify_identity(identity, spec)
    if (identity_before["pid"], identity_before["start_time_ticks"]) != (
        identity_after["pid"],
        identity_after["start_time_ticks"],
    ):
        raise ValueError(f"Inference server restarted during {name}")
    server_log = saved_file(str(run_dir / "server-inference.log"))
    text = server_log.read_text(encoding="utf-8")
    if "REMOTE_INFERENCE_DEVICE=cpu" not in text or "REMOTE_INFERENCE_SEED=42" not in text:
        raise ValueError(f"CPU service log lacks actual device/seed markers: {name}")
    return identity_before, before_path, after_path, server_log


def verify_videos(row: dict[str, Any], run_dir: Path, name: str) -> dict[str, dict[str, Any]]:
    """Require a nonempty preserved video for each declared camera."""
    video_sources = {}
    for camera in CAMERAS:
        video = saved_file(row["videos"][camera])
        if video.parent != run_dir / "videos" or not video.name.endswith(f"-{camera}-env0-eps0.mp4"):
            raise ValueError(f"Camera video source differs for {name}/{camera}")
        if video.stat().st_size == 0:
            raise ValueError(f"Empty video source for {name}/{camera}")
        video_sources[camera] = {**source(video), "bytes": video.stat().st_size}
    return video_sources


def summarize_trial(row: dict[str, Any], spec: dict[str, Any], report: Path) -> dict[str, Any]:
    """Verify and summarize one completed real rollout without altering it."""
    name = spec["name"]
    if (
        spec["task_id"] != TASK_ID
        or spec["inference_device"] != "cpu"
        or spec["inference_seed"] != 42
        or spec["seed"] not in (51, 52)
        or spec["family"] not in ("act", "dp", "pi")
    ):
        raise ValueError(f"Unexpected model trial conditions: {name}")
    if row["name"] != name or row["spec"] != spec or row["status"] != "completed":
        raise ValueError(f"Trial is missing, changed, or incomplete: {name}")
    if row["eval_exit_code"] != 0:
        raise ValueError(f"Evaluator did not exit successfully: {name}")
    run_dir = report.parent / name
    eval_path = saved_file(row["eval_summary"])
    if eval_path != run_dir / "eval_summary.json" or sha256_file(eval_path) != row["eval_summary_sha256"]:
        raise ValueError(f"Evaluator summary path or hash differs for {name}")
    eval_summary = json.loads(eval_path.read_text(encoding="utf-8"))
    metadata = eval_summary["metadata"]
    rollouts = eval_summary["rollouts"]
    if eval_summary["status"] != "completed" or eval_summary["error"] is not None:
        raise ValueError(f"Evaluator did not finish at a task boundary: {name}")
    if len(rollouts) != 1 or eval_summary["summary"]["num_rollouts"] != 1:
        raise ValueError(f"Expected one rollout in {name}")
    if (
        metadata["Task"] != TASK_ID
        or metadata["Seed"] != spec["seed"]
        or metadata["Policy ID"] != name
        or metadata["Action semantics"] != "ee_absolute"
        or metadata["Episode length override (s)"] != spec["episode_length_s"]
        or metadata["Disturbance"]
        or metadata["Device"] != "cuda:0"
        or not metadata["Save video"]
        or metadata["Requested rollouts"] != 1
        or metadata["Num envs"] != 1
        or sorted(metadata["Video cameras"]) != sorted(CAMERAS)
    ):
        raise ValueError(f"Evaluator conditions differ from {name}")
    if spec["family"] in {"act", "dp"}:
        server = metadata["Server metadata"]
        if server["policy"] != spec["family"] or server["checkpoint"] != spec["checkpoint"]:
            raise ValueError(f"ACT/DP server checkpoint differs from {name}")
    elif (
        metadata["Server metadata"]["action_representation"] != "ee_local_relative"
        or metadata["Prompt"] != "press the button"
    ):
        raise ValueError(f"OpenPI policy metadata differs from {name}")
    if spec["family"] in {"act", "pi"}:
        overrides = metadata["Eval overrides"]
        if overrides["n_action_steps"] != 8 or overrides["policy_target_hz"] != 20:
            raise ValueError(f"Policy execution schedule differs from {name}")
    rollout = rollouts[0]
    outcome = rollout["termination_reason"]
    steps = rollout["executed_steps"]
    if (
        outcome not in {"success", "timeout"}
        or rollout["success"] != (outcome == "success")
        or outcome != row["outcome"]
        or steps != row["steps"]
        or not 0 < steps <= round(spec["episode_length_s"] * 120)
    ):
        raise ValueError(f"Task outcome or steps differ from {name}")
    tracking_path, analysis_path, sample_basis = verify_tracking(row, eval_summary, rollout, run_dir, name)
    identity_before, before_path, after_path, server_log = verify_server(row, spec, run_dir, name)
    video_sources = verify_videos(row, run_dir, name)
    training = training_source(spec)
    return {
        "name": name,
        "status": "verified",
        "experiment_id": spec["experiment_id"],
        "task_id": TASK_ID,
        "policy_family": spec["family"],
        "model_variant": spec["model_variant"],
        "env_seed": spec["seed"],
        "inference_seed": spec["inference_seed"],
        "inference_device": spec["inference_device"],
        "environment_device": metadata["Device"],
        "action_semantics": metadata["Action semantics"],
        "episode_length_s": metadata["Episode length override (s)"],
        "attempt_selection": spec["attempt_selection"],
        "outcome": outcome,
        "success": rollout["success"],
        "steps": steps,
        "tracking": rollout["tracking"],
        "tracking_sample_basis": sample_basis,
        "training": training,
        "initial_observation_available": False,
        "server_identity": identity_before,
        "sources": {
            "driver_report": source(report),
            "eval_summary": source(eval_path),
            "tracking_jsonl": source(tracking_path),
            "tracking_analysis": source(analysis_path),
            "runtime_env_cfg": source(saved_file(str(run_dir / "env_cfg.yaml"))),
            "server_identity_before": source(before_path),
            "server_identity_after": source(after_path),
            "server_log": source(server_log),
            "videos": video_sources,
        },
    }


def main() -> int:
    """Require the whole real trial matrix before publishing a JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-json", type=Path, default=OUTPUTS_ROOT / "research/uaquad-20261009/policy/results.json"
    )
    parser.add_argument("--specs-json", type=Path, default=REPO_ROOT / "usage_assets/uaquad/policy_specs.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "usage_assets/uaquad/policy_trials.json")
    args = parser.parse_args()
    output_path = args.output.resolve()
    output_roots = (REPO_ROOT / "usage_assets/uaquad", OUTPUTS_ROOT / "research")
    if not any(output_path.is_relative_to(root) for root in output_roots):
        parser.error("Policy trial summary must be inside usage_assets/uaquad/ or outputs/research/")
    report_path = saved_file(str(args.report_json))
    specs_path = args.specs_json.resolve()
    if not specs_path.is_relative_to(REPO_ROOT / "usage_assets/uaquad"):
        parser.error("Policy specifications must be in usage_assets/uaquad/")
    specs = json.loads(specs_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "completed" or len(specs) != 12 or len(report["results"]) != 12:
        raise ValueError("Publication requires all 12 completed policy trials")
    by_name = {row["name"]: row for row in report["results"]}
    if len(by_name) != 12 or set(by_name) != {spec["name"] for spec in specs}:
        raise ValueError("Driver results differ from the twelve declared policy trials")
    trials = [summarize_trial(by_name[spec["name"]], spec, report_path) for spec in specs]
    payload = {
        "schema_version": 1,
        "verification_status": "passed",
        "coverage": {"expected_count": 12, "verified_count": 12, "missing_names": [], "invalid_names": []},
        "claim_limit": (
            "One optimizer step per checkpoint verifies model and transport interfaces, not learned UAQuad "
            "task performance. UAQuad seed 51 was present in its own one-step training demonstration."
        ),
        "metric_basis": {
            "source": (
                "source/ambench/ambench/evaluation/tracking/collect.py and "
                "source/ambench/ambench/evaluation/tracking/metrics.py"
            ),
            "sampling": (
                "Evaluator tracking records post-step observations for nonterminal steps only. "
                "summarize_records omits the last tracking record when more than one exists; "
                "each numeric metric count then excludes nonfinite values independently. "
                "Executed steps and per-metric sample counts therefore differ."
            ),
            "time": (
                "Tracking execution_time is metric_input_records multiplied by the logged simulator dt; "
                "it is neither the full executed_steps duration nor wall-clock inference time."
            ),
            "base_tilt_rad": (
                "Euclidean norm of Euler XYZ roll and pitch recovered from actual base WXYZ quaternion, "
                "in radians. It is not the geometric angle between the thrust axis and world vertical."
            ),
            "max_tilt_utilization": (
                "Maximum recorded base_tilt_rad. Despite the legacy field name, this is an angle "
                "in radians and is not normalized by an attitude limit."
            ),
            "saturation_rate": (
                "Fraction of metric input records where any reported motor thrust is at or outside "
                "the configured thrust bounds within 1e-5 N, or the actuator reports thrust_saturated. "
                "This incidence can be high with saturation disabled and does not prove applied clipping."
            ),
            "tracking_errors": (
                "Post-step controller desired pose minus measured pose; these are controller tracking "
                "errors, not task-target distance or a measure of task success."
            ),
        },
        "source_report": source(report_path),
        "source_specs": source(specs_path),
        "trials": trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp_path.replace(output_path)
    print(f"Verified 12/12 UAQuad policy trials: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
