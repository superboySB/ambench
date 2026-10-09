# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and safely merge three-camera previews from completed recording trials.

The host entrypoint uses only the standard library. Source video headers and
saved environment snapshots are read in the existing simulation container on
CPU, without starting Isaac Sim. Original episode reports remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

CAMERAS = ("scene_camera", "base_camera", "ee_camera")
PROBE_CODE = """
import json
import sys
from pathlib import Path

import av
import yaml


class SnapshotLoader(yaml.SafeLoader):
    pass


def plain_python_value(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


SnapshotLoader.add_multi_constructor("tag:yaml.org,2002:python/", plain_python_value)
request = json.loads(sys.argv[1])
cfg = yaml.load(Path(request["env_cfg"]).read_text(), Loader=SnapshotLoader)
profile = cfg["robot_profile"]
robot = profile["robot"]
control = profile["control"]
multirotor = robot.get("multirotor") or {}
aerodynamics = multirotor.get("aerodynamics") or {}
camera_poses = {}
for name in request["videos"]:
    camera = cfg["scene_camera_cfg"] if name == "scene_camera" else profile[name]
    offset = camera["offset"]
    camera_poses[name] = {
        "prim_path": camera["prim_path"],
        "position": offset["pos"],
        "quaternion_wxyz": offset["rot"],
        "convention": offset["convention"],
        "width": camera["width"],
        "height": camera["height"],
    }
videos = {}
for name, path in request["videos"].items():
    with av.open(path) as source:
        stream = source.streams.video[0]
        videos[name] = {
            "fps": float(stream.average_rate),
            "frame_count": stream.frames,
            "duration_s": float(stream.duration * stream.time_base),
            "width": stream.width,
            "height": stream.height,
        }
training_robot_id = None
if request.get("training_env_cfg"):
    training_cfg = yaml.load(Path(request["training_env_cfg"]).read_text(), Loader=SnapshotLoader)
    training_robot_id = training_cfg["robot_profile"]["robot"]["robot_id"]
print(json.dumps({
    "robot_profile": {
        "robot_id": robot["robot_id"],
        "pipeline": control["class_type"].rsplit(":", 1)[-1],
        "controller": control["controller"]["class_type"].rsplit(":", 1)[-1],
        "action_mode": control["action_mode"]["_value_"],
    },
    "experiment_conditions": {
        "enable_saturation": cfg["enable_saturation"],
        "enable_aerodynamic_effects": cfg["enable_aerodynamic_effects"],
        "enable_wind_effect": cfg["enable_wind_effect"],
        "wind_force_w": aerodynamics.get("wind_force_w"),
    },
    "actuation": {
        "fully_actuated": multirotor.get("fully_actuated"),
        "thrust_limits_n": (multirotor.get("actuator") or {}).get("thrust_limits"),
    },
    "camera_poses": camera_poses,
    "seed": cfg["seed"],
    "step_dt_s": cfg["sim"]["dt"] * cfg["decimation"],
    "episode_length_s": cfg["episode_length_s"],
    "videos": videos,
    "training_robot_id": training_robot_id,
}, sort_keys=True))
"""


def sha256(path: Path) -> str:
    """Hash a source report or media file without retaining it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source(value: str, repo_root: Path) -> Path:
    """Map an unavailable container report path through its outputs anchor."""
    path = repo_root / value
    if not path.exists() and "outputs" in path.parts:
        path = repo_root / Path(*path.parts[path.parts.index("outputs") :])
    path = path.resolve()
    if not path.exists() or not path.is_relative_to(repo_root):
        raise ValueError(f"Source is missing or outside the repository: {value}")
    return path


def probe_snapshot(
    env_cfg: Path,
    videos: dict[str, Path],
    repo_root: Path,
    container: str,
    container_root: Path,
    training_env_cfg: Path | None = None,
) -> dict:
    """Inspect saved configs and source video headers using existing CPU tools."""
    request = {
        "env_cfg": str(container_root / env_cfg.relative_to(repo_root)),
        "videos": {name: str(container_root / path.relative_to(repo_root)) for name, path in videos.items()},
    }
    if training_env_cfg is not None:
        request["training_env_cfg"] = str(container_root / training_env_cfg.relative_to(repo_root))
    probe = subprocess.run(
        ["docker", "exec", container, "python", "-c", PROBE_CODE, json.dumps(request)],
        capture_output=True,
        text=True,
    )
    if probe.returncode:
        raise RuntimeError(f"CPU snapshot probe failed for {env_cfg}: {probe.stderr[-2000:]}")
    return json.loads(probe.stdout.splitlines()[-1])


def training_metadata(trial: dict, repo_root: Path) -> tuple[dict, Path]:
    """Trace policy training conditions to the supplied provenance and dataset."""
    reference_path = resolve_source(trial["training_reference"], repo_root)
    reference = json.loads(reference_path.read_text())
    if trial["training_robot"] == "EE":
        matching = [row for row in reference["rows"] if row["mode"] == "ee" and row["policy"] == trial["model_variant"]]
        if len(matching) != 1 or matching[0]["checkpoint"] != trial["checkpoint"]:
            raise ValueError(f"Training reference does not identify the served EE checkpoint: {trial['name']}")
        row = matching[0]
        if row["training_steps"] != trial["training_steps"] or row["seed"] != trial["training_seed"]:
            raise ValueError(f"EE training steps or seed differ from the trial: {trial['name']}")
        media = json.loads((repo_root / "usage_assets/animations/manifest.json").read_text())
        source_record = next(record for record in media if record["name"] == "expert_pressbutton")
        data_file = resolve_source(source_record["source"], repo_root)
        dataset = data_file.parents[2]
        info = dataset / "meta/info.json"
        if sha256(info) != row["canonical_info"]["sha256"]:
            raise ValueError("The locally recorded EE dataset differs from the training reference")
        source_dataset = row["canonical_info"]["path"].removesuffix("/meta/info.json")
        dataset_sha256 = row["canonical_info"]["sha256"]
        dataset_hash_basis = "canonical meta/info.json"
        regime = "ee_transfer"
    elif trial["training_robot"] == "UAQuad":
        canonical = reference["canonical_dataset"]
        dataset = resolve_source(canonical["source_relative_path"], repo_root)
        data_file = dataset / "data/chunk-000/file-000.parquet"
        dataset_sha256 = sha256(data_file)
        if (
            dataset_sha256 != trial["source_dataset_sha256"]
            or dataset_sha256 != canonical["sha256"]["data/chunk-000/file-000.parquet"]
        ):
            raise ValueError("The UAQuad training dataset differs from its provenance")
        row = reference["act" if trial["family"] == "act" else "diffusion_policy"]
        if row["optimizer_steps"] != trial["training_steps"] or row["training_seed"] != trial["training_seed"]:
            raise ValueError(f"UAQuad training steps or seed differ from the trial: {trial['name']}")
        if row["checkpoint_container_path"] != trial["checkpoint"]:
            raise ValueError(f"UAQuad training checkpoint differs from the trial: {trial['name']}")
        source_dataset = canonical["source_relative_path"]
        dataset_hash_basis = "canonical data/chunk-000/file-000.parquet"
        regime = "uaquad_one_step"
    else:
        raise ValueError(f"Unsupported training embodiment: {trial['training_robot']}")
    initialized = {
        "ResNet18 IMAGENET1K_V1": True,
        "random ResNet18": False,
        "OpenPI base checkpoint": True,
    }[trial["pretrained_initialization"]]
    return {
        "training_robot": trial["training_robot"],
        "source_dataset": source_dataset,
        "source_dataset_sha256": dataset_sha256,
        "source_dataset_hash_basis": dataset_hash_basis,
        "training_steps": trial["training_steps"],
        "training_initialized_pretrained": initialized,
        "pretrained_initialization": trial["pretrained_initialization"],
        "training_regime": regime,
        "source_training_reference": trial["training_reference"],
        "source_training_reference_sha256": sha256(reference_path),
        "training_env_cfg": str((dataset / "env_cfg.yaml").relative_to(repo_root)),
        "training_env_cfg_sha256": sha256(dataset / "env_cfg.yaml"),
    }, dataset / "env_cfg.yaml"


def build_policy_specs(
    record: dict,
    record_index: int,
    report_path: Path,
    repo_root: Path,
    container: str,
    container_root: Path,
) -> list[dict]:
    """Build model previews from one real evaluator rollout and its saved config."""
    trial = record["spec"]
    summary_path = resolve_source(record["eval_summary"], repo_root)
    summary = json.loads(summary_path.read_text())
    runtime = summary["metadata"]
    if summary["status"] != "completed" or summary["error"] is not None or len(summary["rollouts"]) != 1:
        raise ValueError(f"Expected one completed real model rollout: {trial['name']}")
    if sha256(summary_path) != record["eval_summary_sha256"]:
        raise ValueError(f"Evaluator report changed after validation: {trial['name']}")
    if runtime["Task"] != trial["task_id"] or runtime["Seed"] != trial["seed"]:
        raise ValueError(f"Evaluator task or seed differs from the trial: {trial['name']}")
    if runtime["Policy ID"] != trial["name"] or set(runtime["Video cameras"]) != set(CAMERAS):
        raise ValueError(f"Evaluator identity or cameras differ from the trial: {trial['name']}")
    rollout = summary["rollouts"][0]
    if rollout["termination_reason"] not in {"success", "timeout"}:
        raise ValueError(f"Evaluator did not finish at a task boundary: {trial['name']}")
    if rollout["success"] != (rollout["termination_reason"] == "success"):
        raise ValueError(f"Evaluator task success and termination reason disagree: {trial['name']}")
    videos = {camera: resolve_source(record["videos"][camera], repo_root) for camera in CAMERAS}
    env_cfg = summary_path.parent / "env_cfg.yaml"
    training, training_cfg = training_metadata(trial, repo_root)
    snapshot = probe_snapshot(env_cfg, videos, repo_root, container, container_root, training_cfg)
    expected_training_robot = {"EE": "end_effector", "UAQuad": "ua_quad"}[trial["training_robot"]]
    if snapshot["training_robot_id"] != expected_training_robot or snapshot["robot_profile"]["robot_id"] != "ua_quad":
        raise ValueError(
            f"Saved training/inference robot profiles differ from the declared experiment: {trial['name']}"
        )
    training["training_robot_id"] = snapshot["training_robot_id"]
    if snapshot["seed"] != trial["seed"] or not math.isclose(snapshot["step_dt_s"], 1 / 120):
        raise ValueError(f"Saved model evaluation seed or timing differs: {trial['name']}")
    tracking_path = summary_path.parent / "tracking/tracking.jsonl"
    if sha256(tracking_path) != record["tracking_sha256"]:
        raise ValueError(f"Tracking log changed after validation: {trial['name']}")
    identity = record["server_identity"]
    if identity["environment"] != {"CUDA_VISIBLE_DEVICES": "-1", "PYTHONHASHSEED": str(trial["inference_seed"])}:
        raise ValueError(f"Inference process device or seed differs from the declared CPU trial: {trial['name']}")
    server_log = summary_path.parent / "server-inference.log"
    server_text = server_log.read_text()
    if (
        "REMOTE_INFERENCE_DEVICE=cpu" not in server_text
        or f"REMOTE_INFERENCE_SEED={trial['inference_seed']}" not in server_text
    ):
        raise ValueError(f"Inference server did not report the declared CPU device and seed: {trial['name']}")
    timing = []
    specs = []
    for camera in CAMERAS:
        header = snapshot["videos"][camera]
        if not math.isclose(header["fps"], 30) or header["frame_count"] < 2:
            raise ValueError(f"Expected a nonempty model source video at 30 FPS: {videos[camera]}")
        timing.append((header["fps"], header["frame_count"], header["duration_s"]))
        note = (
            "One complete model attempt; success and timeout are retained without filtering. "
            "Source MP4 is verified 30 FPS with frame_skip=4; simulation runs at 120 Hz. "
            "Reset observation is unavailable in the evaluator report; no initial state is fabricated. "
            "One optimizer step validates software interfaces and does not establish trained task performance."
        )
        if training["training_regime"] == "ee_transfer":
            note += " This EE-trained checkpoint is transferred to the physical UAQuad without UAQuad training."
        else:
            note += " The visual backbone was randomly initialized for this UAQuad one-step CPU training."
        specs.append({
            "name": f"{trial['name']}_{camera}",
            "title": f"{trial['model_variant']} / {camera} / seed{trial['seed']}",
            "source_video": str(videos[camera].relative_to(repo_root)),
            "source_fps": header["fps"],
            "source_report": str(summary_path.relative_to(repo_root)),
            "outcome_path": ["rollouts", 0, "termination_reason"],
            "outcome_labels": {"success": "SUCCESS", "timeout": "TIMEOUT"},
            "metadata": {
                "experiment_id": trial["experiment_id"],
                "category": "uaquad",
                "trial_id": trial["name"],
                "camera": camera,
                "task_id": runtime["Task"],
                "env_seed": runtime["Seed"],
                "inference_seed": trial["inference_seed"],
                "inference_device": trial["inference_device"],
                "training_seed": trial["training_seed"],
                "training_steps": trial["training_steps"],
                "training_reference": trial["training_reference"],
                "episode_index": rollout["rollout_idx"],
                "steps": rollout["executed_steps"],
                "episode_length_s": runtime["Episode length override (s)"],
                "simulation_duration_s": rollout["executed_steps"] * snapshot["step_dt_s"],
                "action_semantics": runtime["Action semantics"],
                "source_time_basis": "source MP4 presentation timestamps (30 FPS), simulation ticks (120 Hz)",
                "attempt_selection": trial["attempt_selection"],
            },
            "recording_metadata": {
                **training,
                "policy_family": trial["family"],
                "model_variant": trial["model_variant"],
                "robot_id": snapshot["robot_profile"]["robot_id"],
                "robot_profile": snapshot["robot_profile"],
                "experiment_conditions": snapshot["experiment_conditions"],
                "actuation": snapshot["actuation"],
                "camera_pose": snapshot["camera_poses"][camera],
                "step_dt_s": snapshot["step_dt_s"],
                "episode_status": summary["status"],
                "successful_episodes": int(rollout["success"]),
                "requested_rollouts": summary["summary"]["num_rollouts"],
                "initial_observation_available": False,
                "recording_video_fps": header["fps"],
                "recording_video_frame_skip": 4,
                "source_video_header": header,
                "source_env_cfg": str(env_cfg.relative_to(repo_root)),
                "source_env_cfg_sha256": sha256(env_cfg),
                "runtime_metadata_source": "eval_summary.json and saved env_cfg.yaml",
                "source_driver_report": str(report_path.relative_to(repo_root)),
                "source_driver_report_sha256": sha256(report_path),
                "source_driver_record_index": record_index,
                "source_tracking": str(tracking_path.relative_to(repo_root)),
                "source_tracking_sha256": sha256(tracking_path),
                "source_server_log": str(server_log.relative_to(repo_root)),
                "source_server_log_sha256": sha256(server_log),
                "server_identity": identity,
                "tracking": rollout.get("tracking"),
                "recording_spec": trial,
            },
            "notes": note,
        })
    if len(set(timing)) != 1:
        raise ValueError(f"The three model source cameras are not synchronized: {trial['name']}")
    return specs


def build_specs(reports: list[Path], repo_root: Path, container: str, container_root: Path) -> list[dict]:
    """Build previews for actual exported outcomes, checking all camera headers."""
    specs = []
    seen_trials = set()
    for report_path in reports:
        driver = json.loads(report_path.read_text())
        for record_index, record in enumerate(driver["results"]):
            if record["status"] != "completed":
                continue
            trial = record["spec"]
            trial_id = trial["name"]
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", trial_id) or trial_id in seen_trials:
                raise ValueError(f"Unsafe or duplicate trial name: {trial_id}")
            seen_trials.add(trial_id)
            if "eval_summary" in record:
                specs.extend(
                    build_policy_specs(record, record_index, report_path, repo_root, container, container_root)
                )
                continue
            outcomes_path = resolve_source(record["outcomes_json"], repo_root)
            session = outcomes_path.parent
            outcomes = json.loads(outcomes_path.read_text())
            if outcomes["status"] not in {"completed", "episode_limit"} or len(outcomes["episodes"]) != 1:
                raise ValueError(f"Expected one finished episode: {outcomes_path}")
            episode = outcomes["episodes"][0]
            if not episode["exported"] or episode["termination_reason"] not in {"success", "timeout"}:
                raise ValueError(f"Episode has no exported success/timeout recording: {outcomes_path}")
            if outcomes["task"] != trial["task_id"] or outcomes["seed"] != trial["seed"]:
                raise ValueError(f"Runtime task or seed differs from the trial spec: {trial_id}")
            if set(outcomes["camera_names"]) != set(CAMERAS):
                raise ValueError(f"Expected all three runtime cameras: {trial_id}")
            videos = {}
            for camera in CAMERAS:
                pattern = f"*{camera}-env{episode['env_id']}-eps{episode['exported_episode_index']}.mp4"
                matches = list((session / "videos").glob(pattern))
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one source video for {trial_id}/{camera}")
                videos[camera] = matches[0]
            env_cfg = session / "env_cfg.yaml"
            snapshot = probe_snapshot(env_cfg, videos, repo_root, container, container_root)
            if snapshot["seed"] != outcomes["seed"] or not math.isclose(snapshot["step_dt_s"], outcomes["step_dt_s"]):
                raise ValueError(f"Saved environment timing/seed differs from the outcome: {trial_id}")
            for field in ("robot_profile", "experiment_conditions", "camera_poses"):
                if field in outcomes and outcomes[field] != snapshot[field]:
                    raise ValueError(f"Runtime {field} differs from its saved environment: {trial_id}")
            telemetry = session / outcomes["telemetry_file"]
            with telemetry.open() as source:
                telemetry_rows = sum(1 for _ in source)
            if telemetry_rows != outcomes["telemetry_rows"] or telemetry_rows != episode["step_count"]:
                raise ValueError(f"Telemetry row count differs from the completed episode: {trial_id}")
            camera_timing = []
            for camera in CAMERAS:
                video = videos[camera]
                header = snapshot["videos"][camera]
                if not math.isclose(header["fps"], 30.0) or header["frame_count"] < 2:
                    raise ValueError(f"Expected a nonempty 30 FPS source video: {video}")
                camera_timing.append((header["fps"], header["frame_count"], header["duration_s"]))
                for field, value in (("video_fps", 30), ("video_frame_skip", 4)):
                    if field in outcomes and outcomes[field] != value:
                        raise ValueError(f"Unexpected runtime {field}: {trial_id}")
                runtime_source = "episode_outcomes.json" if "robot_profile" in outcomes else "env_cfg.yaml fallback"
                name = f"{trial_id}_{camera}"
                notes = (
                    "One complete scripted attempt; success and timeout clips are both retained. "
                    "Canonical observations/telemetry are 120 Hz; source MP4 is verified 30 FPS with frame_skip=4. "
                    "The preview compresses source time; it is not a real-time or aggregate performance result."
                )
                if not snapshot["experiment_conditions"]["enable_saturation"]:
                    notes += " Saturation is disabled: motor commands may exceed physical thrust limits."
                specs.append({
                    "name": name,
                    "title": f"{snapshot['robot_profile']['robot_id']} / {camera} / seed{outcomes['seed']}",
                    "source_video": str(video.relative_to(repo_root)),
                    "source_fps": header["fps"],
                    "source_report": str(outcomes_path.relative_to(repo_root)),
                    "outcome_path": ["episodes", 0, "termination_reason"],
                    "outcome_labels": {"success": "SUCCESS", "timeout": "TIMEOUT"},
                    "metadata": {
                        "experiment_id": trial["experiment_id"],
                        "category": "uaquad",
                        "trial_id": trial_id,
                        "camera": camera,
                        "task_id": outcomes["task"],
                        "env_seed": outcomes["seed"],
                        "episode_index": episode["episode_index"],
                        "steps": episode["step_count"],
                        "episode_length_s": outcomes["episode_length_s"],
                        "simulation_duration_s": episode["simulation_elapsed_s"],
                        "initial_observation": episode["initial_observation"],
                        "action_semantics": snapshot["robot_profile"]["action_mode"],
                        "source_time_basis": (
                            "source MP4 presentation timestamps (30 FPS), telemetry simulation ticks (120 Hz)"
                        ),
                        "attempt_selection": "one complete attempt including a timeout if the task did not succeed",
                    },
                    "recording_metadata": {
                        "policy_family": "scripted",
                        "robot_id": snapshot["robot_profile"]["robot_id"],
                        "robot_profile": snapshot["robot_profile"],
                        "experiment_conditions": snapshot["experiment_conditions"],
                        "actuation": snapshot["actuation"],
                        "camera_pose": snapshot["camera_poses"][camera],
                        "step_dt_s": outcomes["step_dt_s"],
                        "episode_status": outcomes["status"],
                        "successful_episodes": outcomes["successful_episodes"],
                        "requested_successes": outcomes["requested_successes"],
                        "recording_video_fps": header["fps"],
                        "recording_video_frame_skip": 4,
                        "source_video_header": header,
                        "source_env_cfg": str(env_cfg.relative_to(repo_root)),
                        "source_env_cfg_sha256": sha256(env_cfg),
                        "runtime_metadata_source": runtime_source,
                        "source_driver_report": str(report_path.relative_to(repo_root)),
                        "source_driver_report_sha256": sha256(report_path),
                        "source_driver_record_index": record_index,
                        "source_telemetry": str(telemetry.relative_to(repo_root)),
                        "source_telemetry_sha256": sha256(telemetry),
                        "telemetry_rows": telemetry_rows,
                        "recording_spec": trial,
                    },
                    "notes": notes,
                })
            if len(set(camera_timing)) != 1:
                raise ValueError(f"The three source cameras are not synchronized: {trial_id}")
    return specs


def merge_previews(
    specs: list[dict], previews_dirs: list[Path], media_dir: Path, max_bytes: int, repo_root: Path
) -> None:
    """Append verified previews while preserving every existing media byte."""
    incoming = []
    preview_sources = {}
    for previews_dir in previews_dirs:
        for row in json.loads((previews_dir / "manifest.json").read_text()):
            if row["name"] in preview_sources:
                raise ValueError(f"Duplicate preview name across cache directories: {row['name']}")
            preview_sources[row["name"]] = previews_dir
            incoming.append(row)
    by_name = {spec["name"]: spec for spec in specs}
    if len(incoming) != len(by_name) or {row["name"] for row in incoming} != set(by_name):
        raise ValueError("Preview manifest names do not match the source specs")
    manifest_path = media_dir / "manifest.json"
    existing = json.loads(manifest_path.read_text())
    existing_by_name = {row["name"]: row for row in existing}
    if len(existing_by_name) != len(existing):
        raise ValueError("Existing media manifest contains duplicate names")
    preserved = {}
    for row in existing:
        for kind in ("gif", "mp4"):
            path = media_dir / row[kind]
            digest = sha256(path)
            if digest != row[f"{kind}_sha256"]:
                raise ValueError(f"Existing media does not match its recorded hash: {path}")
            preserved[path] = digest
    prepared = []
    for row in incoming:
        spec = by_name[row["name"]]
        previews_dir = preview_sources[row["name"]]
        for field, source in (("source", spec["source_video"]), ("source_report", spec["source_report"])):
            if row[field] != source or sha256(resolve_source(source, repo_root)) != row[f"{field}_sha256"]:
                raise ValueError(
                    f"Cached preview source changed or differs from the frozen spec: {row['name']}/{field}"
                )
        if row["outcome_path"] != spec["outcome_path"] or row["outcome_labels"] != spec["outcome_labels"]:
            raise ValueError(f"Cached preview outcome selector differs from the frozen spec: {row['name']}")
        if not math.isclose(row["source_fps"], spec["source_fps"]):
            raise ValueError(f"Exported source FPS differs from the probed video: {row['name']}")
        if row["source_frame_count"] != spec["recording_metadata"]["source_video_header"]["frame_count"]:
            raise ValueError(f"Exported source frame count differs from the probed video: {row['name']}")
        if not row["mp4_full_decode_passed"]:
            raise ValueError(f"MP4 full decoding was not verified: {row['name']}")
        for field, value in spec["metadata"].items():
            if row.get(field) != value:
                raise ValueError(f"Exported experiment metadata differs for {row['name']}/{field}")
        budget = existing_by_name.get(row["name"], {}).get("preview_byte_budget_bytes", max_bytes)
        merged = {**row, **spec["recording_metadata"], "preview_byte_budget_bytes": budget}
        if row["name"] in existing_by_name and merged != existing_by_name[row["name"]]:
            raise ValueError(f"Refusing to replace an existing experiment record: {row['name']}")
        for kind in ("gif", "mp4"):
            filename = row[kind]
            if not re.fullmatch(rf"[a-z0-9][a-z0-9_-]*\.{kind}", filename):
                raise ValueError(f"Unsafe media filename: {filename}")
            source = previews_dir / filename
            if source.stat().st_size > budget or source.stat().st_size != row[f"{kind}_bytes"]:
                raise ValueError(f"Preview violates its byte budget or recorded size: {source}")
            if sha256(source) != row[f"{kind}_sha256"]:
                raise ValueError(f"Preview does not match its recorded hash: {source}")
            destination = media_dir / filename
            if destination.exists() and sha256(destination) != row[f"{kind}_sha256"]:
                raise ValueError(f"Refusing to overwrite different media: {destination}")
        prepared.append(merged)
    for row in prepared:
        for kind in ("gif", "mp4"):
            destination = media_dir / row[kind]
            if not destination.exists():
                shutil.copyfile(preview_sources[row["name"]] / row[kind], destination)
        if row["name"] not in existing_by_name:
            existing.append(row)
    for path, digest in preserved.items():
        if sha256(path) != digest:
            raise ValueError(f"Previously published media changed: {path}")
    manifest_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    print(f"Verified {len(preserved)} existing media files; manifest now contains {len(existing)} camera records")


def main() -> int:
    """Prepare actual-source specs and optionally merge their verified previews."""
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=repo_root / "usage_assets/uaquad/animation_specs.json")
    parser.add_argument("--probe-container", default="ambench-sim-research")
    parser.add_argument("--container-root", type=Path, default=Path("/workspace/ambench"))
    parser.add_argument("--merge-previews", type=Path, nargs="+")
    parser.add_argument("--media-dir", type=Path, default=repo_root / "usage_assets/animations")
    parser.add_argument("--max-bytes", type=int, default=192 * 1024)
    args = parser.parse_args()
    reports = [resolve_source(str(path), repo_root) for path in args.results_json]
    specs = build_specs(reports, repo_root, args.probe_container, args.container_root)
    if not specs:
        parser.error("No completed recording trials were found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2, sort_keys=True) + "\n")
    print(f"Built {len(specs)} camera specs from {len(specs) // len(CAMERAS)} completed trials: {args.output}")
    if args.merge_previews:
        merge_previews(specs, args.merge_previews, args.media_dir, args.max_bytes, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
