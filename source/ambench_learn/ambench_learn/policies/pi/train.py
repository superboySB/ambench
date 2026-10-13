# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Run provenance-bound JAX OpenPI training without editing the pinned checkout."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig
    from openpi.training.utils import TrainState


@dataclasses.dataclass(frozen=True)
class TrainingRequest:
    """Inputs sufficient to reconstruct the policy and its normalization identity."""

    variant: str
    stage: str
    mode: str
    repo_id: str
    asset_id: str
    dataset_home: str
    export_report: str
    norm_directory: str
    checkpoint_base_dir: str
    exp_name: str
    initial_checkpoint: str
    openpi_root: str
    parent_manifest: str | None = None
    seed: int = 42
    num_workers: int = 2
    fsdp_devices: int = 1


def file_sha256(path: Path) -> str:
    """Hash an artifact without loading a model or putting it in Git."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(directory: Path) -> dict[str, str]:
    """Freeze every file in a finalized dataset or checkpoint subtree."""
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = {
        str(path.relative_to(directory)): file_sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()
    }
    if not files:
        raise ValueError(f"Artifact directory is empty: {directory}")
    return files


def write_json(path: Path, payload: dict) -> None:
    """Update a small provenance receipt atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure_dataset_home(request: TrainingRequest) -> None:
    """Set the pinned LeRobot home before importing its module-level constants."""
    os.environ["HF_LEROBOT_HOME"] = str(Path(request.dataset_home).expanduser().resolve())
    os.environ["WANDB_MODE"] = "disabled"


def build_config(request: TrainingRequest, *, resume: bool = False) -> TrainConfig:
    """Apply paper budgets and an explicit data/asset identity to an official config."""
    for value in (request.repo_id, request.asset_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", value) or ".." in value:
            raise ValueError("Repository and asset IDs must be safe owner/name identifiers.")
    if (
        request.variant not in ("pi0", "pi05")
        or request.stage not in ("mt", "st")
        or request.mode not in ("full", "lora")
    ):
        raise ValueError("Unsupported policy variant, training stage, or tuning mode.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", request.exp_name) or request.exp_name in (".", ".."):
        raise ValueError("Experiment name must be a single safe path component.")
    if request.num_workers < 0 or request.fsdp_devices < 1 or not 0 <= request.seed < 2**32:
        raise ValueError("Worker count, FSDP devices, or training seed is invalid.")
    configure_dataset_home(request)
    from flax import nnx
    from openpi.training import config as openpi_config
    from openpi.training import optimizer, weight_loaders

    imported_config = Path(openpi_config.__file__)
    requested_config = Path(request.openpi_root) / "src/openpi/training/config.py"
    if file_sha256(imported_config) != file_sha256(requested_config):
        raise ValueError("Imported OpenPI configuration differs from the requested pinned source checkout.")
    name = f"{request.variant}_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative"
    original = openpi_config.get_config(name)
    model = original.model
    if request.mode == "lora":
        model = dataclasses.replace(model, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    data = dataclasses.replace(
        original.data,
        repo_id=request.repo_id,
        assets=openpi_config.AssetsConfig(assets_dir=request.norm_directory, asset_id=request.asset_id),
        include_base_image=False,
        action_representation="ee_local_relative",
    )
    steps, batch_size = (40_000, 32) if request.stage == "mt" else (20_000, 8)
    return dataclasses.replace(
        original,
        model=model,
        data=data,
        exp_name=request.exp_name,
        assets_base_dir=request.norm_directory,
        checkpoint_base_dir=request.checkpoint_base_dir,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(Path(request.initial_checkpoint) / "params")),
        num_train_steps=steps,
        batch_size=batch_size,
        seed=request.seed,
        num_workers=request.num_workers,
        fsdp_devices=request.fsdp_devices,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=2.5e-5, decay_steps=steps, decay_lr=2.5e-6
        ),
        ema_decay=0.99 if request.mode == "full" else None,
        freeze_filter=nnx.Nothing if request.mode == "full" else model.get_freeze_filter(),
        save_interval=5000,
        keep_period=5000,
        overwrite=False,
        resume=resume,
        wandb_enabled=False,
        policy_metadata={
            "action_representation": "ee_local_relative",
            "repo_id": request.repo_id,
            "asset_id": request.asset_id,
        },
    )


def dataset_proof(request: TrainingRequest) -> dict:
    """Require successful, selected training episodes and freeze the derived dataset."""
    report_path = Path(request.export_report)
    report = json.loads(report_path.read_text())
    if report.get("repo_id") != request.repo_id or report.get("target_hz") != 20 or report.get("include_base_image"):
        raise ValueError("Export repository, policy rate, or camera layout does not match.")
    if report.get("success_verified") is not True or report.get("require_success") is not True:
        raise ValueError("Training requires an export with independently verified successful episodes.")
    task_counts: dict[str, int] = {}
    robots = set()
    official_families = {
        "PressButton",
        "PegInHole",
        "FrameAssembly",
        "CabinetPickPlace",
        "LemonHarvesting",
        "TossBall",
        "RotateValve",
        "PushSlider",
        "PullLever",
        "OpenDoor",
        "WipeWindow",
        "NDT",
    }
    for source in report["sources"]:
        if source.get("success_verified") is not True or not source.get("source_files_sha256"):
            raise ValueError("Source success or canonical file provenance is missing.")
        selected = source["selected_episode_indices"]
        if selected != list(range(80)) or len(selected) != source["selected_successful_episodes"]:
            raise ValueError(
                "Formal training requires predeclared source episodes 0..79; validation episodes 80..99 are excluded."
            )
        task = source["task_id"]
        if task.split("-", 1)[0] not in official_families or not task.endswith("-Abs-PID-Direct-v0"):
            raise ValueError("Training data must use a maintained task with the absolute EE/PID interface.")
        task_counts[task] = task_counts.get(task, 0) + len(selected)
        robots.add(source["robot_type"])
    if len(robots) != 1 or not task_counts or any(count != 80 for count in task_counts.values()):
        raise ValueError("Each task must supply 80 successful training episodes from one embodiment.")
    if sum(task_counts.values()) != report["total_episodes"]:
        raise ValueError("Export total episode count does not match its selected sources.")
    if request.stage == "st" and len(task_counts) != 1:
        raise ValueError("Single-task adaptation requires exactly one task.")
    if request.stage == "mt" and len(task_counts) < 2:
        raise ValueError("Multi-task adaptation requires at least two tasks.")
    dataset_root = Path(request.dataset_home) / request.repo_id
    info = json.loads((dataset_root / "meta/info.json").read_text())
    export_metadata = info.get("ambench", {}).get("openpi_export", {})
    shapes = {key: tuple(value.get("shape", ())) for key, value in info["features"].items()}
    if (
        info["codebase_version"] != "v2.1"
        or info["fps"] != 20
        or info["total_episodes"] != report["total_episodes"]
        or info["total_frames"] != report["total_frames"]
        or info["total_tasks"] != len(task_counts)
        or export_metadata.get("source_action_semantics") != "ee_absolute"
        or export_metadata.get("include_base_image") is not False
        or shapes.get("actions") != (8,)
        or shapes.get("ee_pos") != (3,)
        or shapes.get("ee_quat") != (4,)
        or shapes.get("gripper_width") != (1,)
        or shapes.get("ee_image") != (384, 384, 3)
    ):
        raise ValueError("Finalized OpenPI metadata does not match the 20Hz EE training contract.")
    robot = next(iter(robots))
    robot_labels = {"ua_quad": "UAQuad", "ua_hexa": "UAHexa", "fa_hexa": "FAHexa", "omni_hexa": "OmniHexa", "ee": "EE"}
    if robot not in robot_labels or any(f"-Am-{robot_labels[robot]}-" not in task for task in task_counts):
        raise ValueError("Task IDs and recorded training embodiment disagree.")
    families = {task.split("-", 1)[0] for task in task_counts}
    if len(families) != len(task_counts):
        raise ValueError("Different task IDs duplicate a task family.")
    return {
        "export_report_sha256": file_sha256(report_path),
        "dataset_files": artifact_inventory(dataset_root),
        "robot_type": robot,
        "task_counts": task_counts,
        "total_episodes": report["total_episodes"],
        "paper_dataset_scope_matched": robot == "fa_hexa" and (
            families == official_families if request.stage == "mt" else len(families) == 1
        ),
        "selection_contract": "source episode indices 0..79; 80..99 reserved for validation",
        "scope_note": "UA-specialized training is new evidence; the paper's learned-policy comparison uses FAHexa.",
    }


def config_snapshot(config: TrainConfig) -> dict:
    """Save the effective model, optimizer, scheduler, data factory, and training settings."""
    snapshot = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if field.name in ("freeze_filter", "data"):
            continue
        if dataclasses.is_dataclass(value):
            value = dataclasses.asdict(value)
        if isinstance(value, Path):
            value = str(value)
        snapshot[field.name] = value
    snapshot["data"] = {
        "factory": type(config.data).__module__ + "." + type(config.data).__qualname__,
        "repo_id": config.data.repo_id,
        "asset_id": config.data.assets.asset_id,
        "assets_dir": config.data.assets.assets_dir,
        "action_representation": config.data.action_representation,
        "include_base_image": config.data.include_base_image,
        "prompt_from_task": config.data.base_config.prompt_from_task,
    }
    snapshot["freeze_filter"] = (
        "Nothing" if "lora" not in config.model.paligemma_variant else "model.get_freeze_filter()"
    )
    return snapshot


def read_checkpoint_step(checkpoint: Path) -> int:
    """Read only the stored optimizer update count, never infer it from a directory name."""
    import numpy as np
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        restored = checkpointer.restore(
            checkpoint / "train_state",
            item={"step": np.asarray(0, dtype=np.int32)},
            transforms={"step": ocp.Transform(original_key="step")},
            restore_args={"step": ocp.RestoreArgs(restore_type=np.ndarray)},
        )
        return int(np.asarray(restored["step"]))
    finally:
        checkpointer.close()


def verify_checkpoint(checkpoint: Path, request: TrainingRequest, norm_sha: str) -> dict:
    """Bind a real training state to its inference parameters and copied normalization."""
    stats = checkpoint / "assets" / request.asset_id / "norm_stats.json"
    if not stats.is_file() or file_sha256(stats) != norm_sha:
        raise ValueError("Checkpoint normalization differs from the verified training statistics.")
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    try:
        metadata = checkpointer.metadata(checkpoint / "params").tree
    finally:
        checkpointer.close()
    shape_rows = []
    pending = [metadata]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            shape_rows.append((tuple(value.shape), str(value.dtype)))
    if not shape_rows:
        raise ValueError("Inference checkpoint has no stored parameter arrays.")
    return {
        "path": str(checkpoint.resolve()),
        "optimizer_updates": read_checkpoint_step(checkpoint),
        "parameter_tensor_count": len(shape_rows),
        "parameter_scalar_count": sum(math.prod(shape) for shape, _ in shape_rows),
        "parameter_dtypes": sorted({dtype for _, dtype in shape_rows}),
        "params_files": artifact_inventory(checkpoint / "params"),
        "norm_stats_sha256": norm_sha,
    }


def compute_statistics(request: TrainingRequest, config: TrainConfig, proof: dict) -> dict:
    """Compute statistics from all training frames, including the last short batch."""
    import numpy as np
    from openpi.shared import normalize
    from openpi.training import data_loader
    from scripts.compute_norm_stats import RemoveStrings
    from torch.utils.data import DataLoader

    stats_path = Path(request.norm_directory) / request.asset_id / "norm_stats.json"
    if stats_path.exists():
        raise FileExistsError("Normalization already exists; validate and reuse its provenance instead of overwriting.")
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.TransformedDataset(
        dataset, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs, RemoveStrings()]
    )
    accumulators = {key: normalize.RunningStats() for key in ("state", "actions")}
    frames = 0
    for batch in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=request.num_workers, drop_last=False):
        for key, accumulator in accumulators.items():
            accumulator.update(np.asarray(batch[key]))
        frames += len(batch["state"])
    normalize.save(stats_path.parent, {key: value.get_statistics() for key, value in accumulators.items()})
    receipt = {"dataset_proof": proof, "frames_used": frames, "norm_stats_sha256": file_sha256(stats_path)}
    write_json(stats_path.parent / "training_norm_provenance.json", receipt)
    return receipt


def validate_statistics(request: TrainingRequest, proof: dict) -> str:
    """Reject stale statistics from another repository, episode selection, or export."""
    import numpy as np
    from openpi.shared import normalize

    directory = Path(request.norm_directory) / request.asset_id
    receipt = json.loads((directory / "training_norm_provenance.json").read_text())
    sha = file_sha256(directory / "norm_stats.json")
    if receipt["dataset_proof"] != proof or receipt["norm_stats_sha256"] != sha:
        raise ValueError("Normalization provenance does not match the selected training dataset.")
    statistics = normalize.load(directory)
    for key in ("state", "actions"):
        value = statistics[key]
        required = (
            (value.mean, value.std, value.q01, value.q99) if request.variant == "pi05" else (value.mean, value.std)
        )
        if any(array is None or np.asarray(array).shape != (8,) or not np.isfinite(array).all() for array in required):
            raise ValueError("EE statistics must contain finite 8D state/action values and required quantiles.")
    return sha


def validate_parent(request: TrainingRequest, proof: dict) -> dict | None:
    """Require a genuinely completed, matching multi-task parent before specialization."""
    if request.stage == "mt":
        if request.parent_manifest is not None:
            raise ValueError("Multi-task initialization must come from base weights, not an ST manifest.")
        return None
    if request.parent_manifest is None:
        raise ValueError("ST adaptation requires a completed MT training manifest.")
    parent = json.loads(Path(request.parent_manifest).read_text())
    previous = TrainingRequest(**parent["request"])
    if (
        parent["status"] != "completed"
        or parent.get("trained") is not True
        or previous.stage != "mt"
        or previous.variant != request.variant
        or previous.mode != request.mode
        or parent["dataset_proof"]["robot_type"] != proof["robot_type"]
        or not set(proof["task_counts"]).issubset(parent["dataset_proof"]["task_counts"])
    ):
        raise ValueError("ST initialization is incompatible with the completed multi-task parent.")
    if request.asset_id == previous.asset_id or request.repo_id == previous.repo_id:
        raise ValueError("ST must use its own single-task dataset repository and normalization asset ID.")
    from ambench_learn.policies.pi.serve import load_serving_config

    _, checkpoint = load_serving_config(Path(request.parent_manifest), Path(request.initial_checkpoint))
    configure_dataset_home(request)
    selected = parent["candidate_checkpoints"][str(checkpoint.resolve())]
    return {
        "manifest_sha256": file_sha256(Path(request.parent_manifest)),
        "checkpoint": selected,
        "run_budget_completed": parent["run_budget_completed"],
        "selected_checkpoint_updates": selected["optimizer_updates"],
    }


def run_training(request: TrainingRequest, *, action: str, resume: bool = False) -> dict:
    """Prepare, validate, or execute a full-budget run with explicit completion evidence."""
    config = build_config(request, resume=resume)
    proof = dataset_proof(request)
    if action == "stats":
        return {"status": "statistics_completed", "trained": False, **compute_statistics(request, config, proof)}
    norm_sha = validate_statistics(request, proof)
    parent = validate_parent(request, proof)
    snapshot = config_snapshot(config)
    if action == "validate":
        return {"status": "configuration_verified", "trained": False, "config": snapshot, "dataset_proof": proof}
    if action != "train":
        raise ValueError("Expected stats, validate, or train action.")
    source_root = Path(request.openpi_root)
    source_files = (
        "scripts/train.py",
        "src/openpi/training/config.py",
        "src/openpi/training/checkpoints.py",
        "src/openpi/policies/am_bench_policy.py",
        "src/openpi/transforms.py",
        "src/openpi/models/pi0.py",
    )
    source_sha = {name: file_sha256(source_root / name) for name in source_files}
    manifest_path = Path(request.checkpoint_base_dir) / "provenance" / f"{config.name}_{request.exp_name}.json"
    contract = {
        "request": dataclasses.asdict(request),
        "config": {**snapshot, "resume": False},
        "dataset_proof": proof,
        "normalization_sha256": norm_sha,
        "parent": parent,
        "openpi_source_sha256": source_sha,
        "initial_params_files": artifact_inventory(Path(request.initial_checkpoint) / "params"),
        "wrapper_source_sha256": {name: file_sha256(Path(__file__).parent / name) for name in ("train.py", "serve.py")},
    }
    if resume:
        existing = json.loads(manifest_path.read_text())
        if existing["contract"] != contract or existing["status"] == "completed":
            raise ValueError("Resume contract changed or the requested training budget is already complete.")
        candidates = sorted(
            (path for path in config.checkpoint_dir.iterdir() if path.is_dir() and path.name.isdecimal()),
            key=lambda p: int(p.name),
        )
        if not candidates:
            raise ValueError("Resume requested without a real saved training state.")
        initial_step = read_checkpoint_step(candidates[-1])
        if initial_step >= config.num_train_steps:
            raise ValueError(
                "The budget is already stored but completion was not verified; review the previous finite-step"
                " evidence."
            )
    else:
        if manifest_path.exists() or config.checkpoint_dir.exists():
            raise FileExistsError(
                "Training provenance or checkpoint directory exists; choose a new run or explicit resume."
            )
        initial_step = 0
    receipt = {
        **contract,
        "contract": contract,
        "status": "running",
        "trained": False,
        "resume_boundary": "Optimizer/EMA/step restored; the pinned JAX trainer rebuilds the shuffled data iterator.",
    }
    if resume:
        write_json(manifest_path.with_name(manifest_path.stem + f"_before_resume_{time.time_ns()}.json"), existing)
    write_json(manifest_path, receipt)
    # The manifest lives outside checkpoint_dir, which the official trainer creates.
    spec = importlib.util.spec_from_file_location("ambench_pinned_openpi_jax_train", source_root / "scripts/train.py")
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the pinned JAX training entrypoint.")
    trainer = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(trainer)
    except BaseException as error:
        receipt.update(status="interrupted", trained=False, error_type=type(error).__name__)
        write_json(manifest_path, receipt)
        raise
    original_step = trainer.train_step
    observed_steps: set[int] = set()
    observer_lock = threading.Lock()
    loss_log = manifest_path.with_name(manifest_path.stem + f"_attempt_{time.time_ns()}_losses.jsonl")
    receipt["loss_log"] = str(loss_log)
    write_json(manifest_path, receipt)
    import jax
    import numpy as np

    def observe(step: Any, metrics: dict[str, Any]) -> None:
        step = int(np.asarray(step))
        values = {key: float(np.asarray(value)) for key, value in metrics.items()}
        if not all(np.isfinite(value) for value in values.values()):
            raise FloatingPointError("Non-finite JAX loss, gradient norm, or parameter norm.")
        with observer_lock:
            if step not in observed_steps:
                with loss_log.open("a") as file:
                    file.write(json.dumps({"optimizer_step": step, **values}) + "\n")
                observed_steps.add(step)

    def checked_step(*args: Any, **kwargs: Any) -> tuple[TrainState, dict[str, Any]]:
        state, metrics = original_step(*args, **kwargs)
        jax.debug.callback(observe, state.step, metrics)
        return state, metrics

    trainer.train_step = checked_step
    try:
        trainer.main(config)
        jax.effects_barrier()
        final_path = config.checkpoint_dir / str(config.num_train_steps - 1)
        final = verify_checkpoint(final_path, request, norm_sha)
        if final["optimizer_updates"] != config.num_train_steps or observed_steps != set(
            range(initial_step + 1, config.num_train_steps + 1)
        ):
            raise ValueError("Stored optimizer step and finite observed updates do not prove completion.")
        candidate_checkpoints = {}
        for path in sorted(config.checkpoint_dir.iterdir()):
            if not path.is_dir() or not path.name.isdecimal():
                continue
            verified = final if path.resolve() == final_path.resolve() else verify_checkpoint(path, request, norm_sha)
            if not 0 < verified["optimizer_updates"] <= config.num_train_steps:
                raise ValueError("A retained candidate has an invalid stored optimizer step.")
            candidate_checkpoints[str(path.resolve())] = verified
        receipt.update(
            status="completed",
            trained=True,
            final_checkpoint=final,
            run_budget_completed=final["optimizer_updates"],
            candidate_checkpoints=candidate_checkpoints,
            finite_update_count=len(observed_steps),
        )
    except BaseException as error:
        receipt.update(
            status="interrupted",
            trained=False,
            error_type=type(error).__name__,
            finite_update_count=len(observed_steps),
        )
        write_json(manifest_path, receipt)
        raise
    receipt["loss_log"] = str(loss_log)
    write_json(manifest_path, receipt)
    return {
        "status": receipt["status"],
        "trained": receipt["trained"],
        "manifest": str(manifest_path),
        "optimizer_updates": final["optimizer_updates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "variant",
        "stage",
        "mode",
        "repo_id",
        "asset_id",
        "dataset_home",
        "export_report",
        "norm_directory",
        "checkpoint_base_dir",
        "exp_name",
        "initial_checkpoint",
        "openpi_root",
    ):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--parent-manifest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--action", choices=("stats", "validate", "train"), default="validate")
    parser.add_argument("--resume", action="store_true")
    arguments = vars(parser.parse_args())
    action, resume = arguments.pop("action"), arguments.pop("resume")
    logging.basicConfig(level=logging.INFO, force=True)
    sys.path.insert(0, arguments["openpi_root"])
    print(json.dumps(run_training(TrainingRequest(**arguments), action=action, resume=resume), indent=2))


if __name__ == "__main__":
    main()
