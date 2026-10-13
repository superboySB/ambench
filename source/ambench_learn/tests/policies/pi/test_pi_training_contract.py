# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU identity and shape checks; these fixtures are not trained OpenPI policies."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ambench_learn.policies.pi.serve import load_serving_config
from ambench_learn.policies.pi.train import (
    TrainingRequest,
    build_config,
    config_snapshot,
    dataset_proof,
    file_sha256,
    read_checkpoint_step,
    validate_parent,
    validate_statistics,
    verify_checkpoint,
    write_json,
)


def fixture_request(tmp_path: Path, **overrides) -> TrainingRequest:
    repo_id = overrides.get("repo_id", "lab/uaquad_train80")
    dataset = tmp_path / "datasets" / repo_id
    dataset.mkdir(parents=True, exist_ok=True)
    (dataset / "fixture.bin").write_bytes(b"tiny test fixture, not robot demonstrations")
    report = {
        "repo_id": repo_id,
        "target_hz": 20,
        "include_base_image": False,
        "require_success": True,
        "success_verified": True,
        "total_episodes": 160,
        "total_frames": 320,
        "sources": [
            {
                "task_id": task,
                "robot_type": "ua_quad",
                "selected_episode_indices": list(range(80)),
                "selected_successful_episodes": 80,
                "success_verified": True,
                "source_files_sha256": {"fixture": "0" * 64},
            }
            for task in ("PressButton-Am-UAQuad-Abs-PID-Direct-v0", "OpenDoor-Am-UAQuad-Abs-PID-Direct-v0")
        ],
    }
    write_json(
        dataset / "meta/info.json",
        {
            "codebase_version": "v2.1",
            "fps": 20,
            "total_episodes": 160,
            "total_frames": 320,
            "total_tasks": 2,
            "features": {
                key: {"shape": shape}
                for key, shape in {
                    "actions": [8],
                    "ee_pos": [3],
                    "ee_quat": [4],
                    "gripper_width": [1],
                    "ee_image": [3, 384, 384],
                }.items()
            },
            "ambench": {"openpi_export": {"source_action_semantics": "ee_absolute", "include_base_image": False}},
        },
    )
    report_path = tmp_path / "export_report.json"
    write_json(report_path, report)
    request = TrainingRequest(
        variant="pi0",
        stage="mt",
        mode="full",
        repo_id=repo_id,
        asset_id="lab/custom_stats",
        dataset_home=str(tmp_path / "datasets"),
        export_report=str(report_path),
        norm_directory=str(tmp_path / "norm"),
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        exp_name="fixture",
        initial_checkpoint=str(tmp_path / "base"),
        openpi_root="/opt/openpi",
        num_workers=0,
    )
    return dataclasses.replace(request, **overrides)


def write_statistics(request: TrainingRequest, proof: dict) -> str:
    from openpi.shared import normalize

    values = np.ones(8, dtype=np.float32)
    statistics = {
        key: normalize.NormStats(mean=values * 0, std=values, q01=-values, q99=values) for key in ("state", "actions")
    }
    directory = Path(request.norm_directory) / request.asset_id
    normalize.save(directory, statistics)
    sha = file_sha256(directory / "norm_stats.json")
    write_json(directory / "training_norm_provenance.json", {"dataset_proof": proof, "norm_stats_sha256": sha})
    return sha


def tiny_checkpoint(path: Path, step: int, request: TrainingRequest) -> None:
    import orbax.checkpoint as ocp
    from flax import nnx, struct

    class FixtureModel(nnx.Module):
        def __init__(self):
            self.kernel = nnx.Param(np.ones((2, 3), dtype=np.float32))

    parameters = nnx.state(FixtureModel())

    @struct.dataclass
    class FixtureState:
        step: object
        params: object
        opt_state: object

    path.mkdir(parents=True)
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path / "train_state", FixtureState(np.asarray(step, np.int32), parameters, {"m": np.zeros(3)}))
    checkpointer.save(path / "params", {"params": parameters})
    checkpointer.close()
    target = path / "assets" / request.asset_id
    target.mkdir(parents=True)
    target.joinpath("norm_stats.json").write_bytes(
        (Path(request.norm_directory) / request.asset_id / "norm_stats.json").read_bytes()
    )


@pytest.mark.parametrize("variant", ["pi0", "pi05"])
@pytest.mark.parametrize("stage", ["mt", "st"])
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_config_budgets_shapes_and_freeze_filter(tmp_path: Path, variant: str, stage: str, mode: str) -> None:
    from flax import nnx
    from flax.nnx.filterlib import to_predicate

    request = fixture_request(tmp_path, variant=variant, stage=stage, mode=mode)
    config = build_config(request)
    assert (config.num_train_steps, config.batch_size) == ((40000, 32) if stage == "mt" else (20000, 8))
    assert config.ema_decay == (0.99 if mode == "full" else None)
    assert config.lr_schedule.peak_lr == 2.5e-5
    assert config.lr_schedule.warmup_steps == 1000
    assert config.lr_schedule.decay_steps == config.num_train_steps
    observation, action = config.model.inputs_spec(batch_size=2)
    assert observation.state.shape == (2, 32)
    assert action.shape == (2, 50, 32)
    assert observation.tokenized_prompt.shape == (2, 200 if variant == "pi05" else 48)
    assert config.data.repo_id == request.repo_id
    assert config.data.assets.asset_id == "lab/custom_stats"
    assert config.data.assets.assets_dir == request.norm_directory
    assert config.data.include_base_image is False
    assert config.data.action_representation == "ee_local_relative"
    predicate = to_predicate(config.freeze_filter)
    parameter = nnx.Param(np.zeros((4, 8), dtype=np.float32)).to_state()
    assert bool(predicate(("PaliGemma", "llm", "kernel"), parameter)) == (mode == "lora")
    assert not predicate(("PaliGemma", "llm", "lora_a"), parameter)
    assert not predicate(("vision", "kernel"), parameter)
    assert json.loads(json.dumps(config_snapshot(config)))["batch_size"] == config.batch_size


@pytest.mark.parametrize("variant", ["pi0", "pi05"])
def test_custom_stats_and_local_relative_roundtrip(tmp_path: Path, variant: str) -> None:
    from openpi import transforms

    request = fixture_request(tmp_path, variant=variant)
    config = build_config(request)
    proof = dataset_proof(request)
    write_statistics(request, proof)
    validate_statistics(request, proof)
    data = config.data.create(config.assets_dirs, config.model)
    absolute = np.tile([1.25, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, -1.0], (50, 1)).astype(np.float32)
    raw = {
        "ee_pos": np.array([1.0, 2.0, 3.0]),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "gripper_width": np.array([0.04]),
        "ee_image": np.zeros((384, 384, 3), dtype=np.uint8),
        "actions": absolute,
        "prompt": "press the button",
    }
    converted = transforms.compose([*data.repack_transforms.inputs, *data.data_transforms.inputs])(raw)
    assert converted["state"].shape == (8,)
    np.testing.assert_allclose(converted["actions"][:, :3], np.tile([0.25, 0, 0], (50, 1)))
    assert not converted["image_mask"]["base_0_rgb"]
    transformed = transforms.compose(data.model_transforms.inputs)(converted)
    assert transformed["actions"].shape == (50, 32)
    assert transformed["state"].shape == (32,)
    assert transformed["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    decoded = transforms.compose(data.data_transforms.outputs)({"actions": transformed["actions"]})
    np.testing.assert_allclose(decoded["actions"], absolute, rtol=0, atol=1e-6)
    assert proof["paper_dataset_scope_matched"] is False


def test_wrong_repo_validation_leak_and_stale_stats_are_rejected(tmp_path: Path) -> None:
    request = fixture_request(tmp_path)
    proof = dataset_proof(request)
    write_statistics(request, proof)
    with pytest.raises(ValueError, match="repository"):
        dataset_proof(dataclasses.replace(request, repo_id="lab/another_dataset"))
    report = json.loads(Path(request.export_report).read_text())
    report["sources"][0]["selected_episode_indices"] = list(range(20, 100))
    write_json(Path(request.export_report), report)
    with pytest.raises(ValueError, match="validation episodes"):
        dataset_proof(request)
    with pytest.raises(ValueError, match="provenance"):
        validate_statistics(request, {**proof, "export_report_sha256": "changed"})


def test_dataset_gate_matches_real_exporter_image_metadata(tmp_path: Path) -> None:
    exporter = Path(__file__).resolve().parents[5] / "scripts/data/export_lerobot_to_openpi.py"
    tree = ast.parse(exporter.read_text())
    build_features = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_features"
    )
    # The exporter uses the newer LeRobot package. Load its actual pure metadata
    # helper while these contract tests run in OpenPI's pinned older environment.
    namespace = {"np": np, "Any": Any}
    exec(compile(ast.Module(body=[build_features], type_ignores=[]), str(exporter), "exec"), namespace)
    features = namespace["build_features"]({
        "ee_image": np.zeros((384, 384, 3), dtype=np.uint8),
        "actions": np.zeros(8, dtype=np.float32),
        "ee_pos": np.zeros(3, dtype=np.float32),
        "ee_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "gripper_width": np.zeros(1, dtype=np.float32),
        "task": "press the button",
    })
    assert features["ee_image"]["shape"] == (3, 384, 384)
    assert features["ee_image"]["names"] == ["channel", "height", "width"]
    request = fixture_request(tmp_path)
    info_path = Path(request.dataset_home) / request.repo_id / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"] = features
    write_json(info_path, info)
    assert dataset_proof(request)["total_episodes"] == 160
    info["features"]["ee_image"]["shape"] = [384, 384, 3]
    write_json(info_path, info)
    with pytest.raises(ValueError, match="training contract"):
        dataset_proof(request)


def completed_fixture_manifest(
    request: TrainingRequest, proof: dict, norm_sha: str, final: dict, candidates: dict
) -> dict:
    """Describe only synthetic checkpoint identity, never a learned policy result."""
    contract = {
        "request": dataclasses.asdict(request),
        "config": config_snapshot(build_config(request)),
        "dataset_proof": proof,
        "normalization_sha256": norm_sha,
        "openpi_source_sha256": {
            "src/openpi/training/config.py": file_sha256(Path(request.openpi_root) / "src/openpi/training/config.py")
        },
        "wrapper_source_sha256": {
            name: file_sha256(Path(__file__).resolve().parents[3] / "ambench_learn/policies/pi" / name)
            for name in ("train.py", "serve.py")
        },
    }
    return {
        **contract,
        "contract": contract,
        "status": "completed",
        "trained": True,
        "final_checkpoint": final,
        "run_budget_completed": final["optimizer_updates"],
        "candidate_checkpoints": candidates,
    }


def test_stored_step_and_serving_contract_use_real_checkpoint_state(tmp_path: Path) -> None:
    request = fixture_request(tmp_path)
    proof = dataset_proof(request)
    norm_sha = write_statistics(request, proof)
    misleading_directory = tmp_path / "39999"
    tiny_checkpoint(misleading_directory, 7, request)
    assert read_checkpoint_step(misleading_directory) == 7
    checkpoint = tmp_path / "named_without_step"
    tiny_checkpoint(checkpoint, 40000, request)
    final = verify_checkpoint(checkpoint, request, norm_sha)
    assert final["parameter_scalar_count"] == 6
    assert final["parameter_tensor_count"] == 1
    candidate = tmp_path / "5000"
    tiny_checkpoint(candidate, 5001, request)
    candidate_proof = verify_checkpoint(candidate, request, norm_sha)
    manifest = completed_fixture_manifest(
        request,
        proof,
        norm_sha,
        final,
        {
            str(checkpoint.resolve()): final,
            str(candidate.resolve()): candidate_proof,
        },
    )
    manifest_path = tmp_path / "training.json"
    write_json(manifest_path, manifest)
    served, selected = load_serving_config(manifest_path)
    assert selected == checkpoint
    assert served.data.assets.asset_id == request.asset_id
    assert served.data.assets.assets_dir == str(checkpoint / "assets")
    assert served.policy_metadata["optimizer_updates"] == 40000
    assert served.policy_metadata["paper_dataset_scope_matched"] is False
    selected_config, selected_path = load_serving_config(manifest_path, candidate)
    assert selected_path == candidate
    assert selected_config.policy_metadata["run_budget_completed"] == 40000
    assert selected_config.policy_metadata["selected_checkpoint_updates"] == 5001
    with pytest.raises(ValueError, match="Checkpoint path"):
        load_serving_config(manifest_path, misleading_directory)
    manifest["status"] = "interrupted"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="complete"):
        load_serving_config(manifest_path)
    manifest["status"] = "completed"
    manifest["request"] = {**manifest["request"], "asset_id": "lab/wrong_stats"}
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="asset ID"):
        load_serving_config(manifest_path)


def test_st_requires_own_dataset_statistics_and_registered_mt_parent(tmp_path: Path) -> None:
    parent_request = fixture_request(tmp_path / "parent")
    parent_proof = dataset_proof(parent_request)
    parent_norm_sha = write_statistics(parent_request, parent_proof)
    final_path = tmp_path / "mt_final"
    selected_path = tmp_path / "mt_selected"
    tiny_checkpoint(final_path, 40000, parent_request)
    tiny_checkpoint(selected_path, 35001, parent_request)
    final = verify_checkpoint(final_path, parent_request, parent_norm_sha)
    selected = verify_checkpoint(selected_path, parent_request, parent_norm_sha)
    parent = completed_fixture_manifest(
        parent_request,
        parent_proof,
        parent_norm_sha,
        final,
        {
            str(final_path.resolve()): final,
            str(selected_path.resolve()): selected,
        },
    )
    parent_path = tmp_path / "parent_manifest.json"
    write_json(parent_path, parent)
    child = fixture_request(
        tmp_path / "child",
        stage="st",
        repo_id="lab/press_train80",
        asset_id="lab/press_stats",
        initial_checkpoint=str(selected_path),
        parent_manifest=str(parent_path),
    )
    report = json.loads(Path(child.export_report).read_text())
    report["sources"] = report["sources"][:1]
    report["total_episodes"] = 80
    report["total_frames"] = 160
    write_json(Path(child.export_report), report)
    info_path = Path(child.dataset_home) / child.repo_id / "meta/info.json"
    info = json.loads(info_path.read_text())
    info.update(total_episodes=80, total_frames=160, total_tasks=1)
    write_json(info_path, info)
    child_proof = dataset_proof(child)
    write_statistics(child, child_proof)
    validate_statistics(child, child_proof)
    relationship = validate_parent(child, child_proof)
    assert relationship["run_budget_completed"] == 40000
    assert relationship["selected_checkpoint_updates"] == 35001
    with pytest.raises(ValueError, match="own single-task"):
        validate_parent(dataclasses.replace(child, asset_id=parent_request.asset_id), child_proof)
    child_norm_receipt = Path(child.norm_directory) / child.asset_id / "training_norm_provenance.json"
    parent_norm_receipt = (
        Path(parent_request.norm_directory) / parent_request.asset_id / "training_norm_provenance.json"
    )
    child_norm_receipt.write_bytes(parent_norm_receipt.read_bytes())
    with pytest.raises(ValueError, match="provenance"):
        validate_statistics(child, child_proof)
