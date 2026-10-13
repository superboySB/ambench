# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Score deployed 8D EE ACT chunks on whole held-out demonstration episodes.

This is offline validation, not a simulation success rate. The VAE receives no
target actions during prediction, and normalization comes from the checkpoint.
Run a separate, previously unused set of environment seeds for the final test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ambench_learn.data.action_resampling import resolve_dataset_action_semantics
from ambench_learn.data.resampled_lerobot_dataset import ResampledLeRobotDataset
from ambench_learn.data.selected_lerobot_dataset import SelectedLeRobotDataset
from ambench_learn.policies.remote.server import ACTBackend


def file_sha256(path: Path) -> str:
    """Hash a checkpoint file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Evaluate every valid held-out action target using saved training stats."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--training-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--validation-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--policy-target-hz", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train_ids = set(args.training_episodes)
    validation_ids = set(args.validation_episodes)
    if len(train_ids) != len(args.training_episodes) or len(validation_ids) != len(args.validation_episodes):
        parser.error("Episode selections must not contain duplicates.")
    if train_ids & validation_ids:
        parser.error("Training and validation episode selections overlap.")
    if min(train_ids | validation_ids) < 0 or args.batch_size < 1:
        parser.error("Episode indices must be nonnegative and batch size positive.")
    if args.output.exists():
        parser.error("Refusing to overwrite an existing validation result.")

    started = time.monotonic()
    torch.manual_seed(42)
    raw_dataset = SelectedLeRobotDataset(args.repo_id, root=args.dataset_root, episodes=args.validation_episodes)
    backend = ACTBackend(args.checkpoint, args.device, None, None)
    training_report_path = backend.checkpoint.parents[2] / "benchmark_dataset_report.json"
    training_report = json.loads(training_report_path.read_text())
    if set(training_report["source_episode_indices"]) != train_ids:
        raise ValueError("Declared training episodes differ from the checkpoint's actual training dataset report.")
    if Path(training_report["dataset_root"]).resolve() != args.dataset_root.resolve():
        raise ValueError("Validation must use the same canonical dataset root as the saved training report.")
    reset = backend.reset(action_semantics=resolve_dataset_action_semantics(raw_dataset))
    dataset = ResampledLeRobotDataset(
        raw_dataset,
        target_fps=args.policy_target_hz,
        action_window_length=backend.config.chunk_size,
        policy_action_representation=reset["action_representation"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    absolute_errors = torch.zeros(8, dtype=torch.float64)
    valid_targets = 0
    observations_scored = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = backend.preprocessor(raw_batch)
            # Prediction sees observations only: the supervised target is never
            # supplied to the ACT VAE posterior during this deployed-mode score.
            observation = {key: batch[key] for key in backend.config.input_features}
            predicted = backend.policy.predict_action_chunk(observation)
            target = batch["action"]
            valid = ~batch["action_is_pad"].to(device=target.device)
            if predicted.shape != target.shape or predicted.shape[-1] != 8:
                raise ValueError(f"Expected equal BxHx8 chunks, got {predicted.shape} and {target.shape}.")
            if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
                raise ValueError("Nonfinite prediction or held-out target.")
            error = (predicted - target).abs() * valid.unsqueeze(-1)
            absolute_errors += error.sum(dim=(0, 1)).to(device="cpu", dtype=torch.float64)
            valid_targets += int(valid.sum().item())
            observations_scored += int(predicted.shape[0])
    if observations_scored != len(dataset) or valid_targets < 1:
        raise ValueError("The complete held-out dataset was not scored.")
    per_dimension = (absolute_errors / valid_targets).tolist()
    report = {
        "status": "passed",
        "evidence_level": "whole_episode_held_out_offline_action_validation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(backend.checkpoint),
        "checkpoint_files_sha256": {
            path.relative_to(backend.checkpoint).as_posix(): file_sha256(path)
            for path in sorted(backend.checkpoint.rglob("*"))
            if path.is_file()
        },
        "dataset_root": str(args.dataset_root),
        "dataset_info_sha256": file_sha256(args.dataset_root / "meta/info.json"),
        "training_episode_indices": args.training_episodes,
        "training_dataset_report_sha256": file_sha256(training_report_path),
        "validation_episode_indices": args.validation_episodes,
        "normalization_source": "saved checkpoint processors; never held-out dataset statistics",
        "vae_prediction_input": "observations only, zero prior latent in eval mode",
        "policy_action_representation": reset["action_representation"],
        "policy_target_hz": args.policy_target_hz,
        "logical_observations_scored": observations_scored,
        "valid_action_targets_scored": valid_targets,
        "mean_normalized_absolute_error": sum(per_dimension) / len(per_dimension),
        "normalized_absolute_error_per_dimension": per_dimension,
        "device": args.device,
        "duration_seconds": time.monotonic() - started,
        "claim_limit": "Offline action error is a checkpoint-selection signal, not closed-loop task success.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "checkpoint", "mean_normalized_absolute_error")}))


if __name__ == "__main__":
    main()
