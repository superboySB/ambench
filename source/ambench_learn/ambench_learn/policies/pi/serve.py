# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve JAX weights with the exact data factory and statistics used in training."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openpi.training.config import TrainConfig

from ambench_learn.policies.pi.train import (
    TrainingRequest,
    build_config,
    config_snapshot,
    file_sha256,
    read_checkpoint_step,
    verify_checkpoint,
)


def load_serving_config(manifest_path: Path, checkpoint: Path | None = None) -> tuple[TrainConfig, Path]:
    """Reject incomplete training, changed weights, and another repository's statistics."""
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed" or manifest.get("trained") is not True:
        raise ValueError("Serving requires a verified complete training manifest.")
    request = TrainingRequest(**manifest["request"])
    config = build_config(request)
    contract = manifest["contract"]
    if (
        contract["request"] != manifest["request"]
        or contract["dataset_proof"] != manifest["dataset_proof"]
        or contract["normalization_sha256"] != manifest["normalization_sha256"]
        or contract["config"] != config_snapshot(config)
    ):
        raise ValueError("Serving repository, asset ID, or resolved training configuration differs from its contract.")
    for relative, expected in contract["openpi_source_sha256"].items():
        if file_sha256(Path(request.openpi_root) / relative) != expected:
            raise ValueError("Pinned OpenPI source differs from the training implementation.")
    for name, expected in contract["wrapper_source_sha256"].items():
        if file_sha256(Path(__file__).parent / name) != expected:
            raise ValueError("Training/serving wrapper differs from the recorded implementation.")
    final_path = Path(manifest["final_checkpoint"]["path"])
    if (
        manifest["run_budget_completed"] != config.num_train_steps
        or read_checkpoint_step(final_path) != config.num_train_steps
    ):
        raise ValueError("The run's final stored state did not complete its declared training budget.")
    selected = final_path if checkpoint is None else checkpoint
    expected = manifest["candidate_checkpoints"].get(str(selected.resolve()))
    if expected is None:
        raise ValueError("Checkpoint path is not a registered candidate from the completed training run.")
    proof = verify_checkpoint(selected, request, manifest["normalization_sha256"])
    if proof != expected or not 0 < proof["optimizer_updates"] <= config.num_train_steps:
        raise ValueError("Selected checkpoint updates or inference parameter hashes changed.")
    # Reconstruct the original asset ID but read its immutable checkpoint copy.
    data = dataclasses.replace(
        config.data,
        assets=dataclasses.replace(config.data.assets, assets_dir=str(selected / "assets")),
    )
    metadata = {
        **config.policy_metadata,
        "variant": request.variant,
        "stage": request.stage,
        "tuning_mode": request.mode,
        "optimizer_updates": proof["optimizer_updates"],
        "run_budget_completed": manifest["run_budget_completed"],
        "selected_checkpoint_updates": proof["optimizer_updates"],
        "normalization_sha256": manifest["normalization_sha256"],
        "robot_type": manifest["dataset_proof"]["robot_type"],
        "task_counts": manifest["dataset_proof"]["task_counts"],
        "paper_dataset_scope_matched": manifest["dataset_proof"]["paper_dataset_scope_matched"],
    }
    return dataclasses.replace(config, data=data, policy_metadata=metadata), selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    config, checkpoint = load_serving_config(args.manifest, args.checkpoint)
    if args.validate_only:
        print(json.dumps({"status": "serving_identity_verified", "metadata": config.policy_metadata}, indent=2))
        return
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    policy = create_trained_policy(config, checkpoint, default_prompt=args.default_prompt)
    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    main()
