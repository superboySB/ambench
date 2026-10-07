# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Build tiny random ACT/DP checkpoints and exercise real policy inference.

Run with the matching Python environment in the policy Docker image. The
generated weights are random and only verify loading, preprocessing, model
forward, decoding, and checkpoint compatibility; they are not task policies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def smoke_act(output_dir: Path, image_size: int) -> dict:
    import lerobot.policies.act.processor_act as lerobot_act_processor
    import numpy as np
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    from ambench_learn.policies.act.se_relative_processor import (
        make_act_pre_post_processors,
    )
    from ambench_learn.policies.remote.server import ACTBackend

    torch.manual_seed(0)
    lerobot_act_processor.make_act_pre_post_processors = make_act_pre_post_processors
    checkpoint = output_dir / "pretrained_model"
    checkpoint.mkdir(parents=True, exist_ok=True)
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
            "observation.images.ee_camera": PolicyFeature(FeatureType.VISUAL, (3, image_size, image_size)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
        device="cpu",
        push_to_hub=False,
        chunk_size=4,
        n_action_steps=2,
        pretrained_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        use_vae=False,
    )
    config.action_representation = "ee_local_relative"
    ACTPolicy(config).save_pretrained(checkpoint)
    stats = {
        "observation.state": {"mean": torch.zeros(8), "std": torch.ones(8)},
        "observation.images.ee_camera": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
        "action": {"mean": torch.zeros(8), "std": torch.ones(8)},
    }
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(checkpoint)
    postprocessor.save_pretrained(checkpoint)

    backend = ACTBackend(checkpoint, "cpu", None, None)
    reset = backend.reset(action_semantics="ee_absolute")
    observation = {
        "observation.state": np.array([[0, 0, 0, 1, 0, 0, 0, 0.5]], dtype=np.float32),
        "observation.images.ee_camera": np.zeros((1, 3, image_size, image_size), dtype=np.float32),
    }
    actions = [backend.infer(observation=observation)["action"] for _ in range(2)]
    if any(action.shape != (1, 8) or not np.isfinite(action).all() for action in actions):
        raise RuntimeError("ACT synthetic model did not return two finite 8D actions.")
    return {
        "policy": "act",
        "checkpoint": str(checkpoint),
        "info": backend.info(),
        "reset": reset,
        "action_shapes": [list(action.shape) for action in actions],
        "model_forward": "passed",
    }


def smoke_dp(output_dir: Path) -> dict:
    import diffusion_policy
    import hydra
    import numpy as np
    from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
    from omegaconf import OmegaConf

    from ambench_learn.policies.remote.server import DPBackend

    config_dir = Path(diffusion_policy.__file__).resolve().parent / "config"
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = hydra.compose(
            config_name="train_diffusion_unet_timm_umi_workspace",
            overrides=[
                "task=umi_drone_ee_pos",
                "task.obs_down_sample_steps=2",
                "task.action_horizon=4",
                "task.shape_meta.obs.camera0_rgb.shape=[3,64,64]",
                "policy.obs_encoder.model_name=resnet18",
                "policy.obs_encoder.pretrained=false",
                "policy.obs_encoder.feature_aggregation=avg",
                "policy.obs_encoder.transforms=null",
                "policy.num_inference_steps=2",
                "policy.noise_scheduler.num_train_timesteps=4",
                "policy.down_dims=[32,64]",
                "policy.diffusion_step_embed_dim=32",
                "training.use_ema=false",
            ],
        )
    workspace = hydra.utils.get_class(config._target_)(config, output_dir=str(output_dir))
    for key, attr in config.task.shape_meta.obs.items():
        if not attr.get("ignore_by_policy", False):
            workspace.model.normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    workspace.model.normalizer["action"] = SingleFieldLinearNormalizer.create_identity()
    checkpoint = output_dir / "latest.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    workspace.save_checkpoint(checkpoint, use_thread=False)

    backend = DPBackend(checkpoint, "cpu")
    observation = {
        "ee_pos": np.zeros(3, dtype=np.float32),
        "ee_quat": np.array([1, 0, 0, 0], dtype=np.float32),
        "gripper_width": np.array([0.5], dtype=np.float32),
        "arm_joint_pos": np.zeros(4, dtype=np.float32),
        "base_pos": np.zeros(3, dtype=np.float32),
        "base_quat": np.array([1, 0, 0, 0], dtype=np.float32),
    }
    reset = backend.reset(observation=observation)
    images = {"ee_camera": np.zeros((64, 64, 3), dtype=np.uint8)}
    actions = backend.infer(timestep=0, observation=observation, images=images)["actions"]
    if actions.shape != (8, 8) or not np.isfinite(actions).all():
        raise RuntimeError("DP synthetic model did not return a finite (8, 8) action plan.")
    return {
        "policy": "dp",
        "checkpoint": str(checkpoint),
        "info": backend.info(),
        "reset": reset,
        "action_shape": list(actions.shape),
        "model_forward": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=("act", "dp"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--act-image-size", type=int, default=64, help="Use 384 to match the AM-Bench EE camera.")
    args = parser.parse_args()
    if args.act_image_size < 32:
        parser.error("--act-image-size must be at least 32.")
    output_dir = args.output_dir.expanduser().resolve()
    report = smoke_act(output_dir, args.act_image_size) if args.policy == "act" else smoke_dp(output_dir)
    report_path = output_dir / "synthetic_inference_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
