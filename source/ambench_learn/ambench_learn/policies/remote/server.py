# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Serve ACT or Diffusion Policy inference to an Isaac Sim container.

Run ``python -m ambench_learn.policies.remote.server --policy act|dp
--checkpoint PATH --host 0.0.0.0 --port 8001 --device cuda:0``.
Only one evaluation client should use a server instance at a time because
both ACT's action queue and DP's observation history are episode state.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ambench_learn.policies.remote.protocol import serve


class ACTBackend:
    """LeRobot ACT model, processors, and queued actions on the policy GPU."""

    policy_name = "act"

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        n_action_steps: int | None,
        temporal_ensemble_coeff: float | None,
    ) -> None:
        import lerobot.policies.act.processor_act as lerobot_act_processor
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.processor.device_processor import DeviceProcessorStep

        from ambench_learn.policies.act.se_relative_processor import (
            make_act_pre_post_processors,
            reconnect_se_relative_absolute_steps,
        )

        pretrained_dir = checkpoint if checkpoint.name == "pretrained_model" else checkpoint / "pretrained_model"
        if not pretrained_dir.is_dir():
            raise FileNotFoundError(f"LeRobot pretrained_model directory not found: {pretrained_dir}")
        lerobot_act_processor.make_act_pre_post_processors = make_act_pre_post_processors
        cli_overrides = [f"--device={device}"]
        if n_action_steps is not None:
            cli_overrides.append(f"--n_action_steps={n_action_steps}")
        if temporal_ensemble_coeff is not None:
            cli_overrides.append(f"--temporal_ensemble_coeff={temporal_ensemble_coeff}")
        self.config = PreTrainedConfig.from_pretrained(pretrained_dir, cli_overrides=cli_overrides)
        policy_cls = get_policy_class(self.config.type)
        self.policy = policy_cls.from_pretrained(pretrained_dir, config=self.config)
        self.preprocessor, self.postprocessor = make_pre_post_processors(self.config, pretrained_path=pretrained_dir)
        for step in self.preprocessor.steps:
            if isinstance(step, DeviceProcessorStep):
                step.device = device
                step.__post_init__()
            elif hasattr(step, "to"):
                step.to(device=device)
        reconnect_se_relative_absolute_steps(self.preprocessor, self.postprocessor)
        self.policy.to(device)
        self.policy.eval()
        self._representation: str | None = None
        self._ready = False
        self._torch = torch

    def info(self) -> dict[str, Any]:
        image_features = {
            key: list(feature.shape) if feature.shape is not None else None
            for key, feature in self.config.image_features.items()
        }
        return {
            "policy": self.policy_name,
            "image_features": image_features,
            "action_representation": getattr(self.config, "action_representation", None),
            "n_action_steps": getattr(self.config, "n_action_steps", None),
            "temporal_ensemble_coeff": getattr(self.config, "temporal_ensemble_coeff", None),
        }

    def reset(self, *, action_semantics: str) -> dict[str, Any]:
        from ambench_learn.policies.act.eval_utils import (
            resolve_act_action_representation,
        )
        from ambench_learn.policies.act.se_relative_processor import (
            BaseJointRelativeTrajectoryProcessorStep,
            EELocalRelativeTrajectoryProcessorStep,
        )

        representation = resolve_act_action_representation(action_semantics, self.config, self.postprocessor)
        if getattr(self.config, "temporal_ensemble_coeff", None) is not None and representation.endswith("_relative"):
            raise ValueError("ACT temporal ensembling is unsupported for relative actions.")
        self.policy.reset()
        for step in self.preprocessor.steps:
            if isinstance(step, (EELocalRelativeTrajectoryProcessorStep, BaseJointRelativeTrajectoryProcessorStep)):
                step.clear_decode_anchor()
        self._representation = representation
        self._ready = True
        state_keys = (
            ["base_pos", "base_quat", "arm_joint_pos", "gripper_width"]
            if representation == "base_joint_relative"
            else ["ee_pos", "ee_quat", "gripper_width"]
        )
        return {"action_representation": representation, "state_keys": state_keys}

    def infer(self, *, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        from ambench_learn.policies.act.se_relative_processor import (
            BaseJointRelativeTrajectoryProcessorStep,
            EELocalRelativeTrajectoryProcessorStep,
        )

        if not self._ready or self._representation is None:
            raise RuntimeError("Call /reset before ACT inference.")
        expected_keys = {"observation.state", *self.config.image_features.keys()}
        if set(observation) != expected_keys:
            raise ValueError(f"ACT observation keys {sorted(observation)} differ from {sorted(expected_keys)}.")
        policy_obs = {key: self._torch.from_numpy(np.asarray(value).copy()) for key, value in observation.items()}
        with self._torch.no_grad():
            processed = self.preprocessor(policy_obs)
            if (
                self._representation.endswith("_relative")
                and hasattr(self.policy, "_action_queue")
                and len(self.policy._action_queue) == 0
            ):
                step_type = (
                    BaseJointRelativeTrajectoryProcessorStep
                    if self._representation == "base_joint_relative"
                    else EELocalRelativeTrajectoryProcessorStep
                )
                relative_step = next((step for step in self.preprocessor.steps if isinstance(step, step_type)), None)
                if relative_step is None or relative_step._last_state is None:
                    raise RuntimeError("ACT relative decoder has no measured chunk-start state.")
                relative_step.cache_decode_anchor()
            action = self.postprocessor(self.policy.select_action(processed))
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return {"action": action.detach().cpu().to(self._torch.float32).numpy()}


class DPBackend:
    """UMI Diffusion Policy model and observation history on the policy GPU."""

    policy_name = "dp"

    def __init__(self, checkpoint: Path, device: str) -> None:
        import dill
        import hydra
        import torch
        from diffusion_policy.common.pytorch_util import dict_apply
        from omegaconf import OmegaConf
        from umi.real_world.real_inference_util import (
            get_real_umi_action,
            get_real_umi_obs_dict,
        )

        from ambench_learn.policies.dp.eval_utils import (
            ObservationBufferManager,
            get_real_base_joint_action,
            get_real_base_joint_obs_dict,
            interpolate_action_sequence,
            interpolate_base_joint_action_sequence,
            prepare_umi_image_observation,
            quat_wxyz_to_axis_angle,
            resolve_dp_execution_schedule,
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(f"Diffusion Policy checkpoint not found: {checkpoint}")
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        with checkpoint.open("rb") as file:
            payload = torch.load(file, map_location="cpu", pickle_module=dill)
        self.cfg = payload["cfg"]
        workspace_cls = hydra.utils.get_class(self.cfg._target_)
        workspace = workspace_cls(self.cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self.policy = workspace.ema_model if self.cfg.training.use_ema else workspace.model
        self.policy.action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        self.policy.to(device)
        self.policy.eval()
        self.device = device
        self.shape_meta = self.cfg.task.shape_meta
        self.action_mode = self.shape_meta.action.get("mode", "ee_pose")
        self.camera_key_mapping = {}
        if "camera0_rgb" in self.shape_meta.obs:
            self.camera_key_mapping["ee_camera"] = "camera0_rgb"
        if "camera1_rgb" in self.shape_meta.obs:
            self.camera_key_mapping["base_camera"] = "camera1_rgb"
        if not self.camera_key_mapping:
            raise ValueError("DP checkpoint must use camera0_rgb or camera1_rgb.")
        self.obs_down_sample_steps = int(self.cfg.task.obs_down_sample_steps)
        _, self.query_frequency, self.prediction_horizon_steps = resolve_dp_execution_schedule(
            int(self.shape_meta.action.horizon), self.obs_down_sample_steps
        )
        self._torch = torch
        self._dict_apply = dict_apply
        self._get_real_umi_action = get_real_umi_action
        self._get_real_umi_obs_dict = get_real_umi_obs_dict
        self._ObservationBufferManager = ObservationBufferManager
        self._get_real_base_joint_action = get_real_base_joint_action
        self._get_real_base_joint_obs_dict = get_real_base_joint_obs_dict
        self._interpolate_action_sequence = interpolate_action_sequence
        self._interpolate_base_joint_action_sequence = interpolate_base_joint_action_sequence
        self._prepare_umi_image_observation = prepare_umi_image_observation
        self._quat_wxyz_to_axis_angle = quat_wxyz_to_axis_angle
        self._ready = False

    def info(self) -> dict[str, Any]:
        camera_shapes = {
            camera_name: list(self.shape_meta.obs[key].shape) for camera_name, key in self.camera_key_mapping.items()
        }
        return {
            "policy": self.policy_name,
            "action_mode": self.action_mode,
            "action_pose_repr": self.cfg.task.pose_repr.action_pose_repr,
            "hydra_workspace": self.cfg._target_,
            "camera_shapes": camera_shapes,
            "obs_down_sample_steps": self.obs_down_sample_steps,
            "query_frequency": self.query_frequency,
            "prediction_horizon_steps": self.prediction_horizon_steps,
        }

    def reset(self, *, observation: dict[str, np.ndarray]) -> dict[str, Any]:
        ee_pos = np.asarray(observation.get("ee_pos", np.zeros(3)), dtype=np.float32)
        ee_quat = np.asarray(observation.get("ee_quat", [1, 0, 0, 0]), dtype=np.float32)
        base_pos = np.asarray(observation.get("base_pos", np.zeros(3)), dtype=np.float32)
        base_quat = np.asarray(observation.get("base_quat", [1, 0, 0, 0]), dtype=np.float32)
        self.episode_start_pose = [np.concatenate([ee_pos, self._quat_wxyz_to_axis_angle(ee_quat)])]
        self.episode_base_start_pose = [np.concatenate([base_pos, self._quat_wxyz_to_axis_angle(base_quat)])]
        first_camera_key = next(iter(self.camera_key_mapping.values()))
        img_horizon = int(self.shape_meta.obs[first_camera_key].horizon)
        low_dim_horizon = next(
            int(attr.horizon) for attr in self.shape_meta.obs.values() if attr.get("type", "low_dim") == "low_dim"
        )
        self.buffer = self._ObservationBufferManager(
            img_horizon, low_dim_horizon, image_keys=list(self.camera_key_mapping.values())
        )
        self.policy.reset()
        self._ready = True
        return {"status": "reset"}

    def infer(
        self, *, timestep: int, observation: dict[str, np.ndarray], images: dict[str, np.ndarray]
    ) -> dict[str, Any]:
        if not self._ready:
            raise RuntimeError("Call /reset before DP inference.")
        if timestep % self.obs_down_sample_steps != 0:
            raise ValueError("DP observation timestep is not on the configured sample grid.")
        current_images = {}
        for camera_name, obs_key in self.camera_key_mapping.items():
            if camera_name not in images:
                raise KeyError(f"Missing DP camera {camera_name}.")
            shape = self.shape_meta.obs[obs_key].shape
            current_images[obs_key] = self._prepare_umi_image_observation(
                images[camera_name], target_size=(int(shape[1]), int(shape[2]))
            )

        ee_pos = np.asarray(observation.get("ee_pos", np.zeros(3)), dtype=np.float32)
        ee_quat = np.asarray(observation.get("ee_quat", [1, 0, 0, 0]), dtype=np.float32)
        gripper = np.asarray(observation["gripper_width"], dtype=np.float32)
        joints = np.asarray(observation.get("arm_joint_pos", np.zeros(4)), dtype=np.float32)
        base_pos = np.asarray(observation.get("base_pos", np.zeros(3)), dtype=np.float32)
        base_quat = np.asarray(observation.get("base_quat", [1, 0, 0, 0]), dtype=np.float32)
        self.buffer.add_observation(
            current_images,
            ee_pos,
            self._quat_wxyz_to_axis_angle(ee_quat),
            gripper,
            robot_joint_pos=joints if "robot0_joint_pos" in self.shape_meta.obs else None,
            robot_base_pos=base_pos if "robot0_base_pos" in self.shape_meta.obs else None,
            robot_base_rot=(
                self._quat_wxyz_to_axis_angle(base_quat)
                if "robot0_base_rot_axis_angle" in self.shape_meta.obs
                else None
            ),
        )
        stacked = self.buffer.get_stacked_observation()
        if timestep % self.query_frequency != 0:
            return {"actions": None}
        if stacked is None:
            raise RuntimeError("DP observation history is empty at the query step.")
        filtered = {key: value for key, value in stacked.items() if key in self.shape_meta.obs}
        if self.action_mode == "base_joint":
            obs_np = self._get_real_base_joint_obs_dict(
                env_obs=filtered,
                shape_meta=self.shape_meta,
                obs_pose_repr=self.cfg.task.pose_repr.obs_pose_repr,
            )
        else:
            base_start = self.episode_base_start_pose
            if "robot0_base_pos" not in filtered or "robot0_base_rot_axis_angle" not in filtered:
                base_start = None
            obs_np = self._get_real_umi_obs_dict(
                env_obs=filtered,
                shape_meta=self.shape_meta,
                obs_pose_repr=self.cfg.task.pose_repr.obs_pose_repr,
                episode_start_pose=self.episode_start_pose,
                episode_base_start_pose=base_start,
            )
        obs_torch = self._dict_apply(
            obs_np, lambda value: self._torch.from_numpy(value).unsqueeze(0).to(device=self.device)
        )
        for key, attr in self.shape_meta.obs.items():
            if attr.get("ignore_by_policy", False):
                obs_torch.pop(key, None)
        with self._torch.no_grad():
            raw_action = self.policy.predict_action(obs_torch)["action_pred"][0].detach().cpu().numpy()
            if self.action_mode == "base_joint":
                decoded = self._get_real_base_joint_action(
                    raw_action, stacked, self.cfg.task.pose_repr.action_pose_repr
                )
            else:
                decoded = self._get_real_umi_action(raw_action, stacked, self.cfg.task.pose_repr.action_pose_repr)
        if self.action_mode == "base_joint":
            actions = self._interpolate_base_joint_action_sequence(decoded, self.prediction_horizon_steps)
        else:
            actions = self._interpolate_action_sequence(decoded, self.prediction_horizon_steps, output_type="quat")
        return {"actions": np.asarray(actions, dtype=np.float32)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, choices=("act", "dp"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-action-steps", type=int, default=None, help="ACT inference override.")
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None, help="ACT inference override.")
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if args.policy == "act":
        backend = ACTBackend(checkpoint, args.device, args.n_action_steps, args.temporal_ensemble_coeff)
    else:
        backend = DPBackend(checkpoint, args.device)
    serve(backend, args.host, args.port)


if __name__ == "__main__":
    main()
