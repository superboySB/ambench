# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise actual training termination, stochastic resume, and accumulation."""

from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path

import dill
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

UMI_ROOT = (
    Path(__file__).resolve().parents[3] / "ambench_learn" / "policies" / "dp" / "universal_manipulation_interface"
)
sys.path.insert(0, str(UMI_ROOT))

from diffusion_policy.dataset.base_dataset import BaseDataset  # noqa: E402
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner  # noqa: E402
from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (  # noqa: E402
    TrainDiffusionUnetImageWorkspace,
)


class SmallDataset(BaseDataset):
    """Use disjoint episodes and stochastic samples, without simulation assets."""

    def __init__(self, stochastic: bool = True, alternate_split: bool = False):
        self.stochastic = stochastic
        self.train_episode_indices = (0, 1, 2) if not alternate_split else (0, 1, 3)
        self.val_episode_indices = (3, 4) if not alternate_split else (2, 4)
        self.indices = list(range(13))

    def get_validation_dataset(self) -> SmallDataset:
        validation = copy.copy(self)
        validation.indices = list(range(100, 105))
        return validation

    def get_normalizer(self) -> torch.Tensor:
        return torch.tensor(float(np.mean(self.indices)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        value = self.indices[index] / 100
        if self.stochastic:
            value += (random.random() + np.random.random() + torch.rand(()).item()) / 100
        state = torch.tensor([value, value + 0.1], dtype=torch.float32)
        action = torch.stack([torch.linspace(value, value + 0.2, 10)] * 2)
        return {"obs": {"state": state}, "action": action}


class SmallPolicy(torch.nn.Module):
    """Use dropout and diffusion-like noise so RNG restoration is observable."""

    validation_calls = 0
    validation_grad_enabled = []

    def __init__(self, obs_encoder: dict, stochastic: bool = True):
        super().__init__()
        self.obs_encoder = torch.nn.Linear(2, 4)
        self.model = torch.nn.Sequential(torch.nn.Dropout(0.2 if stochastic else 0.0), torch.nn.Linear(4, 10))
        self.register_buffer("normalizer", torch.zeros(()))
        self.stochastic = stochastic

    def set_normalizer(self, normalizer: torch.Tensor) -> None:
        self.normalizer.copy_(normalizer)

    def predict_action(self, obs: dict, prefix=None) -> dict:
        action = self.model(self.obs_encoder(obs["state"]))
        return {"action_pred": action[:, None].expand(-1, 2, -1)}

    def forward(self, batch: dict) -> torch.Tensor:
        if not self.training:
            type(self).validation_calls += 1
            type(self).validation_grad_enabled.append(torch.is_grad_enabled())
        prediction = self.predict_action(batch["obs"])["action_pred"]
        if self.stochastic:
            prediction = prediction + torch.randn_like(prediction) * 0.01
        return torch.nn.functional.mse_loss(prediction, batch["action"])


class NoopRunner(BaseImageRunner):
    def run(self, policy) -> dict:
        # A real evaluation runner may consume all three random streams.
        random.random()
        np.random.random()
        torch.rand(())
        return {}


def training_config(*, workers: int = 0, stochastic: bool = True, epochs: int = 4):
    module = __name__
    return OmegaConf.create({
        "_target_": "diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace",
        "policy": {
            "_target_": f"{module}.SmallPolicy",
            "obs_encoder": {"pretrained": False},
            "stochastic": stochastic,
        },
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "lr": 0.002,
            "betas": [0.9, 0.99],
            "weight_decay": 0.0001,
        },
        "ema": {
            "_target_": "diffusion_policy.model.diffusion.ema_model.EMAModel",
            "update_after_step": 0,
            "inv_gamma": 1.0,
            "power": 0.75,
            "max_value": 0.9999,
        },
        "task": {
            "dataset": {"_target_": f"{module}.SmallDataset", "stochastic": stochastic},
            "env_runner": {"_target_": f"{module}.NoopRunner"},
        },
        "dataloader": {
            "batch_size": 4,
            "num_workers": workers,
            "shuffle": stochastic,
            "persistent_workers": workers > 0,
        },
        "val_dataloader": {
            "batch_size": 2,
            "num_workers": workers,
            "shuffle": True,
            "persistent_workers": workers > 0,
        },
        "training": {
            "device": "cpu",
            "seed": 17,
            "debug": False,
            "resume": False,
            "use_ema": True,
            "freeze_encoder": False,
            "num_epochs": epochs,
            "gradient_accumulate_every": 3,
            "lr_scheduler": "cosine",
            "lr_warmup_steps": 1,
            "max_train_steps": None,
            "max_val_steps": None,
            "checkpoint_every": 10,
            "val_every": 1,
            "rollout_every": 1,
            "sample_every": 1,
            "tqdm_interval_sec": 100,
        },
        "logging": {"mode": "disabled", "project": "unit-tests"},
        "checkpoint": {
            "topk": {"monitor_key": "val_loss", "mode": "min", "k": 2, "format_str": "epoch={epoch}.ckpt"},
            "save_last_ckpt": True,
            "save_last_snapshot": False,
        },
    })


def read_payload(directory: Path) -> dict:
    with (directory / "checkpoints/latest.ckpt").open("rb") as file:
        return torch.load(file, map_location="cpu", pickle_module=dill)


def assert_nested_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("workers", [0, 2])
def test_stochastic_resume_matches_uninterrupted_training(tmp_path: Path, workers: int) -> None:
    config = training_config(workers=workers)
    full = tmp_path / "full"
    TrainDiffusionUnetImageWorkspace(config, output_dir=str(full)).run()
    full_payload = read_payload(full)

    interrupted = tmp_path / "interrupted"
    partial_config = copy.deepcopy(config)
    partial_config.training.checkpoint_every = 1
    workspace = TrainDiffusionUnetImageWorkspace(partial_config, output_dir=str(interrupted))
    original_save = workspace.save_checkpoint

    class PlannedInterruption(Exception):
        pass

    def save_then_interrupt(*args, **kwargs):
        result = original_save(*args, **kwargs)
        if workspace.epoch == 2 and not args and "path" not in kwargs:
            raise PlannedInterruption
        return result

    workspace.save_checkpoint = save_then_interrupt
    with pytest.raises(PlannedInterruption):
        workspace.run()
    partial_payload = read_payload(interrupted)
    assert dill.loads(partial_payload["pickles"]["epoch"]) == 2
    assert dill.loads(partial_payload["pickles"]["optimizer_step"]) == 4
    assert "optimizer" in partial_payload["state_dicts"]

    resume_config = copy.deepcopy(config)
    resume_config.training.resume = True
    resumed = TrainDiffusionUnetImageWorkspace(resume_config, output_dir=str(interrupted))
    resumed.run()
    resumed_payload = read_payload(interrupted)
    assert_nested_equal(full_payload["state_dicts"], resumed_payload["state_dicts"])
    for key in (
        "epoch",
        "global_step",
        "optimizer_step",
        "lr_scheduler_state",
        "ema_state",
        "rng_states",
        "dataset_split",
    ):
        assert_nested_equal(dill.loads(full_payload["pickles"][key]), dill.loads(resumed_payload["pickles"][key]))
    assert resumed.epoch == 4
    assert resumed.global_step == 16
    assert resumed.optimizer_step == 8
    assert resumed.ema_state["optimization_step"] == 8
    assert not list(interrupted.rglob("*.tmp"))


def test_final_epoch_full_validation_and_inference_reload(tmp_path: Path) -> None:
    SmallPolicy.validation_calls = 0
    SmallPolicy.validation_grad_enabled = []
    cfg = training_config(epochs=3)
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(tmp_path))
    workspace.run()
    payload = read_payload(tmp_path)
    assert dill.loads(payload["pickles"]["epoch"]) == 3
    assert dill.loads(payload["pickles"]["global_step"]) == 12
    assert dill.loads(payload["pickles"]["optimizer_step"]) == 6
    assert "optimizer" in payload["state_dicts"]
    logs = [json.loads(line) for line in (tmp_path / "logs.json.txt").read_text().splitlines()]
    assert len(logs) == 3
    assert all(log["val_samples"] == 5 and log["val_batches"] == 3 and log["val_complete"] for log in logs)
    assert SmallPolicy.validation_calls == 9
    assert not any(SmallPolicy.validation_grad_enabled)
    assert workspace.model.normalizer.item() == 6.0
    assert workspace.dataset_split["train_episode_indices"] == [0, 1, 2]
    assert workspace.dataset_split["val_episode_indices"] == [3, 4]
    # This is the same constructor + load_payload contract used by DP serving.
    loaded = TrainDiffusionUnetImageWorkspace(payload["cfg"])
    loaded.load_payload(payload)
    assert_nested_equal(workspace.model.state_dict(), loaded.model.state_dict())
    assert_nested_equal(workspace.ema_model.state_dict(), loaded.ema_model.state_dict())


def test_accumulation_matches_manual_example_weighted_adamw_updates(tmp_path: Path) -> None:
    cfg = training_config(stochastic=False, epochs=1)
    cfg.training.lr_scheduler = "constant"
    cfg.training.use_ema = False
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(tmp_path))
    reference = copy.deepcopy(workspace.model)
    reference.set_normalizer(torch.tensor(6.0))
    reference_optimizer = torch.optim.AdamW(
        [{"params": reference.model.parameters()}, {"params": reference.obs_encoder.parameters(), "lr": 0.002}],
        lr=0.002,
        betas=(0.9, 0.99),
        weight_decay=0.0001,
    )
    dataset = SmallDataset(stochastic=False)
    # The first accumulated window contains 12 examples; the last contains one.
    for indices in (range(12), range(12, 13)):
        samples = [dataset[index] for index in indices]
        batch = {
            "obs": {"state": torch.stack([sample["obs"]["state"] for sample in samples])},
            "action": torch.stack([sample["action"] for sample in samples]),
        }
        reference_optimizer.zero_grad(set_to_none=True)
        reference(batch).backward()
        reference_optimizer.step()
    workspace.run()
    for actual, expected in zip(workspace.model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-8)
    assert workspace.optimizer_step == 2
    assert workspace.global_step == 4


def test_resume_rejects_changed_episode_split_and_legacy_checkpoint(tmp_path: Path) -> None:
    cfg = training_config(epochs=1)
    TrainDiffusionUnetImageWorkspace(cfg, output_dir=str(tmp_path)).run()
    resume = copy.deepcopy(cfg)
    resume.training.resume = True
    resume.task.dataset.alternate_split = True
    with pytest.raises(ValueError, match="dataset split"):
        TrainDiffusionUnetImageWorkspace(resume, output_dir=str(tmp_path)).run()
    resume.task.dataset.alternate_split = False
    payload = read_payload(tmp_path)
    del payload["state_dicts"]["optimizer"]
    with (tmp_path / "checkpoints/latest.ckpt").open("wb") as file:
        torch.save(payload, file, pickle_module=dill)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        TrainDiffusionUnetImageWorkspace(resume, output_dir=str(tmp_path)).run()
