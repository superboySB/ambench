"""Train image diffusion policies with complete, resumable epoch checkpoints."""

from __future__ import annotations

import copy
import math
import os
import pathlib
import pickle
import random

import hydra
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseDataset, BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

OmegaConf.register_new_resolver("eval", eval, replace=True)


def capture_rng_state() -> dict:
    """Capture the stochastic training streams at an epoch boundary."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict) -> None:
    """Restore the streams without advancing the next training sample."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def log_action_mse(step_log: dict, category: str, predicted: torch.Tensor, target: torch.Tensor) -> None:
    """Retain task-space diagnostics for EE and BaseJoint action layouts."""
    step_log[f"{category}_action_mse_error"] = torch.nn.functional.mse_loss(predicted, target).item()
    if predicted.shape[-1] % 10 == 0:
        predicted = predicted.reshape(*predicted.shape[:2], -1, 10)
        target = target.reshape(*target.shape[:2], -1, 10)
        slices = {"pos": slice(0, 3), "rot": slice(3, 9), "width": slice(9, 10)}
    elif predicted.shape[-1] == 14:
        slices = {"base_pos": slice(0, 3), "base_rot": slice(3, 9), "arm": slice(9, 13), "width": slice(13, 14)}
    else:
        raise ValueError(f"Unsupported action dimension for action MSE: {predicted.shape[-1]}")
    for name, selection in slices.items():
        step_log[f"{category}_action_mse_error_{name}"] = torch.nn.functional.mse_loss(
            predicted[..., selection], target[..., selection]
        ).item()


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = (
        "global_step",
        "optimizer_step",
        "epoch",
        "training_state_version",
        "lr_scheduler_state",
        "ema_state",
        "rng_states",
        "dataset_split",
        "topk_state",
    )
    exclude_keys = ()

    def __init__(self, cfg: OmegaConf, output_dir: str | None = None):
        super().__init__(cfg, output_dir=output_dir)
        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)
        random.seed(cfg.training.seed)
        self.model = hydra.utils.instantiate(cfg.policy)
        self.ema_model = copy.deepcopy(self.model) if cfg.training.use_ema else None

        encoder_lr = cfg.optimizer.lr * (0.1 if cfg.policy.obs_encoder.pretrained else 1.0)
        encoder_params = [param for param in self.model.obs_encoder.parameters() if param.requires_grad]
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop("_target_")
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.model.model.parameters()},
                {"params": encoder_params, "lr": encoder_lr},
            ],
            **optimizer_cfg,
        )
        # epoch counts completed epochs; optimizer_step counts actual updates.
        self.global_step = 0
        self.optimizer_step = 0
        self.epoch = 0
        self.training_state_version = 2
        self.lr_scheduler_state = None
        self.ema_state = None
        self.rng_states = None
        self.dataset_split = None
        self.topk_state = {}

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        exclude_keys=None,
        include_keys=None,
        use_thread=False,
    ) -> str:
        """Atomically write a complete checkpoint before training continues."""
        checkpoint_path = self.get_checkpoint_path(tag) if path is None else pathlib.Path(path)
        temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
        super().save_checkpoint(
            path=temporary_path,
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            use_thread=False,
        )
        os.replace(temporary_path, checkpoint_path)
        return str(checkpoint_path.absolute())

    def run(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
        accumulation = int(cfg.training.gradient_accumulate_every)
        if accumulation < 1:
            raise ValueError("gradient_accumulate_every must be positive.")
        # Restart workers at epoch boundaries so resume reproduces worker RNGs.
        cfg.dataloader.persistent_workers = False
        cfg.val_dataloader.persistent_workers = False
        cfg.val_dataloader.shuffle = False
        OmegaConf.update(cfg, "val_dataloader.drop_last", False, force_add=True)
        self.cfg = cfg
        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        logging_enabled = cfg.logging.mode != "disabled"
        accelerator = Accelerator(
            cpu=str(cfg.training.device) == "cpu",
            log_with="wandb" if logging_enabled else None,
        )
        if logging_enabled:
            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            project = wandb_cfg.pop("project")
            accelerator.init_trackers(
                project_name=project,
                config=OmegaConf.to_container(cfg, resolve=True),
                init_kwargs={"wandb": wandb_cfg},
            )

        dataset = hydra.utils.instantiate(cfg.task.dataset)
        if not isinstance(dataset, (BaseImageDataset, BaseDataset)):
            raise TypeError("Expected an image or UMI dataset.")
        val_dataset = dataset.get_validation_dataset()
        split = {"train_samples": len(dataset), "val_samples": len(val_dataset)}
        if hasattr(dataset, "train_episode_indices") and hasattr(dataset, "val_episode_indices"):
            split["train_episode_indices"] = list(dataset.train_episode_indices)
            split["val_episode_indices"] = list(dataset.val_episode_indices)
        elif hasattr(dataset, "val_mask"):
            split["train_episode_indices"] = np.flatnonzero(~dataset.val_mask).tolist()
            split["val_episode_indices"] = np.flatnonzero(dataset.val_mask).tolist()
        if set(split.get("train_episode_indices", [])) & set(split.get("val_episode_indices", [])):
            raise ValueError("Training and validation episodes overlap.")
        if not len(dataset):
            raise ValueError("The training dataset is empty.")

        train_generator = torch.Generator()
        val_generator = torch.Generator()
        train_dataloader = DataLoader(dataset, generator=train_generator, **cfg.dataloader)
        val_dataloader = DataLoader(val_dataset, generator=val_generator, **cfg.val_dataloader)
        normalizer_path = pathlib.Path(self.output_dir) / "normalizer.pkl"
        if accelerator.is_main_process and not (cfg.training.resume and normalizer_path.is_file()):
            with normalizer_path.open("wb") as file:
                pickle.dump(dataset.get_normalizer(), file)
        accelerator.wait_for_everyone()
        with normalizer_path.open("rb") as file:
            normalizer = pickle.load(file)
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        if cfg.training.resume:
            checkpoint_path = self.get_checkpoint_path()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
            payload = self.load_checkpoint(
                path=checkpoint_path,
                map_location="cpu",
                include_keys=self.include_keys,
            )
            required = {"optimizer_step", "training_state_version", "lr_scheduler_state", "rng_states", "dataset_split"}
            if "optimizer" not in payload["state_dicts"] or not required.issubset(payload["pickles"]):
                raise ValueError("This inference/legacy checkpoint lacks complete resumable training state.")
            if self.training_state_version != 2 or self.dataset_split != split:
                raise ValueError("Resume training state version or dataset split does not match.")
        else:
            self.dataset_split = split

        train_dataloader, val_dataloader, self.model, self.optimizer = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer
        )
        device = accelerator.device
        if self.ema_model is not None:
            self.ema_model.to(device)
        batches_per_epoch = len(train_dataloader)
        if cfg.training.max_train_steps is not None:
            batches_per_epoch = min(batches_per_epoch, int(cfg.training.max_train_steps))
        if batches_per_epoch < 1:
            raise ValueError("At least one training microbatch is required.")
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=math.ceil(batches_per_epoch / accumulation) * cfg.training.num_epochs,
        )
        if self.lr_scheduler_state is not None:
            lr_scheduler.load_state_dict(self.lr_scheduler_state)
            for group, lr in zip(self.optimizer.param_groups, lr_scheduler.get_last_lr()):
                group["lr"] = lr
        ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model) if self.ema_model is not None else None
        if ema is not None and self.ema_state is not None:
            ema.optimization_step = self.ema_state["optimization_step"]
            ema.decay = self.ema_state["decay"]
        env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        if not isinstance(env_runner, BaseImageRunner):
            raise TypeError("Expected an image environment runner.")
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )
        topk_manager.path_value_map = self.topk_state
        if self.rng_states is not None:
            if len(self.rng_states) != accelerator.num_processes:
                raise ValueError("Resume requires the same distributed process count.")
            restore_rng_state(self.rng_states[accelerator.process_index])

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for epoch_index in range(self.epoch, cfg.training.num_epochs):
                train_generator.manual_seed(int(cfg.training.seed) + epoch_index)
                val_generator.manual_seed(int(cfg.training.seed) + 1_000_000)
                self.model.train()
                if cfg.training.freeze_encoder:
                    encoder = accelerator.unwrap_model(self.model).obs_encoder
                    encoder.eval()
                    encoder.requires_grad_(False)
                self.optimizer.zero_grad(set_to_none=True)
                window_samples = 0
                train_totals = torch.zeros(2, dtype=torch.float64, device=device)
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {epoch_index}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                    disable=not accelerator.is_main_process,
                ) as tepoch:
                    for batch_index, batch in enumerate(tepoch):
                        if batch_index >= batches_per_epoch:
                            break
                        batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                        batch_samples = len(batch["action"])
                        raw_loss = self.model(batch)
                        if not torch.isfinite(raw_loss):
                            raise FloatingPointError("Non-finite diffusion training loss.")
                        # Weight by examples, including a short final accumulation window.
                        accelerator.backward(raw_loss * batch_samples)
                        window_samples += batch_samples
                        self.global_step += 1
                        is_window_end = (batch_index + 1) % accumulation == 0
                        is_epoch_end = batch_index + 1 == batches_per_epoch
                        if is_window_end or is_epoch_end:
                            for parameter in self.model.parameters():
                                if parameter.grad is not None:
                                    parameter.grad.div_(window_samples)
                            self.optimizer.step()
                            if not accelerator.optimizer_step_was_skipped:
                                self.optimizer_step += 1
                                lr_scheduler.step()
                                if ema is not None:
                                    ema.step(accelerator.unwrap_model(self.model))
                            self.optimizer.zero_grad(set_to_none=True)
                            window_samples = 0
                        train_totals += torch.tensor(
                            [raw_loss.detach().item() * batch_samples, batch_samples], device=device
                        )
                        tepoch.set_postfix(loss=raw_loss.detach().item(), refresh=False)
                        train_sampling_batch = batch

                train_totals = accelerator.reduce(train_totals, reduction="sum")
                step_log = {
                    "train_loss": (train_totals[0] / train_totals[1]).item(),
                    "global_step": self.global_step,
                    "optimizer_step": self.optimizer_step,
                    "epoch": epoch_index,
                    "completed_epochs": epoch_index + 1,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "train_samples": int(train_totals[1].item()),
                    "gradient_accumulate_every": accumulation,
                    "microbatch_size": int(cfg.dataloader.batch_size),
                }
                policy = self.ema_model if self.ema_model is not None else accelerator.unwrap_model(self.model)
                policy.eval()
                # Diagnostics use the real held-out split and do not perturb training RNG.
                training_rng = capture_rng_state()
                validation_seed = int(cfg.training.seed) + 2_000_000 + accelerator.process_index
                random.seed(validation_seed)
                np.random.seed(validation_seed)
                torch.manual_seed(validation_seed)
                step_log["validation_seed"] = validation_seed
                step_log["validation_uses_ema"] = self.ema_model is not None
                if epoch_index % cfg.training.val_every == 0 and len(val_dataset):
                    val_total = 0.0
                    val_samples = 0
                    val_batches = 0
                    with torch.no_grad():
                        for batch_index, batch in enumerate(val_dataloader):
                            if cfg.training.max_val_steps is not None and batch_index >= cfg.training.max_val_steps:
                                break
                            batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                            loss = policy(batch)
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Non-finite diffusion validation loss.")
                            losses = accelerator.gather_for_metrics(loss.detach().repeat(len(batch["action"])))
                            val_total += losses.sum().item()
                            val_samples += len(losses)
                            val_batches += 1
                    step_log.update(
                        val_loss=val_total / val_samples if val_samples else float("nan"),
                        val_samples=val_samples,
                        val_batches=val_batches,
                        val_complete=val_samples == len(val_dataset),
                    )
                if epoch_index % cfg.training.rollout_every == 0:
                    step_log.update(env_runner.run(policy))
                if epoch_index % cfg.training.sample_every == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        sampled = policy.predict_action(train_sampling_batch["obs"], None)["action_pred"]
                        log_action_mse(step_log, "train", sampled, train_sampling_batch["action"])
                        if len(val_dataset):
                            batch = next(iter(val_dataloader))
                            batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))
                            sampled = policy.predict_action(batch["obs"], None)["action_pred"]
                            log_action_mse(step_log, "val", sampled, batch["action"])
                restore_rng_state(training_rng)
                self.epoch = epoch_index + 1
                self.lr_scheduler_state = lr_scheduler.state_dict()
                if ema is not None:
                    self.ema_state = {"optimization_step": ema.optimization_step, "decay": ema.decay}
                local_rng = capture_rng_state()
                if accelerator.num_processes > 1:
                    self.rng_states = [None] * accelerator.num_processes
                    torch.distributed.all_gather_object(self.rng_states, local_rng)
                else:
                    self.rng_states = [local_rng]
                save_epoch = epoch_index % cfg.training.checkpoint_every == 0
                final_epoch = self.epoch == cfg.training.num_epochs
                if (save_epoch or final_epoch) and accelerator.is_main_process:
                    wrapped_model = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    metric_dict = {key.replace("/", "_"): value for key, value in step_log.items()}
                    topk_path = (
                        topk_manager.get_ckpt_path(metric_dict) if topk_manager.monitor_key in metric_dict else None
                    )
                    self.topk_state = topk_manager.path_value_map
                    if topk_path is not None:
                        self.save_checkpoint(path=topk_path)
                    if cfg.checkpoint.save_last_ckpt or final_epoch:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    self.model = wrapped_model
                accelerator.wait_for_everyone()
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg: OmegaConf) -> None:
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
