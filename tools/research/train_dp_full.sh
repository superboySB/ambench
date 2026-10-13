#!/usr/bin/env bash
set -euo pipefail

# Paper Appendix B.2: pretrained CLIP ViT-B/16, full [256,512,1024] U-Net,
# two observations, 16 actions at20Hz, DDIM50/16, AdamW3e-4, 40 full epochs.
# Microbatch8 x accumulation8 retains effective batch64 on the local16GiB GPU.
CONTAINER=ambench-policy-training
EPOCHS=40
MICROBATCH_SIZE=8
ACCUMULATION=8
WORKERS=2
SEED=42
SOURCE_PATH=/workspace/ambench/source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface
PYTHON_SOURCE=/workspace/ambench/source/ambench:/workspace/ambench/source/ambench_learn
CHECK_ARGS=()
CHECK_ENV=()

if [[ $# -gt 0 && "$1" == --check ]]; then
  CHECK_ARGS=(--cfg job --resolve)
  CHECK_ENV=(-e CUDA_VISIBLE_DEVICES=-1)
  shift
fi

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [--check] /data/datasets/CANONICAL.zarr.zip /data/checkpoints/NEW_RUN" >&2
  exit 2
fi
ZARR_DATASET="$1"
OUTPUT_DIR="$2"
VALIDATION_EPISODES="$(python3 -c 'import json; print(json.dumps(list(range(80,100))))')"

docker exec --workdir /workspace/ambench \
  "${CHECK_ENV[@]}" \
  -e "PYTHONPATH=$PYTHON_SOURCE:$SOURCE_PATH" \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e WANDB_MODE=disabled \
  "$CONTAINER" /opt/venvs/dp/bin/python -m ambench_learn.policies.dp.train \
  task=umi_drone_ee_pos task.dataset_path="$ZARR_DATASET" \
  task.obs_down_sample_steps=6 task.action_horizon=16 \
  task.pose_repr.obs_pose_repr=relative task.pose_repr.action_pose_repr=relative \
  "+task.dataset.val_episode_indices=$VALIDATION_EPISODES" \
  +task.dataset.normalizer_num_workers=0 \
  policy.obs_encoder.model_name=vit_base_patch16_clip_224.openai \
  policy.obs_encoder.pretrained=true policy.obs_encoder.frozen=false \
  +policy.obs_encoder.normalize_rgb=true +policy.obs_encoder.eval_crop_ratio=0.95 \
  'policy.down_dims=[256,512,1024]' policy.num_inference_steps=16 \
  policy.noise_scheduler.num_train_timesteps=50 \
  training.num_epochs="$EPOCHS" training.gradient_accumulate_every="$ACCUMULATION" \
  training.device=cuda:0 training.seed="$SEED" training.use_ema=true \
  training.lr_warmup_steps=2000 training.max_train_steps=null training.max_val_steps=null \
  training.val_every=1 training.checkpoint_every=5 \
  dataloader.batch_size="$MICROBATCH_SIZE" dataloader.num_workers="$WORKERS" \
  dataloader.persistent_workers=false val_dataloader.batch_size="$MICROBATCH_SIZE" \
  val_dataloader.num_workers=1 val_dataloader.persistent_workers=false \
  checkpoint.topk.monitor_key=val_loss checkpoint.topk.k=4 \
  'checkpoint.topk.format_str="epoch={epoch:04d}-val_loss={val_loss:.5f}.ckpt"' \
  exp_name=uaquad_dp_full logging.name=uaquad_dp_full logging.mode=disabled \
  hydra.run.dir="$OUTPUT_DIR" "${CHECK_ARGS[@]}"
