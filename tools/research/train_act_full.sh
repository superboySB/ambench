#!/usr/bin/env bash
set -euo pipefail

# Paper Appendix B.2: ImageNet ResNet18, 20 Hz, chunk16/exec8, batch8,
# AdamW lr1e-5, backbone lr1e-5, weight decay1e-4, KL10, 20,000 updates.
# The source dataset has 100 successful episodes: train0..79, validation80..99.
CONTAINER=ambench-policy-training
STEPS=20000
BATCH_SIZE=8
WORKERS=2
SEED=42
SAVE_FREQUENCY=5000

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 /data/datasets/CANONICAL /data/checkpoints/NEW_RUN REPO_ID" >&2
  exit 2
fi
CANONICAL_DATASET="$1"
OUTPUT_DIR="$2"
REPO_ID="$3"
TRAIN_EPISODES="$(python3 -c 'import json; print(json.dumps(list(range(80))))')"

docker exec --workdir /workspace/ambench \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e WANDB_MODE=disabled \
  "$CONTAINER" /opt/venvs/act/bin/python -m ambench_learn.policies.act.train \
  --dataset.repo_id="$REPO_ID" --dataset.root="$CANONICAL_DATASET" \
  --dataset.episodes="$TRAIN_EPISODES" --dataset.use_imagenet_stats=true \
  --policy.type=act --policy.device=cuda:0 --policy.push_to_hub=false \
  '--policy.input_features={"observation.state":{"type":"STATE","shape":[8]},"observation.images.ee_camera":{"type":"VISUAL","shape":[3,384,384]}}' \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.vision_backbone=resnet18 --policy.chunk_size=16 --policy.n_action_steps=8 \
  --policy.optimizer_lr=1e-5 --policy.optimizer_lr_backbone=1e-5 \
  --policy.optimizer_weight_decay=1e-4 --policy.kl_weight=10 \
  --batch_size="$BATCH_SIZE" --steps="$STEPS" --num_workers="$WORKERS" \
  --seed="$SEED" --save_freq="$SAVE_FREQUENCY" --log_freq=100 \
  --output_dir="$OUTPUT_DIR" --job_name=uaquad_act_full --wandb.enable=false \
  --policy_target_hz=20 --policy_action_representation=ee_local_relative
