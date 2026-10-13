#!/usr/bin/env bash
set -euo pipefail

# Run inside the policy Docker container. Prepare stats and validate before train.
VARIANT="pi05"
STAGE="mt"
MODE="full"
ACTION="validate"
REPO_ID="am_bench/uaquad_four_tasks_train80_20hz"
ASSET_ID="am_bench/uaquad_four_tasks_train80_20hz"
DATASET_HOME="/data/datasets/fulltrain-20261013/openpi"
EXPORT_REPORT="$DATASET_HOME/$REPO_ID/ambench_openpi_export_report.json"
NORM_DIRECTORY="/data/assets/fulltrain-20261013/openpi"
CHECKPOINT_BASE_DIR="/data/checkpoints/fulltrain-20261013/openpi"
EXP_NAME="uaquad_four_tasks_pi05_full_mt"
INITIAL_CHECKPOINT="/data/cache/openpi/openpi-assets/checkpoints/pi05_base"
OPENPI_ROOT="/opt/openpi"
SEED=42
FSDP_DEVICES=1

export PYTHONPATH="/workspace/ambench/source/ambench_learn:$OPENPI_ROOT"
/opt/openpi/.venv/bin/python -m ambench_learn.policies.pi.train \
  --variant "$VARIANT" --stage "$STAGE" --mode "$MODE" --action "$ACTION" \
  --repo-id "$REPO_ID" --asset-id "$ASSET_ID" --dataset-home "$DATASET_HOME" \
  --export-report "$EXPORT_REPORT" --norm-directory "$NORM_DIRECTORY" \
  --checkpoint-base-dir "$CHECKPOINT_BASE_DIR" --exp-name "$EXP_NAME" \
  --initial-checkpoint "$INITIAL_CHECKPOINT" --openpi-root "$OPENPI_ROOT" \
  --seed "$SEED" --num-workers 2 --fsdp-devices "$FSDP_DEVICES" "$@"
