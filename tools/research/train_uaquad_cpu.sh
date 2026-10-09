#!/usr/bin/env bash
# The two validated container paths are quoted before remote shell expansion.
# shellcheck disable=SC2029
set -euo pipefail

SSH_TARGET=tencent-86
SSH_OPTIONS=(-p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)

CHECK_ONLY=false
if [[ $# -gt 0 && "$1" == --check ]]; then
  CHECK_ONLY=true
  shift
fi
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 [--check] /data/datasets/UAQUAD_SESSION/lerobot /data/checkpoints/NEW_RUN_DIR" >&2
  exit 2
fi

CANONICAL_DATASET="$1"
RUN_DIR="$2"
if [[ ! "$CANONICAL_DATASET" =~ ^/data/datasets/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]]; then
  echo "Canonical input must be a container path below /data/datasets" >&2
  exit 2
fi
if [[ ! "$RUN_DIR" =~ ^/data/checkpoints/[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$ ]]; then
  echo "New run directory must be a container path below /data/checkpoints" >&2
  exit 2
fi
for path in "$CANONICAL_DATASET" "$RUN_DIR"; do
  if [[ "$path" == *"/../"* || "$path" == */.. || "$path" == *"/./"* || "$path" == */. ]]; then
    echo "Dot path segments are not supported: $path" >&2
    exit 2
  fi
done

printf -v REMOTE_ARGS '%q ' "$CHECK_ONLY" "$CANONICAL_DATASET" "$RUN_DIR"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "bash -s -- $REMOTE_ARGS" <<'REMOTE'
set -euo pipefail

CHECK_ONLY="$1"
CANONICAL_DATASET="$2"
RUN_DIR="$3"
REMOTE_ROOT=/diff/dzp_is_sb/ambench-research
IMAGE=ambench:research-policy
EXPECTED_IMAGE_ID=sha256:ba710cfc14e200c62efe0effbe3b6b265af7d004fbb38890c3f4cd30c7d7cac8
CPU_COUNT=8
MEMORY_LIMIT=20g
STAGE_TIMEOUT=900s

HOST_DATASET="$REMOTE_ROOT${CANONICAL_DATASET#/data}"
HOST_RUN_DIR="$REMOTE_ROOT${RUN_DIR#/data}"
INFO_JSON="$HOST_DATASET/meta/info.json"
ACT_OUTPUT="$RUN_DIR/act"
DP_OUTPUT="$RUN_DIR/dp"
ZARR_PATH="$RUN_DIR/uaquad_pressbutton_ee.zarr.zip"

docker info >/dev/null
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
if [[ "$IMAGE_ID" != "$EXPECTED_IMAGE_ID" ]]; then
  echo "Policy image changed: expected $EXPECTED_IMAGE_ID, got $IMAGE_ID" >&2
  exit 1
fi
if [[ ! -s "$INFO_JSON" ]]; then
  echo "Canonical LeRobot meta/info.json is missing: $CANONICAL_DATASET" >&2
  exit 1
fi
if ! grep -Eq '"robot_type"[[:space:]]*:[[:space:]]*"ua_quad"' "$INFO_JSON" || \
   ! grep -Eq '"action_semantics"[[:space:]]*:[[:space:]]*"ee_absolute"' "$INFO_JSON"; then
  echo "Expected a UAQuad ee_absolute canonical dataset" >&2
  exit 1
fi
if [[ -e "$HOST_RUN_DIR" || -L "$HOST_RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run directory: $RUN_DIR" >&2
  exit 1
fi

POLICY_CHECKPOINT_MOUNT="$(docker inspect ambench-policy-research --format '{{range .Mounts}}{{if eq .Destination "/data/checkpoints"}}{{.Source}}{{end}}{{end}}')"
if [[ "$POLICY_CHECKPOINT_MOUNT" != "$REMOTE_ROOT/checkpoints" ]]; then
  echo "The existing policy container does not share /data/checkpoints with this run" >&2
  exit 1
fi
if [[ "$CHECK_ONLY" == true ]]; then
  printf 'CHECK_OK input=%s output=%s image=%s cpu=%s memory=%s network=none timeout=%s\n' \
    "$CANONICAL_DATASET" "$RUN_DIR" "$IMAGE_ID" "$CPU_COUNT" "$MEMORY_LIMIT" "$STAGE_TIMEOUT"
  exit 0
fi

mkdir -p "$(dirname "$HOST_RUN_DIR")"
mkdir "$HOST_RUN_DIR"
mkdir "$HOST_RUN_DIR/logs"
RUN_NAME="${RUN_DIR##*/}"

run_stage() {
  local stage="$1"
  local container_name="ambench-uaquad-${RUN_NAME}-${stage}"
  local log_path="$HOST_RUN_DIR/logs/${stage}.log"
  local -a statuses
  shift

  printf 'START_UTC=%s STAGE=%s\n' "$(date -u +%FT%TZ)" "$stage" | tee "$log_path"
  set +e
  timeout --signal=TERM --kill-after=30s "$STAGE_TIMEOUT" docker run --rm \
    --name "$container_name" --runtime=runc --network=none \
    --cpus="$CPU_COUNT" --memory="$MEMORY_LIMIT" --memory-swap="$MEMORY_LIMIT" \
    -e CUDA_VISIBLE_DEVICES=-1 -e HF_HUB_OFFLINE=1 \
    "$@" 2>&1 | tee -a "$log_path"
  statuses=("${PIPESTATUS[@]}")
  set -e
  if [[ "${statuses[0]}" -ne 0 || "${statuses[1]}" -ne 0 ]]; then
    docker rm -f "$container_name" >/dev/null 2>&1 || true
    printf 'FAILED_UTC=%s STAGE=%s DOCKER_STATUS=%s LOG_STATUS=%s\n' \
      "$(date -u +%FT%TZ)" "$stage" "${statuses[0]}" "${statuses[1]}" | tee -a "$log_path" >&2
    return 1
  fi
  printf 'END_UTC=%s STAGE=%s STATUS=0\n' "$(date -u +%FT%TZ)" "$stage" | tee -a "$log_path"
}

run_stage act_train \
  --shm-size=2g -e TRANSFORMERS_OFFLINE=1 \
  -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=1 \
  --mount "type=bind,src=$REMOTE_ROOT/source,dst=/workspace/ambench,readonly" \
  --mount "type=bind,src=$REMOTE_ROOT/datasets,dst=/data/datasets,readonly" \
  --mount "type=bind,src=$REMOTE_ROOT/checkpoints,dst=/data/checkpoints" \
  --mount "type=bind,src=$REMOTE_ROOT/outputs,dst=/data/outputs" \
  --workdir /workspace/ambench --entrypoint /opt/venvs/act/bin/python \
  "$IMAGE" -m ambench_learn.policies.act.train \
  --dataset.repo_id=am_bench/pressbutton_ee_absolute --dataset.root="$CANONICAL_DATASET" \
  --dataset.use_imagenet_stats=true --policy.type=act \
  '--policy.input_features={"observation.state":{"type":"STATE","shape":[8]},"observation.images.ee_camera":{"type":"VISUAL","shape":[3,384,384]}}' \
  --policy.chunk_size=16 --policy.n_action_steps=8 \
  --policy.pretrained_backbone_weights=null --policy.device=cpu \
  --policy.push_to_hub=false --output_dir="$ACT_OUTPUT" \
  --job_name=uaquad_pressbutton_act_cpu --batch_size=1 --steps=1 --seed=42 \
  --num_workers=0 --save_freq=1 --wandb.enable=false \
  --policy_target_hz=20 --policy_action_representation=ee_local_relative

run_stage dp_convert \
  --shm-size=2g -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 \
  --mount "type=bind,src=$REMOTE_ROOT/source,dst=/workspace/ambench,readonly" \
  --mount "type=bind,src=$REMOTE_ROOT/datasets,dst=/data/datasets,readonly" \
  --mount "type=bind,src=$REMOTE_ROOT/checkpoints,dst=/data/checkpoints" \
  --workdir /workspace/ambench --entrypoint /opt/venvs/dp/bin/python \
  "$IMAGE" scripts/data/dp/lerobot_to_zarr.py \
  --input_path "$CANONICAL_DATASET" --output_path "$ZARR_PATH" --omit_base_image

run_stage dp_validate \
  -e OPENBLAS_NUM_THREADS=1 \
  --mount "type=bind,src=$REMOTE_ROOT/source,dst=/workspace/ambench,readonly" \
  --mount "type=bind,src=$REMOTE_ROOT/checkpoints,dst=/data/checkpoints,readonly" \
  --workdir /workspace/ambench --entrypoint /opt/venvs/dp/bin/python \
  "$IMAGE" scripts/data/dp/validate_zarr.py "$ZARR_PATH" --image_size 224

run_stage dp_train \
  --shm-size=2g -e TRANSFORMERS_OFFLINE=1 -e WANDB_MODE=disabled \
  -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 \
  --mount "type=bind,src=$REMOTE_ROOT/source,dst=/workspace/ambench,readonly" \
  --mount "type=bind,src=$REMOTE_ROOT/checkpoints,dst=/data/checkpoints" \
  --mount "type=bind,src=$REMOTE_ROOT/outputs,dst=/data/outputs" \
  --workdir /workspace/ambench/source/ambench_learn/ambench_learn/policies/dp/universal_manipulation_interface \
  --entrypoint /opt/venvs/dp/bin/python "$IMAGE" train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace \
  task=umi_drone_ee_pos task.dataset_path="$ZARR_PATH" \
  task.obs_down_sample_steps=6 task.action_horizon=16 \
  task.pose_repr.obs_pose_repr=relative task.pose_repr.action_pose_repr=relative \
  policy.obs_encoder.model_name=resnet18 policy.obs_encoder.pretrained=false \
  policy.obs_encoder.feature_aggregation=avg policy.obs_encoder.transforms=null \
  'policy.down_dims=[64,128]' policy.diffusion_step_embed_dim=64 \
  policy.num_inference_steps=4 policy.noise_scheduler.num_train_timesteps=8 \
  training.num_epochs=1 training.max_train_steps=1 training.max_val_steps=1 \
  training.device=cpu training.seed=42 \
  dataloader.batch_size=1 dataloader.num_workers=0 dataloader.persistent_workers=false \
  val_dataloader.batch_size=1 val_dataloader.num_workers=0 val_dataloader.persistent_workers=false \
  exp_name=uaquad_dp_cpu logging.name=uaquad_dp_cpu logging.mode=disabled \
  hydra.run.dir="$DP_OUTPUT"

timeout --signal=TERM --kill-after=30s "$STAGE_TIMEOUT" docker run --rm \
  --runtime=runc --network=none --cpus="$CPU_COUNT" --memory="$MEMORY_LIMIT" --memory-swap="$MEMORY_LIMIT" \
  -e CUDA_VISIBLE_DEVICES=-1 \
  --mount "type=bind,src=$REMOTE_ROOT/datasets,dst=/data/datasets,readonly" \
  --mount "type=bind,src=$REMOTE_ROOT/checkpoints,dst=/data/checkpoints,readonly" \
  --entrypoint sha256sum "$IMAGE" \
  "$CANONICAL_DATASET/meta/info.json" "$CANONICAL_DATASET/data/chunk-000/file-000.parquet" \
  "$ZARR_PATH" "$ACT_OUTPUT/checkpoints/000001/pretrained_model/config.json" \
  "$ACT_OUTPUT/checkpoints/000001/pretrained_model/model.safetensors" \
  "$DP_OUTPUT/.hydra/config.yaml" "$DP_OUTPUT/checkpoints/latest.ckpt" \
  > "$HOST_RUN_DIR/sha256.txt"

printf 'ACT_CHECKPOINT=%s/checkpoints/000001/pretrained_model\n' "$ACT_OUTPUT"
printf 'DP_CHECKPOINT=%s/checkpoints/latest.ckpt\n' "$DP_OUTPUT"
printf 'SHA256_MANIFEST=%s/sha256.txt\n' "$RUN_DIR"
REMOTE
