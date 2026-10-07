#!/usr/bin/env bash
# SSH command arguments below are intentionally expanded on the local side.
# shellcheck disable=SC2029
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_TARGET=tencent-86
REMOTE_ROOT=/diff/dzp_is_sb/ambench-research
IMAGE=ambench:research-policy
CONTAINER=ambench-policy-research
EXPORT_DIR="$HOME/.cache/ambench-research"
SSH_OPTIONS=(-p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)
RSYNC_SSH='ssh -p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new'
GPU_REQUEST="${1:-auto}"

if [[ $# -gt 1 || ( "$GPU_REQUEST" != auto && ! "$GPU_REQUEST" =~ ^[0-9]+$ ) ]]; then
  echo "Usage: $0 [GPU_ID]  (default: smallest idle GPU ID)" >&2
  exit 2
fi

command -v zstd >/dev/null || { echo "zstd is required locally to export the Docker image" >&2; exit 1; }
command -v rsync >/dev/null || { echo "rsync is required locally to transfer the image" >&2; exit 1; }

GPU_ID="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "bash -s -- '$GPU_REQUEST'" <<'REMOTE'
set -euo pipefail
requested="$1"
rows="$(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)"
running="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true)"
selected=""
while IFS=, read -r index uuid memory utilization; do
  index="${index//[[:space:]]/}"
  uuid="${uuid//[[:space:]]/}"
  memory="${memory//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  [[ "$index" =~ ^[0-9]+$ && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] || continue
  [[ "$requested" == auto || "$requested" == "$index" ]] || continue
  (( memory <= 64 && utilization == 0 )) || continue
  [[ "$running" != *"$uuid"* ]] || continue
  if [[ -z "$selected" || "$index" -lt "$selected" ]]; then
    selected="$index"
  fi
done <<< "$rows"
if [[ -z "$selected" ]]; then
  echo "No idle GPU matches request '$requested' (requires <=64 MiB, 0% utilization and no compute process)." >&2
  nvidia-smi >&2
  exit 1
fi
printf '%s\n' "$selected"
REMOTE
)"
printf 'Selected idle %s GPU %s\n' "$SSH_TARGET" "$GPU_ID"

REMOTE_DOCKER="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  'if docker info >/dev/null 2>&1; then echo docker; elif sudo -n docker info >/dev/null 2>&1; then echo "sudo -n docker"; else echo "Docker unavailable without password" >&2; exit 1; fi')"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  "$PROJECT_ROOT/tools/research/build_images.sh" policy
fi
IMAGE_ID="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
IMAGE_FINGERPRINT="$(docker image inspect "$IMAGE" --format '{{json .Config}} {{json .RootFS.Layers}}' | sha256sum | cut -d ' ' -f 1)"
IMAGE_SUFFIX="${IMAGE_ID#sha256:}"
ARCHIVE="ambench-policy-${IMAGE_SUFFIX:0:16}.tar.zst"
mkdir -p "$EXPORT_DIR"

ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  "mkdir -p '$REMOTE_ROOT/images' '$REMOTE_ROOT/source/source' '$REMOTE_ROOT/source/scripts' '$REMOTE_ROOT/datasets' '$REMOTE_ROOT/checkpoints' '$REMOTE_ROOT/outputs' '$REMOTE_ROOT/cache' '$REMOTE_ROOT/assets'"

# The image contains installed packages; the small source bind mount keeps the
# policy service and data conversion scripts on the same research branch.
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' --exclude .pytest_cache -e "$RSYNC_SSH" \
  "$PROJECT_ROOT/source/" "$SSH_TARGET:$REMOTE_ROOT/source/source/"
rsync -a --delete --exclude __pycache__ --exclude '*.pyc' -e "$RSYNC_SSH" \
  "$PROJECT_ROOT/scripts/" "$SSH_TARGET:$REMOTE_ROOT/source/scripts/"

REMOTE_IMAGE_ID="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  "$REMOTE_DOCKER image inspect '$IMAGE' --format '{{.Id}}' 2>/dev/null || true")"
REMOTE_IMAGE_FINGERPRINT="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  "$REMOTE_DOCKER image inspect '$IMAGE' --format '{{json .Config}} {{json .RootFS.Layers}}' 2>/dev/null | sha256sum | cut -d ' ' -f 1")"
if [[ "$REMOTE_IMAGE_ID" != "$IMAGE_ID" && "$REMOTE_IMAGE_FINGERPRINT" != "$IMAGE_FINGERPRINT" ]]; then
  if [[ ! -s "$EXPORT_DIR/$ARCHIVE" ]]; then
    printf 'Compressing %s to %s\n' "$IMAGE" "$EXPORT_DIR/$ARCHIVE"
    PARTIAL_ARCHIVE="$EXPORT_DIR/$ARCHIVE.partial.$$"
    trap 'rm -f "$PARTIAL_ARCHIVE"' EXIT
    docker save "$IMAGE" | zstd -T0 -3 -o "$PARTIAL_ARCHIVE"
    zstd -t "$PARTIAL_ARCHIVE"
    mv "$PARTIAL_ARCHIVE" "$EXPORT_DIR/$ARCHIVE"
    trap - EXIT
  fi
  rsync -ah --partial --info=progress2 -e "$RSYNC_SSH" \
    "$EXPORT_DIR/$ARCHIVE" "$SSH_TARGET:$REMOTE_ROOT/images/$ARCHIVE"
  ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
    "if command -v zstd >/dev/null 2>&1; then zstd -dc '$REMOTE_ROOT/images/$ARCHIVE' | $REMOTE_DOCKER load; else $REMOTE_DOCKER load -i '$REMOTE_ROOT/images/$ARCHIVE'; fi"
  REMOTE_IMAGE_ID="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
    "$REMOTE_DOCKER image inspect '$IMAGE' --format '{{.Id}}'")"
  REMOTE_IMAGE_FINGERPRINT="$(ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
    "$REMOTE_DOCKER image inspect '$IMAGE' --format '{{json .Config}} {{json .RootFS.Layers}}' | sha256sum | cut -d ' ' -f 1")"
  [[ "$REMOTE_IMAGE_ID" == "$IMAGE_ID" || "$REMOTE_IMAGE_FINGERPRINT" == "$IMAGE_FINGERPRINT" ]] || {
    echo "Loaded remote image contents do not match the local image" >&2
    exit 1
  }
fi

printf -v RUN_ARGS '%q ' "$GPU_ID" "$CONTAINER" "$REMOTE_ROOT" "$IMAGE"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "bash -s -- $RUN_ARGS" <<'REMOTE'
set -euo pipefail
gpu_id="$1"
container="$2"
remote_root="$3"
image="$4"
gpu_still_idle=false
running="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true)"
while IFS=, read -r index uuid memory utilization; do
  index="${index//[[:space:]]/}"
  uuid="${uuid//[[:space:]]/}"
  memory="${memory//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  if [[ "$index" == "$gpu_id" && "$memory" =~ ^[0-9]+$ && "$utilization" =~ ^[0-9]+$ ]] \
      && (( memory <= 64 && utilization == 0 )) && [[ "$running" != *"$uuid"* ]]; then
    gpu_still_idle=true
  fi
done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
if [[ "$gpu_still_idle" != true ]]; then
  echo "GPU $gpu_id became busy during upload; rerun deployment to select another idle GPU." >&2
  exit 1
fi
if docker info >/dev/null 2>&1; then
  docker_command=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
  docker_command=(sudo -n docker)
else
  echo 'Docker unavailable without password' >&2
  exit 1
fi
if "${docker_command[@]}" container inspect "$container" >/dev/null 2>&1; then
  "${docker_command[@]}" rm -f "$container" >/dev/null
fi
"${docker_command[@]}" run -d \
  --name "$container" \
  --gpus "device=$gpu_id" \
  --ipc host \
  --shm-size 32g \
  --init \
  -p 127.0.0.1:8000:8000 \
  -p 127.0.0.1:8001:8001 \
  -e HF_HOME=/data/cache/huggingface \
  -e HF_LEROBOT_HOME=/data/datasets/openpi \
  -e TORCH_HOME=/data/cache/torch \
  -e OPENPI_DATA_HOME=/data/cache/openpi \
  -e WANDB_MODE=offline \
  -v "$remote_root/source:/workspace/ambench:ro" \
  -v "$remote_root/datasets:/data/datasets" \
  -v "$remote_root/checkpoints:/data/checkpoints" \
  -v "$remote_root/assets:/data/assets" \
  -v "$remote_root/outputs:/data/outputs" \
  -v "$remote_root/cache:/data/cache" \
  -w /workspace/ambench \
  "$image" sleep infinity
"${docker_command[@]}" exec "$container" nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
REMOTE

printf 'Policy container ready: %s:%s (GPU %s)\n' "$SSH_TARGET" "$CONTAINER" "$GPU_ID"
