#!/usr/bin/env bash
# The model name is validated locally before remote shell expansion.
# shellcheck disable=SC2029
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_TARGET=tencent-86
CONTAINER=ambench-policy-research
IMAGE=ambench:research-policy
REMOTE_CACHE=/diff/dzp_is_sb/ambench-research/cache/openpi/openpi-assets/checkpoints
SSH_OPTIONS=(-p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)
RSYNC_SSH='ssh -p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new'

if [[ $# -ne 1 || ( "$1" != pi0 && "$1" != pi05 ) ]]; then
  echo "Usage: $0 pi0|pi05" >&2
  exit 2
fi

if [[ "$1" == pi0 ]]; then
  CHECKPOINT=pi0_base
else
  CHECKPOINT=pi05_base
fi

LOCAL_CACHE="$PROJECT_ROOT/outputs/openpi-cache"
LOCAL_CHECKPOINT="$LOCAL_CACHE/openpi-assets/checkpoints/$CHECKPOINT"
REMOTE_CHECKPOINT="$REMOTE_CACHE/$CHECKPOINT"
mkdir -p "$LOCAL_CACHE"

# Public GCS transfers much faster from this workstation. Use the same policy
# Docker without GPU, then send only verified bytes to the remote cache.
echo "Downloading and verifying official $CHECKPOINT with the local policy Docker."
docker run --rm --user "$(id -u):$(id -g)" --network host -i \
  -e HOME=/tmp -e JAX_PLATFORMS=cpu -e OPENPI_DATA_HOME=/data/cache/openpi \
  -v "$LOCAL_CACHE:/data/cache/openpi" \
  -v "$PROJECT_ROOT/outputs:/data/outputs" \
  --entrypoint /opt/openpi/.venv/bin/python "$IMAGE" - "$1" \
  < "$PROJECT_ROOT/tools/research/fetch_openpi_base.py" 2>&1 \
  | tee "$PROJECT_ROOT/outputs/openpi-$1-base-local.log"

if ! ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "test -d '$REMOTE_CHECKPOINT'"; then
  ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_CACHE'"
  rsync -a --partial --append-verify --info=progress2 -e "$RSYNC_SSH" \
    "$LOCAL_CHECKPOINT/" "$SSH_TARGET:$REMOTE_CHECKPOINT.incoming/" \
    2>&1 | tee "$PROJECT_ROOT/outputs/openpi-$1-base-transfer.log"
  ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
    "test ! -e '$REMOTE_CHECKPOINT' && mv '$REMOTE_CHECKPOINT.incoming' '$REMOTE_CHECKPOINT'"
fi

echo "Verifying $CHECKPOINT again in the remote policy Docker."
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  "docker exec -i -e JAX_PLATFORMS=cpu '$CONTAINER' /opt/openpi/.venv/bin/python - '$1'" \
  < "$PROJECT_ROOT/tools/research/fetch_openpi_base.py" 2>&1 \
  | tee "$PROJECT_ROOT/outputs/openpi-$1-base-remote.log"
