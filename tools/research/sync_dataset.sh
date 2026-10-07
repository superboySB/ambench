#!/usr/bin/env bash
# SSH path arguments are validated and intentionally expanded locally.
# shellcheck disable=SC2029
set -euo pipefail

SSH_TARGET=tencent-86
REMOTE_DATASETS=/diff/dzp_is_sb/ambench-research/datasets
SSH_OPTIONS=(-p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 LOCAL_SESSION_DIR [REMOTE_NAME]" >&2
  exit 2
fi

LOCAL_SESSION="$(realpath "$1")"
REMOTE_NAME="${2:-$(basename "$LOCAL_SESSION")}"
if [[ ! -f "$LOCAL_SESSION/meta/info.json" && ! -f "$LOCAL_SESSION/lerobot/meta/info.json" ]]; then
  echo "Expected canonical LeRobot meta/info.json in $LOCAL_SESSION or its lerobot/ child" >&2
  exit 1
fi
if [[ ! "$REMOTE_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "REMOTE_NAME must contain only letters, digits, dot, underscore or hyphen" >&2
  exit 2
fi

REMOTE_DEST="$REMOTE_DATASETS/$REMOTE_NAME"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  "mkdir -p '$REMOTE_DATASETS' && test ! -e '$REMOTE_DEST'" || {
  echo "Remote dataset already exists (or SSH failed): $SSH_TARGET:$REMOTE_DEST" >&2
  exit 1
}

rsync -a --partial --info=progress2 \
  -e "ssh -p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new" \
  "$LOCAL_SESSION/" "$SSH_TARGET:$REMOTE_DEST/"
printf 'Dataset uploaded to %s:%s\n' "$SSH_TARGET" "$REMOTE_DEST"
