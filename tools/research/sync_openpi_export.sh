#!/usr/bin/env bash
# SSH path arguments are validated and intentionally expanded locally.
# shellcheck disable=SC2029
set -euo pipefail

SSH_TARGET=tencent-86
REMOTE_OPENPI_HOME=/diff/dzp_is_sb/ambench-research/datasets/openpi
SSH_OPTIONS=(-p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 LOCAL_EXPORT_REPO_ROOT NAMESPACE/NAME" >&2
  exit 2
fi

LOCAL_EXPORT="$(realpath "$1")"
REPO_ID="$2"
if [[ ! -f "$LOCAL_EXPORT/meta/info.json" || ! -f "$LOCAL_EXPORT/ambench_openpi_export_report.json" ]]; then
  echo "Expected meta/info.json and ambench_openpi_export_report.json at $LOCAL_EXPORT" >&2
  exit 1
fi
if [[ ! "$REPO_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "REPO_ID must be NAMESPACE/NAME using letters, digits, dot, underscore or hyphen" >&2
  exit 2
fi
RECORDED_REPO_ID="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("repo_id", ""))' "$LOCAL_EXPORT/ambench_openpi_export_report.json")"
if [[ "$RECORDED_REPO_ID" != "$REPO_ID" ]]; then
  echo "Export metadata repo_id '$RECORDED_REPO_ID' does not match destination '$REPO_ID'" >&2
  exit 1
fi

REMOTE_DEST="$REMOTE_OPENPI_HOME/$REPO_ID"
REMOTE_PARENT="${REMOTE_DEST%/*}"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  "mkdir -p '$REMOTE_PARENT' && test ! -e '$REMOTE_DEST'" || {
  echo "Remote OpenPI export already exists (or SSH failed): $SSH_TARGET:$REMOTE_DEST" >&2
  exit 1
}

rsync -a --partial --info=progress2 \
  -e "ssh -p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new" \
  "$LOCAL_EXPORT/" "$SSH_TARGET:$REMOTE_DEST/"
printf 'OpenPI export uploaded to %s:%s\n' "$SSH_TARGET" "$REMOTE_DEST"
