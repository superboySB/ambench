#!/usr/bin/env bash
set -euo pipefail

SSH_TARGET=tencent-86
SSH_OPTIONS=(-p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)

echo "Forwarding local 127.0.0.1:8001 (ACT/DP) and :8000 (OpenPI) to $SSH_TARGET; keep this process running."
exec ssh "${SSH_OPTIONS[@]}" \
  -o ExitOnForwardFailure=yes \
  -N \
  -L 127.0.0.1:8001:127.0.0.1:8001 \
  -L 127.0.0.1:8000:127.0.0.1:8000 \
  "$SSH_TARGET"
