#!/usr/bin/env bash
# The selected cleanup scope is expanded locally into the remote shell.
# shellcheck disable=SC2029
set -euo pipefail

SSH_TARGET=tencent-86
REMOTE_ROOT=/diff/dzp_is_sb/ambench-research
EXPORT_DIR="$HOME/.cache/ambench-research"
SSH_OPTIONS=(-p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)

if [[ $# -ne 1 || ( "$1" != local && "$1" != remote && "$1" != all ) ]]; then
  echo "Usage: $0 local|remote|all" >&2
  echo "Removes only AM-Bench research containers, image tags and policy image archives; preserves data and caches." >&2
  exit 2
fi

if [[ "$1" == local || "$1" == all ]]; then
  for container in ambench-sim-research ambench-policy-research; do
    if docker container inspect "$container" >/dev/null 2>&1; then
      docker rm -f "$container"
    fi
  done
  for image in ambench:research-sim ambench:research-policy; do
    if docker image inspect "$image" >/dev/null 2>&1; then
      docker image rm "$image"
    fi
  done
  if [[ -d "$EXPORT_DIR" ]]; then
    find "$EXPORT_DIR" -maxdepth 1 -type f \
      -regextype posix-extended \
      -regex '.*/ambench-policy-[0-9a-f]{16}\.tar\.zst' -print -delete
  fi
fi

if [[ "$1" == remote || "$1" == all ]]; then
  ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "bash -s -- '$REMOTE_ROOT'" <<'REMOTE'
set -euo pipefail
remote_root="$1"
if docker info >/dev/null 2>&1; then
  docker_command=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
  docker_command=(sudo -n docker)
else
  echo 'Docker unavailable without password' >&2
  exit 1
fi
if "${docker_command[@]}" container inspect ambench-policy-research >/dev/null 2>&1; then
  "${docker_command[@]}" rm -f ambench-policy-research
fi
if "${docker_command[@]}" image inspect ambench:research-policy >/dev/null 2>&1; then
  "${docker_command[@]}" image rm ambench:research-policy
fi
if [[ -d "$remote_root/images" ]]; then
  find "$remote_root/images" -maxdepth 1 -type f \
    -regextype posix-extended \
    -regex '.*/ambench-policy-[0-9a-f]{16}\.tar\.zst' -print -delete
fi
REMOTE
fi
