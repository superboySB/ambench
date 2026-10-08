#!/usr/bin/env bash
# The validated policy arguments are expanded locally into the remote shell.
# shellcheck disable=SC2029
set -euo pipefail

SSH_TARGET=tencent-86
SSH_OPTIONS=(-p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)
INFERENCE_SEED=42
INFERENCE_DEVICE=cuda:0

if [[ $# -gt 0 && "$1" == --cpu ]]; then
  INFERENCE_DEVICE=cpu
  shift
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 [--cpu] act|dp CHECKPOINT | pi CONFIG CHECKPOINT" >&2
  exit 2
fi

POLICY="$1"
if [[ "$POLICY" == pi ]]; then
  [[ $# -eq 3 ]] || { echo "Usage: $0 pi CONFIG CHECKPOINT" >&2; exit 2; }
  CONFIG="$2"
  CHECKPOINT="$3"
  [[ "$CONFIG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
    echo "OpenPI config name contains unsupported characters" >&2
    exit 2
  }
elif [[ "$POLICY" == act || "$POLICY" == dp ]]; then
  [[ $# -eq 2 ]] || { echo "Usage: $0 $POLICY CHECKPOINT" >&2; exit 2; }
  CONFIG=""
  CHECKPOINT="$2"
else
  echo "Unknown policy: $POLICY" >&2
  exit 2
fi

printf -v REMOTE_ARGS '%q ' "$POLICY" "$CHECKPOINT" "$CONFIG" "$INFERENCE_SEED" "$INFERENCE_DEVICE"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "bash -s -- $REMOTE_ARGS" <<'REMOTE'
set -euo pipefail
policy="$1"
checkpoint="$2"
config="$3"
inference_seed="$4"
inference_device="$5"
container=ambench-policy-research

if docker info >/dev/null 2>&1; then
  docker_command=(docker)
elif sudo -n docker info >/dev/null 2>&1; then
  docker_command=(sudo -n docker)
else
  echo "Docker unavailable without password" >&2
  exit 1
fi

"${docker_command[@]}" container inspect "$container" >/dev/null
"${docker_command[@]}" exec -i "$container" bash -s -- "$policy" "$checkpoint" "$config" "$inference_seed" "$inference_device" <<'CONTAINER'
set -euo pipefail
policy="$1"
checkpoint="$2"
config="$3"
inference_seed="$4"
inference_device="$5"
visible_devices=0
if [[ "$inference_device" == cpu ]]; then
  visible_devices=-1
fi
pid_file=/data/outputs/policy-server.pid
log_file="/data/outputs/policy-server-${policy}.log"
seed_runner='
import random
import runpy
import sys

import numpy as np
import torch

target, seed, device = sys.argv[1:4]
del sys.argv[1:4]
seed = int(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if device == "cpu":
    torch.set_num_threads(8)
    assert not torch.cuda.is_available(), "CPU mode must hide all CUDA devices"
sys.argv[0] = target
print(f"REMOTE_INFERENCE_SEED={seed}", flush=True)
print(f"REMOTE_INFERENCE_DEVICE={device}", flush=True)
if target.endswith(".py"):
    runpy.run_path(target, run_name="__main__")
else:
    runpy.run_module(target, run_name="__main__")
'

if [[ -s "$pid_file" ]]; then
  previous_pid="$(cat "$pid_file")"
  if [[ "$previous_pid" =~ ^[0-9]+$ && -r "/proc/$previous_pid/cmdline" ]]; then
    previous_command="$(tr '\0' ' ' < "/proc/$previous_pid/cmdline")"
    if [[ "$previous_command" == *ambench_learn.policies.remote.server* || "$previous_command" == *serve_policy.py* ]]; then
      kill "$previous_pid"
      for _ in {1..15}; do
        kill -0 "$previous_pid" 2>/dev/null || break
        sleep 1
      done
      kill -0 "$previous_pid" 2>/dev/null && {
        echo "Previous policy server did not stop: PID $previous_pid" >&2
        exit 1
      }
    fi
  fi
fi

case "$policy" in
  act)
    [[ -e "$checkpoint" ]] || { echo "Missing ACT checkpoint: $checkpoint" >&2; exit 1; }
    nohup env CUDA_VISIBLE_DEVICES="$visible_devices" PYTHONHASHSEED="$inference_seed" /opt/venvs/act/bin/python \
      -c "$seed_runner" ambench_learn.policies.remote.server "$inference_seed" "$inference_device" \
      --policy act --checkpoint "$checkpoint" --host 0.0.0.0 --port 8001 --device "$inference_device" \
      > "$log_file" 2>&1 < /dev/null &
    ;;
  dp)
    [[ -e "$checkpoint" ]] || { echo "Missing DP checkpoint: $checkpoint" >&2; exit 1; }
    nohup env CUDA_VISIBLE_DEVICES="$visible_devices" PYTHONHASHSEED="$inference_seed" /opt/venvs/dp/bin/python \
      -c "$seed_runner" ambench_learn.policies.remote.server "$inference_seed" "$inference_device" \
      --policy dp --checkpoint "$checkpoint" --host 0.0.0.0 --port 8001 --device "$inference_device" \
      > "$log_file" 2>&1 < /dev/null &
    ;;
  pi)
    [[ -e "$checkpoint" || "$checkpoint" == gs://* ]] || {
      echo "Missing OpenPI checkpoint: $checkpoint" >&2
      exit 1
    }
    cd /opt/openpi
    # OpenPI wraps sample_actions with torch.compile(max-autotune). Disable the
    # first-request compile for reproducible short inference checks.
    nohup env CUDA_VISIBLE_DEVICES="$visible_devices" TORCHDYNAMO_DISABLE=1 PYTHONHASHSEED="$inference_seed" /opt/openpi/.venv/bin/python \
      -c "$seed_runner" scripts/serve_policy.py "$inference_seed" "$inference_device" \
      --port 8000 policy:checkpoint "--policy.config=$config" "--policy.dir=$checkpoint" \
      > "$log_file" 2>&1 < /dev/null &
    ;;
esac
printf '%s\n' "$!" > "$pid_file"
sleep 3
if ! kill -0 "$(cat "$pid_file")" 2>/dev/null; then
  cat "$log_file" >&2
  exit 1
fi
printf 'Started %s server on %s, PID %s; log: %s\n' "$policy" "$inference_device" "$(cat "$pid_file")" "$log_file"
CONTAINER
REMOTE
