#!/usr/bin/env bash
# The validated policy arguments are expanded locally into the remote shell.
# shellcheck disable=SC2029
set -euo pipefail

SSH_TARGET=tencent-86
SSH_OPTIONS=(-p 22 -o BatchMode=yes -o PasswordAuthentication=no -o PreferredAuthentications=publickey -o StrictHostKeyChecking=accept-new)

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 act|dp CHECKPOINT | pi CONFIG CHECKPOINT" >&2
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

printf -v REMOTE_ARGS '%q ' "$POLICY" "$CHECKPOINT" "$CONFIG"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "bash -s -- $REMOTE_ARGS" <<'REMOTE'
set -euo pipefail
policy="$1"
checkpoint="$2"
config="$3"
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
"${docker_command[@]}" exec -i "$container" bash -s -- "$policy" "$checkpoint" "$config" <<'CONTAINER'
set -euo pipefail
policy="$1"
checkpoint="$2"
config="$3"
pid_file=/data/outputs/policy-server.pid
log_file="/data/outputs/policy-server-${policy}.log"

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
    nohup /opt/venvs/act/bin/python -m ambench_learn.policies.remote.server \
      --policy act --checkpoint "$checkpoint" --host 0.0.0.0 --port 8001 --device cuda:0 \
      > "$log_file" 2>&1 < /dev/null &
    ;;
  dp)
    [[ -e "$checkpoint" ]] || { echo "Missing DP checkpoint: $checkpoint" >&2; exit 1; }
    nohup /opt/venvs/dp/bin/python -m ambench_learn.policies.remote.server \
      --policy dp --checkpoint "$checkpoint" --host 0.0.0.0 --port 8001 --device cuda:0 \
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
    nohup env TORCHDYNAMO_DISABLE=1 /opt/openpi/.venv/bin/python scripts/serve_policy.py \
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
printf 'Started %s server PID %s; log: %s\n' "$policy" "$(cat "$pid_file")" "$log_file"
CONTAINER
REMOTE
