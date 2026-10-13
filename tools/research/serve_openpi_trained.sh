#!/usr/bin/env bash
set -euo pipefail

# Run inside the policy Docker container; SSH forwards this loopback endpoint.
MANIFEST="/data/checkpoints/fulltrain-20261013/openpi/provenance/pi05_am_bench_multitask_openpi_original_20hz_h50_ee_local_relative_uaquad_four_tasks_pi05_full_mt.json"
HOST="127.0.0.1"
PORT=8000
OPENPI_ROOT="/opt/openpi"

export PYTHONPATH="/workspace/ambench/source/ambench_learn:$OPENPI_ROOT"
/opt/openpi/.venv/bin/python -m ambench_learn.policies.pi.serve \
  --manifest "$MANIFEST" --host "$HOST" --port "$PORT" "$@"
