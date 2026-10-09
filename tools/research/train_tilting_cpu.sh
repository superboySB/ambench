#!/usr/bin/env bash
set -euo pipefail

# The same bounded CPU recipe is used with OmniHexa's canonical dataset.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$SCRIPT_DIR/train_uaquad_cpu.sh" --robot-type omni_hexa "$@"
