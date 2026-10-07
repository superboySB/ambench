#!/usr/bin/env bash
set -euo pipefail

REPO=/workspace/ambench
WORKDIR=/workspace/ambench-run

if [[ ! -d "$REPO/scripts" || ! -d "$REPO/source" ]]; then
  echo "AM-Bench source mount is missing at $REPO" >&2
  exit 1
fi

# acados writes generated C, a model JSON file and its compile directory
# relative to the current directory. Keep those files in this writable Docker
# volume while preserving the normal relative paths used by repo commands.
for name in scripts source ext assets datasets outputs videos tools docker usage_assets \
  README.md note.md usage.md pyproject.toml; do
  if [[ ! -e "$REPO/$name" ]]; then
    continue
  fi
  if [[ -L "$WORKDIR/$name" ]]; then
    [[ "$(readlink "$WORKDIR/$name")" == "$REPO/$name" ]] || {
      echo "Unexpected link in writable workdir: $WORKDIR/$name" >&2
      exit 1
    }
  elif [[ -e "$WORKDIR/$name" ]]; then
    echo "Unexpected entry in writable workdir: $WORKDIR/$name" >&2
    exit 1
  else
    ln -s "$REPO/$name" "$WORKDIR/$name"
  fi
done

cd "$WORKDIR"
exec "$@"
