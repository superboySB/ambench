#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_TARGET="${1:-all}"
BUILD_MODE="${2:-cached}"
BUILD_FLAGS=(--progress=plain)

if [[ $# -gt 2 || ( "$BUILD_MODE" != cached && "$BUILD_MODE" != clean ) ]]; then
  echo "Usage: $0 [sim|policy|all] [cached|clean]" >&2
  exit 2
fi
if [[ "$BUILD_MODE" == clean ]]; then
  BUILD_FLAGS+=(--no-cache --pull)
fi

REVISION="$(git -C "$PROJECT_ROOT" rev-parse HEAD)"
SUBMODULES="$(git -C "$PROJECT_ROOT" submodule status --recursive | tr '\n' ';')"
SOURCE_DIRTY=false
if [[ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]]; then
  SOURCE_DIRTY=true
fi
BUILD_FLAGS+=(--build-arg "AMBENCH_REVISION=$REVISION"
  --build-arg "AMBENCH_SUBMODULES=$SUBMODULES"
  --build-arg "AMBENCH_SOURCE_DIRTY=$SOURCE_DIRTY")

write_manifest() {
  local target="$1"
  local image="ambench:research-$target"
  local image_id
  image_id="$(docker image inspect "$image" --format '{{.Id}}')"
  local manifest_dir="$PROJECT_ROOT/outputs/docker-builds/${image_id#sha256:}"
  mkdir -p "$manifest_dir"
  docker image inspect "$image" > "$manifest_dir/image-inspect.json"
  printf '%s\n' "$BUILD_MODE" > "$manifest_dir/build-mode.txt"
  printf 'Built %s at %s (source %s, dirty=%s); manifest: %s\n' \
    "$image" "$image_id" "$REVISION" "$SOURCE_DIRTY" "$manifest_dir"
}

build_sim() {
  test -f "$PROJECT_ROOT/ext/pyroki/pyproject.toml" || {
    echo "Missing pinned ext/pyroki submodule; run git submodule update --init ext/pyroki ext/acados ext/openpi" >&2
    exit 1
  }
  test -f "$PROJECT_ROOT/ext/acados/CMakeLists.txt" || {
    echo "Missing pinned ext/acados submodule; run git submodule update --init ext/pyroki ext/acados ext/openpi" >&2
    exit 1
  }
  docker build "${BUILD_FLAGS[@]}" \
    -f "$PROJECT_ROOT/docker/Dockerfile.sim" \
    -t ambench:research-sim "$PROJECT_ROOT"
  write_manifest sim
}

build_policy() {
  test -f "$PROJECT_ROOT/ext/openpi/uv.lock" || {
    echo "Missing pinned ext/openpi submodule; run git submodule update --init ext/pyroki ext/acados ext/openpi" >&2
    exit 1
  }
  docker build "${BUILD_FLAGS[@]}" \
    -f "$PROJECT_ROOT/docker/Dockerfile.policy" \
    -t ambench:research-policy "$PROJECT_ROOT"
  write_manifest policy
}

case "$BUILD_TARGET" in
  sim) build_sim ;;
  policy) build_policy ;;
  all) build_sim; build_policy ;;
  *)
    echo "Usage: $0 [sim|policy|all] [cached|clean]" >&2
    exit 2
    ;;
esac
