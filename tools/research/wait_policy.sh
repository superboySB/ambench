#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 act|dp|pi [TIMEOUT_SECONDS]" >&2
  exit 2
fi
POLICY="$1"
TIMEOUT_SECONDS="${2:-300}"
[[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ && "$TIMEOUT_SECONDS" -gt 0 ]] || {
  echo "TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
}

case "$POLICY" in
  act|dp) URL=http://127.0.0.1:8001/health ;;
  pi) URL=http://127.0.0.1:8000/healthz ;;
  *) echo "Unknown policy: $POLICY" >&2; exit 2 ;;
esac

END_TIME=$((SECONDS + TIMEOUT_SECONDS))
while (( SECONDS < END_TIME )); do
  if curl --fail --silent --show-error --max-time 2 "$URL" >/dev/null 2>&1; then
    printf '%s is ready at %s\n' "$POLICY" "$URL"
    exit 0
  fi
  sleep 2
done
echo "$POLICY did not become ready at $URL within $TIMEOUT_SECONDS seconds. Check /data/outputs/policy-server-${POLICY}.log in the remote container." >&2
exit 1
