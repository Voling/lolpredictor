#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && { pwd -W 2> /dev/null || pwd; })"
bucket="${1:?usage: stage_s3.sh <models bucket>}"
run="$(aws s3 cp "s3://$bucket/current.json" - | sed -E 's/.*"run": *"([^"]+)".*/\1/')"
[[ "$run" =~ ^[0-9]{8}-[0-9]{6}$ ]] || { echo "current.json names no run" >&2; exit 1; }
rm -rf "$root/deploy/artifacts"
mkdir -p "$root/deploy/artifacts/runs/$run" "$root/deploy/artifacts/serve"
aws s3 sync "s3://$bucket/runs/$run/models" "$root/deploy/artifacts/models" --only-show-errors
aws s3 sync "s3://$bucket/runs/$run/processed" "$root/deploy/artifacts/processed" --only-show-errors
aws s3 cp "s3://$bucket/runs/$run/manifest.json" "$root/deploy/artifacts/runs/$run/manifest.json" --only-show-errors
printf '{"run": "%s"}' "$run" > "$root/deploy/artifacts/serve/current.json"
echo "$run"
