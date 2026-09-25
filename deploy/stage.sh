#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
run="${1:-$(sed -E 's/.*"run": *"([^"]+)".*/\1/' "$root/data/serve/current.json")}"
source="$root/data/runs/$run"
[ -d "$source/serve" ] || { echo "no served artifacts for run $run" >&2; exit 1; }
rm -rf "$root/deploy/artifacts"
mkdir -p "$root/deploy/artifacts/runs/$run" "$root/deploy/artifacts/serve"
cp -r "$source/serve/models" "$source/serve/processed" "$root/deploy/artifacts/"
cp "$source/manifest.json" "$root/deploy/artifacts/runs/$run/"
printf '{"run": "%s"}' "$run" > "$root/deploy/artifacts/serve/current.json"
echo "$run"
