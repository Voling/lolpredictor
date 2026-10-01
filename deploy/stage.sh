#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
run="${1:-$(sed -E 's/.*"run": *"([^"]+)".*/\1/' "$root/data/serve/current.json")}"
source="$root/data/runs/$run"
[ -d "$source/serve" ] || { echo "no served artifacts for run $run" >&2; exit 1; }
[ -f "$source/serve/models/style_vectors.npz" ] || { echo "run $run has no style_vectors.npz, run python -m synergy pack first" >&2; exit 1; }
rm -rf "$root/deploy/artifacts"
mkdir -p "$root/deploy/artifacts/runs/$run" "$root/deploy/artifacts/serve" "$root/deploy/artifacts/processed"
cp -r "$source/serve/models" "$root/deploy/artifacts/"
for name in player_names.parquet player_profiles.parquet propensity_report.json; do cp "$source/serve/processed/$name" "$root/deploy/artifacts/processed/"; done
cp "$source/manifest.json" "$root/deploy/artifacts/runs/$run/"
printf '{"run": "%s"}' "$run" > "$root/deploy/artifacts/serve/current.json"
echo "$run"
