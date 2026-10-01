#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && { pwd -W 2> /dev/null || pwd; })"
run="${1:-$(sed -E 's/.*"run": *"([^"]+)".*/\1/' "$root/data/serve/current.json")}"
bucket="${MODEL_BUCKET:-$(terraform -chdir="$root/deploy/terraform" output -raw models_bucket)}"
source="$root/data/runs/$run"
[ -d "$source/serve" ] || { echo "no served artifacts for run $run" >&2; exit 1; }
aws s3 sync "$source/serve/models" "s3://$bucket/runs/$run/models" --delete --only-show-errors
aws s3 sync "$source/serve/processed" "s3://$bucket/runs/$run/processed" --delete --only-show-errors
[ -d "$source/serve/evaluator" ] && aws s3 sync "$source/serve/evaluator" "s3://$bucket/runs/$run/evaluator" --delete --only-show-errors
aws s3 cp "$source/manifest.json" "s3://$bucket/runs/$run/manifest.json" --only-show-errors
printf '{"run": "%s"}' "$run" | aws s3 cp - "s3://$bucket/current.json" --only-show-errors
echo "published run $run, the next backend deploy serves it"
