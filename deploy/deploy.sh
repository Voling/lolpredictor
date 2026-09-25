#!/usr/bin/env bash
set -euo pipefail
export MSYS_NO_PATHCONV=1
root="$(cd "$(dirname "$0")/.." && pwd)"
tf="$root/deploy/terraform"
approve=()
[ "${AUTO_APPROVE:-0}" = "1" ] && approve=(-auto-approve)

run="$(bash "$root/deploy/stage.sh" "${1:-}")"
echo "deploying run $run"

terraform -chdir="$tf" init -input=false
terraform -chdir="$tf" apply -input=false -auto-approve -target=aws_ecr_repository.api -target=aws_ecr_lifecycle_policy.api
repository="$(terraform -chdir="$tf" output -raw repository_url)"
region="$(terraform -chdir="$tf" console <<< 'var.region' | tr -d '"')"

aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "${repository%%/*}"
docker build --platform linux/amd64 --provenance=false -f "$root/deploy/lambda.Dockerfile" -t "$repository:$run" "$root"
docker push "$repository:$run"
digest="$(docker inspect --format '{{index .RepoDigests 0}}' "$repository:$run")"

terraform -chdir="$tf" apply -input=false "${approve[@]}" -var "image_uri=$digest"
bucket="$(terraform -chdir="$tf" output -raw bucket)"
distribution="$(terraform -chdir="$tf" output -raw distribution_id)"

(cd "$root/frontend" && STATIC_EXPORT=1 npx next build)
aws s3 sync "$root/frontend/out" "s3://$bucket" --delete
aws cloudfront create-invalidation --distribution-id "$distribution" --paths "/*" > /dev/null
echo "live at $(terraform -chdir="$tf" output -raw site_url)"
