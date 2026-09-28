#!/usr/bin/env bash
set -euo pipefail
export MSYS_NO_PATHCONV=1
root="$(cd "$(dirname "$0")/.." && { pwd -W 2> /dev/null || pwd; })"
tf="$root/deploy/terraform"
approve=()
[ "${AUTO_APPROVE:-0}" = "1" ] && approve=(-auto-approve)

run="$(bash "$root/deploy/stage.sh" "${1:-}")"
echo "deploying run $run"

terraform -chdir="$tf" init -input=false
terraform -chdir="$tf" apply -input=false -auto-approve -target=aws_ecr_repository.api -target=aws_ecr_lifecycle_policy.api
repository="$(terraform -chdir="$tf" output -raw repository_url)"
region="$(terraform -chdir="$tf" console <<< 'var.region' | tr -d '"')"
parameter="$(terraform -chdir="$tf" console <<< 'local.riot_key_parameter' | tr -d '"')"

key_file="$(mktemp)"
trap 'rm -f "$key_file"' EXIT
sed -n 's/^RIOT_API_KEY=//p' "$root/.env" | tr -d '\r\n' > "$key_file"
[ -s "$key_file" ] || { echo "RIOT_API_KEY is missing from .env" >&2; exit 1; }
key_path="$key_file"
command -v cygpath > /dev/null && key_path="$(cygpath -w "$key_file")"
aws ssm put-parameter --region "$region" --name "$parameter" --type SecureString --overwrite --value "file://$key_path" > /dev/null
rm -f "$key_file"

aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "${repository%%/*}"
docker build --platform linux/amd64 --provenance=false -f "$root/deploy/lambda.Dockerfile" -t "$repository:$run" "$root"
docker push "$repository:$run"
digest="$(docker inspect --format '{{index .RepoDigests 0}}' "$repository:$run")"

terraform -chdir="$tf" apply -input=false "${approve[@]}" -var "image_uri=$digest"
aws lambda update-function-code --region "$region" --function-name "$(terraform -chdir="$tf" output -raw function_name)" --image-uri "$digest" > /dev/null
aws lambda wait function-updated --region "$region" --function-name "$(terraform -chdir="$tf" output -raw function_name)"
bucket="$(terraform -chdir="$tf" output -raw bucket)"
distribution="$(terraform -chdir="$tf" output -raw distribution_id)"
cognito_client="$(terraform -chdir="$tf" output -raw cognito_client_id)"

(cd "$root/frontend" && STATIC_EXPORT=1 NEXT_PUBLIC_COGNITO_REGION="$region" NEXT_PUBLIC_COGNITO_CLIENT_ID="$cognito_client" npx next build)
aws s3 sync "$root/frontend/out" "s3://$bucket" --delete
aws cloudfront create-invalidation --distribution-id "$distribution" --paths "/*" > /dev/null
echo "live at $(terraform -chdir="$tf" output -raw site_url)"
echo "contributed games land in $(terraform -chdir="$tf" output -raw contributed_store), put it in CONTRIBUTED_STORE in .env"
