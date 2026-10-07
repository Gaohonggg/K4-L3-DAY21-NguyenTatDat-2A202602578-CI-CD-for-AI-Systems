#!/usr/bin/env bash
# Run as ubuntu on the Ubuntu 22.04 EC2 instance after uploading the project.
set -euo pipefail

project_dir="${1:-/opt/income-api}"
if [[ "$(id -un)" != ubuntu ]]; then
  echo "Run this script as the ubuntu SSH user."
  exit 1
fi
if [[ "$project_dir" != /opt/income-api ]]; then
  echo "The supplied systemd unit requires /opt/income-api."
  exit 1
fi
cd "$project_dir"
test -f requirements-serve.txt
test -f infra/systemd/income-api.service
test -f infra/systemd/income-api.env.example

sudo apt-get update
sudo apt-get install -y python3-venv curl ca-certificates
/usr/bin/python3.10 --version

if [[ ! -x "$HOME/.local/bin/uv" ]]; then
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl -fsSL https://astral.sh/uv/0.11.18/install.sh -o "$installer"
  sh "$installer"
fi
uv_bin="$HOME/.local/bin/uv"
"$uv_bin" --version
if [[ ! -d .venv ]]; then
  # A managed Python under /home/ubuntu would be blocked by ProtectHome=true.
  "$uv_bin" venv --python /usr/bin/python3.10 --no-python-downloads .venv
fi
"$uv_bin" pip install --python .venv/bin/python -r requirements-serve.txt
"$uv_bin" pip check --python .venv/bin/python

# Show the instance role identity, never its temporary credentials.
.venv/bin/python - <<'PY'
import json
import boto3
identity = boto3.client("sts", region_name="ap-southeast-1").get_caller_identity()
print(json.dumps(identity, indent=2))
if ":assumed-role/IncomeApiDay21Role/" not in identity["Arn"]:
    raise SystemExit("Expected the EC2 instance role IncomeApiDay21Role.")
PY

sudo install -d -o ubuntu -g ubuntu -m 755 "$project_dir/models"
sudo install -m 600 infra/systemd/income-api.env.example /etc/income-api.env
sudo install -m 644 infra/systemd/income-api.service /etc/systemd/system/income-api.service
sudo systemctl daemon-reload
sudo systemctl enable income-api
sudo systemctl restart income-api

for attempt in $(seq 1 30); do
  if curl -fsS --max-time 3 http://127.0.0.1:8080/healthz; then
    printf '\n'
    curl -fsS --max-time 3 http://127.0.0.1:8080/version
    printf '\n'
    sudo systemctl status income-api --no-pager
    exit 0
  fi
  sleep 2
done
echo "API failed to become ready. Service logs:"
sudo journalctl -u income-api -n 80 --no-pager
exit 1
