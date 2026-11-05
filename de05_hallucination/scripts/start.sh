#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/uvicorn ]]; then
  echo 'Dependencies are missing. Run: bash scripts/install.sh' >&2
  exit 1
fi
exec .venv/bin/uvicorn hallucination_lab.app:app --host 127.0.0.1 --port "${APP_PORT:-8000}"
