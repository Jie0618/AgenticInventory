#!/usr/bin/env bash
# Main entry for perishable Stage1; Python adapters implement the backend.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN:-python3}" -u -m experiments.launch perishable 1 "$@"
