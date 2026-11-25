#!/usr/bin/env bash
# Main entry for lead_time_perishable Stage2; Python adapters implement the backend.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN:-python3}" -u -m experiments.launch lead_time_perishable 2 "$@"
