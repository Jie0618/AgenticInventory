#!/usr/bin/env bash
# Complete Stage1 -> Stage2 -> Stage3 workflow, using each stage's main entry.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SETTING="${ROOT}/experiments/perishable"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: [DRY_RUN=1] [WORKERS=60] [STAGE1=...] [STAGE2=...] [STAGE3=...] bash experiments/perishable/run_stage123.sh"
  exit 0
fi
if [[ "$#" != 0 ]]; then
  echo "Use environment overrides for the complete workflow; pass stage options to <stage>/run.sh." >&2
  exit 2
fi
export EXPERIMENT_PIPELINE=1
bash "${SETTING}/stage1_single_distribution_sr/run.sh"
bash "${SETTING}/stage2_multi_distribution_sr/run.sh"
bash "${SETTING}/stage3_multi_perishable_sr/run.sh"
