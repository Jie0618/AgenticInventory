#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../../../.." && pwd)
CAMPAIGN_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
CONFIG=${CONFIG:-$CAMPAIGN_DIR/configs/standard.json}
POLICY_SET=${POLICY_SET:-$CAMPAIGN_DIR/policies/sets/mc_final.json}
RUN_DIR=${RUN_DIR:-$CAMPAIGN_DIR/runs/mc_final}
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
COMMAND=("$PYTHON_BIN" -u -m experiments.benchmark_campaign
  --setting perishable --campaign-dir "$CAMPAIGN_DIR"
  --config "$CONFIG" --policy-set "$POLICY_SET" --run-dir "$RUN_DIR")
exec "${COMMAND[@]}" "$@"
