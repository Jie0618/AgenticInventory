# Lead Time + Backlog (lead_time_backlog)

Demonstrates policy search and comparison against baselines in an inventory system with both lead times and limited customer patience. Costs include holding, backlog carry, and abandonment.

| Stage | Configuration |
| --- | --- |
| Stage1/2 search anchor | `(L,K)=(3,3)` |
| Stage3 search configurations | `(L,K)∈{(2,2),(3,3),(1,4),(4,1),(5,3),(3,5)}` |
| Benchmark evaluation grid | 50 demand distributions × 8 cost profiles × 9 configurations (`L,K∈{3,5,10}`), totaling 3,600 instances per policy |

Complete the [environment setup](../../README.md#environment-and-quick-start) and [API configuration](../../configs/README.md) first, then run all commands below from the repository root. See the [policy search and evaluation guide](../README.md) for parameter overrides, computing resources, and resuming interrupted runs.

## Policies from the Paper

Section EC.6.2, Eq. (EC.40) of the paper gives the following policies; see [inventory models and policy formulas](../../README.md#four-inventory-settings-and-policies-from-the-paper) for their expressions. Policy IDs in the evaluation set match the names below.

| Policy | Fitted parameters |
| --- | --- |
| BATS | `S1,S2,K` |
| CBATS | `S1,S2,K,C` |

## Policy Search

Run the complete three-stage search:

```bash
WORKERS=60 bash experiments/lead_time_backlog/run_stage123.sh
```

You can also run each stage separately, with Stage2 consuming all Stage1 results and Stage3 consuming Stage2:

```bash
WORKERS=60 bash experiments/lead_time_backlog/stage1_single_distribution_sr/run.sh
WORKERS=60 bash experiments/lead_time_backlog/stage2_multi_distribution_sr/run.sh
WORKERS=60 bash experiments/lead_time_backlog/stage3_multi_context_sr/run.sh
```

Default output locations relative to this directory:

| Stage | Output directory |
| --- | --- |
| Stage1 | `stage1_single_distribution_sr/results/l3k3_balanced_10_standard/` |
| Stage2 | `stage2_multi_distribution_sr/results/l3k3_balanced_10_standard/` |
| Stage3 | `stage3_multi_context_sr/results/stage3_ml_md_run/` |

To preview only the launch commands, prefix the commands above with `DRY_RUN=1`.

## Policy Evaluation

Refit policy parameters and compare against baselines on independent demand paths:

```bash
RUN_DIR="$PWD/tmp/lead_time_backlog_mc_new" \
bash experiments/lead_time_backlog/benchmark_campaigns/inventory_benchmark_family/scripts/run_mc_final.sh \
  --workers 60
```

The default evaluation set is `benchmark_campaigns/inventory_benchmark_family/policies/sets/mc_final.json`, and no API is required. To evaluate newly discovered policies, specify the selected policy set with `POLICY_SET`; see [policy selection and comparison](../README.md#selecting-policies-and-comparing-with-baselines) for instructions.

## Plotting Results

Generate comparison plots from saved evaluation results:

```bash
PYTHONPATH=src:. python experiments/lead_time_backlog/benchmark_campaigns/inventory_benchmark_family/plotting/mc_final.py \
  --output-dir tmp/figures/lead_time_backlog
```

Plotting reads the summaries provided with the repository by default; use `--run-root` to specify a new directory containing complete evaluation results.
