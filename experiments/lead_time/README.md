# Lead Time (lead_time)

Demonstrates policy search and comparison against baselines in a nonperishable, lost-sales inventory system with lead times.

| Stage | Configuration |
| --- | --- |
| Stage1/2 search anchor | `L=3` |
| Stage3 search configurations | `L=3,5,8,10` |
| Benchmark evaluation grid | 50 demand distributions × 4 cost profiles × 14 lead times (`L=2,…,15`), totaling 2,800 instances per policy |

Complete the [environment setup](../../README.md#environment-and-quick-start) and [API configuration](../../configs/README.md) first, then run all commands below from the repository root. See the [policy search and evaluation guide](../README.md) for parameter overrides, computing resources, and resuming interrupted runs.

## Policies from the Paper

Section 5.1, Eqs. (31)–(32) of the paper give the following policies; see [inventory models and policy formulas](../../README.md#four-inventory-settings-and-policies-from-the-paper) for their expressions. Policy IDs in the evaluation set match the names below.

| Policy | Fitted parameters |
| --- | --- |
| SCDS | `S,F,K,C` |
| IA-SCDS | `S,F,K,C` |

## Policy Search

Run the complete three-stage search:

```bash
WORKERS=60 bash experiments/lead_time/run_stage123.sh
```

You can also run each stage separately, with Stage2 consuming all Stage1 results and Stage3 consuming Stage2:

```bash
WORKERS=60 bash experiments/lead_time/stage1_single_distribution_sr/run.sh
WORKERS=60 bash experiments/lead_time/stage2_multi_distribution_sr/run.sh
WORKERS=60 bash experiments/lead_time/stage3_multi_leadtime_sr/run.sh
```

Default output locations relative to this directory:

| Stage | Output directory |
| --- | --- |
| Stage1 | `stage1_single_distribution_sr/results/l3_ten_case_production/` |
| Stage2 | `stage2_multi_distribution_sr/results/l3_ten_case_standard/` |
| Stage3 | `stage3_multi_leadtime_sr/results/stage3_ml_md_run/` |

To preview only the launch commands, prefix the commands above with `DRY_RUN=1`.

## Policy Evaluation

Refit policy parameters and compare against baselines on independent demand paths:

```bash
RUN_DIR="$PWD/tmp/lead_time_mc_new" \
bash experiments/lead_time/benchmark_campaigns/inventory_benchmark_family/scripts/run_mc_final.sh \
  --workers 60
```

The default evaluation set is `benchmark_campaigns/inventory_benchmark_family/policies/sets/mc_final.json`, and no API is required. To evaluate newly discovered policies, specify the selected policy set with `POLICY_SET`; see [policy selection and comparison](../README.md#selecting-policies-and-comparing-with-baselines) for instructions.

## Plotting Results

Generate comparison plots from saved evaluation results:

```bash
PYTHONPATH=src:. python experiments/lead_time/benchmark_campaigns/inventory_benchmark_family/plotting/mc_final.py \
  --output-dir tmp/figures/lead_time
```

Plotting reads the summaries provided with the repository by default; use `--run-root` to specify a new directory containing complete evaluation results.
