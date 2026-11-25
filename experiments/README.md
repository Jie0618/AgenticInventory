# Policy Search and Evaluation Guide

This directory demonstrates how to use the agentic model to search for policies and evaluate their performance in four inventory settings. Search proceeds in three stages: exploring policies for a single demand distribution, searching for shared structures across demand distributions, and extending to policy families across system configurations. During search, the agent calls LLMs to propose candidate policies, reflects on numerical evaluation feedback, and updates the policy population; the optimizer fits the parameters of candidate policies.

The following sections explain how to configure and launch search, select discovered policies, and compare them against baselines using independent Monte Carlo evaluation. The four settings are lead time (L), perishability (P), lead time with backlog (L+B), and lead time with perishability (L+P).

Run all commands from the repository root. First activate the environment following the [environment setup instructions](../README.md#environment-and-quick-start), and fill in the API configuration required for search as described in [configs/README.md](../configs/README.md). Bash entry points set import paths automatically.

## Configuration and Computing Resources

The root-level [configs/](../configs/README.md) directory configures only LLM services; experiment parameters are already provided in each setting's stage and campaign configurations, which do not require API keys.

| Configuration location (relative to each setting) | Controls |
| --- | --- |
| `stage1_*/configs/`, `stage2_*/configs/` | Search iterations, parameter optimization, evaluation budgets, and memory |
| `stage3_*/configs/` | Search across contexts and evaluation budgets |
| `benchmark_campaigns/inventory_benchmark_family/configs/standard.json` | Full MC path counts, horizon, contexts, concurrency, and memory |

The full configurations target servers, with training concurrency set to 60 by default. On smaller machines, first reduce concurrency using `WORKERS` for training or `--workers` for benchmarks; also check `memory_reserve_gb` in the training configuration and `memory_safety_reserve_gb` in the benchmark configuration, some of which reserve 100 GB by default. Reducing concurrency alone does not resolve waits for memory admission; set a reasonable reserve based on available memory and account for each task's memory requirements.

For an initial run, reduce path counts, horizon, search iterations, or contexts to explore the search and evaluation workflow with a smaller budget, and save results in a new output directory. Stage configuration overrides are described below; for benchmarks, use `CONFIG=/actual/path/config.json` to select a copy. Relative configuration paths are resolved by the corresponding entry point.

## Entry Points and Stage Responsibilities

| Entry point | Responsibility |
| --- | --- |
| `<setting>/run_stage123.sh` | Run the three stages sequentially, stopping on failure |
| `<setting>/stage1_single_distribution_sr/run.sh` | Search separately in ten environments and prepare references for each case |
| `<setting>/stage2_multi_distribution_sr/run.sh` | Import Stage1 frontiers and search for shared structures across demands |
| `<setting>/stage3_*/run.sh` | Import Stage2 frontiers and search for policy families across contexts |
| `launch.py` | Default commands, paths, and process environments for the four settings |
| `dispatch.py` | Bind Stage1/2 backends to setting-specific inputs |
| `training_inputs.py` | Load and validate the ten frozen training environments for each setting |
| `benchmark_campaign.py` | Coordinate independent MC comparisons of frozen policies |

The setting names are `lead_time`, `perishable`, `lead_time_backlog`, and `lead_time_perishable`. Single-feature settings use the `suite/single/workflow` backends, while joint settings use the MC-only `mc1/mc2` backends; Stage3 shares a common family runner.

## Launching Three-Stage Policy Search

```bash
WORKERS=60 bash experiments/lead_time/run_stage123.sh
```

The full pipeline script accepts only environment variable overrides; additional CLI arguments cause an error. You can explicitly set the three output roots, and inputs and outputs are connected automatically:

```bash
STAGE1="$PWD/tmp/lead_time_training/stage1" \
STAGE2="$PWD/tmp/lead_time_training/stage2" \
STAGE3="$PWD/tmp/lead_time_training/stage3" \
WORKERS=60 bash experiments/lead_time/run_stage123.sh
```

Default outputs are stored under `results/<name>/` in each stage:

| Setting | Stage1 name | Stage2 name |
| --- | --- | --- |
| lead_time | `l3_ten_case_production` | `l3_ten_case_standard` |
| perishable | `m3_n22_stationary_headroom_ten_case_production` | `M3_ten_case_standard` |
| lead_time_backlog | `l3k3_balanced_10_standard` | Same as Stage1 |
| lead_time_perishable | `l3m3_balanced_10` | Same as Stage1 |

All Stage3 runs use the name `stage3_ml_md_run`. Checkpoints are stored at `<root>/<case>/checkpoint.json` for Stage1, `<root>/checkpoint.json` for Stage2, and `<root>/CHECKPOINT.json` for Stage3; note the capitalization.

## Individual Stages and Single Cases

```bash
WORKERS=60 bash experiments/lead_time/stage1_single_distribution_sr/run.sh \
  --output-root "$PWD/tmp/lead_time_training/stage1"

WORKERS=60 bash experiments/lead_time/stage2_multi_distribution_sr/run.sh \
  --checkpoint-root "$PWD/tmp/lead_time_training/stage1" \
  --output-dir "$PWD/tmp/lead_time_training/stage2"

WORKERS=60 bash experiments/lead_time/stage3_multi_leadtime_sr/run.sh \
  --stage2-checkpoint "$PWD/tmp/lead_time_training/stage2/checkpoint.json" \
  --output-dir "$PWD/tmp/lead_time_training/stage3"
```

Stage2 requires completed Stage1 runs for all ten environments, and Stage3 requires a completed, importable Stage2 run. Before running a single case, list its name:

```bash
PYTHONPATH=src:. python -c \
  "from experiments.lead_time.configs import cases; print('\n'.join(c.name for c in cases()))"
```

Use `--case <actual_name>` to select a Stage1 environment. Wrappers for single-feature settings append the case name to the output root; joint settings also organize case results under the root directory. Running only one case is insufficient to launch the default Stage2.

## Parameter Overrides

| Environment variable | Purpose |
| --- | --- |
| `PYTHON_BIN` | Bash entry points default to `python3`; direct calls to `launch.py` default to the current Python interpreter |
| `WORKERS` | Training concurrency, default 60; benchmarks use `--workers` |
| `STAGE1/STAGE2/STAGE3` | Default output locations and dependencies between stages |
| `MUTATION_CONFIG/REFLECTION_CONFIG` | Configuration for the two LLM roles |
| `OUTPUT_DIR`, Stage1's `OUTPUT_ROOT` | Output overrides for individual stages |
| `CHECKPOINT_ROOT` | Input root for an individual Stage2 run |
| `STAGE2_CHECKPOINT/SEED_ARTIFACTS` | Input checkpoint for an individual Stage3 run |
| `EVALUATION_CONFIG/EVAL_CONFIG` | Evaluation configuration for individual stages |
| `ITERATIONS` | Iteration count for an individual Stage3 run |
| `FIDELITY` | Evaluation fidelity for individual joint-setting Stage1/2 and Stage3 runs |
| `LOG_FILE` | Append training output to the specified file |
| `DRY_RUN=1` | Print only the training backend commands |

The full pipeline script sets `EXPERIMENT_PIPELINE=1` and ignores the individual-stage overrides above; call each `run.sh` separately to customize individual stages. `PYTHON_BIN`, `WORKERS`, `MUTATION_CONFIG`, `REFLECTION_CONFIG`, and the three `STAGE` paths also apply to the full pipeline. Explicit CLI arguments are appended after defaults; for ordinary single-value options, the last value takes precedence. The evaluation configuration options for each backend are:

- Single-feature Stage1/2: `--cyclic-evaluation-config`.
- Joint-setting Stage1/2: `--evaluation-config`.
- Stage3: `--config`.

See the actual entry point's `--help` for supported arguments. The training launcher limits inner BLAS/OpenMP threads to avoid nested parallelism across processes.

## Preview and Validation

```bash
DRY_RUN=1 WORKERS=60 bash experiments/perishable/run_stage123.sh
WORKERS=60 bash experiments/lead_time/stage1_single_distribution_sr/run.sh --dry-run
WORKERS=60 bash experiments/perishable/stage3_multi_perishable_sr/run.sh \
  --dry-run
```

`DRY_RUN=1` only prints commands without validating dependencies, inputs, or API keys, making it useful for checking paths before training artifacts exist. The backend's `--dry-run` executes planning logic: Stage3 reads and validates a completed Stage2 checkpoint, without making LLM requests or performing numerical evaluation.

By default, Stage3 imports structure seeds from the completed Stage2 parameter-count and complexity frontiers, then requests 6 templates per seed from the LLM, requiring at least 5 valid templates. Deduplicated templates form the initial population, whose parameters are refitted and evaluated across contexts before subsequent search; the original seeds are not automatically included in the initial population. Missing LLM configuration or insufficient initialization templates stops the run without falling back to manual presets; saved valid templates can be reused on retry. `--iterations 0` disables only subsequent search iterations; initialization and initial-family evaluation still run.

For joint-setting Stage2, use `--dry-run --validate-inputs` to check completed Stage1 protocols, imported frontiers, and LLM configuration. Planning checks do not replace actual numerical and API validation.

## Resuming Interrupted Runs

| Stage | Resume behavior |
| --- | --- |
| Single-feature Stage1 suite | `--skip-existing` skips nonempty case directories without checking completion |
| Single-case Stage1 / joint-setting Stage1 | Search is rejected if a checkpoint already exists; use a new output location |
| Stage2 | `--resume` reads lowercase `checkpoint.json` |
| Stage3 | `--resume` reads uppercase `CHECKPOINT.json` and validates identity |
| Benchmark | Rerunning with the same configuration and `RUN_DIR` automatically reuses valid results; no `--resume` is needed |

```bash
WORKERS=60 bash experiments/lead_time/stage2_multi_distribution_sr/run.sh --resume
WORKERS=60 bash experiments/lead_time/stage3_multi_leadtime_sr/run.sh --resume
```

The examples use default paths; custom runs must provide the same inputs, outputs, and configuration again. Generally use a new directory when the model, policies, or sampling protocol changes, rather than manually editing the manifest to bypass checks. The full pipeline is not a unified resume entry point.

Common Stage1/2 artifacts include checkpoints, `REPORT.md`, evaluation tasks, and audits; single-feature Stage1 writes `events.jsonl` and `llm_calls.jsonl` by default. Stage3 primarily writes `RUN_MANIFEST.json`, `CHECKPOINT.json`, `FEATURE_REGISTRY.json`, `initialization/`, `proposals/`, `reflections/`, `audit/`, and task/reference caches. The presence of files does not imply that the full run completed successfully.

## Selecting Policies and Comparing with Baselines

After search, save selected policy structures in a policy set, then let the benchmark fit their parameters and compare them against baselines on independent demand paths. You can also use the frozen policy sets provided in the repository to try the evaluation workflow on its own.

The campaign path for each setting is `experiments/<setting>/benchmark_campaigns/inventory_benchmark_family/`:

| Subpath | Purpose |
| --- | --- |
| `configs/standard.json` | Full budgets, contexts, concurrency, and memory |
| `policies/sets/mc_final.json` | Frozen policy set |
| `policies/assets/` (if present) | AST/motif assets |
| `mc_model.py` | Setting-specific numerics, policy resolution, and cache adapters |
| `scripts/run_mc_final.sh` | Run entry point |
| `plotting/mc_final.py` | Plots for the full comparison |
| `runs/` | Results and reports |

```bash
RUN_DIR="$PWD/tmp/perishable_mc_new" \
bash experiments/perishable/benchmark_campaigns/inventory_benchmark_family/scripts/run_mc_final.sh \
  --workers 60 --dry-run
```

Remove `--dry-run` to execute. The scripts support the environment variables `CONFIG`, `POLICY_SET`, `RUN_DIR`, and `PYTHON_BIN`; set concurrency with the CLI option `--workers`, rather than the training variable `WORKERS`.

All settings support `--method-ids`, `--instances`, and `--contexts`; contexts are integers for single-feature settings and labels such as `L3_K3` or `L3_M3` for joint settings. Single-feature settings support `--limit-instances`; L+P also provides `--smoke`, which requires a separate output under `tmp/` and is not shared across all four settings.

A full comparison generates `RESULTS.csv`, `SUMMARY.csv`, `OVERALL.csv`, reports, and identity records. Method subsets in joint settings may produce only `PARTIAL_RESULTS.csv`/`PROGRESS.json`, or `SUPPLEMENT_*` for backlog, and do not constitute a full comparison. The full empirical reference includes the best independent evaluation cost among all policies in the full comparison.

Reports can be rebuilt for new runs with complete per-case data:

```bash
PYTHONPATH=src:. python -m benchmarks.inventory_benchmark_family.tools.build_mc_report \
  --setting lead_time \
  --model-module experiments.lead_time.benchmark_campaigns.inventory_benchmark_family.mc_model \
  --run-dir tmp/lead_time_mc_new
```

The compact summaries retained in the repository support plotting, but cannot reconstruct per-case reports or resume tasks. `SUMMARY.csv` contains configuration scores, `OVERALL.csv` contains hierarchical aggregates, and `RESULTS.csv` contains per-case/policy measurements.

### Evaluating Newly Discovered Policies

Export a policy set from Stage3's `CHECKPOINT.json`, then pass it to the evaluation entry point through `POLICY_SET`. For example, compare the two completed candidates with the lowest training scores from an L+B search:

```bash
PYTHONPATH=src:. python -m benchmarks.inventory_benchmark_family.tools.freeze_policies \
  --checkpoint experiments/lead_time_backlog/stage3_multi_context_sr/results/stage3_ml_md_run/CHECKPOINT.json \
  --setting lead_time_backlog --top-k 2 \
  --output tmp/lead_time_backlog_policies/sets/search.json

POLICY_SET="$PWD/tmp/lead_time_backlog_policies/sets/search.json" \
RUN_DIR="$PWD/tmp/lead_time_backlog_search_mc" \
bash experiments/lead_time_backlog/benchmark_campaigns/inventory_benchmark_family/scripts/run_mc_final.sh \
  --workers 60
```

All four settings support this workflow; replace `--setting`, the checkpoint path, and the corresponding evaluation script. Omitting `--top-k` selects the candidate with the lowest training score; alternatively, use `--family-ids <family_id_or_full_hash> ...` to specify candidates. The tool selects only completed candidates with finite scores, exports their structures and parameter bounds together with the setting's baselines, and checks compatibility with the evaluation contexts. New checkpoints include policy definitions; older versions require retaining `proposals/` in the same search directory.

The export command prints the mapping between policy IDs and original names; use `--method-ids <policy_ID> ...` during evaluation to run only selected methods. The benchmark refits parameters, and final costs are measured on independent demand paths. New sets do not overwrite the paper's policy sets in the repository; a full comparison directly produces score tables, while the paper's plotting entry points are intended for the original policy sets.

For custom evaluation configurations, add `--config /actual/path/config.json` to the export command and set `CONFIG=/actual/path/config.json` for the evaluation script. Existing policy sets can still be repackaged using `--input <policy_set.json> --output <new_path.json>`.
