# Quick Examples

Install and activate the environment following the [environment setup instructions](../README.md#environment-and-quick-start). Run from the repository root; no API, training artifacts, or environment variables are required.

[compare_policies.py](compare_policies.py) uses the repository optimizer's `fast` mode to fit the paper's policies and selected baselines for L and P, then evaluates them on independent demand paths and prints separate rankings and parameters. The examples add import paths automatically, so the repository does not need to be installed. Case definitions, including demand and costs, are in `comparison_case()`.

## Fitting Policies from the Paper

```bash
# Run the paper policies and selected baselines for L and P by default
python examples/compare_policies.py

# Run the paper policies and baselines for one setting
python examples/compare_policies.py --setting lead_time
python examples/compare_policies.py --setting perishable

# Fit a single policy
python examples/compare_policies.py --setting lead_time --policy SCDS
python examples/compare_policies.py --setting perishable --policy RLCS

# Fit only the specified policies
python examples/compare_policies.py --setting lead_time --policies cbs SCDS IA-SCDS
python examples/compare_policies.py --setting perishable --policies bsp_low EABS RLCS
```

`--setting` supports `all` (default), `lead_time`, and `perishable`. `--policy` and `--policies` are equivalent and both accept one or more policy names; when omitted, all example policies for the setting are run. When `--setting` is omitted, policies are run in their respective settings; `base_stock` belongs to both L and P, so specify `--setting` to run it in only one setting.

| Setting | Paper policies | Baselines |
| --- | --- | --- |
| L: `lead_time` | SCDS, IA-SCDS | `co`, `base_stock`, `cbs`, `weighted_tilted_pic` |
| P: `perishable` | EABS, RLCS | `base_stock`, `bsp_low`, `weighted_base_stock` |

Classical references include CO (Constant Order), Base Stock, and CBS (Capped Base Stock); Weighted Tilted PIC, BSP-low, and Weighted Base Stock are also selected as stronger formula baselines. All policies load their frozen definitions from the corresponding campaigns; policies within a setting use the same fast configuration and shared demand paths.

Both settings share the frozen demand distribution `random_pmf_4_m12_alpha1`, with uneven discrete probabilities on `0,…,12` and most probability mass concentrated around 4, 6, and 12. The full PMF is in the [demand catalog](../benchmarks/inventory_benchmark_family/demand_catalogs/inventory_50/distributions.json).

| Setting | Structure | Shortage cost | Expiry cost | Fitting paths | Fast proposal budget |
| --- | --- | --- | --- | --- | --- |
| L | `L=6` | 20 | 0 | 4,096 | 4,096 |
| P | `M=2` | 9 | 16 | 4,096 | 4,096 |

Shared configuration: empty initial state, a 100-period horizon, an action cap of `n=20`, holding cost 1, and no inventory projection capacity. The example parameter domain for RLCS removes saturated regions that are equivalent after action projection; its policy formula and full experiment definition remain consistent.

The fixed default seeds are `123` for fitting, `125` for the optimizer, and `20261125` for independent evaluation. All policies in the same setting bind to the same fitting demand paths, with the first 512 paths used for candidate screening; final evaluation uses 50,000 independent paths. Selecting individual policies or changing their input order does not change their fitting seeds or paths. `--seed` controls the fitting seed, and the optimizer uses `seed+2`; `--evaluation-seed` separately controls final evaluation.

After each setting finishes, cost ranking and fitted parameter tables are printed in the terminal, without creating output directories or result files. `Rank` orders policies by independent evaluation `MC/period` in ascending order; costs are displayed to six decimal places, with ties sharing a rank (e.g., `1, 2, 2, 4`). The parameter table uses the same order. `MC SE/period` reports the evaluation standard error.

The per-period costs obtained by running the default commands are shown below; lower is better:

| Setting | Best selected baseline | Paper policies and costs |
| --- | --- | --- |
| L | CBS: 20.293585 | SCDS: 20.091240; IA-SCDS: 20.099474 |
| P | BSP-low / Weighted Base Stock: 10.090474 | EABS: 10.088021; RLCS: 10.087242 |
