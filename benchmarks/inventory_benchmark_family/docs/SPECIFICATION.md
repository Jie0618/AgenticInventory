# Inventory benchmark family specification

## Design principle

A benchmark instance is the tuple

```text
(setting, structural context, demand distribution, economic profile).
```

The setting is part of the identity. A lost-sales lead-time instance and a
perishable instance with the same demand and `h,b` are not the same instance.

## Shared demand backbone

All suites use the 50 PMFs in
`benchmarks/inventory_benchmark_family/demand_catalogs/inventory_50/distributions.json`.
The shared loader validates the 50 unique IDs and normalized probabilities.

## Model-scoped economics

- Lead time: `h,b`.
- Perishable: `h,b,w`.
- Lead time + perishable: `h,b,w`.
- Lead time + backlog: `h,b,g,a`.

In retained-backlog models, `b` is an economic scale for constructing `g` and
`a`, not an entry-shortage charge. Transition cost is `h*I + g*B_next + a*A`
(plus `k*q` if ordering economics are enabled). Only immediately lost demand
in lost-sales models incurs `b*lost_sales`.
The corrected backlog convention is part of case/cache identity; old-cost
references and results cannot be reused as current-cost measurements.

`ordering` remains zero in this benchmark. Adding ordering economics requires a new
profile set and therefore a new suite or benchmark-family version.

All profiles set `h=1` and encode relative costs. This removes irrelevant
common scale while retaining the policy tradeoffs.

## Structural contexts

- Lead time: `L=1,...,8`, retained for continuity with the frozen lead-time suite.
- Perishable: `M in {2,3,4,5}`.
- Lead time + perishable:
  `(2,2),(3,3),(1,4),(4,1),(5,3),(3,5)`.
- Lead time + backlog:
  `(2,2),(3,3),(1,4),(4,1),(5,3),(3,5)`, interpreted as `(L,K)`.

The composed anchors cover balanced, lead-time-dominant, and
perishability/backlog-dominant state shapes.
These are suite defaults, not the context lists of every campaign. Current
formal MC configurations explicitly select L/M=2–15 for the single-feature
settings and nine contexts for each combined setting.

For result-level compatibility, the lead-time suite also freezes the case
ordering, case IDs, `kind`, and default state caps
`{1:50,2:40,3:22,4:14,5:8,6:5,7:4,8:3}`. This fixes the input geometry and
preserves MC streams in runners whose stable seed includes the instance ID.

## Versioning rules

The following changes require a new version:

- any PMF or distribution ID;
- any economic ratio or profile assignment;
- default structural contexts;
- transition semantics or event order;
- case ID construction.

Changes to `n`, horizon, MC paths, optimizer budget, worker count, or reporting
do not change benchmark instance identity, but must be recorded in each run
manifest.

## Baselines and references

Baseline catalogs and references are setting-scoped. A lead-time baseline may
be transferred as a candidate policy, but it does not become a native
perishable or backlog baseline merely because its AST validates. Each suite's
formal MC campaign must declare its own policy set and empirical reference
contract. Training's Exact/DP evaluators live in `src`; they are independent
of the MC benchmark entry points.

## Reporting fitness

The shared aggregation is `R = 0.60 mean + 0.40 CVaR80`, using fractional
upper-tail probability mass. Compute R over instance gaps within a structural
context, then apply R again to equally weighted context scores when reporting
cross-configuration fitness. This is distinct from pooling every instance.
The final report reference is the best observed cost per instance across
successfully measured policies. Search can use a declared fixed reference
and signed gaps; sharing the aggregation formula does not equate these scores.
