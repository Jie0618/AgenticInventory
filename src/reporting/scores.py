"""Best-observed-policy reporting using the shared mean/CVaR metric."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from inventory.common.fitness import metrics


def cross_configuration_metrics(scores):
    """Aggregate configuration fitness on the shared configuration set."""
    common = sorted(set.intersection(*(set(values) for values in scores.values()))) if scores else []
    if not common:
        raise ValueError("Cross-configuration fitness requires shared configurations")
    return common, {policy: metrics(values[context] for context in common)
                    for policy, values in scores.items()}


def load_scores(run_dir, context_fields):
    """Read compact configuration/overall tables without loading measurements."""
    tables = []
    for filename, fields in (("SUMMARY.csv", context_fields), ("OVERALL.csv", ())):
        path = Path(run_dir) / filename
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Empty score table: {path}")
        seen = set()
        for row in rows:
            policy = row.get("policy_id", row.get("policy"))
            if not policy:
                raise ValueError(f"Missing policy identifier: {path}")
            for field in fields:
                row[field] = int(row[field])
            key = (policy, *(row[field] for field in fields))
            if key in seen:
                raise ValueError(f"Duplicate score: {path}: {key}")
            seen.add(key)
            for field in ("fitness", "mean_gap_pct", "cvar80_gap_pct"):
                row[field] = float(row[field])
                if not math.isfinite(row[field]):
                    raise ValueError(f"Nonfinite score: {path}: {key}")
            for field in ("count", "cases", "contexts"):
                if field in row:
                    row[field] = int(row[field])
                    if row[field] <= 0:
                        raise ValueError(f"Nonpositive score count: {path}: {key}")
        tables.append(rows)
    if {r.get("policy_id", r.get("policy")) for r in tables[0]} != {
            r.get("policy_id", r.get("policy")) for r in tables[1]}:
        raise ValueError("Configuration and overall policy rosters differ")
    return tuple(tables)


def configuration_scores(rows, context_field):
    """Score measured gaps within contexts, then equally weight shared contexts."""
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (row["instance_id"], row["policy_id"])
        if key in seen:
            raise ValueError("Duplicate instance-policy results")
        seen.add(key)
        groups[(int(row[context_field]), row["policy_id"])].append(
            float(row["relative_gap_pct"]))
    contexts = [{context_field: context, "policy_id": pid, **metrics(values)}
                for (context, pid), values in sorted(groups.items())]
    scores = defaultdict(dict)
    for row in contexts:
        scores[row["policy_id"]][row[context_field]] = row["fitness"]
    _, aggregate = cross_configuration_metrics(scores)
    overall = [{"policy_id": pid, **values} for pid, values in sorted(aggregate.items())]
    return contexts, overall


def score_rows(rows, axis, *, reference_costs=None):
    """Use all successfully measured formal policies for the reference.

    As in the simple campaigns, each method uses its successful measurements;
    absent/TLE measurements are not imputed. Counts are retained in the tables.
    An explicit reference_costs mapping preserves a frozen final comparison.
    """
    references = {}
    seen = set()
    for row in rows:
        key_pair = row['case'], row['policy']
        if key_pair in seen:
            raise ValueError('Duplicate case-policy result')
        seen.add(key_pair)
        if not math.isfinite(float(row['cost'])) or float(row['cost']) < 0:
            raise ValueError('Invalid measured cost')
        key = row['case']
        candidate = (float(row['cost']), row['policy'])
        references[key] = min(references.get(key, candidate), candidate)
    if reference_costs is not None:
        if set(reference_costs) != set(references) or any(
                not math.isfinite(float(c)) or float(c) <= 0 for c in reference_costs.values()):
            raise ValueError('Frozen reference must cover every case with a positive finite cost')
        references = {case: (float(cost), "frozen_official_reference")
                      for case, cost in reference_costs.items()}
    records = []
    grouped = defaultdict(list)
    for row in rows:
        cost, policy = references[row['case']]
        if cost <= 0:
            raise ValueError('Nonpositive best-observed reference')
        record = dict(row, reference_cost=cost, reference_policy=policy,
                      gap_pct=100 * (float(row['cost']) / cost - 1))
        records.append(record)
        grouped[(int(row['L']), int(row[axis]), row['policy'])].append(record['gap_pct'])
    contexts = [dict(L=l, **{axis:k}, policy=p, **metrics(gaps))
                for (l,k,p), gaps in sorted(grouped.items())]
    scores = defaultdict(dict)
    for row in contexts:
        scores[row['policy']][(row['L'], row[axis])] = row['fitness']
    common, aggregate = cross_configuration_metrics(scores)
    overall = [dict(policy=p, contexts=len(common),
                    cases=sum(r['count'] for r in contexts if r['policy'] == p), **m)
               for p,m in aggregate.items()]
    return records, contexts, overall


def write_scores(output, rows, axis, *, report_note=""):
    """Write raw measurements and the compact score tables used by figures."""
    from .tables import write_csv
    reference_path = output / 'REFERENCE_COSTS.json'
    reference_metadata = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    references = reference_metadata.get('costs')
    records, contexts, overall = score_rows(rows, axis, reference_costs=references)
    if reference_metadata.get('reference_rule'):
        for record in records:
            record['reference_policy'] = 'all_formal_policies'
    write_csv(output/'RESULTS.csv', records)
    write_csv(output/'SUMMARY.csv', contexts)
    write_csv(output/'OVERALL.csv', overall)
    if all(key in records[0] for key in ('b', 'g', 'a')):
        costs = defaultdict(list)
        for r in records:
            costs[(r['b'], r['g'], r['a'], r['policy'])].append(r['gap_pct'])
        write_csv(output/'COST_SUMMARY.csv',
                  [dict(zip(('b','g','a','policy'), key), **metrics(gaps))
                   for key,gaps in sorted(costs.items())])
    definition = ('Reference: per-case minimum of all successfully measured policies, including all Our policies. '
                  'Gap = 100*(cost/reference-1). Context fitness = 0.6*mean gap + '
                  '0.4*CVaR80 gap (fractional boundary mass). Overall applies the same operator '
                  'to context fitness, with equal context weights. Missing/TLE measurements are '
                  'excluded per policy, not imputed; see counts. Empirical reference, not an oracle.')
    if references is not None:
        scope = ('all retained formal policies in REFERENCE_COSTS.json'
                 if reference_metadata.get('reference_rule') else 'frozen REFERENCE_COSTS.json')
        definition = ('Reference: ' + scope + '. ' +
                      definition[definition.index('Gap = '):])
    if report_note:
        definition += ' ' + report_note
    lines = ['# Final MC mean–CVaR80 scores', '', definition, '',
             '| Policy | Cases | Overall fitness | Mean context fitness | CVaR80 context fitness |',
             '|---|---:|---:|---:|---:|']
    lines += [f"| {r['policy']} | {r['cases']} | {r['fitness']:.6f} | {r['mean_gap_pct']:.6f} | {r['cvar80_gap_pct']:.6f} |"
              for r in sorted(overall, key=lambda r:r['fitness'])]
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (output/'README.md').write_text('# Final MC results\n\n'+definition+'\n\n'
        'RESULTS.csv retains all raw measurements and recomputed gaps. SUMMARY.csv contains context scores; '
        'OVERALL.csv contains hierarchical cross-context scores (its mean_gap_pct/cvar80_gap_pct '
        'columns summarize context fitness, not pooled case gaps). Plotting reads SUMMARY.csv and OVERALL.csv '
        'directly; no context-splitting step is needed. figures/ contains images only.\n')
    return records, contexts, overall
