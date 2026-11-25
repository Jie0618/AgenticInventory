"""Setting-owned MC campaign plans and local execution."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib import import_module
from pathlib import Path

from benchmarks.inventory_benchmark_family.evaluate_mc import ROOT, _environment, run_isolated_tasks
from runtime.artifacts import atomic_json

def run_joint_campaign(setting, argv):
    """Coordinate both joint settings; numerical differences stay in adapters."""
    import fcntl
    from benchmarks.inventory_benchmark_family.mc_protocol import (
        load_campaign_config, load_campaign_selection, compatible_resume_protocol, _policy_payload)
    from benchmarks.inventory_benchmark_family.tools.build_mc_report import write_joint_report, write_partial_results
    model_module = f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model"
    model = import_module(model_module)
    backlog = setting == "lead_time_backlog"
    parser = argparse.ArgumentParser(description="Single-machine case-policy MC comparison")
    parser.add_argument('--campaign-dir', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--policy-set', '--policies', dest='policies', type=Path)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--method-ids', nargs='+')
    parser.add_argument('--instances', nargs='+')
    parser.add_argument('--contexts', nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    parser.set_defaults(resume_source=[], allow_search_budget_change=False, policy_group='all', smoke=False, plot=False)
    if not backlog:
        parser.add_argument('--resume-source', type=Path, action='append')
        parser.add_argument('--allow-search-budget-change', action='store_true')
        parser.add_argument('--policy-group', choices=('all', 'formula', 'computational'))
        parser.add_argument('--smoke', action='store_true')
        parser.add_argument('--plot', action='store_true')
    args = parser.parse_args(argv)
    if args.campaign_dir is None and any(p is None for p in (args.config, args.policies, args.run_dir)):
        parser.error('provide --campaign-dir or all of --config, --policy-set and --run-dir')
    if args.campaign_dir is not None:
        base = args.campaign_dir.resolve()
        args.config = args.config or base / 'configs/standard.json'
        args.policies = args.policies or base / 'policies/sets/mc_final.json'
        args.run_dir = args.run_dir or base / 'runs/mc_final'
    if args.method_ids and (len(set(args.method_ids)) != len(args.method_ids) or args.policy_group != 'all'):
        raise ValueError('Use unique method IDs and either --method-ids or --policy-group')
    settings = load_campaign_config(args.config)
    geometry = list(settings['contexts'])
    configured_workers = settings.pop('workers')
    workers = configured_workers if args.workers is None else args.workers
    if workers < 1:
        raise ValueError('workers must be positive')
    keys = ('state_action_cap', 'horizon', 'proposal_paths', 'strict_paths', 'evaluation_paths', 'task_timeout_seconds')
    if backlog:
        keys += ('candidate_batch_size', 'myopic_inner_paths', 'myopic_path_batch_size', 'myopic_action_batch_size')
        model.profiles(settings)
    else:
        model.validate_search_overrides(settings)
    if any(type(settings[k]) is not int or settings[k] < 1 for k in keys):
        raise ValueError('MC path counts, capacities and task budgets must be positive integers')
    proposal_budget = settings['proposal_budget']
    if proposal_budget is not None and (type(proposal_budget) is not int or proposal_budget < 1):
        raise ValueError('proposal_budget must be null or a positive integer')
    if settings['task_memory_gb'] < 0 or settings['fidelity'] not in ('fast', 'standard', 'certify'):
        raise ValueError('Invalid task memory limit or fitting fidelity')
    axis = 'K' if backlog else 'M'
    def context(case):
        return f'L{case.lead_time}_{axis}{case.config.backlog.patience if backlog else case.shelf_life}'
    if args.contexts:
        valid = {f'L{l}_{axis}{k}': [l,k] for l,k in geometry}
        if len(set(args.contexts)) != len(args.contexts) or not set(args.contexts) <= valid.keys():
            raise ValueError('Invalid or duplicate context shard')
        settings['contexts'] = [valid[c] for c in args.contexts]
    if args.smoke:
        if not args.run_dir.resolve().is_relative_to(ROOT / 'tmp'):
            raise ValueError('--smoke requires an explicit --run-dir under repository tmp/')
        settings.update(contexts=[[3,3]], horizon=4, fidelity='fast', proposal_budget=8,
                        proposal_paths=2, strict_paths=3, evaluation_paths=4, task_timeout_seconds=60, task_memory_gb=4)
        workers = min(workers, 2)
    frozen = load_campaign_selection(args.policies, model.BASELINES)
    sr_ids = tuple(p['id'] for p in frozen['policies'] if p.get('type') != 'registered_baseline')
    baseline_only = frozen.get('selection_kind') == 'baselines_only' and not sr_ids
    model.validate_campaign({**settings, 'contexts': geometry}, frozen, sr_ids, baseline_only=baseline_only)
    comparison_ids = model.BASELINES + sr_ids
    method_ids = tuple(p for p in comparison_ids if not args.method_ids or p in args.method_ids)
    if args.method_ids and not set(args.method_ids) <= set(method_ids):
        raise ValueError('Unknown method IDs in the loaded policy set')
    selected = list(model.cases(settings))
    if args.instances:
        requested = set(args.instances)
        selected = [case for case in selected if case.name in requested]
        if len(requested) != len(args.instances) or {case.name for case in selected} != requested:
            raise ValueError('Unknown or duplicate instance shard')
    if not selected:
        raise ValueError('Empty case shard')
    if args.smoke:
        selected = selected[:1]
    representatives = {}
    for case in selected:
        representatives.setdefault(context(case), case)
    resolved = {key: model.resolve(case, frozen) for key,case in representatives.items()}
    scheduled = {key: {p['id'] for p in policies if p['id'] in method_ids and
                      (args.policy_group == 'all' or p['execution'] == args.policy_group)}
                 for key,policies in resolved.items()}
    identity = dict(protocol=model.PROTOCOL, settings=settings, frozen_selection=frozen,
                    resolved_policies=resolved, case_signatures={c.name:c.case_signature for c in selected},
                    model_module=model_module)
    if backlog:
        identity.update(non_sof_status=model.NON_SOF_STATUS,
                        reference=f'per-case empirical minimum of {", ".join(comparison_ids)}; unit case weights')
        if method_ids != comparison_ids:
            identity['scheduled_method_ids'] = list(method_ids)
    else:
        identity.update(smoke=args.smoke, reference='per-case minimum final MC cost of all registered methods including SR; unit case weights')
        if args.method_ids:
            identity['scheduled_method_ids'] = list(method_ids)
    plan = dict(status='plan', contexts=list(resolved), cases=len(selected), baseline_ids=model.BASELINES,
                sr_ids=sr_ids, scheduled_method_ids=method_ids, policy_group=args.policy_group, smoke=args.smoke,
                tasks=sum(len(scheduled[context(c)]) for c in selected), task_granularity='case_policy',
                workers=workers, settings=settings)
    if backlog:
        plan.update(economic_profiles=len(model.profiles(settings)), non_sof_status=model.NON_SOF_STATUS)
    else:
        plan['effective_computational_search'] = {key: {
            p['id']: model.effective_search_settings(representatives[key], p, settings)
            for p in policies if p['id'] in scheduled[key] and p['execution'] == 'computational'}
            for key,policies in resolved.items()}
    print(json.dumps(plan), flush=True)
    if args.dry_run:
        return
    sources = []
    source_names = {}
    source_paths = [path.resolve() for path in args.resume_source]
    if len(set(source_paths)) != len(source_paths):
        raise ValueError('Repeated resume source')
    for source in source_paths:
        if source == args.run_dir.resolve():
            raise ValueError('Resume source must be separate from output')
        source_identity = json.loads((source / 'MC_PROTOCOL.json').read_text())
        if not compatible_resume_protocol(source_identity, identity, args.allow_search_budget_change):
            raise ValueError('Resume source differs in protocol, settings, cases or policies')
        sources.append((source, source_identity))
        source_names[source] = {p.name for p in (source / 'results').glob('*.json')}
    output = args.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol_path = output / 'MC_PROTOCOL.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != identity:
            raise ValueError('Run identity changed; use a new run directory')
        atomic_json(identity, protocol_path)
        if sources:
            records = [{'source_directory': str(source),
                        'search_budget_change_allowed': args.allow_search_budget_change,
                        'source_settings': source_identity['settings']}
                       for source, source_identity in sources]
            # Preserve the existing single-source provenance format.
            provenance = records[0] if len(records) == 1 else {'sources': records}
            provenance_path = output / 'RESUME_SOURCE.json'
            if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
                raise ValueError('Resume source changed')
            atomic_json(provenance, provenance_path)
        policy_snapshot = _policy_payload(args.policies)
        atomic_json({**policy_snapshot, 'selection': frozen, 'resolved': resolved}, output / 'POLICIES.json')
        result_dir, scratch = output / 'results', output / '.scratch'
        result_dir.mkdir(exist_ok=True)
        scratch.mkdir(exist_ok=True)
        rows, pending = [], []
        for case in selected:
            for policy in resolved[context(case)]:
                if backlog and policy['id'] not in method_ids:
                    continue
                policy = model.resolve_case_policy(case, policy)
                path = result_dir / (model.digest([case.name, policy['id']]) + '.json')
                if path.exists():
                    row = json.loads(path.read_text())
                    model.validate_cached(row, case, policy, settings)
                    rows.append(row)
                    continue
                matches = []
                for source, source_identity in sources:
                    if path.name in source_names[source]:
                        row = json.loads((source / 'results' / path.name).read_text())
                        model.validate_cached(row, case, policy, source_identity['settings'])
                        imported = dict(row)
                        if args.allow_search_budget_change and 'search_settings' in row:
                            imported['imported_search_settings'] = row['search_settings']
                        matches.append(imported)
                if len(matches) > 1:
                    raise ValueError(f'Overlapping resume sources for {case.name}/{policy["id"]}')
                if matches:
                    atomic_json(matches[0], path)
                    rows.append(json.loads(path.read_text()))
                elif policy['id'] in scheduled[context(case)]:
                    pending.append((case, policy, path))
        print(json.dumps(dict(status='started', cached=len(rows), pending=len(pending))), flush=True)
        run_isolated_tasks(pending, workers=workers, settings=settings,
                           scratch=scratch, setting=setting, model_module=model_module,
                           validate_cached=model.validate_cached, rows=rows, output=output)
        if backlog and method_ids != model.BASELINES + sr_ids:
            write_partial_results(output / 'SUPPLEMENT_RESULTS.csv', rows)
            atomic_json(dict(status='complete', cases=len(selected), method_ids=list(method_ids),
                             tasks=len(rows), full_comparison=False), output / 'SUPPLEMENT_SUMMARY.json')
        elif len(rows) != len(selected) * len(model.BASELINES + sr_ids):
            if rows:
                write_partial_results(output / 'PARTIAL_RESULTS.csv', rows)
            total = len(selected) * len(model.BASELINES + sr_ids)
            atomic_json(dict(completed=len(rows), expected=total, policy_group=args.policy_group,
                             scheduled_method_ids=identity.get('scheduled_method_ids'),
                             pending_full_comparison=total-len(rows)), output / 'PROGRESS.json')
        else:
            write_joint_report(output, rows, selected, model=model, method_ids=model.BASELINES + sr_ids)
            if (args.plot and not baseline_only and not args.smoke
                    and len(resolved) == 9 and sr_ids == model.SR_IDS):
                subprocess.run([sys.executable, "-m",
                                f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.plotting.mc_final",
                                "--run-root", str(output)], cwd=ROOT, env=_environment(), check=True)
        print(json.dumps(dict(status='complete', tasks=len(rows))), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--setting", required=True, choices=(
        "lead_time", "perishable", "lead_time_backlog", "lead_time_perishable"))
    args, remaining = parser.parse_known_args(argv)
    if args.setting in ("lead_time_backlog", "lead_time_perishable"):
        run_joint_campaign(args.setting, remaining)
    else:
        from benchmarks.inventory_benchmark_family.evaluate_mc import main as run
        run(setting=args.setting, model_module=(
            f"experiments.{args.setting}.benchmark_campaigns.inventory_benchmark_family.mc_model"),
            argv=remaining)


if __name__ == "__main__":
    main()
