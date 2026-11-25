"""lead_time_backlog: case/policy resolution and setting-specific MC numerics."""
from runtime.artifacts import seed_from_text, stable_key
import itertools
import json
import math
import time
import numpy as np
from dataclasses import asdict
from benchmarks.inventory_benchmark_family.demand_catalogs.inventory_50 import load_demands
from benchmarks.inventory_benchmark_family.profiles import EconomicProfile
from inventory.core import DeterministicLeadTime, DeterministicBacklog, inventory_case
from inventory.policy.ast import add, sub, mul, maximum, state, dist, ast_check, ast_to_formula, cap, const, parameter
from inventory.policy.family import materialize_family_for_case
from inventory.policy.family_domain import InventoryFamilyDomain
from optimization import ParameterSpec, OptimizationConfig, optimize_policy_constants
from optimization.policies.parameters import apply_cap_modes
from optimization.policies.baselines.lead_time_backlog import baselines_for_lead_time_backlog, related_model_baselines
from sr.family.proposals import family_from_json
from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator, TwoFidelityComposedMCEvaluator
from benchmarks.baselines.lead_time_backlog.myopic import Myopic, BACKEND
from benchmarks.baselines.lead_time_backlog.gbs import actions_many as gbs_actions_many, BACKEND as GBS_BACKEND

SETTING = "lead_time_backlog"

SOF_IDS = ('constant_order', 'base_stock', 'capped_base_stock', 'sq', 'qr', 'gbs', 'codex-astra-urgency-correction')


BASELINES = SOF_IDS + ('myopic',)


SR_IDS = ('BATS', 'CBATS', 'SR-best')


METHOD_IDS = BASELINES + SR_IDS


PROTOCOL = 'ltb-mc-final-9contexts-no-entry-shortage-sr3-cyclic-cap'


# Immutable numerical seed salt; preserve it across file and display renames.
BANK_PROTOCOL = 'ltb-mc-final-9contexts-cost8-sof3-myopic-sr4'


NON_SOF_STATUS = {
    'included': ['myopic'],
    'reference': 'Bu and Li (2026), SSRN 7457520, section 5 equation (5)',
    'specialization': 'fixed L, deterministic K, repository full cost and finite capacity',
    'evidence': 'docs/baselines/leadtime_backlog.md',
}


def codex_policy(case=None, *, lead_time=0, shelf_life=0, patience=0, mean=None):
    """Frozen external formula; only coefficients are fitted for each case."""
    lead = case.lead_time if case is not None else lead_time
    patience = case.config.backlog.patience if case is not None else patience
    if case is not None:
        mean = sum(float(v) * float(p) for v, p in zip(case.demand.values, case.demand.probs))
    def total(nodes):
        return add(*nodes) if len(nodes) > 1 else nodes[0] if nodes else const(0)
    inventory = state('x')
    pipeline = total([state(f'p{i}') for i in range(1, lead + 1)])
    backlog = total([state(f'b{i}') for i in range(1, patience + 1)])
    motifs = {}
    urgency = maximum(sub(sub(add(dist('mu'), backlog), inventory),
                          state('p1') if lead else const(0)), const(0))
    ast = add(sub(sub(parameter('a'), inventory), pipeline), backlog, mul(parameter('b'), urgency))
    allowed = {'x'}
    allowed.update(f'p{i}' for i in range(1, lead + 1))
    allowed.update(f'b{i}' for i in range(1, patience + 1))
    checked = ast_check(ast, motifs=motifs, allowed_state_names=frozenset(allowed))
    upper = 500. if mean is None else min(500., 200. + mean * (lead + 1))
    specs = [ParameterSpec('a', kind='continuous', lower=0, upper=upper, default=upper / 2),
             ParameterSpec('b', kind='continuous', lower=-1.5, upper=1.5, default=0)]
    if set(checked.stats.parameters) != {p.name for p in specs}:
        raise ValueError('Frozen parameter membership mismatch')
    return dict(id='codex-astra-urgency-correction', type='baseline', execution='formula', method_class='SOF',
                provenance='AI-generated baseline; frozen Codex source formulas', ast=checked.ast,
                motifs=motifs, formula=ast_to_formula(checked.ast),
                parameter_specs=[asdict(p) for p in specs], source='codex-astra')


def validate_campaign(settings, frozen, sr_ids, *, baseline_only):
    from benchmarks.inventory_benchmark_family.mc_protocol import validate_joint_selection
    validate_joint_selection(
        settings, frozen, sr_ids, baseline_ids=BASELINES,
        reviewed_rosters=(SR_IDS,), baseline_only=baseline_only)
    if len({p['id'] for p in frozen['policies'] if p.get('family')}) != len(sr_ids):
        raise ValueError('Duplicate frozen SR selection')


def resolve_case_policy(case, policy):
    """Resolve case-dependent domains before fitting or validating cached results."""
    if policy.get('source') != 'codex-astra':
        return policy
    resolved = codex_policy(case)
    if resolved['id'] != policy['id']:
        raise ValueError('Codex policy ID does not match the setting')
    return resolved


def digest(value):
    return stable_key(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def seed(case, bank, global_seed):
    bank = {'BATS': 'SR-3', 'CBATS': 'SR-4'}.get(bank, bank)
    return seed_from_text(f'{BANK_PROTOCOL}|{global_seed}|{case}|{bank}'.encode())


def profiles(settings):
    axes = [settings[k] for k in ('shortage_over_holding', 'backlog_over_shortage', 'abandonment_over_shortage')]
    if any(not a or len(set(a)) != len(a) or any(v <= 0 for v in a) for a in axes):
        raise ValueError('Cost axes must be nonempty, positive and unique')
    return tuple(EconomicProfile(f'bh{b:g}__gb{g:g}__ab{a:g}'.replace('.', 'p'), b,
                                 backlog_over_shortage=g, abandonment_over_shortage=a)
                 for b, g, a in itertools.product(*axes))


def cases(settings, *, only_name=None):
    economics = profiles(settings)
    for lead, patience in settings['contexts']:
        prefix = f'ltb_mc_final__L{lead}_K{patience}__'
        if only_name is not None and not only_name.startswith(prefix):
            continue
        for demand in load_demands():
            demand_prefix = prefix+demand.demand_id+'__'
            if only_name is not None and not only_name.startswith(demand_prefix):
                continue
            for profile in economics:
                name = demand_prefix+profile.profile_id
                if only_name is not None and only_name != name:
                    continue
                yield inventory_case(
                    name=name,
                    demand=demand.demand, n=settings['state_action_cap'],
                    costs=profile.costs(), lead_time=DeterministicLeadTime(lead),
                    backlog=DeterministicBacklog(patience), capacity=settings['state_action_cap'],
                    kind=f'benchmark:lead_time_backlog:{demand.family}')


def context(case):
    return f'L{case.lead_time}_K{case.config.backlog.patience}'


def resolve(case, frozen):
    output = []
    names = dict(constant_order='constant_order', net_base_stock='base_stock',
                 net_capped_base_stock='capped_base_stock')
    for spec in baselines_for_lead_time_backlog(case.lead_time, case.config.backlog.patience, case.n):
        if spec.name not in names:
            continue
        output.append(dict(id=names[spec.name], type='baseline', execution='formula',
                           method_class='SOF',
                           provenance='transferred_policy_family',
                           reference=spec.reference, ast=spec.ast, motifs=dict(spec.motifs),
                           formula=spec.formula, parameter_specs=[asdict(p) for p in spec.parameter_specs]))
    for spec in related_model_baselines(case.lead_time, case.config.backlog.patience, case.n):
        output.append(dict(id=spec.name, type='baseline', execution='formula', method_class='SOF',
                           provenance='published_policy_related_model_transfer', reference=spec.reference,
                           ast=spec.ast, motifs=dict(spec.motifs), formula=spec.formula,
                           parameter_specs=[asdict(p) for p in spec.parameter_specs]))
    from benchmarks.baselines.lead_time_backlog import computational_baselines
    dedicated = {spec.name: spec for spec in computational_baselines(case)}
    output.append(dedicated['gbs'].definition())
    registered = [r for r in frozen['policies'] if r.get('type') == 'registered_baseline']
    if any(r.get('name') != 'codex' for r in registered):
        raise ValueError('Unknown registered formula baseline')
    codex = codex_policy(case)
    if len(registered) != 1 or registered[0]['id'] != codex['id']:
        raise ValueError('Expected exactly one registered Codex baseline')
    output.append(codex)
    output.append(dedicated['myopic'].definition())
    domain = InventoryFamilyDomain.from_cases({context(case): case}, training_case_count=8, cyclic_cap=True)
    # Cyclic replaces only the CO/CBS cap constant, never the state target.
    for policy in output:
        if policy['id'] not in {'constant_order', 'capped_base_stock'}:
            continue
        limit_name = 'Q' if policy['id'] == 'constant_order' else 'Qmax'
        arg = const(float(case.n)) if limit_name == 'Q' else policy['ast']['left']
        policy['ast'] = cap(arg, parameter(limit_name), mode='cyclic_integer')
        policy['parameter_specs'] = [asdict(p) for p in apply_cap_modes(
            [ParameterSpec(**p) for p in policy['parameter_specs']], policy['ast'], policy['motifs'])]
        policy['formula'] = ast_to_formula(policy['ast'])
    for row in frozen['policies']:
        if row.get('type') == 'registered_baseline':
            continue
        family = family_from_json(row['family'], domain=domain)
        ast, motifs = materialize_family_for_case(family, case)
        checked = ast_check(ast, motifs=motifs, allowed_state_names=case.schema.names)
        specs = []
        for p in family.parameters:
            integer = p.kind in {'target', 'cap', 'lag'} and float(p.lower).is_integer() and float(p.upper).is_integer()
            default = (p.lower + p.upper)/2
            specs.append(ParameterSpec(p.name, kind='integer' if integer else 'continuous',
                                       lower=p.lower, upper=p.upper, default=round(default) if integer else default))
        if set(checked.stats.parameters) != {p.name for p in specs}:
            raise ValueError(f'Unused or missing parameter: {row["id"]}')
        specs = apply_cap_modes(specs, checked.ast, motifs)
        output.append(dict(id=row['id'], type='sr', execution='formula', method_class='SOF',
                           provenance='training_only_frozen_selection', ast=ast, motifs=motifs,
                           formula=ast_to_formula(ast), parameter_specs=[asdict(p) for p in specs],
                           source=row.get('selection', {})))
    selected_ids = tuple(row['id'] for row in frozen['policies'] if row.get('family'))
    expected = BASELINES + selected_ids
    if tuple(p['id'] for p in output) != expected:
        raise ValueError('Expected registered baselines and the frozen Stage3 representatives')
    return output



class BatchAuditedMC(ComposedMonteCarloPolicyCostEvaluator):
    """Record real kernel batch sizes, rather than wrapper evaluate_many calls."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_statistics = {'kernel_calls': 0, 'max_candidate_batch': 0,
                                 'multi_candidate_calls': 0, 'candidate_points': 0}

    def _evaluate_canonical_many(self, params_batch, *, pathwise=False):
        stats = self.batch_statistics
        stats['kernel_calls'] += 1
        stats['candidate_points'] += len(params_batch)
        stats['max_candidate_batch'] = max(stats['max_candidate_batch'], len(params_batch))
        stats['multi_candidate_calls'] += int(len(params_batch) > 1)
        return super()._evaluate_canonical_many(params_batch, pathwise=pathwise)


class GBSBatchMC(BatchAuditedMC):
    """Keep canonical batched transitions, override only paper-specific actions."""

    def _prepare_canonical_actions_many(self, states, params_batch):
        return states, params_batch

    def _canonical_actions_many(self, prepared, remaining_horizon):
        states, params = prepared
        return gbs_actions_many(states, params, self.exact_evaluator.case)

    def _evaluate_compiled_many(self, params_batch, *, pathwise=False):
        # The dummy AST is only an evaluator-construction detail. Never use
        # its compiled action map; GBS must preserve ceiling, not AST rint.
        return self._evaluate_canonical_many(params_batch, pathwise=pathwise)


def evaluate(case, policy, settings):
    policy = resolve_case_policy(case, policy)
    started = time.perf_counter()
    banks = {k: seed(case.name, k, settings['seed']) for k in ('proposal', 'strict', 'final')}
    def evaluator(paths, bank_seed):
        cls = GBSBatchMC if policy['execution'] == 'gbs' else BatchAuditedMC
        return cls.from_case(const(0) if policy['execution']=='gbs' else policy['ast'],
            case, settings['horizon'], motifs=policy.get('motifs',{}),
            paths=paths, seed=bank_seed, batch_size=settings['candidate_batch_size'])
    extra = {}
    if policy['execution'] in {'formula', 'gbs'}:
        low = evaluator(settings['proposal_paths'], banks['proposal'])
        high = evaluator(settings['strict_paths'], banks['strict'])
        specs = apply_cap_modes(tuple(ParameterSpec(**p) for p in policy['parameter_specs']),
                                policy.get('ast', const(0)), policy.get('motifs', {}))
        fit = optimize_policy_constants(specs,
            TwoFidelityComposedMCEvaluator(high=high, low=low),
            config=OptimizationConfig(mode=settings['fidelity'], objective='policy_cost',
                total_evaluations=settings['proposal_budget'], action_cap=case.n,
                random_seed=seed(case.name, policy['id'], settings['seed'])))
        final = evaluator(settings['evaluation_paths'], banks['final'])
        costs = final.simulate_path_costs(fit.best_params)
        extra.update(params=dict(fit.best_params), tuning_cost=float(fit.best_cost),
                     optimizer_status=fit.status, nfev=fit.total_evaluations, optimizer_route=fit.route,
                     execution_backend=GBS_BACKEND if policy['execution']=='gbs' else 'composed_canonical_candidate_x_path_batch',
                     batch_statistics={k:e.batch_statistics for k,e in [('proposal',low),('strict',high),('final',final)]})
        if not math.isfinite(fit.best_cost):
            raise ValueError('Invalid fitted cost')
    elif policy['execution'] == 'myopic':
        # Use exactly the formula evaluator's outer bank generator for CRN.
        # This object only supplies demand paths; it never evaluates myopic.
        final = BatchAuditedMC.from_case(const(0), case, settings['horizon'],
            paths=settings['evaluation_paths'], seed=banks['final'])
        planner = Myopic(case, settings, inner_seed=seed(case.name, 'myopic_inner', settings['seed']))
        costs = planner.simulate(final._demands)
        extra.update(params={}, tuning_cost=None, optimizer_status='parameter_free', nfev=0,
                     optimizer_route='parameter_free', execution_backend=BACKEND,
                     inner_bank_seed=planner.inner_seed,
                     batch_statistics={'final':planner.statistics})
    else:
        raise ValueError(f'Unsupported execution {policy["execution"]}; refusing scalar fallback')
    value = float(np.mean(costs))
    if not math.isfinite(value) or value < 0:
        raise ValueError('Invalid MC cost')
    return dict(case=case.name, case_signature=case.case_signature,
                horizon=settings['horizon'],
                L=case.lead_time, K=case.config.backlog.patience, policy=policy['id'],
                type=policy['type'], method_class=policy['method_class'], provenance=policy['provenance'],
                h=case.costs.holding, b=case.costs.shortage, g=case.costs.backlog,
                a=case.costs.abandonment, cost=value, cost_per_period=value/settings['horizon'],
                cost_standard_error=float(np.std(costs, ddof=1)/np.sqrt(len(costs))) if len(costs)>1 else 0.,
                num_parameters=len(policy['parameter_specs']), bank_seeds=banks,
                evaluation_paths=settings['evaluation_paths'], elapsed_seconds=time.perf_counter()-started,
                **extra)


def validate_cached(row, case, policy, settings):
    expected = dict(case=case.name, case_signature=case.case_signature,
                    horizon=settings['horizon'],
                    L=case.lead_time, K=case.config.backlog.patience, policy=policy['id'], type=policy['type'],
                    method_class=policy['method_class'], provenance=policy['provenance'],
                    evaluation_paths=settings['evaluation_paths'],
                    execution_backend=(BACKEND if policy['execution']=='myopic' else
                                       GBS_BACKEND if policy['execution']=='gbs' else 'composed_canonical_candidate_x_path_batch'),
                    bank_seeds={k:seed(case.name,k,settings['seed']) for k in ('proposal','strict','final')})
    if any(row.get(k) != v for k,v in expected.items()):
        raise ValueError(f'Cached result identity mismatch: {case.name}/{policy["id"]}')
    if policy['execution']=='myopic':
        if (row.get('inner_bank_seed') != seed(case.name,'myopic_inner',settings['seed'])
                or row.get('nfev') != 0 or row.get('optimizer_route') != 'parameter_free'):
            raise ValueError('Cached myopic bank or execution identity mismatch')
    if not math.isfinite(row['cost']) or row['cost'] < 0 or set(row['params']) != {p['name'] for p in policy['parameter_specs']}:
        raise ValueError('Cached cost or parameter membership invalid')
    for p in policy['parameter_specs']:
        v = row['params'][p['name']]
        if not math.isfinite(v) or not p['lower'] <= v <= p['upper'] or (p['kind']=='integer' and v != round(v)):
            raise ValueError('Cached parameter outside frozen domain')
