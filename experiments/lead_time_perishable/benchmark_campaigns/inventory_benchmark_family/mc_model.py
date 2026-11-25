"""lead_time_perishable: case/policy resolution and setting-specific MC numerics."""
from runtime.artifacts import seed_from_text, stable_key
import json
import re
import math
import time
import numpy as np
from dataclasses import asdict
from benchmarks.baselines.lead_time_perishable.catalog import computational_baselines, baseline_specs
from benchmarks.inventory_benchmark_family.suites import StructuralContext, cases as suite_cases
from inventory.policy.ast import add, sub, mul, minimum, maximum, state, dist, motif, ast_check, ast_to_formula, cap, const, parameter
from inventory.policy.family import materialize_family_for_case
from inventory.policy.family_domain import InventoryFamilyDomain
from optimization import ParameterSpec, OptimizationConfig, optimize_policy_constants
from optimization.policies.parameters import apply_cap_modes
from optimization.policies.baselines.lead_time_perishable import bsp_ws3
from sr.family.proposals import family_from_json
from benchmarks.baselines.lead_time_perishable.batched import BatchedLeadTimePerishableMC, BatchedLeadTimePerishableProvider
from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator, TwoFidelityComposedMCEvaluator

SETTING = "lead_time_perishable"

BASELINES = ('constant_order', 'base_stock', 'copsq', 'capped_base_stock',
             'bsp_low', 'weighted_base_stock', 'bsp_2d', 'bsp_oir', 'ewa', 'bsp_low_ew', 'pil',
             'apil', 'sqmax_ew', 'bsp_ws3', 'codex-astra-capped-survival')


SR_IDS = ('ETIS', 'SCBS')


OPTIONAL_SR_IDS = ()


PROTOCOL = 'ltp-mc-final-9contexts-stage3-100-cyclic-cap'


# Immutable numerical seed salt; preserve it across file and display renames.
BANK_PROTOCOL = 'ltp-mc-final-9contexts-50rounds'


def codex_policy(case=None, *, lead_time=0, shelf_life=0, patience=0, mean=None):
    """Frozen external formula; only coefficients are fitted for each case."""
    lead = case.lead_time if case is not None else lead_time
    shelf = case.shelf_life if case is not None else shelf_life
    if case is not None:
        mean = sum(float(v) * float(p) for v, p in zip(case.demand.values, case.demand.probs))
    def total(nodes):
        return add(*nodes) if len(nodes) > 1 else nodes[0] if nodes else const(0)
    inventory = total([state(f'a{i}') for i in range(1, shelf + 1)])
    pipeline = total([state(f'p{i}') for i in range(1, lead + 1)])
    if not 1 <= shelf <= 10:
        raise ValueError("Codex fixed SOF template requires 1 <= shelf_life <= 10")
    # Fixed ten-level template; selectors beyond M are zero, never new parameters.
    excess = const(0)
    for j in range(10, 0, -1):
        bucket = state(f'a{j}') if j <= shelf else const(0)
        excess = maximum(const(0), add(sub(bucket, dist('mu')), excess))
    motifs = {'astra_expiry_excess': excess}
    ast = minimum(parameter('c'), add(sub(sub(parameter('a'), inventory), pipeline),
                                    mul(parameter('b'), motif('astra_expiry_excess'))))
    allowed = {f'a{i}' for i in range(1, shelf + 1)} if shelf else {'x'}
    allowed.update(f'p{i}' for i in range(1, lead + 1))
    checked = ast_check(ast, motifs=motifs, allowed_state_names=frozenset(allowed))
    upper = 500. if mean is None else min(500., 200. + mean * (lead + 1))
    specs = [ParameterSpec('a', kind='continuous', lower=0, upper=upper, default=upper / 2),
             ParameterSpec('b', kind='continuous', lower=-1.5, upper=1.5, default=0)]
    specs.append(ParameterSpec('c', kind='continuous', lower=0, upper=80, default=40))
    if set(checked.stats.parameters) != {p.name for p in specs}:
        raise ValueError('Frozen parameter membership mismatch')
    return dict(id='codex-astra-capped-survival', type='baseline', execution='formula', method_class='SOF',
                provenance='AI-generated baseline; frozen Codex source formulas', ast=checked.ast,
                motifs=motifs, formula=ast_to_formula(checked.ast),
                parameter_specs=[asdict(p) for p in specs], source='codex-astra')


def validate_campaign(settings, frozen, sr_ids, *, baseline_only):
    from benchmarks.inventory_benchmark_family.mc_protocol import validate_joint_selection
    validate_joint_selection(
        settings, frozen, sr_ids, baseline_ids=BASELINES,
        reviewed_rosters=(SR_IDS, SR_IDS + OPTIONAL_SR_IDS), baseline_only=baseline_only)


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
    # Preserve optimizer streams when changing public policy names.
    bank = {'ETIS': 'cyclic-ETIS-S', 'SCBS': 'cyclic-SCBS'}.get(bank, bank)
    return seed_from_text(f'{BANK_PROTOCOL}|{global_seed}|{case}|{bank}'.encode())


def cases(settings):
    contexts = [StructuralContext(f'L{l}_M{m}', lead_time=l, shelf_life=m) for l, m in settings['contexts']]
    return suite_cases('lead_time_perishable', contexts=contexts, n=settings['state_action_cap'])


def resolve(case, frozen):
    output = []
    for spec in (*baseline_specs(case), bsp_ws3(case.lead_time, case.shelf_life, case.n)):
        if spec.name not in BASELINES:
            continue
        output.append({'id': spec.name, 'type': 'baseline', 'execution': 'formula',
                       'ast': spec.ast, 'motifs': dict(spec.motifs), 'formula': spec.formula,
                       'parameter_specs': [asdict(p) for p in spec.parameter_specs]})
    for spec in computational_baselines(case.lead_time, case.shelf_life, case.n):
        output.append({'id': spec.name, 'type': 'baseline', 'execution': 'computational',
                       'formula': spec.formula, 'implementation_version': spec.implementation_version,
                       'parameter_specs': [asdict(p) for p in spec.parameter_specs]})
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
    registered = [r for r in frozen['policies'] if r.get('type') == 'registered_baseline']
    if any(r.get('name') != 'codex' for r in registered):
        raise ValueError('Unknown registered formula baseline')
    codex = codex_policy(case)
    if len(registered) != 1 or registered[0]['id'] != codex['id']:
        raise ValueError('Expected exactly one registered Codex baseline')
    output.append(codex)
    for row in frozen['policies']:
        if row.get('type') == 'registered_baseline':
            continue
        domain = InventoryFamilyDomain.from_cases({f'L{case.lead_time}_M{case.shelf_life}': case}, training_case_count=8, cyclic_cap=True)
        family = family_from_json(row['family'], domain=domain)
        ast, motifs = materialize_family_for_case(family, case)
        check = ast_check(ast, motifs=motifs, allowed_state_names=case.schema.allowed_state_names)
        specs = []
        for p in family.parameters:
            integer = p.kind in {'target', 'cap', 'lag'} and float(p.lower).is_integer() and float(p.upper).is_integer()
            default = (p.lower + p.upper) / 2
            specs.append(ParameterSpec(p.name, kind='integer' if integer else 'continuous',
                                       lower=p.lower, upper=p.upper, default=round(default) if integer else default))
        if 'parameter_specs' in row:
            specs = [ParameterSpec(**p) for p in row['parameter_specs']]
        if set(check.stats.parameters) != {p.name for p in specs}:
            raise ValueError(f'Unused or missing family parameter: {row["id"]}')
        specs = apply_cap_modes(specs, check.ast, motifs)
        output.append({'id': row['id'], 'type': 'sr', 'execution': 'formula',
                       'ast': ast, 'motifs': motifs, 'formula': ast_to_formula(ast),
                       'parameter_specs': [asdict(p) for p in specs], 'source': row.get('selection', {})})
    selected_ids = tuple(r['id'] for r in frozen['policies'] if r.get('type') != 'registered_baseline')
    if frozen.get('selection_kind') != 'search_candidates' and any(p not in SR_IDS + OPTIONAL_SR_IDS for p in selected_ids):
        raise ValueError('Unregistered SR policy')
    expected = BASELINES + selected_ids
    if {r['id'] for r in output} != set(expected) or len(output) != len(expected):
        raise ValueError('Policy membership must match the registered baselines and SR families')
    return output



SEARCH_KEYS = ('fidelity', 'proposal_budget', 'proposal_paths', 'strict_paths')
COMPUTATIONAL_IDS = ('ewa', 'bsp_low_ew', 'pil')


def validate_search_overrides(settings):
    overrides = settings.get('computational_search_overrides', {})
    if not isinstance(overrides, dict):
        raise ValueError('computational_search_overrides must be an object')
    for context, values in overrides.items():
        if not re.fullmatch(r'L(?:3|5|10)_M(?:3|5|10)', context):
            raise ValueError(f'Invalid search override context: {context}')
        if not isinstance(values, dict) or not values or set(values) - set(SEARCH_KEYS):
            raise ValueError(f'Invalid search override keys: {context}')
        for key, value in values.items():
            if key == 'fidelity':
                if value not in ('fast', 'standard', 'certify'):
                    raise ValueError('Invalid search override fidelity')
            elif type(value) is not int or value <= 0:
                raise ValueError(f'Search override {key} must be a positive integer')

def effective_search_settings(case, policy, settings):
    effective = {key: settings[key] for key in SEARCH_KEYS}
    if policy['id'] in COMPUTATIONAL_IDS and policy['execution'] == 'computational':
        context = f'L{case.lead_time}_M{case.shelf_life}'
        override = settings.get('computational_search_overrides', {}).get(context, {})
        for key, value in override.items():
            # Tiny smoke/custom settings must never be enlarged by production overrides.
            effective[key] = value if key == 'fidelity' or effective[key] is None else min(effective[key], value)
    return effective


class CampaignTwoFidelityMC(TwoFidelityComposedMCEvaluator):
    def uses_special_initial_anchors(self, specs):
        """Avoid expensive redundant center/upper anchors for BSP search.

        The BSP implementation has a wide three-integer domain.  Its default
        and lower-bound anchors are sufficient to seed the low-fidelity
        proposal search; center/upper anchors can force a very large exact
        projection before proposal screening begins.  Other evaluators keep
        the shared controller's existing behavior.
        """
        names = tuple(getattr(spec, 'name', '') for spec in specs)
        if names == ('S1', 'S2', 'tau'):
            return True
        high_hook = getattr(self.high, 'uses_special_initial_anchors', None)
        return bool(high_hook(specs)) if callable(high_hook) else False

    @property
    def optimization_route_hint(self):
        high = getattr(self.high, 'optimization_route_hint', None)
        low = getattr(self.low, 'optimization_route_hint', None)
        if high != low:
            raise ValueError('Inconsistent two-fidelity optimization routes')
        return high


def evaluate(case, policy, settings):
    policy = resolve_case_policy(case, policy)
    started = time.perf_counter()
    validate_search_overrides(settings)
    search = effective_search_settings(case, policy, settings)
    if policy['execution'] == 'computational':
        spec = next(s for s in computational_baselines(case.lead_time, case.shelf_life, case.n)
                    if s.name == policy['id'])
        provider = BatchedLeadTimePerishableProvider(case, spec.name)
        def evaluator(paths, bank_seed):
            return BatchedLeadTimePerishableMC(case, settings['horizon'], provider,
                                              paths=paths, seed=bank_seed)
    else:
        def evaluator(paths, bank_seed):
            return ComposedMonteCarloPolicyCostEvaluator.from_case(
                policy['ast'], case, settings['horizon'], motifs=policy['motifs'],
                paths=paths, seed=bank_seed, batch_size=64)
    banks = {name: seed(case.name, name, settings['seed']) for name in ('proposal', 'strict', 'final')}
    low = evaluator(search['proposal_paths'], banks['proposal'])
    high = evaluator(search['strict_paths'], banks['strict'])
    specs = tuple(ParameterSpec(**p) for p in policy['parameter_specs'])
    if policy['execution'] != 'computational':
        specs = apply_cap_modes(specs, policy['ast'], policy['motifs'])
    fit = optimize_policy_constants(specs,
        CampaignTwoFidelityMC(high=high, low=low),
        config=OptimizationConfig(mode=search['fidelity'], objective='policy_cost',
            total_evaluations=search['proposal_budget'], action_cap=case.n,
            random_seed=seed(case.name, policy['id'], settings['seed'])))
    final = evaluator(settings['evaluation_paths'], banks['final'])
    path_costs = final.simulate_path_costs(fit.best_params)
    value = float(np.mean(path_costs))
    if not math.isfinite(value) or value < 0 or not math.isfinite(fit.best_cost):
        raise ValueError('Nonfinite or negative MC cost')
    return {'case': case.name, 'case_signature': case.case_signature,
            'horizon': settings['horizon'],
            'L': case.lead_time, 'M': case.shelf_life, 'policy': policy['id'], 'type': policy['type'],
            'cost': value, 'cost_per_period': value / settings['horizon'],
            'cost_standard_error': float(np.std(path_costs, ddof=1) / np.sqrt(len(path_costs))) if len(path_costs) > 1 else 0.0,
            'params': dict(fit.best_params), 'num_parameters': len(policy['parameter_specs']),
            'tuning_cost': float(fit.best_cost), 'optimizer_status': fit.status,
            'nfev': fit.total_evaluations, 'optimizer_route': fit.route, 'bank_seeds': banks,
            'search_settings': search,
            'execution_backend': ('ltp_mean_demand_batched_mc' if policy['id'] in {'apil', 'sqmax_ew'} else
                                  'ltp_exact_projection_batched_mc')
                if policy['execution'] == 'computational' else 'composed_canonical_batched_mc',
            'batch_statistics': {'proposal': low.stats, 'strict': high.stats,
                                 'final': final.stats, 'projection': provider.projector.stats,
                                 'certified_inventory_error_bound': provider.projection_error,
                                 'complete_projection_rechecks': provider.rounding_rechecks,
                                 'complete_projection': provider._rounding_projector.stats
                                     if provider._rounding_projector is not None else {}}
                if policy['execution'] == 'computational' else {},
            'evaluation_paths': settings['evaluation_paths'], 'elapsed_seconds': time.perf_counter() - started}


def validate_cached(row, case, policy, settings):
    expected = {'case': case.name, 'case_signature': case.case_signature,
                'horizon': settings['horizon'],
                'policy': policy['id'], 'type': policy['type'], 'L': case.lead_time, 'M': case.shelf_life,
                'evaluation_paths': settings['evaluation_paths'],
                'bank_seeds': {k: seed(case.name, k, settings['seed']) for k in ('proposal', 'strict', 'final')}}
    if any(row.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Cached result identity mismatch: {case.name}/{policy["id"]}')
    if ('search_settings' in row
            and row['search_settings'] != row.get('imported_search_settings', effective_search_settings(case, policy, settings))):
        raise ValueError('Cached result search settings mismatch')
    if (not math.isfinite(row['cost']) or row['cost'] < 0
            or set(row['params']) != {p['name'] for p in policy['parameter_specs']}):
        raise ValueError('Cached result cost/parameters invalid')
    for p in policy['parameter_specs']:
        value = row['params'][p['name']]
        if (not math.isfinite(value) or not p['lower'] <= value <= p['upper']
                or (p['kind'] == 'integer' and value != round(value))):
            raise ValueError('Cached parameter outside its frozen domain')
