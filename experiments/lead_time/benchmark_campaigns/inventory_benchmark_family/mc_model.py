"""Lead-time catalog and rolling MC numerics; no campaign workflow."""

import numpy as np
from dataclasses import asdict, replace

from benchmarks.baselines.lead_time import ComputationalActionAdapter, resolve_benchmark_policy
from benchmarks.baselines.lead_time.policy_sets import ComputationalPolicyDefinition, ResolvedComputationalPolicy
from benchmarks.inventory_benchmark_family.mc_protocol import MCModel, formula_definition
from benchmarks.inventory_benchmark_family.mc_runtime import MCPolicy
from benchmarks.inventory_benchmark_family.mc_protocol import resolve_formula_policy
from inventory.policy.ast import add, sub, mul, minimum, state, ast_check, ast_to_formula, cap, const, parameter
from optimization.models.types import ParameterSpec
from optimization.backends.lead_time.mc_evaluator import MonteCarloPolicyCostEvaluator
from optimization.backends.lead_time.mc_adapters import ScaledMCEvaluator, TwoFidelityMCEvaluator
from optimization.backends.lead_time.rolling import rolling_reference
from optimization.policies.parameters import apply_cap_modes


def codex_policy(case=None, *, lead_time=0, shelf_life=0, patience=0, mean=None):
    """Frozen external formula; only coefficients are fitted for each case."""
    lead = case.lead_time if case is not None else lead_time
    if case is not None:
        mean = sum(float(v) * float(p) for v, p in zip(case.demand.values, case.demand.probs))
    def total(nodes):
        return add(*nodes) if len(nodes) > 1 else nodes[0] if nodes else const(0)
    inventory = state('x')
    pipeline = total([state(f'p{i}') for i in range(1, lead + 1)])
    motifs = {}
    ast = minimum(parameter('c'), sub(sub(parameter('a'), inventory),
                                    mul(sub(const(1), parameter('b')), pipeline)))
    allowed = {'x', *(f'p{i}' for i in range(1, lead + 1))}
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


def resolve_policy(definition, context, n, *, case=None):
    if definition.type == "formula_family":
        from benchmarks.inventory_benchmark_family.mc_protocol import resolve_family_policy
        return resolve_family_policy(definition, "lead_time", context, n, case=case)
    if definition.type == "registered_baseline" and definition.registered_name == "codex":
        spec = codex_policy(case, lead_time=context)
        return MCPolicy(definition.id, definition.type, spec['ast'],
                        tuple(ParameterSpec(**p) for p in spec['parameter_specs']),
                        spec['motifs'], spec['formula'])
    if definition.type == "registered_baseline" and definition.registered_name == "co":
        return MCPolicy(definition.id, definition.type,
                        cap(const(float(n)), parameter("rate"), "cyclic_integer"),
                        (ParameterSpec("rate", kind="continuous", lower=0, upper=n, default=n/3),),
                        {}, "min(n, round(rate*(t+1))-round(rate*t))")
    if definition.type == "registered_baseline" and definition.registered_name == "cbs":
        original = resolve_policy(replace(definition, registered_name="capped_base_stock"), context, n)
        limit = "C"
        node = original.ast
        branch = node["right"] if node["left"].get("name") == limit else node["left"]
        return replace(original, ast=cap(branch, parameter(limit), "cyclic_integer"),
                       parameter_specs=tuple(replace(p, kind="continuous") if p.name == limit else p for p in original.parameter_specs),
                       formula=f"cyclic CBS: round({limit}*(t+1))-round({limit}*t)")
    if definition.type == "formula_ast":
        if definition.id.startswith("sr_d5a758e4"):
            from optimization.policies.parameters import parameter_spec
            mean = case.demand.mean if case is not None else float(n)
            declarations = definition.family["parameters"]
            specs = tuple(
                ParameterSpec(p["name"], kind="continuous", lower=0,
                              upper=(context + 1) * n + 2 * mean, default=n)
                if p["name"].startswith("S1") else
                ParameterSpec(p["name"], kind="continuous", lower=p["lower"],
                              upper=p["upper"], default=p["lower"])
                if p["kind"] in {"scale", "weight"} else
                parameter_spec(p["name"], context, n)
                for p in declarations)
            definition = replace(definition, parameter_specs=specs)
        return resolve_formula_policy(definition, "lead_time", context, n)
    if definition.type == "registered_baseline" and definition.registered_name == "constant_order":
        from benchmarks.baselines.cyclic_co import resolve_cyclic
        return resolve_cyclic(definition.id, "lead_time", n)
    row = ComputationalPolicyDefinition(definition.id, definition.type, definition.registered_name) if definition.type == "computational_baseline" else formula_definition(definition)
    policy = resolve_benchmark_policy(row, context, n)
    spec = policy.computational_spec if isinstance(policy, ResolvedComputationalPolicy) else None
    specs = definition.parameter_specs if getattr(definition, 'parameter_specs', None) is not None else policy.parameter_specs
    if {p.name for p in specs} != {p.name for p in policy.parameter_specs}:
        raise ValueError("frozen parameter domains do not match the AST")
    return MCPolicy(policy.id, policy.type, policy.ast, apply_cap_modes(specs, policy.ast, policy.motifs), policy.motifs, policy.formula, spec)


class RollingMC:
    def __init__(self, case, config):
        self.case = case
        self.reference = rolling_reference(case)
        self.config = config
        self.adapters = {}

    def evaluator(self, policy, bank):
        if policy.execution:
            from benchmarks.baselines.cyclic_co import CyclicLeadTimeMC, schedule_for
            return CyclicLeadTimeMC(policy.ast, self.reference, bank, policy.motifs,
                                   horizon=bank.horizon, schedule=schedule_for(policy),
                                   use_exact_anchor_control_variate=False,
                                   use_common_random_numbers=True, collect_pathwise_costs=False,
                                   batch_size=int(self.config["candidate_batch_size"]))
        if policy.id not in self.adapters:
            self.adapters[policy.id] = (ComputationalActionAdapter(policy.computational_spec.make_provider(self.reference), self.reference) if policy.computational_spec is not None else None)
        return MonteCarloPolicyCostEvaluator(policy.ast, self.reference, bank, policy.motifs, horizon=bank.horizon, action_evaluator=self.adapters[policy.id], use_exact_anchor_control_variate=False, use_common_random_numbers=True, collect_pathwise_costs=False, batch_size=int(self.config["candidate_batch_size"]))

    def fitting_evaluator(self, high, low, scale):
        return TwoFidelityMCEvaluator(high, ScaledMCEvaluator(low, scale))

    def path_costs(self, evaluator, params, bank):
        return evaluator.simulate_path_costs(params, bank=bank)

    def bank_arrays(self, bank):
        return {"demand": bank.demand_indices, "policy_uniforms": np.asarray(bank.policy_uniforms)}

    def policy_metadata(self, policy):
        if policy.computational_spec is None:
            return None
        spec = policy.computational_spec
        return {name: getattr(spec, name) for name in ("paper_title", "authors", "year", "reference_url", "implementation_version")}


MODEL = MCModel("lead_time", "lead_time", "L", "20260827", "standard.json", "mc_final.json", resolve_policy, RollingMC, archived_suites=("inventory_benchmark_50x4_mc_best_baseline",), randomized_policies=True)
