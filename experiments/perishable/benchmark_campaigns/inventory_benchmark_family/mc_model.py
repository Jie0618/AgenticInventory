"""Perishable policy resolution and canonical transition MC numerics."""

from __future__ import annotations

from dataclasses import asdict, replace

from benchmarks.baselines.perishable import computational_baselines
from benchmarks.inventory_benchmark_family.mc_protocol import MCModel
from benchmarks.inventory_benchmark_family.mc_runtime import MCPolicy
from benchmarks.inventory_benchmark_family.mc_protocol import resolve_formula_policy
from inventory.policy.ast import add, sub, mul, maximum, state, dist, ast_check, ast_to_formula, cap, const, parameter
from optimization.models.types import ParameterSpec
from optimization.evaluators.composed import ComposedMonteCarloPolicyCostEvaluator, ComputationalMonteCarloPolicyCostEvaluator, TwoFidelityComposedMCEvaluator
from optimization.policies.baselines.perishable import baselines_for_perishable


def codex_policy(case=None, *, lead_time=0, shelf_life=0, patience=0, mean=None):
    """Frozen external formula; only coefficients are fitted for each case."""
    lead = case.lead_time if case is not None else lead_time
    shelf = case.shelf_life if case is not None else shelf_life
    if case is not None:
        mean = sum(float(v) * float(p) for v, p in zip(case.demand.values, case.demand.probs))
    def total(nodes):
        return add(*nodes) if len(nodes) > 1 else nodes[0] if nodes else const(0)
    inventory = total([state(f'a{i}') for i in range(1, shelf + 1)])
    motifs = {}
    ast = add(sub(parameter('a'), inventory),
              mul(parameter('b'), maximum(sub(dist('mu'), inventory), const(0))))
    allowed = {f'a{i}' for i in range(1, shelf + 1)} if shelf else {'x'}
    allowed.update(f'p{i}' for i in range(1, lead + 1))
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


def resolve_policy(definition, context, n, *, case=None):
    if definition.type == "formula_family":
        from benchmarks.inventory_benchmark_family.mc_protocol import resolve_family_policy
        return resolve_family_policy(definition, "perishable", context, n, case=case)
    if definition.type == "registered_baseline" and definition.registered_name == "codex":
        spec = codex_policy(case, shelf_life=context)
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
        limit = "Qmax"
        node = original.ast
        branch = node["right"] if node["left"].get("name") == limit else node["left"]
        return replace(original, ast=cap(branch, parameter(limit), "cyclic_integer"),
                       parameter_specs=tuple(replace(p, kind="continuous") if p.name == limit else p for p in original.parameter_specs),
                       formula=f"cyclic CBS: round({limit}*(t+1))-round({limit}*t)")
    if definition.type == "formula_ast":
        if definition.id == "EABS":
            specs = tuple(ParameterSpec(p["name"], kind="continuous", lower=p["lower"],
                                        upper=1.5 * (context * n + 1) if p["kind"] == "target" else p["upper"],
                                        default=p["lower"])
                          for p in definition.family["parameters"])
            definition = replace(definition, parameter_specs=specs)
        elif definition.id.startswith("sr_3d0308f9"):
            sigma = case.demand.std if case is not None else float(n)
            specs = tuple(ParameterSpec(p["name"], kind="continuous", lower=p["lower"],
                                        upper=(context * n + 2 * sigma if p["kind"] == "target" else
                                               n + 2 * sigma if p["kind"] == "cap" else p["upper"]),
                                        default=n if p["kind"] == "target" else max(1, n // 3) if p["kind"] == "cap" else 0.)
                          for p in definition.family["parameters"])
            definition = replace(definition, parameter_specs=specs)
        return resolve_formula_policy(definition, "perishable", context, n)
    if definition.type == "registered_baseline" and definition.registered_name == "constant_order":
        from benchmarks.baselines.cyclic_co import resolve_cyclic
        return resolve_cyclic(definition.id, "perishable", n)
    if definition.type == "computational_baseline":
        spec = next(row for row in computational_baselines() if row.name == definition.registered_name)
        return MCPolicy(definition.id, definition.type, {"op": "const", "value": 0.0}, spec.parameter_specs, {}, spec.display_name, spec)
    if definition.type == "registered_baseline":
        spec = next(row for row in baselines_for_perishable(context, n, include_extended=True) if row.name == definition.registered_name)
        return MCPolicy(definition.id, definition.type, spec.ast, spec.parameter_specs, dict(spec.motifs), spec.formula)
    raise ValueError(f"unsupported policy type: {definition.type}")


class PerishableMC:
    def __init__(self, case, config):
        self.case, self.config = case, config

    def evaluator(self, policy, bank):
        if policy.execution:
            from benchmarks.baselines.cyclic_co import CyclicPerishableMC, schedule_for
            return CyclicPerishableMC(self.case, bank, schedule_for(policy),
                                      batch_size=int(self.config["candidate_batch_size"]))
        if policy.computational_spec is not None:
            evaluator = ComputationalMonteCarloPolicyCostEvaluator(self.case, bank.horizon, policy.computational_spec.make_provider(self.case), paths=bank.num_paths, seed=bank.seed, require_batch=True)
        else:
            evaluator = ComposedMonteCarloPolicyCostEvaluator.from_case(policy.ast, self.case, bank.horizon, motifs=policy.motifs, paths=bank.num_paths, seed=bank.seed, batch_size=int(self.config["candidate_batch_size"]))
        evaluator.bind_demand_paths(bank.demand_values)
        if policy.computational_spec is not None and bank.policy_uniforms is not None:
            evaluator.bind_policy_uniforms(bank.policy_uniforms)
        return evaluator

    def fitting_evaluator(self, high, low, scale):
        if scale != 1:
            raise ValueError("perishable MC uses the full proposal horizon")
        return TwoFidelityComposedMCEvaluator(high, low)

    def path_costs(self, evaluator, params, bank):
        return evaluator.simulate_path_costs(params)

    def bank_arrays(self, bank):
        arrays = {"demand": bank.demand_values}
        if bank.policy_uniforms is not None:
            arrays["policy_uniforms"] = bank.policy_uniforms
        return arrays

    def policy_metadata(self, policy):
        spec = policy.computational_spec
        return None if spec is None else {name: getattr(spec, name) for name in ("paper", "doi", "implementation_version", "implementation_note")}


def result_is_current(row):
    versions = {spec.implementation_version for spec in computational_baselines()}
    for policy in row.get("policies", ()):
        if policy.get("policy_type") == "computational_baseline":
            if policy.get("computational_baseline", {}).get("implementation_version") not in versions:
                return False
    return True


# This seed salt is numerical protocol data, not a display/version label.
# Keep its original bytes so fitted and evaluation demand banks remain reusable.
MODEL = MCModel("perishable", "shelf_life", "M", "perishable-mc-final", "standard.json", "mc_final.json", resolve_policy, PerishableMC, fixed_computational=True, randomized_policies=True, result_is_current=result_is_current, extra_case_fields=("w", "case_signature"))
