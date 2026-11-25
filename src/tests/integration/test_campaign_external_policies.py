"""External campaign formulas keep fixed structures and case-specific domains."""
import unittest
import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from inventory.policy.ast import evaluate_ast
from experiments.lead_time_perishable.benchmark_campaigns.inventory_benchmark_family.mc_model import codex_policy


@pytest.mark.parametrize("setting", ("lead_time", "perishable", "lead_time_backlog", "lead_time_perishable"))
def test_external_manifest_preserves_parameter_domains(setting):
    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model")
    manifest = Path(model.__file__).parent / "policies/sets/mc_final.json"
    external, = [row for row in json.loads(manifest.read_text())["policies"] if row.get("external")]
    lead = 0 if setting == "perishable" else 3
    for mean in (0, 0.5, 2.75, 200):
        policy = model.codex_policy(lead_time=lead, shelf_life=3, patience=3, mean=mean)
        assert policy["id"] == external["id"]
        assert {p["name"] for p in policy["parameter_specs"]} == set(external["parameter_domains"])
        for parameter in policy["parameter_specs"]:
            domain = external["parameter_domains"][parameter["name"]]
            assert parameter["kind"] == domain["kind"]
            assert parameter["lower"] == domain["lower"]
            assert parameter["upper"] == (min(500, 200 + mean * (lead + 1))
                                           if parameter["name"] == "a" else domain["upper"])


@pytest.mark.parametrize("setting", ("lead_time_backlog", "lead_time_perishable"))
def test_external_resolution_rejects_wrong_policy_identity(setting):
    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model")
    settings = json.loads((Path(model.__file__).parent / "configs/standard.json").read_text())
    case = next(iter(model.cases(settings)))
    policy = model.codex_policy(case)
    assert model.resolve_case_policy(case, policy) == policy
    with pytest.raises(ValueError, match="policy ID does not match"):
        model.resolve_case_policy(case, dict(policy, id="wrong-setting"))


def test_ltp_certify_search_budget_is_shared_by_every_method_and_context():
    from benchmarks.inventory_benchmark_family.mc_protocol import load_campaign_config
    from experiments.lead_time_perishable.benchmark_campaigns.inventory_benchmark_family import mc_model
    from types import SimpleNamespace

    settings = load_campaign_config(Path(mc_model.__file__).parent / "configs/standard.json")
    expected = dict(fidelity="certify", proposal_budget=None,
                    proposal_paths=128, strict_paths=1000)
    assert not settings.get("computational_search_overrides")
    assert settings["evaluation_paths"] == 8000
    for lead, shelf in settings["contexts"]:
        case = SimpleNamespace(lead_time=lead, shelf_life=shelf)
        for method in mc_model.BASELINES + mc_model.SR_IDS:
            policy = dict(id=method, execution="computational" if method in mc_model.COMPUTATIONAL_IDS else "formula")
            assert mc_model.effective_search_settings(case, policy, settings) == expected


class TestCodexFixedTemplate(unittest.TestCase):
    def test_fixed_template_matches_original_prefix_excess(self):
        rng = np.random.default_rng(20261121)
        for shelf in (3, 5, 10):
            policy = codex_policy(lead_time=3, shelf_life=shelf, mean=0.5)
            self.assertEqual(len(policy['parameter_specs']), 3)
            for mean in (0.0, 0.5, 2.75):
                for _ in range(30):
                    buckets = rng.integers(0, 21, size=shelf)
                    state = {f'a{j}': int(v) for j, v in enumerate(buckets, 1)}
                    pipeline = rng.integers(0, 21, size=3)
                    state.update({f'p{j}': int(v) for j, v in enumerate(pipeline, 1)})
                    params = dict(a=float(rng.uniform(0, 200)),
                                  b=float(rng.uniform(-1.5, 1.5)),
                                  c=float(rng.uniform(0, 80)))
                    excess = max(0.0, float(np.max(
                        np.cumsum(buckets) - mean * np.arange(1, shelf + 1))))
                    actual_excess = evaluate_ast(
                        policy['motifs']['astra_expiry_excess'], state, {},
                        dist_features={'mu': mean})
                    self.assertAlmostEqual(actual_excess, excess, places=12)
                    actual = evaluate_ast(policy['ast'], state, params,
                                          dist_features={'mu': mean}, motifs=policy['motifs'])
                    expected = min(params['c'], params['a'] - sum(buckets)
                                   - sum(pipeline) + params['b'] * excess)
                    self.assertEqual(np.clip(np.rint(actual), 0, 20),
                                     np.clip(np.rint(expected), 0, 20))

    def test_ten_fixed_linear_selectors_and_scope_guard(self):
        def levels(node):
            if not isinstance(node, dict):
                return 0
            return int(node.get('op') == 'maximum') + sum(
                levels(child) for value in node.values()
                for child in (value if isinstance(value, list) else [value]))
        for shelf in (3, 5, 10):
            policy = codex_policy(lead_time=3, shelf_life=shelf)
            self.assertEqual(levels(policy['motifs']['astra_expiry_excess']), 10)
        for shelf in (0, 11):
            with self.assertRaisesRegex(ValueError, 'fixed SOF template'):
                codex_policy(lead_time=3, shelf_life=shelf)


@pytest.mark.parametrize("setting", ("lead_time", "perishable", "lead_time_backlog"))
def test_frozen_external_formulas_match_source_actions(setting):
    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model")
    rng = np.random.default_rng(20261122)
    mean = 2.75
    for size in (2, 3, 5):
        policy = model.codex_policy(lead_time=size if setting != "perishable" else 0,
                                   shelf_life=size, patience=size, mean=mean)
        for _ in range(30):
            state = {f"p{j}": int(rng.integers(0, 21)) for j in range(1, size + 1)}
            state.update({f"b{j}": int(rng.integers(0, 21)) for j in range(1, size + 1)})
            state.update({f"a{j}": int(rng.integers(0, 21)) for j in range(1, size + 1)})
            state["x"] = int(rng.integers(0, 21))
            params = dict(a=float(rng.uniform(0, 200)), b=float(rng.uniform(-1.5, 1.5)),
                          c=float(rng.uniform(0, 80)))
            inventory = state["x"]
            pipeline = sum(state[f"p{j}"] for j in range(1, size + 1))
            backlog = sum(state[f"b{j}"] for j in range(1, size + 1))
            if setting == "lead_time":
                expected = min(params["c"], params["a"] - inventory - (1 - params["b"]) * pipeline)
            elif setting == "perishable":
                inventory = sum(state[f"a{j}"] for j in range(1, size + 1))
                expected = params["a"] - inventory + params["b"] * max(mean - inventory, 0)
            else:
                expected = (params["a"] - inventory - pipeline + backlog + params["b"]
                            * max(mean + backlog - inventory - state["p1"], 0))
            actual = evaluate_ast(policy["ast"], state, params, dist_features={"mu": mean},
                                  motifs=policy["motifs"])
            assert actual == pytest.approx(expected)
            assert np.clip(np.rint(actual), 0, 20) == np.clip(np.rint(expected), 0, 20)


@pytest.mark.parametrize("setting", ("lead_time", "perishable"))
def test_baselines_use_cyclic_caps_and_current_policy_ids_for_seeds(setting):
    from benchmarks.inventory_benchmark_family.mc_protocol import load_policy_set
    from inventory.policy.ast import has_cyclic_cap
    from runtime.artifacts import seed_from_text

    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model").MODEL
    _, definitions = load_policy_set(model.data_dir / "policies/sets/mc_final.json")
    for name in ("co", "cbs"):
        definition, = [row for row in definitions if row.id == name]
        policy = model.resolve_policy(definition, 3, 20)
        assert has_cyclic_cap(policy.ast, policy.motifs)
        assert model.seed("fit", "example", name) == seed_from_text(
            model.seed_namespace + "|fit|example|" + name)


@pytest.mark.parametrize("setting,prefix", (
    ("lead_time", "sr_d5a758e4"),
    ("perishable", "EABS"),
    ("perishable", "sr_3d0308f9"),
))
@pytest.mark.parametrize("context", (2, 3, 15))
def test_pareto_families_fit_the_paper_expression(setting, prefix, context):
    from benchmarks.inventory_benchmark_family.mc_protocol import load_policy_set
    from inventory.policy.ast import compile_policy

    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model").MODEL
    config = json.loads((model.data_dir / "configs/standard.json").read_text())
    case = model.cases(config, contexts=(context,))[0]
    _, definitions = load_policy_set(model.data_dir / "policies/sets/mc_final.json")
    definition, = [d for d in definitions if d.id.startswith(prefix)]
    policy = model.resolve_policy(definition, context, case.n, case=case)
    expected_names = ({"C", "S", "S1", "K"} if setting == "lead_time" else
                      {"S", "omega1", "omega2"} if prefix == "EABS" else
                      {"C", "S", "omega1", "omega2"})
    assert {p.name for p in policy.parameter_specs} == expected_names
    rng = np.random.default_rng(193)
    for period in range(64):
        params = {p.name: float(rng.uniform(p.lower, p.upper)) for p in policy.parameter_specs}
        if "C" in params:
            params["C"] = 2.5
        state = {key: int(rng.integers(case.n + 1)) for key in case.schema.names}
        state["time"] = period
        xi = round(2.5 * (period + 1)) - round(2.5 * period)
        if setting == "lead_time":
            inventory = state["x"] + sum(state[f"p{i}"] for i in range(1, context + 1))
            width = (context + 4) // 5
            tail = sum(state[f"p{i}"] for i in range(context - width + 1, context + 1))
            expected = min(params["K"] * (params["S"] - inventory), params["S1"] - tail, xi)
        else:
            state[f"a{context}"] = 0
            inventory = sum(state[f"a{i}"] for i in range(1, context))
            if prefix == "EABS":
                expected = params["S"] - params["omega1"] * inventory + params["omega2"] * state["a1"]
            else:
                expected = min(xi, params["S"] - params["omega1"] * inventory
                               - params["omega2"] * min(state[f"a{context-1}"], case.demand.mean))
        features = {"mu": case.demand.mean, "sigma": case.demand.std}
        assert evaluate_ast(policy.ast, state, params, features, motifs=policy.motifs) == pytest.approx(expected)
        action = compile_policy(policy.ast, params, motifs=policy.motifs, action_cap=case.n,
                                allowed_state_names=case.schema.names)
        assert action(state, features, {}) == min(case.n, max(0, round(expected)))


@pytest.mark.parametrize("setting", ("lead_time", "perishable", "lead_time_perishable", "lead_time_backlog"))
def test_representative_templates_match_paper_formulas_and_domains(setting):
    """Compare resolved ASTs and compiled actions with the eight paper equations."""
    from inventory.policy.ast import compile_policy
    from benchmarks.inventory_benchmark_family.mc_protocol import (
        load_campaign_config, load_campaign_selection, load_policy_set)

    model = import_module(f"experiments.{setting}.benchmark_campaigns.inventory_benchmark_family.mc_model")
    root = Path(model.__file__).parent
    config = load_campaign_config(root / "configs/standard.json")
    manifest = json.loads((root / "policies/sets/mc_final.json").read_text())
    names = {
        "lead_time": {"SCDS": {"S", "F", "K", "C"}, "IA-SCDS": {"S", "F", "K", "C"}},
        "perishable": {"EABS": {"S", "omega1", "omega2"}, "RLCS": {"S", "omega1", "gamma", "C"}},
        "lead_time_perishable": {"SCBS": {"S", "K", "C"}, "ETIS": {"S", "omega1", "omega2", "omega3", "C"}},
        "lead_time_backlog": {"BATS": {"S1", "S2", "K"}, "CBATS": {"S1", "S2", "K", "C"}},
    }[setting]
    assert {r['id']: r.get('display_name') for r in manifest['policies'] if r['id'] in names} == {
        name: name for name in names}
    if setting in ("lead_time", "perishable"):
        contexts = range(2, 16)
        _, definitions = load_policy_set(root / "policies/sets/mc_final.json")
    else:
        contexts = config['contexts']
        frozen = load_campaign_selection(root / "policies/sets/mc_final.json", model.BASELINES)
    rng = np.random.default_rng(2411)
    for context in contexts:
        if setting in ("lead_time", "perishable"):
            case = model.MODEL.cases(config, contexts=(context,))[0]
            policies = [model.MODEL.resolve_policy(d, context, case.n, case=case)
                        for d in definitions if d.id in names]
            policies = [dict(id=p.id, ast=p.ast, motifs=p.motifs,
                             specs=p.parameter_specs) for p in policies]
        else:
            case = next(iter(model.cases({**config, 'contexts': [context]})))
            from optimization.models.types import ParameterSpec
            policies = [dict(p, specs=tuple(ParameterSpec(**s) for s in p['parameter_specs']))
                        for p in model.resolve(case, frozen) if p['id'] in names]
        assert {p['id'] for p in policies} == set(names)
        for policy in policies:
            name = policy['id']
            specs = policy['specs']
            assert {s.name for s in specs} == names[name]
            assert all(s.lower == 0 and s.upper > 0 for s in specs)
            assert all(s.kind == 'continuous' for s in specs if s.name in {'C', 'K', 'omega1', 'omega2', 'omega3', 'gamma'})
            if name in {'EABS', 'ETIS'}:
                assert next(s for s in specs if s.name == 'S').kind == 'continuous'
            for index in range(48 if name == 'RLCS' else 24):
                params = {s.name: (int(rng.integers(int(s.lower), int(s.upper)+1)) if s.kind == 'integer'
                                  else float(rng.uniform(s.lower, s.upper))) for s in specs}
                if 'C' in params:
                    params['C'] = 2.5  # Exercises both halves of the cyclic rounding boundary.
                if name in {'SCDS', 'IA-SCDS', 'SCBS'} and index == 0:
                    params['K'] = 0
                state = {key: int(rng.integers(case.n+1)) for key in case.schema.names}
                period = index
                if name == 'RLCS' and index >= 24:
                    # Empty, low and full stock across all cyclic rounding phases.
                    params = dict(S=10.0, C=2.5, omega1=0.4, gamma=0.2)
                    inventory = (0, 2, 20)[(index - 24) // 8]
                    state = {key: 0 for key in case.schema.names}
                    state['a1'] = inventory
                    period = (index - 24) % 8
                state['time'] = period
                lead = case.lead_time
                inventory = sum(state[f'a{i}'] for i in range(1, case.shelf_life+1)) if case.shelf_life else state['x']
                pipeline = sum(state[f'p{i}'] for i in range(1, lead+1))
                gross = inventory+pipeline
                width = max(1, lead//2)
                tail = sum((width-j)*state[f'p{lead-j}'] for j in range(width))/(width*(width+1)/2) if lead else 0
                xi = round(params['C']*(period+1))-round(params['C']*period) if 'C' in params else None
                if name in {'SCDS', 'IA-SCDS'}:
                    summary = state[f'p{lead}'] if name == 'SCDS' else tail
                    expected = min(xi, params['K']*min(params['S']-gross, params['F']-summary))
                elif name == 'EABS':
                    expected = params['S']-params['omega1']*inventory+params['omega2']*state['a1']
                elif name == 'RLCS':
                    lifetime = sum(i*state[f'a{i}'] for i in range(1, case.shelf_life+1))
                    expected = min(xi, params['S']-params['omega1']*inventory-params['gamma']*lifetime/(1+inventory))
                elif name == 'SCBS':
                    expected = min(xi, params['K']*(params['S']-gross))
                elif name == 'ETIS':
                    expected = min(xi, params['S']-params['omega1']*gross+params['omega2']*state['a1']-params['omega3']*tail)
                else:
                    backlog = sum(state[f'b{i}'] for i in range(1, case.config.backlog.patience+1))
                    net = gross-backlog
                    short = params['K']*(state['b1']+max(params['S1']-net-backlog, 0))
                    expected = min(short, params['S2']-net)
                    if name == 'CBATS':
                        expected = min(xi, expected)
                actual = evaluate_ast(policy['ast'], state, params, motifs=policy['motifs'])
                assert actual == pytest.approx(expected, abs=1e-10), (name, context, index)
                action = compile_policy(policy['ast'], params, motifs=policy['motifs'],
                                        action_cap=case.n, allowed_state_names=case.schema.names)
                assert action(state, {}, {}) == min(case.n, max(0, round(expected)))
