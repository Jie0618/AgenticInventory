from __future__ import annotations

import json
import unittest

from inventory.common.distributions import DiscreteDemand
from inventory.core import (
    DeterministicBacklog,
    DeterministicLeadTime,
    FixedShelfLife,
    InventoryCase,
    InventoryModelConfig,
    LostSales,
    TransitionCosts,
)
from inventory.policy.ast import ast_check, motif, parameter, state
from inventory.policy.ast.core import COST_FEATURES, DISTRIBUTION_FEATURES
from inventory.policy.domain import policy_domain_for_case
from inventory.policy.family import PolicyFamily
from inventory.policy.family_domain import InventoryFamilyDomain
from inventory.policy.motifs import motifs_for_config
from optimization import InventorySearchContext
from evaluation.backends.lead_time.dp import solve_dp
from optimization.policies.parameters import target_roles_for_motifs
from sr.core.gp import target_gap
from sr.core.inventory import InventorySearchBindings
from sr.core.models import SearchCandidate, SearchConfig, SearchIndividual
from sr.llm import SearchDomain, propose_asts


def _case(name: str, config: InventoryModelConfig) -> InventoryCase:
    return InventoryCase(
        name,
        config,
        TransitionCosts(holding=1.0, shortage=4.0, expiry=2.0, backlog=0.5),
        DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="two-point"),
        n=3,
    )


class _Client:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.prompt = ""

    def complete(self, prompt: str) -> str:
        self.prompt = prompt
        return json.dumps(self.response)


class SearchDomainPromptTests(unittest.TestCase):
    def test_family_domain_materializes_non_pipeline_component_compositions(self) -> None:
        configurations = (
            InventoryModelConfig(
                perishability=FixedShelfLife(2), backlog=LostSales(), capacity=3
            ),
            InventoryModelConfig(backlog=DeterministicBacklog(2), capacity=3),
            InventoryModelConfig(
                perishability=FixedShelfLife(2),
                backlog=DeterministicBacklog(1),
                capacity=3,
            ),
        )
        for index, config in enumerate(configurations):
            with self.subTest(config=config):
                case = _case(f"family-{index}", config)
                domain = InventoryFamilyDomain.from_cases(
                    {f"context-{index}": case}, training_case_count=1
                )
                family = PolicyFamily(
                    f"seed-{index}", motif("inventory_position"), {}, ()
                )
                domain.validate_family(family)
                self.assertTrue(domain.semantic_hash(family))
                self.assertGreater(
                    domain.complexity_stats(family)["expanded_complexity_max"],
                    0.0,
                )
                self.assertEqual(domain.common_features, {})
                self.assertNotIn("total_pipeline", domain.common_motif_names)

    def test_domain_is_derived_from_enabled_components_and_schema(self) -> None:
        case = _case(
            "composed",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(2),
                perishability=FixedShelfLife(3),
                backlog=DeterministicBacklog(2),
                capacity=3,
            ),
        )
        domain = SearchDomain.from_spec(policy_domain_for_case(case))
        self.assertEqual(
            domain.components, ("lead_time", "perishability", "backlog")
        )
        self.assertEqual(domain.state_names, case.schema.names)
        self.assertIn("p2", domain.state_names)
        self.assertIn("a3", domain.state_names)
        self.assertIn("b2", domain.state_names)

        payload = domain.prompt_payload(motifs_for_config(case.config))
        self.assertIn("effective_net_inventory_position", payload["motifs"])
        self.assertIn("expiring_inventory", payload["motifs"])
        self.assertIn("oldest_backlog", payload["motifs"])

    def test_lead_time_only_domain_preserves_optimized_case_vocabulary(self) -> None:
        case = _case(
            "lead-time",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(2),
                backlog=LostSales(),
                capacity=3,
            ),
        )
        domain = SearchDomain.from_spec(policy_domain_for_case(case))
        self.assertEqual(domain.components, ("lead_time",))
        self.assertEqual(domain.state_names, ("x", "p1", "p2"))
        self.assertIn("p1 arrives first", " ".join(domain.rules))
        self.assertIn("demand is lost", " ".join(domain.rules))
        self.assertEqual(set(domain.distribution_names), DISTRIBUTION_FEATURES)
        self.assertLessEqual(set(domain.cost_names), COST_FEATURES)
        self.assertIn("p_zero", domain.distribution_names)
        self.assertTrue(
            {"support_min", "support_max", "d_max"}.isdisjoint(domain.distribution_names)
        )
        self.assertNotIn("q95", domain.distribution_names)
        self.assertNotIn("values", domain.distribution_names)
        self.assertNotIn("expiry", domain.cost_names)
        self.assertNotIn("backlog", domain.cost_names)

    def test_proposal_rejects_state_outside_case_schema(self) -> None:
        case = _case(
            "perishable",
            InventoryModelConfig(
                perishability=FixedShelfLife(2),
                backlog=LostSales(),
                capacity=3,
            ),
        )
        motifs = motifs_for_config(case.config)
        checked = ast_check(parameter("S"), motifs=motifs)
        candidate = SearchCandidate(
            ast=checked.ast,
            ast_hash=checked.ast_hash,
            formula="S",
            macro_complexity=1.0,
            expanded_complexity=1.0,
            parameter_names=("S",),
            source="seed",
        )
        parent = SearchIndividual(
            candidate=candidate,
            params={"S": 1.0},
            policy_cost=1.0,
            dp_cost=1.0,
            gap_pct=0.0,
            status="ok",
            route="test",
            optimizer_seconds=0.0,
            completed_at=0.0,
            result_path="",
        )
        client = _Client(
            {
                "proposals": [
                    {
                        "action": "replace_root",
                        "ast": {"op": "state", "name": "p1"},
                    }
                ]
            }
        )
        proposals = propose_asts(
            client,
            (parent,),
            motifs,
            SearchConfig(llm_proposals_per_iteration=1),
            SearchDomain.from_spec(policy_domain_for_case(case)),
        )
        self.assertEqual(proposals, ())
        self.assertIn("expiring_inventory", client.prompt)
        self.assertNotIn('"state_names": ["x"', client.prompt)

    def test_case_bindings_reuse_one_candidate_and_optimizer_interface(self) -> None:
        case = _case(
            "perishable-backlog",
            InventoryModelConfig(
                perishability=FixedShelfLife(2),
                backlog=DeterministicBacklog(1),
                capacity=2,
            ),
        )
        context = InventorySearchContext.build(case, horizon=2)
        bindings = InventorySearchBindings.build(context)
        self.assertFalse(context.backend.optimized)
        self.assertEqual(bindings.target_roles["expiring_inventory"], "E")
        self.assertEqual(bindings.target_roles["total_backlog"], "B")
        candidate = bindings.candidate(
            target_gap("E", "expiring_inventory"), source="test-seed"
        )
        self.assertEqual(candidate.parameter_names, ("E",))
        result = bindings.fit(candidate, mode="fast", random_seed=7)
        self.assertTrue(result.success)
        self.assertEqual(
            result.optimizer_metadata["evaluation_backend"],
            "composed_finite_mdp",
        )

    def test_case_bindings_fit_with_a_dynamic_reflection_motif(self) -> None:
        case = _case(
            "lead-time-reflection",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=LostSales(),
                capacity=2,
            ),
        )
        context = InventorySearchContext.build(case, horizon=2)
        bindings = InventorySearchBindings.build(context)
        dynamic_motifs = {**bindings.motifs, "reflection_net": state("x")}
        candidate = bindings.candidate(
            target_gap("S", "reflection_net"),
            source="reflection",
            motifs=dynamic_motifs,
        )
        result = bindings.fit(
            candidate,
            mode="fast",
            random_seed=3,
            motifs=dynamic_motifs,
        )
        self.assertTrue(result.success)

    def test_target_role_assignment_is_component_neutral(self) -> None:
        roles = target_roles_for_motifs(
            {
                "pipeline_lag_moment": {"op": "const", "value": 0},
                "oldest_backlog": {"op": "const", "value": 0},
                "mean_remaining_life": {"op": "const", "value": 0},
                "inventory_position": {"op": "const", "value": 0},
            }
        )
        self.assertEqual(roles["pipeline_lag_moment"], "T")
        self.assertEqual(roles["oldest_backlog"], "B")
        self.assertEqual(roles["mean_remaining_life"], "E")
        self.assertEqual(roles["inventory_position"], "S")

    def test_legacy_reference_cannot_be_bound_to_a_different_case(self) -> None:
        case = InventoryCase(
            "expected",
            InventoryModelConfig(
                lead_time=DeterministicLeadTime(1),
                backlog=LostSales(),
                capacity=3,
            ),
            TransitionCosts(holding=1.0, shortage=4.0),
            DiscreteDemand.from_pmf({0: 0.5, 1: 0.5}, name="expected"),
            n=3,
        )
        wrong_demand = DiscreteDemand.from_pmf({0: 0.25, 1: 0.75}, name="wrong")
        legacy_reference = solve_dp(1, 2, 3, wrong_demand, case.h, case.b)
        with self.assertRaisesRegex(ValueError, "reference inputs do not match"):
            InventorySearchContext.from_reference(case, 2, legacy_reference)


if __name__ == "__main__":
    unittest.main()
