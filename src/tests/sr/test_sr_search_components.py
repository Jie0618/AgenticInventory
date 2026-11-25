from __future__ import annotations

import random
import json
import unittest
from dataclasses import dataclass
from unittest.mock import patch
from pathlib import Path

from inventory.policy.ast import const, minimum, motif, parameter, state
from inventory.common.distributions import DiscreteDemand
from inventory.core import DeterministicLeadTime, LostSales, TransitionCosts
from inventory.core.factory import inventory_case
from inventory.policy.motifs.lead_time import lead_time_motifs
from sr.core.models import ScreenedCandidate, SearchCandidate, SearchConfig, SearchIndividual
from sr.core.pareto import (
    candidate_on_dual_frontier,
    complexity_frontier,
    complexity_profile,
    nondominated,
    parameter_frontier,
    parameter_profile,
)
from sr.core.population import IslandPopulation
from sr.inventory.seeds import initial_seed_asts as _initial_seed_asts, target_gap
from sr.core.inventory import InventorySearchBindings
from optimization.policies.parameters import parameter_spec
from sr.core.grammar import expanded_complexity
from sr.core.selection import select_promotions
from sr.llm import SearchDomain, reflect_motifs
from sr.inventory.single import SymbolicSearch
from sr.analysis.reflection import motif_promotion_evidence
from sr.llm import JSONChatClient, LLMConfig, LLMHTTPError
from sr.inventory.certify_single import certify_seed


def _case(lead_time: int, n: int):
    return inventory_case(
        name=f"test-L{lead_time}-n{n}",
        demand=DiscreteDemand.from_pmf({0: 0.5, 4: 0.5}, name="test"),
        n=n,
        costs=TransitionCosts(holding=1.0, shortage=3.0),
        lead_time=DeterministicLeadTime(lead_time),
        backlog=LostSales(),
        capacity=n,
    )


def make_candidate(ast, *, motifs, lead_time, n, config, source, **kwargs):
    bindings = InventorySearchBindings.build_context_free(_case(lead_time, n), config)
    return bindings.candidate(
        ast, motifs={**bindings.motifs, **motifs}, source=source, **kwargs
    )


def initial_seed_asts(lead_time: int, n: int):
    return _initial_seed_asts(_case(lead_time, n))


def generate_gp_descendants(parents, *, lead_time, n, config, motifs, **kwargs):
    bindings = InventorySearchBindings.build_context_free(_case(lead_time, n), config)
    return bindings.descendants(parents, motifs=motifs, **kwargs)


def _individual(candidate: SearchCandidate, gap: float) -> SearchIndividual:
    return SearchIndividual(candidate, {}, 100.0 + gap, 100.0, gap, "ok", "test", 0.1, 1.0, "test.json")


class SearchComponentTests(unittest.TestCase):
    @staticmethod
    def _lead_time_domain() -> SearchDomain:
        return SearchDomain(
            name="lead_time",
            objective="minimize inventory cost",
            state_names=("x", "p1", "p2", "p3"),
            distribution_names=("mu", "cv"),
            cost_names=("h", "b"),
        )

    def test_fast_nondominated_matches_quadratic_reference_with_ties(self) -> None:
        @dataclass(frozen=True)
        class Point:
            ast_hash: str
            x: float
            y: float

        def objective(row: Point) -> tuple[float, float]:
            return (row.x, row.y)

        def quadratic(rows: list[Point]) -> tuple[Point, ...]:
            unique: dict[tuple[float, float], Point] = {}
            for row in rows:
                point = objective(row)
                if point not in unique or row.ast_hash < unique[point].ast_hash:
                    unique[point] = row
            values = list(unique.values())
            front = [
                row
                for row in values
                if not any(
                    other is not row
                    and other.x <= row.x
                    and other.y <= row.y
                    and (other.x < row.x or other.y < row.y)
                    for other in values
                )
            ]
            return tuple(sorted(front, key=lambda row: (row.x, row.y, row.ast_hash)))

        rng = random.Random(20260814)
        for size in (0, 1, 2, 10, 100):
            for trial in range(30):
                rows = [
                    Point(f"h-{trial:02d}-{index:03d}", float(rng.randrange(12)), float(rng.randrange(12)))
                    for index in range(size)
                ]
                rng.shuffle(rows)
                self.assertEqual(nondominated(rows, objective), quadratic(rows))

    def test_stable_hash_path_does_not_build_repr(self) -> None:
        class HashOnly:
            ast_hash = "safe"

            def __repr__(self) -> str:
                raise AssertionError("repr must not be evaluated when ast_hash exists")

        item = HashOnly()
        self.assertEqual(nondominated((item,), lambda _: (0.0, 0.0)), (item,))

    def test_candidate_frontier_membership_matches_quadratic_augmented_frontier(self) -> None:
        def row(index: int, gap: float, parameters: int, complexity: float, *, ast_hash: str | None = None) -> SearchIndividual:
            candidate = SearchCandidate(
                ast=const(index),
                ast_hash=ast_hash or f"hash-{index:04d}",
                formula=str(index),
                macro_complexity=complexity,
                expanded_complexity=complexity,
                parameter_names=tuple(f"C{slot}" for slot in range(parameters)),
                source="property-test",
            )
            return _individual(candidate, gap)

        def quadratic(rows: list[SearchIndividual], use_complexity: bool) -> set[str]:
            def point(item: SearchIndividual) -> tuple[float, float]:
                size = item.expanded_complexity if use_complexity else item.num_parameters
                return (float(item.gap_pct), float(size))

            unique: dict[tuple[float, float], SearchIndividual] = {}
            for item in rows:
                coordinate = point(item)
                if coordinate not in unique or item.ast_hash < unique[coordinate].ast_hash:
                    unique[coordinate] = item
            values = list(unique.values())
            return {
                item.ast_hash
                for item in values
                if not any(
                    other is not item
                    and point(other)[0] <= point(item)[0]
                    and point(other)[1] <= point(item)[1]
                    and point(other) != point(item)
                    for other in values
                )
            }

        rng = random.Random(9147)
        for trial in range(100):
            existing = [
                row(
                    trial * 100 + index,
                    float(rng.randrange(20)) / 4.0,
                    rng.randrange(7),
                    float(rng.randrange(20)) / 2.0,
                )
                for index in range(rng.randrange(1, 80))
            ]
            # Occasionally exercise replacement semantics: the archive may
            # still contain the older result with the same structural hash.
            replaced_hash = existing[0].ast_hash if trial % 4 == 0 else None
            child = row(
                100_000 + trial,
                float(rng.randrange(20)) / 4.0,
                rng.randrange(7),
                float(rng.randrange(20)) / 2.0,
                ast_hash=replaced_hash,
            )
            augmented = [*existing, child]
            expected = child.ast_hash in (
                quadratic(augmented, False) | quadratic(augmented, True)
            )
            self.assertEqual(candidate_on_dual_frontier(existing, child), expected)

    def test_search_config_rejects_invalid_resource_and_structure_limits(self) -> None:
        invalid = (
            {"max_depth": 0},
            {"max_nodes": 0},
            {"max_expanded_complexity": 0},
            {"max_expanded_complexity": float("nan")},
            {"max_expanded_complexity": float("inf")},
            {"screening_points_low_dim": 0},
            {"reflection_iteration_interval": 0},
            {"max_outstanding_tasks": 0},
            {"iteration_wait_seconds": -1},
            {"iteration_idle_active_workers": -1},
            {"max_outstanding_tasks": 3, "standard_promotions_per_iteration": 4},
            {"max_parameters": -1},
            {"migration_count": -1},
            {"memory_reserve_gb": -1},
            {"parent_bandit_temperature": 0},
            {"operator_min_probability": 0.2},
            {"iterations": 1.5},
            {"island_count": True},
            {"random_seed": 2.5},
            {"iteration_wait_seconds": float("nan")},
            {"iteration_wait_seconds": True},
            {"iteration_wait_seconds": "1.0"},
            {"memory_reserve_gb": float("inf")},
            {"parent_bandit_exploration": float("nan")},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SearchConfig(**values)

        normalized = SearchConfig(iterations=2.0, random_seed=3.0)
        self.assertEqual(normalized.iterations, 2)
        self.assertEqual(normalized.random_seed, 3)

    def test_llm_audit_log_records_prompt_response_without_api_key(self) -> None:
        class FakeAuditClient(JSONChatClient):
            def _complete_once(self, prompt: str, *, request_timeout_seconds: float) -> str:
                return '{"ok":true}'

        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            audit = Path(temporary) / "llm_calls.jsonl"
            client = FakeAuditClient(
                LLMConfig("https://example.test/v1", "secret-value", "test-model"),
                audit_log=audit,
                call_kind="mutation",
            )
            self.assertEqual(client.complete("hello"), '{"ok":true}')
            row = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(row["prompt"], "hello")
            self.assertEqual(row["response"], '{"ok":true}')
            self.assertEqual(row["call_kind"], "mutation")
            self.assertNotIn("secret-value", audit.read_text(encoding="utf-8"))

    def test_llm_config_uses_short_request_timeout_and_retry_backoff_defaults(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "llm.json"
            path.write_text(
                json.dumps(
                    {
                        "base_url": "https://example.test/v1",
                        "api_key": "secret-value",
                        "model": "test-model",
                    }
                ),
                encoding="utf-8",
            )
            config = LLMConfig.from_json(path)
            self.assertEqual(config.timeout_seconds, 150.0)
            self.assertEqual(config.retry_initial_delay_seconds, 2.0)
            self.assertEqual(config.retry_max_delay_seconds, 20.0)
            self.assertEqual(config.retry_backoff_multiplier, 2.0)
            self.assertTrue(config.stream)

    def test_llm_streaming_request_reassembles_sse_json(self) -> None:
        class FakeResponse:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(
                    (
                        b": keepalive\n",
                        b"\n",
                        b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n',
                        b'data: {"choices":[{"index":0,"delta":{"content":"{\\\"ok\\\":"}}]}\n',
                        b'data: {"choices":[{"index":1,"delta":{"content":"ignored"}},{"index":0,"delta":{"content":"true}"}}]}\n',
                        b'data: {"choices":[],"usage":{"completion_tokens":3}}\n',
                        b"data: [DONE]\n",
                    )
                )

        client = JSONChatClient(
            LLMConfig("https://example.test/v1", "secret-value", "test-model")
        )
        with patch("urllib.request.urlopen", return_value=FakeResponse()) as open_url:
            self.assertEqual(client.complete("hello"), '{"ok":true}')
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertTrue(payload["stream"])
        self.assertEqual(request.headers["Accept"], "text/event-stream")

    def test_llm_streaming_supports_content_blocks(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(
                    (
                        b'data: {"choices":[{"index":0,"delta":{"content":[{"type":"text","text":"{\\\"ok\\\":"},{"type":"text","text":"true}"}]}}]}\n',
                        b"data: [DONE]\n",
                    )
                )

        client = JSONChatClient(
            LLMConfig("https://example.test/v1", "secret-value", "test-model")
        )
        with patch("urllib.request.urlopen", return_value=FakeResponse()):
            self.assertEqual(client.complete("hello"), '{"ok":true}')

    def test_llm_falls_back_to_nonstream_when_stream_is_cut(self) -> None:
        class StreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter(
                    (b'data: {"choices":[{"index":0,"delta":{"content":"{\\\"partial\\\":"}}]}\n',)
                )

        class JSONResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{\\\"ok\\\":true}"}}]}'

        responses = iter((StreamResponse(), JSONResponse()))
        client = JSONChatClient(
            LLMConfig(
                "https://example.test/v1",
                "secret-value",
                "test-model",
                max_retries=1,
                retry_initial_delay_seconds=0.0,
            )
        )
        with patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: next(responses)) as open_url:
            self.assertEqual(client.complete("hello"), '{"ok":true}')
        self.assertEqual(open_url.call_count, 2)
        fallback_request = open_url.call_args_list[1].args[0]
        fallback_payload = json.loads(fallback_request.data)
        self.assertFalse(fallback_payload["stream"])
        self.assertEqual(fallback_request.headers["Accept"], "application/json")

    def test_llm_rejects_done_stream_without_text(self) -> None:
        class EmptyStreamResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def __iter__(self):
                return iter((b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n', b"data: [DONE]\n"))

        class EmptyJSONResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":""}}]}'

        client = JSONChatClient(
            LLMConfig(
                "https://example.test/v1",
                "secret-value",
                "test-model",
                max_retries=1,
            )
        )
        responses = iter((EmptyStreamResponse(), EmptyJSONResponse()))
        with patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: next(responses)):
            with self.assertRaisesRegex(RuntimeError, "without text content"):
                client.complete("hello")

    def test_llm_nonstreaming_compatibility_mode(self) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"{\\\"ok\\\":true}"}}]}'

        client = JSONChatClient(
            LLMConfig(
                "https://example.test/v1",
                "secret-value",
                "test-model",
                stream=False,
            )
        )
        with patch("urllib.request.urlopen", return_value=FakeResponse()) as open_url:
            self.assertEqual(client.complete("hello"), '{"ok":true}')
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertFalse(payload["stream"])
        self.assertEqual(request.headers["Accept"], "application/json")

    def test_llm_retries_transient_failures_and_audits_every_attempt(self) -> None:
        class FlakyClient(JSONChatClient):
            calls = 0

            def _complete_once(self, prompt: str, *, request_timeout_seconds: float) -> str:
                self.calls += 1
                if self.calls < 3:
                    raise urllib.error.URLError("temporarily unavailable")
                return '{"recovered":true}'

        import urllib.error
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            audit = Path(temporary) / "llm_calls.jsonl"
            client = FlakyClient(
                LLMConfig(
                    "https://example.test/v1",
                    "secret-value",
                    "test-model",
                    max_retries=4,
                    retry_initial_delay_seconds=0.0,
                ),
                audit_log=audit,
                call_kind="reflection",
            )
            self.assertEqual(client.complete("hello"), '{"recovered":true}')
            rows = [json.loads(line) for line in audit.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["status"] for row in rows], ["attempt_failed", "attempt_failed", "completed"])
            self.assertEqual(rows[-1]["attempts"], 3)
            self.assertTrue(all("secret-value" not in json.dumps(row) for row in rows))

    def test_llm_does_not_retry_permanent_http_error(self) -> None:
        class UnauthorizedClient(JSONChatClient):
            calls = 0

            def _complete_once(self, prompt: str, *, request_timeout_seconds: float) -> str:
                self.calls += 1
                raise LLMHTTPError(401, "invalid credential")

        client = UnauthorizedClient(
            LLMConfig("https://example.test/v1", "secret-value", "test-model")
        )
        with self.assertRaisesRegex(RuntimeError, "after 1 attempt"):
            client.complete("hello")
        self.assertEqual(client.calls, 1)

    def test_certify_seed_depends_on_case_and_ast_not_frontier_order(self) -> None:
        self.assertEqual(certify_seed("case-a", "ast-1"), certify_seed("case-a", "ast-1"))
        self.assertNotEqual(certify_seed("case-a", "ast-1"), certify_seed("case-a", "ast-2"))
        self.assertNotEqual(certify_seed("case-a", "ast-1"), certify_seed("case-b", "ast-1"))

    def test_all_registered_sof_baselines_fit_search_limits(self) -> None:
        motifs = lead_time_motifs(3)
        rows = [
            make_candidate(ast, motifs=motifs, lead_time=3, n=22, config=SearchConfig(), source=source)
            for source, ast in initial_seed_asts(3, 22)
        ]
        baseline_names = {row.source for row in rows if row.source.startswith("baseline:")}
        self.assertEqual(len(baseline_names), 8)  # BS duplicates primitive IP-gap.
        self.assertIn("baseline:weighted_tilted_pic", baseline_names)
        self.assertIn("baseline:aips_discounted_pipeline", baseline_names)
        self.assertIn("baseline:aips_hybrid", baseline_names)
        for lead_time, n in ((1, 50), (2, 40), (3, 22), (4, 14)):
            lead_time_motif_map = lead_time_motifs(lead_time)
            lead_time_rows = [
                make_candidate(
                    ast,
                    motifs=lead_time_motif_map,
                    lead_time=lead_time,
                    n=n,
                    config=SearchConfig(),
                    source=source,
                )
                for source, ast in initial_seed_asts(lead_time, n)
            ]
            self.assertTrue(all(row.expanded_complexity <= 150.0 for row in lead_time_rows))

    def test_expanded_complexity_limit_rejects_macro_hidden_bloat(self) -> None:
        huge_motif = {"op": "add", "args": [state("x") for _ in range(160)]}
        motifs = {"huge": huge_motif}
        with self.assertRaisesRegex(ValueError, "AST nodes .* exceeds 64|expanded complexity .* exceeds 150"):
            make_candidate(
                motif("huge"),
                motifs=motifs,
                lead_time=1,
                n=10,
                config=SearchConfig(max_expanded_complexity=150),
                source="too-large",
            )
        accepted = make_candidate(
            motif("huge"),
            motifs=motifs,
            lead_time=1,
            n=10,
            config=SearchConfig(max_expanded_complexity=200, max_nodes=200),
            source="within-limit",
        )
        self.assertLess(accepted.macro_complexity, 10)
        self.assertGreater(accepted.expanded_complexity, 150)

    def test_parameter_roles_and_expanded_complexity(self) -> None:
        self.assertEqual(parameter_spec("T2", 3, 22).kind, "integer")
        self.assertEqual(parameter_spec("alpha1", 3, 22).kind, "continuous")
        self.assertEqual(parameter_spec("s", 3, 22).kind, "integer")
        self.assertEqual(parameter_spec("omega1", 3, 22).kind, "continuous")
        with self.assertRaises(ValueError):
            parameter_spec("mystery", 3, 22)
        ast = target_gap("T", "pipeline_lag_moment")
        self.assertGreater(expanded_complexity(ast, lead_time_motifs(3)), 13.5)

    def test_dual_frontiers_are_distinct(self) -> None:
        motifs = lead_time_motifs(3)
        config = SearchConfig()
        candidates = [
            make_candidate(target_gap("S", "inventory_position"), motifs=motifs, lead_time=3, n=22, config=config, source="a"),
            make_candidate(target_gap("T", "pipeline_lag_moment"), motifs=motifs, lead_time=3, n=22, config=config, source="b"),
            make_candidate(minimum(target_gap("S", "inventory_position"), target_gap("T", "pipeline_lag_moment")), motifs=motifs, lead_time=3, n=22, config=config, source="c"),
        ]
        rows = [_individual(candidates[0], 2.0), _individual(candidates[1], 1.0), _individual(candidates[2], 0.1)]
        self.assertEqual({row.gap_pct for row in parameter_frontier(rows)}, {1.0, 0.1})
        self.assertEqual({row.gap_pct for row in complexity_frontier(rows)}, {2.0, 1.0, 0.1})

    def test_profiles_keep_one_best_formula_per_budget(self) -> None:
        motifs = lead_time_motifs(3)
        config = SearchConfig()
        one_a = make_candidate(target_gap("S", "inventory_position"), motifs=motifs, lead_time=3, n=22, config=config, source="1a")
        one_b = make_candidate(target_gap("T", "pipeline_lag_moment"), motifs=motifs, lead_time=3, n=22, config=config, source="1b")
        two = make_candidate(minimum(target_gap("S", "inventory_position"), target_gap("T", "pipeline_lag_moment")), motifs=motifs, lead_time=3, n=22, config=config, source="2")
        rows = [_individual(one_a, 2.0), _individual(one_b, 1.0), _individual(two, 0.1)]
        self.assertEqual([row.num_parameters for row in parameter_profile(rows)], [1, 2])
        self.assertEqual([row.gap_pct for row in parameter_profile(rows)], [1.0, 0.1])
        complexities = [row.expanded_complexity for row in complexity_profile(rows)]
        self.assertEqual(complexities, sorted(set(complexities)))

    def test_promotion_capacity_fills_with_distinct_asts_sharing_behavior(self) -> None:
        motifs = lead_time_motifs(3)
        config = SearchConfig()
        candidates = [
            make_candidate(
                target_gap(f"S{index}", "inventory_position"),
                motifs=motifs,
                lead_time=3,
                n=22,
                config=config,
                source=str(index),
            )
            for index in range(6)
        ]
        rows = [
            ScreenedCandidate(candidate, float(index), {}, "shared-behavior")
            for index, candidate in enumerate(candidates)
        ]
        promoted = select_promotions(rows, 5)
        self.assertEqual(len(promoted), 5)
        self.assertEqual(len({row.candidate.ast_hash for row in promoted}), 5)

    def test_strict_frontier_keeps_one_representative_per_objective_point(self) -> None:
        motifs = lead_time_motifs(3)
        config = SearchConfig()
        left = make_candidate(target_gap("S", "inventory_position"), motifs=motifs, lead_time=3, n=22, config=config, source="left")
        right = make_candidate(target_gap("T", "inventory_position"), motifs=motifs, lead_time=3, n=22, config=config, source="right")
        rows = [_individual(left, 1.0), _individual(right, 1.0)]
        self.assertEqual(len(parameter_frontier(rows)), 1)

    def test_gp_produces_legal_descendants_and_population_keeps_buckets(self) -> None:
        motifs = lead_time_motifs(3)
        config = SearchConfig(island_count=2, island_max_size=6)
        left = make_candidate(target_gap("S", "inventory_position"), motifs=motifs, lead_time=3, n=22, config=config, source="left")
        right = make_candidate(target_gap("T", "pipeline_lag_moment"), motifs=motifs, lead_time=3, n=22, config=config, source="right")
        parents = [_individual(left, 2.0), _individual(right, 1.5)]
        descendants = generate_gp_descendants(parents, count=20, motifs=motifs, lead_time=3, n=22, config=config, rng=random.Random(3), iteration=1, island=0)
        self.assertGreaterEqual(len(descendants), 10)
        self.assertTrue(all(row.num_parameters <= config.max_parameters for row in descendants))
        population = IslandPopulation(config)
        for row in parents + [_individual(candidate, 1.0 + i / 10) for i, candidate in enumerate(descendants[:8])]:
            population.add(row)
        self.assertLessEqual(len(population.islands[0]), config.island_max_size)
        self.assertTrue(population.public_frontier())

    def test_bandit_rewards_evolvable_parent_and_pending_is_virtual_visit(self) -> None:
        motifs = lead_time_motifs(3)
        config = SearchConfig(island_count=1, parent_bandit_temperature=0.1)
        parent_a = make_candidate(target_gap("S", "inventory_position"), motifs=motifs, lead_time=3, n=22, config=config, source="a")
        parent_b = make_candidate(target_gap("T", "pipeline_lag_moment"), motifs=motifs, lead_time=3, n=22, config=config, source="b")
        population = IslandPopulation(config)
        population.add(_individual(parent_a, 2.0))
        population.add(_individual(parent_b, 2.0))
        child = make_candidate(
            minimum(parent_a.ast, parameter("C")),
            motifs=motifs,
            lead_time=3,
            n=22,
            config=config,
            source="gp:cap",
            parents=(parent_a.ast_hash,),
        )
        population.add(_individual(child, 0.5))
        self.assertGreater(population.parent_stats[parent_a.ast_hash].mean_reward, 0)
        self.assertEqual(population.operator_stats["cap"].completed, 1)
        unpenalized = population.parent_weights((_individual(parent_a, 2.0), _individual(parent_b, 2.0)))
        penalized = population.parent_weights(
            (_individual(parent_a, 2.0), _individual(parent_b, 2.0)),
            {parent_a.ast_hash: 100},
        )
        self.assertGreater(unpenalized[parent_a.ast_hash], unpenalized[parent_b.ast_hash])
        self.assertLess(penalized[parent_a.ast_hash], unpenalized[parent_a.ast_hash])

    def test_hybrid_parent_selection_is_reproducible_and_diverse(self) -> None:
        motifs = lead_time_motifs(3)
        config = SearchConfig(island_count=1, island_max_size=100)
        population = IslandPopulation(config)
        for index in range(36):
            candidate = make_candidate(
                minimum(target_gap(f"S{index}", "inventory_position"), parameter(f"C{index % 4}")),
                motifs=motifs,
                lead_time=3,
                n=22,
                config=config,
                source="seed",
                iteration=index,
                detail={"behavior_signature": f"behavior-{index % 9}"},
            )
            population.add(_individual(candidate, 1.0 + index / 100.0))
        first = population.parents(0, 24, random.Random(17))
        second = population.parents(0, 24, random.Random(17))
        self.assertEqual([row.ast_hash for row in first], [row.ast_hash for row in second])
        self.assertEqual(len(first), 24)
        best_two = {
            row.ast_hash
            for row in sorted(population.archive.values(), key=lambda row: (row.gap_pct, row.num_parameters, row.expanded_complexity, row.ast_hash))[:2]
        }
        self.assertTrue(best_two.issubset({row.ast_hash for row in first}))
        self.assertGreaterEqual(len({row.candidate.detail["behavior_signature"] for row in first}), 6)
        self.assertEqual(len(population.parents(0, 3, random.Random(17))), 3)

    def test_reflection_motif_requires_two_independent_improvements(self) -> None:
        motifs = dict(lead_time_motifs(3))
        motifs["trial"] = motifs["pipeline_lag_moment"]
        config = SearchConfig()
        first = make_candidate(target_gap("S", "trial"), motifs=motifs, lead_time=3, n=22, config=config, source="reflection", parents=("parent-a",))
        second = make_candidate(minimum(target_gap("S", "trial"), parameter("C")), motifs=motifs, lead_time=3, n=22, config=config, source="reflection-child", parents=("parent-b",))
        one = _individual(first, 0.9)
        two = _individual(second, 0.8)
        self.assertFalse(motif_promotion_evidence("trial", 1.0, (one,)))
        self.assertTrue(motif_promotion_evidence("trial", 1.0, (one, two)))

    def test_reflection_reuses_the_selected_parent_set(self) -> None:
        motifs = lead_time_motifs(3)
        config = SearchConfig(island_count=1)
        population = IslandPopulation(config)
        for index in range(30):
            candidate = make_candidate(
                minimum(target_gap(f"S{index}", "inventory_position"), parameter(f"C{index % 5}")),
                motifs=motifs,
                lead_time=3,
                n=22,
                config=config,
                source="seed",
                iteration=index,
                detail={
                    "behavior_signature": f"behavior-{index % 10}",
                    "bellman_regret_pct": float(index % 7),
                },
            )
            population.add(_individual(candidate, 0.1 + index / 10.0))
        search = SymbolicSearch.__new__(SymbolicSearch)
        search.population = population
        search._selected_parents = population.parents(0, 24, random.Random(17))
        rows = search._reflection_representatives()
        self.assertEqual(rows, search._selected_parents)
        self.assertEqual(len(rows), 24)
        self.assertGreaterEqual(
            len({row.candidate.detail["behavior_signature"] for row in rows}),
            5,
        )

    def test_reflection_rejects_one_bad_motif_without_losing_valid_sibling(self) -> None:
        class FakeClient:
            def complete(self, prompt: str) -> str:
                return '{"motifs": [{"name": "bad", "ast": {"op": "motif", "name": "invented"}}, {"name": "on_hand", "ast": {"op": "state", "name": "x"}, "rationale": "retain stock"}]}'

        rows = reflect_motifs(
            FakeClient(),
            (_individual(make_candidate(state("x"), motifs=lead_time_motifs(3),
                lead_time=3, n=22, config=SearchConfig(), source="parent"), 1.),),
            lead_time_motifs(3),
            version=4,
            domain=self._lead_time_domain(),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].name, "reflection_v4_1_on_hand")

    def test_reflection_normalizes_only_known_ast_aliases(self) -> None:
        class FakeClient:
            def complete(self, prompt: str) -> str:
                return '{"motifs":[{"name":"gap","ast":{"op":"maximum","args":[{"op":"neg","args":[{"op":"inventory_position"}]},{"op":"constant","value":0}]}}]}'

        rows = reflect_motifs(
            FakeClient(),
            (_individual(make_candidate({"op": "neg", "arg": motif("inventory_position")},
                motifs=lead_time_motifs(3), lead_time=3, n=22, config=SearchConfig(), source="parent"), 1.),),
            lead_time_motifs(3),
            version=5,
            domain=self._lead_time_domain(),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].ast["op"], "neg")
        self.assertEqual(
            rows[0].ast["arg"],
            {"op": "motif", "name": "inventory_position"},
        )


if __name__ == "__main__":
    unittest.main()
