"""Schema-driven Bellman-advantage screening for inventory policies."""

from __future__ import annotations

from runtime.artifacts import stable_key
import itertools
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from inventory.policy.ast import AST, evaluate_ast
from optimization.batch.eval import _evaluate_ast_batch
from optimization.batch.features import _round_actions
from optimization.batch.samples import _BatchSamples
from optimization.models.types import ParameterSpec
from optimization.policies.parameters import target_roles_for_motifs
from optimization.search_context import InventorySearchContext
from optimization.evaluators import cost_features_for_case
from sr.core.models import ScreenedCandidate, SearchCandidate, SearchConfig


class BellmanScreen:
    """Fast screen whose vocabulary and parameter domains come from a case."""

    def __init__(
        self,
        diagnostic_json: Path,
        *,
        context: InventorySearchContext,
        motifs: Mapping[str, AST],
        random_seed: int,
    ) -> None:
        payload = json.loads(Path(diagnostic_json).read_text(encoding="utf-8"))
        self.context = context
        self.action_cap = int(context.case.n)
        self.reference_cost = max(float(payload["dp_average_cost"]), 1.0e-12)
        self.rows = tuple(payload["representative_states"])
        self.horizon = int(payload["horizon"])
        self.periods = (np.asarray([row["period"] for row in self.rows], dtype=float)
                        if all("period" in row for row in self.rows) else None)
        self.states = np.asarray([row["state"] for row in self.rows], dtype=float)
        if self.states.ndim != 2 or self.states.shape[1] != context.case.schema.dimension:
            raise ValueError("diagnostic states do not match the canonical case schema")
        self.weights = np.asarray([row["probability"] for row in self.rows], dtype=float)
        self.weights /= self.weights.sum()
        self.advantages = np.asarray([row["q_cost_advantage"] for row in self.rows], dtype=float)
        self.motifs = dict(motifs)
        self.target_roles = target_roles_for_motifs(self.motifs)
        self.distribution_features = dict(context.case.demand.feature_dict())
        self.cost_features = cost_features_for_case(context.case)
        self.random_seed = int(random_seed)

    def parent_diagnostics(self, parents):
        """Evaluate fitted parents, without refitting, on the same fixed bank."""
        from sr.core.diagnostics import decision_records
        from inventory.policy.ast import evaluate_ast
        actions = {}
        for parent in parents:
            values = []
            for row in self.rows:
                # Old diagnostic files cannot establish period-sensitive actions.
                if "period" not in row:
                    raise ValueError("diagnostic bank lacks periods; regenerate inputs in a new directory")
                state = dict(row["state_features"])
                state["time"] = int(row["period"])
                raw = evaluate_ast(
                    parent.candidate.ast, state, parent.params,
                    self.distribution_features, self.cost_features, self.motifs,
                )
                values.append(int(_round_actions(np.asarray([raw]), "round", self.action_cap)[0]))
            actions[parent.ast_hash] = values
        return decision_records(self.rows, actions, case=self.context.case.name)

    def parent_behavior_signature(self, parent):
        from sr.core.diagnostics import action_signature
        actions = []
        for row in self.rows:
            if "period" not in row:
                raise ValueError("diagnostic bank lacks periods; regenerate inputs in a new directory")
            state = {**row["state_features"], "time": int(row["period"])}
            raw = evaluate_ast(parent.candidate.ast, state, parent.params,
                               self.distribution_features, self.cost_features, self.motifs)
            if not np.all(np.isfinite(raw)):
                raise ValueError("diagnostic parent formula produced NaN or Inf")
            actions.append(int(_round_actions(np.asarray([raw]), "round", self.action_cap)[0]))
        return action_signature([
            [(row["period"], row["state"], row["probability"]) for row in self.rows], actions,
        ])

    def score(self, candidate: SearchCandidate, config: SearchConfig) -> ScreenedCandidate:
        specs = self.context.parameter_specs(candidate.parameter_names)
        points = self._parameter_points(candidate, specs, config)
        state_count = len(self.rows)
        best_regret = float("inf")
        best_params: dict[str, float] = {}
        best_actions = np.zeros(state_count, dtype=int)
        for start in range(0, len(points), 512):
            chunk = points[start : start + 512]
            repeated_states = np.tile(self.states, (len(chunk), 1))
            batch = _BatchSamples.from_state_array(
                repeated_states,
                remaining_horizon=1.0,
                state_names=self.context.case.schema.names,
            )
            if self.periods is not None:
                from dataclasses import replace
                batch = replace(batch, features={
                    **batch.features, "time": np.tile(self.periods, len(chunk)),
                    "remaining_horizon": np.tile(self.horizon - self.periods, len(chunk)),
                })
            params = {
                name: np.repeat(chunk[:, index], state_count)
                for index, name in enumerate(candidate.parameter_names)
            }
            raw = _evaluate_ast_batch(
                candidate.ast,
                batch,
                params,
                self.distribution_features,
                self.cost_features,
                self.motifs,
            )
            actions = _round_actions(raw, "round", self.action_cap).astype(int).reshape(len(chunk), state_count)
            selected = np.take_along_axis(self.advantages[None, :, :], actions[:, :, None], axis=2)[:, :, 0]
            regrets = selected @ self.weights
            index = int(np.argmin(regrets))
            if float(regrets[index]) < best_regret:
                best_regret = float(regrets[index])
                best_actions = actions[index].copy()
                best_params = {
                    name: float(chunk[index, column])
                    for column, name in enumerate(candidate.parameter_names)
                }
        signature = (
            f"constant-action:{int(best_actions[0])}"
            if np.all(best_actions == best_actions[0])
            else stable_key(best_actions.astype(np.int16).tobytes())
        )
        return ScreenedCandidate(
            candidate=candidate,
            bellman_regret_pct=100.0 * best_regret / self.reference_cost,
            screen_params=best_params,
            behavior_signature=signature,
        )

    def _parameter_points(
        self,
        candidate: SearchCandidate,
        specs: Sequence[ParameterSpec],
        config: SearchConfig,
    ) -> np.ndarray:
        dimension = len(specs)
        if dimension == 0:
            return np.empty((1, 0), dtype=float)
        anchors = [self._anchors_for_spec(spec) for spec in specs]
        budget = config.screening_points_low_dim if dimension <= 2 else config.screening_points_high_dim
        base_count = min(int(budget), math.prod(len(values) for values in anchors))
        products = list(itertools.islice(itertools.product(*anchors), base_count))
        rng = np.random.default_rng(self.random_seed ^ int(candidate.ast_hash[:16], 16))
        random_points = np.empty((max(0, int(budget) - len(products)), dimension), dtype=float)
        for column, spec in enumerate(specs):
            if spec.kind == "integer":
                random_points[:, column] = rng.integers(int(spec.lower), int(spec.upper) + 1, size=len(random_points))
            elif spec.kind == "grid":
                random_points[:, column] = rng.choice(np.asarray(spec.values, dtype=float), size=len(random_points))
            else:
                random_points[:, column] = rng.uniform(spec.lower, spec.upper, size=len(random_points))
        base = np.asarray(products[:budget], dtype=float).reshape(-1, dimension)
        points = np.vstack((base, random_points)) if len(random_points) else base
        return np.unique(points, axis=0)

    def _anchors_for_spec(self, spec: ParameterSpec) -> tuple[float, ...]:
        values = {float(spec.lower), float(spec.upper), float(spec.initial_value())}
        role = str(spec.name).split("_", 1)[0][:1]
        matching_motifs = [
            name for name, target_role in self.target_roles.items()
            if target_role == role
        ]
        if role in {"C", "r"} or str(spec.name).startswith("cap"):
            values.update(float(row["dp_action"]) for row in self.rows)
        for row in self.rows:
            state_values = {
                name: float(value)
                for name, value in zip(
                    self.context.case.schema.names,
                    row["state"],
                )
            }
            action = float(row["dp_action"])
            for motif_name in matching_motifs:
                target = evaluate_ast(
                    self.motifs[motif_name],
                    state_values,
                    {},
                    self.distribution_features,
                    self.cost_features,
                    self.motifs,
                ) + action
                values.add(float(np.clip(target, spec.lower, spec.upper)))
        if spec.kind == "continuous":
            values.update(np.linspace(spec.lower, spec.upper, 5).tolist())
        return tuple(sorted(values))


__all__ = ["BellmanScreen"]
