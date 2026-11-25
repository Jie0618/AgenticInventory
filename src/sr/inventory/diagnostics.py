"""Fixed simulated occupancy and fitted-parent actions without solving a DP."""
from __future__ import annotations

from runtime.artifacts import seed_from_text, stable_key
import json
import math
from collections import Counter
from dataclasses import replace

import numpy as np

from optimization.batch.eval import _evaluate_ast_batch
from optimization.batch.features import _round_actions
from optimization.batch.samples import _BatchSamples
from optimization.evaluators.composed import cost_features_for_case
from sr.core.diagnostics import DIAGNOSTIC_PROTOCOL, decision_records


class MonteCarloDiagnostics:
    """One case's parent-independent bank, lazily sampled and reused each round."""

    PATHS = 16
    SAMPLE_PERIODS = 16

    def __init__(self, case, horizon):
        if horizon <= 0:
            raise ValueError("diagnostic horizon must be positive")
        self.case = case
        self.horizon = int(horizon)
        self.identity = {
            "protocol": DIAGNOSTIC_PROTOCOL, "case_signature": case.case_signature,
            "horizon": self.horizon, "paths": self.PATHS, "sample_periods": self.SAMPLE_PERIODS,
            "reference_policies": ["cyclic_mean_order", "mean_lead_time_base_stock"],
        }
        token = json.dumps(self.identity, sort_keys=True).encode()
        self.signature = stable_key(token)
        self.seed = seed_from_text(b"diagnostic|" + token)
        self._rows = None
        self._batch = None

    @property
    def rows(self):
        if self._rows is None:
            self._rows = self._sample()
        return self._rows

    def _sample(self):
        periods = set(np.linspace(round(.2 * (self.horizon - 1)), round(.8 * (self.horizon - 1)),
                                  min(self.SAMPLE_PERIODS, self.horizon)).round().astype(int))
        rng = np.random.default_rng(self.seed)
        demands = rng.choice(self.case.demand.values, size=(self.PATHS, max(periods) + 1),
                             p=self.case.demand.probs)
        kernel = self.case.kernel
        mean = min(float(self.case.demand.mean), self.case.n)
        target = math.ceil((self.case.lead_time + 1) * mean)
        counts = Counter()
        for path, demand_path in enumerate(demands):
            current = self.case.initial_state
            for period, demand in enumerate(demand_path):
                if period in periods:
                    counts[(period, current.flatten(self.case.schema))] += 1
                if period == max(periods):
                    break
                if path % 2 == 0:
                    action = round(mean * (period + 1)) - round(mean * period)
                else:
                    position = sum(current.inventory) + sum(current.pipeline) - sum(current.backlog)
                    action = target - position
                action = max(0, min(self.case.n, int(action)))
                current = kernel.sample(current, action, int(demand), rng=rng).next_state
        total = sum(counts.values())
        return tuple({"period": int(period), "state": list(state),
                      "probability": count / total}
                     for (period, state), count in sorted(counts.items()))

    def actions(self, ast, params, motifs):
        """Evaluate fitted constants on the bank, with canonical time and rounding."""
        if self._batch is None:
            rows = self.rows
            batch = _BatchSamples.from_state_array(
                np.asarray([row["state"] for row in rows], dtype=np.int32),
                remaining_horizon=1., state_names=self.case.schema.names,
            )
            periods = np.asarray([row["period"] for row in rows], dtype=float)
            self._batch = replace(batch, features={
                **batch.features, "time": periods, "remaining_horizon": self.horizon - periods,
            })
        values = _evaluate_ast_batch(
            ast, self._batch, {name: np.full(len(self.rows), value) for name, value in params.items()},
            self.case.demand.feature_dict(), cost_features_for_case(self.case), motifs,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("diagnostic parent formula produced NaN or Inf")
        return _round_actions(values, "round", self.case.n).astype(int).tolist()

    def records(self, actions):
        result = decision_records(self.rows, actions, case=self.case.name,
                                  occupancy_source="fixed_mc_reference")
        return {**result, "bank_signature": self.signature, "bank_seed": self.seed,
                "sampling": self.identity}
