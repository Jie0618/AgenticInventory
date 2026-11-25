"""Lead-time DP diagnostic objective used by parameter fitting."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.backends.lead_time.dp import DPResult
from inventory.policy.ast import AST, compile_policy, evaluate_ast

from optimization.batch.analysis import _BatchEvaluationContext
from optimization.batch.eval import _evaluate_ast_batch
from optimization.batch.features import _BatchEvaluationUnsupported, _round_actions
from optimization.models.types import OptimizationConfig, SupervisedSample
from optimization.support.utils import _dp_initial_cost, _initial_state, _loss_name
from optimization.backends.lead_time.policy_cost import evaluate_ast_policy_cost_exact, evaluate_policy_cost_exact


class _Objective:
    """Callable loss object used by all optimizer strategies."""

    def __init__(
        self,
        ast: AST,
        samples: Sequence[SupervisedSample],
        batch: "_BatchSamples",
        dist_features: Mapping[str, Any],
        cost_features: Mapping[str, float],
        motifs: Mapping[str, AST],
        param_names: Sequence[str],
        config: OptimizationConfig,
        dp_result: DPResult | None,
        dp_optimal_cost: float | None = None,
    ) -> None:
        self.ast = ast
        self.samples = samples
        self.dist_features = dist_features
        self.cost_features = cost_features
        self.motifs = motifs
        self.param_names = tuple(param_names)
        self.config = config
        self.dp_result = dp_result
        self.initial_state = _initial_state(dp_result, config.initial_state)
        self.dp_optimal_cost = (
            _dp_initial_cost(dp_result, self.initial_state)
            if dp_optimal_cost is None
            else max(float(dp_optimal_cost), 1.0e-12)
        )
        self.mse_normalizer = self._target_second_moment()
        self.batch = batch
        self.evaluation_context = _BatchEvaluationContext(batch, motifs)
        self.calls = 0
        self.batch_fallback_count = 0
        self.batch_fallback_reasons: list[str] = []

    def __call__(self, theta: Sequence[float]) -> float:
        self.calls += 1
        params = self.params(theta)
        objective = _loss_name(self.config)
        if objective == "action_mse":
            return self.action_mse(params)
        if objective == "raw_mse":
            return self.raw_mse(params)
        if objective in {"policy_cost", "policy_reward"}:
            return self.normalized_policy_cost(params)
        if objective in {"weighted", "weighted_normalized", "mse_policy_weighted"}:
            return self.weighted_loss(params)
        raise ValueError(f"Unknown optimization objective: {self.config.objective}")

    def params(self, theta: Sequence[float]) -> dict[str, float]:
        """Convert a vector into `{parameter_name: value}`."""

        return {name: float(value) for name, value in zip(self.param_names, theta)}

    def raw_mse(self, params: Mapping[str, float]) -> float:
        """MSE between raw real-valued AST output and target action."""

        try:
            raw = _evaluate_ast_batch(
                self.ast,
                self.batch,
                params,
                self.dist_features,
                self.cost_features,
                self.motifs,
                context=self.evaluation_context,
            )
            pred = np.maximum(np.nan_to_num(raw.astype(float), nan=0.0, posinf=0.0, neginf=0.0), 0.0)
            errors = (pred - self.batch.targets) ** 2
            return float(np.dot(self.batch.weights, errors) / max(float(np.sum(self.batch.weights)), 1.0e-12))
        except _BatchEvaluationUnsupported as exc:
            self._record_batch_fallback("raw_mse", exc)

        total_weight = 0.0
        total_error = 0.0
        for sample in self.samples:
            raw = evaluate_ast(self.ast, sample.state, params, self.dist_features, self.cost_features, self.motifs)
            if not np.isfinite(raw):
                raw = 0.0
            pred = max(float(raw), 0.0)
            total_weight += sample.weight
            total_error += sample.weight * (pred - sample.target_action) ** 2
        return float(total_error / max(total_weight, 1.0e-12))

    def action_mse(self, params: Mapping[str, float]) -> float:
        """MSE between compiled integer policy action and target action."""

        try:
            raw = _evaluate_ast_batch(
                self.ast,
                self.batch,
                params,
                self.dist_features,
                self.cost_features,
                self.motifs,
                context=self.evaluation_context,
            )
            pred = _round_actions(raw, self.config.rounding, self.config.action_cap)
            errors = (pred.astype(float) - self.batch.targets) ** 2
            return float(np.dot(self.batch.weights, errors) / max(float(np.sum(self.batch.weights)), 1.0e-12))
        except _BatchEvaluationUnsupported as exc:
            self._record_batch_fallback("action_mse", exc)

        policy = compile_policy(
            self.ast,
            params,
            motifs=self.motifs,
            rounding=self.config.rounding,
            action_cap=self.config.action_cap,
        )
        total_weight = 0.0
        total_error = 0.0
        for sample in self.samples:
            pred = float(policy(sample.state, self.dist_features, self.cost_features))
            total_weight += sample.weight
            total_error += sample.weight * (pred - sample.target_action) ** 2
        return float(total_error / max(total_weight, 1.0e-12))

    def selected_mse(self, params: Mapping[str, float]) -> float:
        """Return raw or rounded-action MSE according to `config.mse_kind`."""

        kind = self.config.mse_kind.strip().lower()
        if kind in {"raw", "raw_mse"}:
            return self.raw_mse(params)
        if kind in {"action", "action_mse"}:
            return self.action_mse(params)
        raise ValueError(f"Unknown mse_kind: {self.config.mse_kind}")

    def normalized_mse(self, params: Mapping[str, float]) -> float:
        """Return selected MSE divided by target action second moment."""

        return self.selected_mse(params) / self.mse_normalizer

    def policy_cost(self, params: Mapping[str, float]) -> float:
        """Return exact finite-horizon expected cost for the compiled policy."""

        if self.dp_result is None:
            raise ValueError("policy cost reporting requires dp_result")
        try:
            return evaluate_ast_policy_cost_exact(
                self.ast,
                self.dp_result,
                params,
                motifs=self.motifs,
                rounding=self.config.rounding,
                action_cap=self.config.action_cap,
                initial_state=self.initial_state,
            )
        except _BatchEvaluationUnsupported as exc:
            self._record_batch_fallback("policy_cost", exc)
        policy = compile_policy(
            self.ast,
            params,
            motifs=self.motifs,
            rounding=self.config.rounding,
            action_cap=self.config.action_cap,
        )
        return evaluate_policy_cost_exact(policy, self.dp_result, self.initial_state)

    def normalized_policy_cost(self, params: Mapping[str, float]) -> float:
        """Return policy cost divided by the DP optimal cost at initial state."""

        return self.policy_cost(params) / self.dp_optimal_cost

    def weighted_loss(self, params: Mapping[str, float]) -> float:
        """Return normalized weighted sum of imitation loss and policy cost."""

        return self.config.mse_weight * self.normalized_mse(params) + self.config.policy_weight * self.normalized_policy_cost(params)

    def _target_second_moment(self) -> float:
        """Compute normalization scale for supervised action MSE."""

        total_weight = sum(sample.weight for sample in self.samples)
        second_moment = sum(sample.weight * sample.target_action**2 for sample in self.samples)
        return float(second_moment / max(total_weight, 1.0e-12) + 1.0e-12)

    def _record_batch_fallback(self, metric: str, exc: Exception) -> None:
        """Remember that a metric had to fall back to scalar evaluation."""

        self.batch_fallback_count += 1
        reason = f"{metric}: {exc}"
        if reason not in self.batch_fallback_reasons:
            self.batch_fallback_reasons.append(reason)
