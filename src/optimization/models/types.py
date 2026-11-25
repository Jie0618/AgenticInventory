from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from inventory.common.numeric import integer_scalar, integer_tuple


@dataclass(frozen=True, init=False)
class ParameterSpec:
    """Canonical search domain for one policy constant.

    The constructor accepts both the new ``kind/lower/upper/values/default``
    vocabulary and the former ``low/high/grid/initial`` vocabulary.  Values are
    stored only in the new representation so every optimizer sees the same
    parameter semantics.  ``monotone_group`` is opt-in metadata for policy
    families with a proven suffix-min equivalence; ordinary parameters leave
    it unset and retain independent canonicalization.
    """

    name: str
    kind: Literal["integer", "grid", "continuous"]
    lower: float
    upper: float
    values: tuple[float, ...] | None
    scale: Literal["linear", "log"]
    default: float | None
    monotone_group: str | None
    monotone_index: int | None

    def __init__(
        self,
        name: str,
        low: float | None = None,
        high: float | None = None,
        initial: float | None = None,
        grid: Sequence[float] | None = None,
        *,
        kind: Literal["integer", "grid", "continuous"] | None = None,
        lower: float | None = None,
        upper: float | None = None,
        values: Sequence[float] | None = None,
        scale: Literal["linear", "log"] = "linear",
        default: float | None = None,
        monotone_group: str | None = None,
        monotone_index: int | None = None,
    ) -> None:
        if not isinstance(name, str):
            raise ValueError("Parameter name must be a string")
        name_value = name
        if not name_value.strip():
            raise ValueError("Parameter name must be non-empty")
        lower_value = lower if lower is not None else low
        upper_value = upper if upper is not None else high
        if lower_value is None or upper_value is None:
            raise TypeError("ParameterSpec requires lower/upper (or legacy low/high)")
        if isinstance(lower_value, (bool, str, bytes)) or isinstance(
            upper_value, (bool, str, bytes)
        ):
            raise ValueError("Parameter bounds must be finite numbers")
        lower_float = float(lower_value)
        upper_float = float(upper_value)
        if not np.isfinite(lower_float) or not np.isfinite(upper_float):
            raise ValueError("Parameter bounds must be finite")
        values_value = values if values is not None else grid
        inferred_kind = kind
        if inferred_kind is None:
            inferred_kind = "grid" if values_value is not None else "continuous"
        if inferred_kind not in {"integer", "grid", "continuous"}:
            raise ValueError(f"Unknown parameter kind: {inferred_kind}")
        if scale not in {"linear", "log"}:
            raise ValueError(f"Unknown parameter scale: {scale}")
        if lower_float > upper_float:
            raise ValueError("Parameter lower bound must not exceed upper bound")
        if inferred_kind == "integer" and (
            not lower_float.is_integer() or not upper_float.is_integer()
        ):
            raise ValueError("Integer parameter bounds must be integers")
        if scale == "log" and lower_float <= 0.0:
            raise ValueError("Log-scale parameters require lower > 0")
        canonical_values = None if values_value is None else tuple(float(value) for value in values_value)
        if inferred_kind == "grid" and not canonical_values:
            raise ValueError("Grid parameters require non-empty values")
        if canonical_values is not None:
            if any(not np.isfinite(value) for value in canonical_values):
                raise ValueError("Grid parameter values must be finite")
            if any(value < lower_float or value > upper_float for value in canonical_values):
                raise ValueError("Grid parameter values must lie within lower/upper bounds")
            if len(set(canonical_values)) != len(canonical_values):
                raise ValueError("Grid parameter values must be unique")
        chosen_default = default if default is not None else initial
        if isinstance(chosen_default, (bool, str, bytes)):
            raise ValueError("Parameter default must be a finite number")
        default_float = None if chosen_default is None else float(chosen_default)
        if default_float is not None and not np.isfinite(default_float):
            raise ValueError("Parameter default must be finite")
        if (monotone_group is None) != (monotone_index is None):
            raise ValueError("monotone_group and monotone_index must be specified together")
        if monotone_group is not None and not isinstance(monotone_group, str):
            raise ValueError("monotone_group must be a string")
        monotone_group_value = monotone_group
        if monotone_group_value is not None and not monotone_group_value.strip():
            raise ValueError("monotone_group must be non-empty")
        monotone_index_value = (
            None
            if monotone_index is None
            else integer_scalar(monotone_index, name="monotone_index", minimum=0)
        )
        object.__setattr__(self, "name", name_value)
        object.__setattr__(self, "kind", inferred_kind)
        object.__setattr__(self, "lower", lower_float)
        object.__setattr__(self, "upper", upper_float)
        object.__setattr__(self, "values", canonical_values)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "default", default_float)
        object.__setattr__(self, "monotone_group", monotone_group_value)
        object.__setattr__(self, "monotone_index", monotone_index_value)

    @property
    def low(self) -> float:
        return self.lower

    @property
    def high(self) -> float:
        return self.upper

    @property
    def initial(self) -> float | None:
        return self.default

    @property
    def grid(self) -> tuple[float, ...] | None:
        return self.values

    def initial_value(self) -> float:
        """Return the optimizer initial value clipped into `[low, high]`."""

        raw = 0.5 * (self.lower + self.upper) if self.default is None else self.default
        if self.kind == "integer":
            return float(np.clip(np.rint(raw), self.lower, self.upper))
        if self.kind == "grid":
            assert self.values is not None
            return min(self.values, key=lambda value: abs(value - raw))
        return float(np.clip(raw, self.lower, self.upper))

    def grid_values(self, count: int) -> np.ndarray:
        """Return candidate grid values used for warm starts or exhaustive search."""

        if self.kind == "integer":
            values = np.arange(int(np.ceil(self.lower)), int(np.floor(self.upper)) + 1, dtype=float)
        elif self.kind == "grid":
            values = np.asarray(self.values, dtype=float)
        else:
            if self.scale == "log":
                values = np.geomspace(self.lower, self.upper, max(2, int(count)))
            else:
                values = np.linspace(self.lower, self.upper, max(2, int(count)))
        return np.asarray([np.clip(value, self.lower, self.upper) for value in values], dtype=float)

    def cardinality(self) -> int | None:
        if self.kind == "integer":
            return max(0, int(np.floor(self.upper)) - int(np.ceil(self.lower)) + 1)
        if self.kind == "grid":
            return len(self.values or ())
        return None


@dataclass(frozen=True)
class MonteCarloProposalConfig:
    """Fixed-R Monte Carlo settings used only during proposal screening.

    ``paths=None`` resolves to the mode default (256/512/1024).  The
    stationary-probe limits are deliberately part of configuration so worker
    reports can reproduce exactly why a candidate fell back to MC.
    """

    enabled: bool = True
    # ``auto`` measures candidate windows and compares the observed expected
    # time of probe-first against direct MC.  The decision is reversible and
    # scoped by search phase; MC-first windows retain sentinel probes.
    # ``always`` is useful for diagnostics; ``never`` preserves the
    # deterministic no-MC path when enabled=False is not convenient for a
    # caller that still wants probe metadata.
    routing_mode: Literal["auto", "always", "never", "probe_fallback"] = "auto"
    lazy_init: bool = True
    warmup_candidates: int = 64
    # Retained for serialized-config compatibility and abort telemetry. Route
    # selection is now governed by the measured cost ratios below.
    abort_rate_no_mc: float = 0.05
    abort_rate_mc_first: float = 0.50
    cost_switch_to_mc_ratio: float = 1.15
    cost_switch_to_probe_ratio: float = 0.85
    # A small MC-first sentinel sample is noisy. Require this many consecutive
    # complete MC-first windows to agree before returning to probe-first.
    cost_switch_to_probe_confirmations: int = 3
    sentinel_candidates: int = 8
    min_mc_observations: int = 4
    paths: int | None = None
    seed: int = 0
    use_common_random_numbers: bool = True
    use_exact_anchor_control_variate: bool = True
    stationary_probe_enabled: bool = True
    stationary_probe_max_periods: int | None = None
    stationary_probe_max_active_states: int | None = None
    # ``None`` means resolve the mode-specific default in the optimization
    # pipeline.  Keeping these optional preserves an explicit override while
    # allowing fast/standard/certify to use different safety budgets.
    stationary_probe_max_generated_branches: int | None = None
    stationary_probe_max_seconds: float | None = None
    stationary_probe_growth_guard_enabled: bool = True
    stationary_probe_growth_ratio: float = 2.0
    stationary_probe_growth_hits: int = 2
    stationary_probe_growth_min_states: int = 2_000
    collect_pathwise_costs: bool = False
    collect_mc_standard_error: bool = True

    def __post_init__(self) -> None:
        boolean_fields = (
            "enabled",
            "lazy_init",
            "use_common_random_numbers",
            "use_exact_anchor_control_variate",
            "stationary_probe_enabled",
            "stationary_probe_growth_guard_enabled",
            "collect_pathwise_costs",
            "collect_mc_standard_error",
        )
        for name in boolean_fields:
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean")
            object.__setattr__(self, name, bool(value))
        if not isinstance(self.routing_mode, str):
            raise ValueError("routing_mode must be a string")
        if self.routing_mode not in {"auto", "always", "never", "probe_fallback"}:
            raise ValueError("routing_mode must be auto, always, never, or probe_fallback")
        positive_integers = (
            "warmup_candidates",
            "cost_switch_to_probe_confirmations",
            "sentinel_candidates",
            "min_mc_observations",
            "stationary_probe_growth_hits",
            "stationary_probe_growth_min_states",
        )
        for name in positive_integers:
            object.__setattr__(
                self, name, integer_scalar(getattr(self, name), name=name, minimum=1)
            )
        optional_positive_integers = (
            "paths",
            "stationary_probe_max_periods",
            "stationary_probe_max_active_states",
            "stationary_probe_max_generated_branches",
        )
        for name in optional_positive_integers:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, integer_scalar(value, name=name, minimum=1)
                )
        object.__setattr__(self, "seed", integer_scalar(self.seed, name="seed", minimum=0))
        finite_floats = (
            "abort_rate_no_mc",
            "abort_rate_mc_first",
            "cost_switch_to_mc_ratio",
            "cost_switch_to_probe_ratio",
            "stationary_probe_growth_ratio",
        )
        for name in finite_floats:
            raw_value = getattr(self, name)
            if isinstance(raw_value, (bool, np.bool_, str, bytes)):
                raise ValueError(f"{name} must be a finite number")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.stationary_probe_max_seconds is not None:
            if isinstance(
                self.stationary_probe_max_seconds, (bool, np.bool_, str, bytes)
            ):
                raise ValueError("stationary_probe_max_seconds must be a finite number")
            seconds = float(self.stationary_probe_max_seconds)
            if not math.isfinite(seconds):
                raise ValueError("stationary_probe_max_seconds must be finite")
            object.__setattr__(self, "stationary_probe_max_seconds", seconds)
        if not 0.0 <= float(self.abort_rate_no_mc) <= 1.0:
            raise ValueError("abort_rate_no_mc must lie in [0, 1]")
        if not 0.0 <= float(self.abort_rate_mc_first) <= 1.0:
            raise ValueError("abort_rate_mc_first must lie in [0, 1]")
        if float(self.abort_rate_no_mc) > float(self.abort_rate_mc_first):
            raise ValueError("abort_rate_no_mc must not exceed abort_rate_mc_first")
        if float(self.cost_switch_to_mc_ratio) < 1.0:
            raise ValueError("cost_switch_to_mc_ratio must be at least 1")
        if not 0.0 < float(self.cost_switch_to_probe_ratio) <= 1.0:
            raise ValueError("cost_switch_to_probe_ratio must lie in (0, 1]")
        if self.sentinel_candidates > self.warmup_candidates:
            raise ValueError("sentinel_candidates must not exceed warmup_candidates")
        if (
            self.stationary_probe_max_seconds is not None
            and float(self.stationary_probe_max_seconds) <= 0.0
        ):
            raise ValueError("stationary_probe_max_seconds must be positive")
        if float(self.stationary_probe_growth_ratio) <= 1.0:
            raise ValueError("stationary_probe_growth_ratio must exceed 1")


@dataclass(frozen=True)
class OptimizationConfig:
    """Unified constant-optimizer configuration.

    New code selects only ``mode`` (fast/standard/certify), an optional exact
    evaluation budget, and a seed.  The older fields remain temporarily for
    diagnostic API compatibility, but the standard optimizer rejects any
    search objective other than policy cost.
    """

    strategy: str = "auto"
    mode: Literal["fast", "standard", "certify"] = "standard"
    total_evaluations: int | None = None
    maxiter: int = 80
    refine_maxiter: int = 80
    grid_points: int = 7
    hybrid_top_k: int = 3
    rounding: str = "round"
    action_cap: int | None = None
    objective: str = "policy_cost"
    loss: str | None = None
    mse_kind: str = "raw"
    mse_weight: float = 1.0
    policy_weight: float = 1.0
    initial_state: tuple[int, ...] | None = None
    require_policy_cost: bool = True
    max_grid_evals: int | None = None
    grid_sample_size: int = 1000
    random_seed: int = 0
    mc_proposal: MonteCarloProposalConfig | None = None

    def __post_init__(self) -> None:
        for name in ("strategy", "mode", "rounding", "objective", "mse_kind"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if self.loss is not None and not isinstance(self.loss, str):
            raise ValueError("loss must be a string when supplied")
        if not isinstance(self.require_policy_cost, (bool, np.bool_)):
            raise ValueError("require_policy_cost must be a boolean")
        object.__setattr__(self, "require_policy_cost", bool(self.require_policy_cost))
        if self.mode not in {"fast", "standard", "certify"}:
            raise ValueError("mode must be fast, standard, or certify")
        if not str(self.strategy).strip():
            raise ValueError("strategy must be non-empty")
        if self.rounding not in {"round", "floor", "ceil", "none"}:
            raise ValueError("rounding must be round, floor, ceil, or none")
        if not str(self.objective).strip():
            raise ValueError("objective must be non-empty")
        if self.loss is not None and not str(self.loss).strip():
            raise ValueError("loss must be non-empty when supplied")
        if self.mse_kind not in {"raw", "raw_mse", "action", "action_mse"}:
            raise ValueError("mse_kind must be raw/raw_mse or action/action_mse")
        if self.mc_proposal is not None and not isinstance(
            self.mc_proposal, MonteCarloProposalConfig
        ):
            raise ValueError("mc_proposal must be a MonteCarloProposalConfig")

        positive_integers = (
            "maxiter",
            "refine_maxiter",
            "grid_points",
            "hybrid_top_k",
            "grid_sample_size",
        )
        for name in positive_integers:
            object.__setattr__(
                self, name, integer_scalar(getattr(self, name), name=name, minimum=1)
            )
        for name in ("total_evaluations", "max_grid_evals"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self, name, integer_scalar(value, name=name, minimum=1)
                )
        if self.action_cap is not None:
            object.__setattr__(
                self,
                "action_cap",
                integer_scalar(self.action_cap, name="action_cap", minimum=0),
            )
        object.__setattr__(
            self,
            "random_seed",
            integer_scalar(self.random_seed, name="random_seed", minimum=0),
        )
        if self.initial_state is not None:
            object.__setattr__(
                self,
                "initial_state",
                integer_tuple(self.initial_state, name="initial_state", minimum=0),
            )
        for name in ("mse_weight", "policy_weight"):
            raw_value = getattr(self, name)
            if isinstance(raw_value, (bool, np.bool_, str, bytes)):
                raise ValueError(f"{name} must be a finite number")
            value = float(raw_value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class EvaluationResult:
    cost: float
    completed: bool = True
    early_stopped: bool = False
    visited_states: int = 0
    reachable_actions: Mapping[tuple[int, ...], int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyOptimizationResult:
    best_params: dict[str, float]
    best_cost: float
    status: str
    mode: str
    route: str
    total_evaluations: int
    completed_evaluations: int
    early_stopped_evaluations: int
    unique_parameter_points: int
    cache_hits: int
    best_found_at_evaluation: int
    optimizer_runs: tuple[Mapping[str, Any], ...]
    convergence_report: Mapping[str, Any]
    elapsed_seconds: float

@dataclass(frozen=True)
class SupervisedSample:
    """One supervised action target used for constant fitting."""

    state: Mapping[str, Any]
    target_action: float
    weight: float = 1.0

@dataclass(frozen=True)
class FitResult:
    """Summary returned after fitting formula constants."""

    params: dict[str, float]
    objective_value: float
    action_mse: float
    raw_mse: float
    policy_cost: float | None
    normalized_mse: float
    normalized_policy_cost: float | None
    optimizer: str
    success: bool
    nfev: int
    elapsed_seconds: float
    formula: str
    param_names: tuple[str, ...]
    batch_fallback_count: int = 0
    batch_fallback_reasons: tuple[str, ...] = ()
    status: str = ""
    route: str = ""
    optimizer_metadata: Mapping[str, Any] = field(default_factory=dict)
