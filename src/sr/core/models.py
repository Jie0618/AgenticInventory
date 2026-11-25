"""Domain models for asynchronous symbolic inventory-policy search."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from inventory.common.numeric import integer_scalar
from inventory.policy.ast import AST
from .diagnostics import DIAGNOSTIC_PROTOCOL


@dataclass(frozen=True)
class SearchConfig:
    iterations: int = 100
    island_count: int = 6
    island_max_size: int = 100
    llm_proposals_per_iteration: int = 20
    gp_descendants_per_slot: int = 0
    standard_promotions_per_iteration: int = 100
    max_outstanding_tasks: int = 240
    iteration_wait_seconds: float = 60.0
    iteration_idle_active_workers: int = 24
    max_parameters: int = 8
    # The current weighted-Tilted-PIC baseline has canonical macro depth 9.
    # Ten preserves all seven baseline families while still bounding growth.
    max_depth: int = 10
    max_nodes: int = 64
    max_expanded_complexity: float = 150.0
    screening_points_low_dim: int = 1024
    screening_points_high_dim: int = 256
    reflection_iteration_interval: int = 5
    search_control_protocol: str = "paper-ec3"
    diagnostic_protocol: str = DIAGNOSTIC_PROTOCOL
    parent_bandit_exploration: float = 0.8
    parent_bandit_temperature: float = 0.5
    parent_bandit_newcomer_bonus: float = 0.5
    operator_bandit_temperature: float = 0.5
    operator_min_probability: float = 0.03
    migration_interval: int = 5
    migration_count: int = 4
    optimizer_workers: int = 60
    evaluation_hard_timeout_seconds: float = 900.0
    memory_reserve_gb: float = 100.0
    random_seed: int = 20260811
    cyclic_cap: bool = False

    def __post_init__(self) -> None:
        if self.search_control_protocol != "paper-ec3":
            raise ValueError("unsupported search control protocol")
        if self.diagnostic_protocol != DIAGNOSTIC_PROTOCOL:
            raise ValueError("unsupported diagnostic protocol")
        if not isinstance(self.cyclic_cap, bool):
            raise ValueError("cyclic_cap must be boolean")
        positive_integers = {
            "iterations": self.iterations,
            "island_count": self.island_count,
            "island_max_size": self.island_max_size,
            "llm_proposals_per_iteration": self.llm_proposals_per_iteration,
            "standard_promotions_per_iteration": self.standard_promotions_per_iteration,
            "max_outstanding_tasks": self.max_outstanding_tasks,
            "optimizer_workers": self.optimizer_workers,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "screening_points_low_dim": self.screening_points_low_dim,
            "screening_points_high_dim": self.screening_points_high_dim,
            "reflection_iteration_interval": self.reflection_iteration_interval,
            "migration_interval": self.migration_interval,
        }
        for name, value in positive_integers.items():
            object.__setattr__(self, name, integer_scalar(value, name=name, minimum=1))
        nonnegative_integers = {
            "gp_descendants_per_slot": self.gp_descendants_per_slot,
            "iteration_idle_active_workers": self.iteration_idle_active_workers,
            "max_parameters": self.max_parameters,
            "migration_count": self.migration_count,
        }
        for name, value in nonnegative_integers.items():
            object.__setattr__(self, name, integer_scalar(value, name=name, minimum=0))
        object.__setattr__(
            self,
            "random_seed",
            integer_scalar(self.random_seed, name="random_seed", minimum=0),
        )
        finite_floats = (
            "iteration_wait_seconds",
            "max_expanded_complexity",
            "parent_bandit_exploration",
            "parent_bandit_temperature",
            "parent_bandit_newcomer_bonus",
            "operator_bandit_temperature",
            "operator_min_probability",
            "evaluation_hard_timeout_seconds",
            "memory_reserve_gb",
        )
        for name in finite_floats:
            raw_value = getattr(self, name)
            if isinstance(raw_value, (bool, str, bytes)):
                raise ValueError(f"{name} must be a finite number")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.iteration_wait_seconds < 0:
            raise ValueError("iteration_wait_seconds must be nonnegative")
        if self.max_outstanding_tasks < self.standard_promotions_per_iteration:
            raise ValueError("max_outstanding_tasks must fit one promoted batch")
        if self.max_expanded_complexity <= 0:
            raise ValueError("max_expanded_complexity must be finite and positive")
        if self.memory_reserve_gb < 0:
            raise ValueError("memory_reserve_gb must be nonnegative")
        if self.evaluation_hard_timeout_seconds <= 0:
            raise ValueError("evaluation_hard_timeout_seconds must be positive")
        if self.parent_bandit_exploration < 0 or self.parent_bandit_newcomer_bonus < 0:
            raise ValueError("bandit exploration and newcomer bonus must be nonnegative")
        if self.parent_bandit_temperature <= 0 or self.operator_bandit_temperature <= 0:
            raise ValueError("bandit temperatures must be positive")
        if not 0 <= self.operator_min_probability < 1.0 / (10 if self.cyclic_cap else 9):
            raise ValueError("operator_min_probability exceeds the operator count limit")


@dataclass(frozen=True)
class SearchCandidate:
    ast: AST
    ast_hash: str
    formula: str
    macro_complexity: float
    expanded_complexity: float
    parameter_names: tuple[str, ...]
    source: str
    parents: tuple[str, ...] = ()
    iteration: int = 0
    island: int = 0
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def num_parameters(self) -> int:
        return len(self.parameter_names)


@dataclass(frozen=True)
class ScreenedCandidate:
    candidate: SearchCandidate
    bellman_regret_pct: float
    screen_params: Mapping[str, float]
    behavior_signature: str


@dataclass(frozen=True)
class SearchIndividual:
    candidate: SearchCandidate
    params: Mapping[str, float]
    policy_cost: float
    dp_cost: float
    gap_pct: float
    status: str
    route: str
    optimizer_seconds: float
    completed_at: float
    result_path: str
    optimizer_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ast_hash(self) -> str:
        return self.candidate.ast_hash

    @property
    def num_parameters(self) -> int:
        return self.candidate.num_parameters

    @property
    def expanded_complexity(self) -> float:
        return self.candidate.expanded_complexity


@dataclass(frozen=True)
class ReflectionMotif:
    name: str
    ast: AST
    rationale: str
    version: int
    status: str = "trial"
