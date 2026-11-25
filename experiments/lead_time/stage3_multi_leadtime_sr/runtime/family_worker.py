"""Benchmark adapter from a :class:`PolicyFamily` to the canonical MC optimizer.

The adapter deliberately keeps Stage III orchestration independent from the
benchmark runners.  A task is evaluated with the same AST kernels and
optimizer used by Stages I/II; only the family parameters are converted to the
canonical ``ParameterSpec`` representation.
"""
from __future__ import annotations

from runtime.artifacts import seed_from_text, stable_key
import json
import os
import threading
import time
from pathlib import Path
from typing import Mapping
import fcntl

from ..cases import cases
from optimization.backends.lead_time.mc_evaluator import DemandTrajectoryBank, MonteCarloPolicyCostEvaluator
from optimization.backends.lead_time.mc_adapters import ScaledMCEvaluator, TwoFidelityMCEvaluator
from benchmarks.baselines.lead_time import (
    ComputationalPolicyDefinition,
    ResolvedComputationalPolicy,
    computational_baselines,
    resolve_benchmark_policy,
)
from optimization.policies.baselines.lead_time import baselines_for_lead_time
from optimization.policies.policy_sets_json import PolicyDefinition
from benchmarks.baselines.lead_time.evaluator import ComputationalActionAdapter
from optimization.backends.lead_time.rolling import rolling_reference
from optimization.models.types import MonteCarloProposalConfig, OptimizationConfig, ParameterSpec
from optimization.solvers.policy_cost import optimize_policy_constants

from inventory.policy.family import PolicyFamily, materialize_family_for_case
from inventory.policy.family_domain import InventoryFamilyDomain
from sr.family.models import CaseMetric, FamilyTask
from sr.family.identity import case_suite_signature
from sr.family.composed_reference import catalog_signature, select_reference_baselines


_OPTIMIZER_LOCK = threading.Lock()
_REFERENCE_PROTOCOL = "lead-time-fixed-bank-reference"


def _seed(*tokens: object) -> int:
    return seed_from_text("|".join(map(str, tokens)).encode())


def _reference_definitions(names):
    catalog = {
        spec.name: PolicyDefinition(spec.name, "registered_baseline", registered_name=spec.name)
        for spec in baselines_for_lead_time(1, 1)
    }
    catalog.update({
        spec.name: ComputationalPolicyDefinition(spec.name, "computational_baseline", spec.name)
        for spec in computational_baselines(1, 1)
    })
    return select_reference_baselines(catalog, names)


def _specs(family: PolicyFamily) -> tuple[ParameterSpec, ...]:
    from inventory.policy.ast import cap_parameter_modes
    modes = cap_parameter_modes(family.ast, family.motifs)
    specs = []
    for parameter in family.parameters:
        integer_like = parameter.kind in {"target", "cap", "lag"}
        integral_bounds = float(parameter.lower).is_integer() and float(parameter.upper).is_integer()
        kind = "integer" if integer_like and integral_bounds else "continuous"
        if parameter.name in modes:
            kind = "integer" if modes[parameter.name] == "integer" else "continuous"
        midpoint = (float(parameter.lower) + float(parameter.upper)) / 2.0
        default = float(round(midpoint)) if kind == "integer" else midpoint
        specs.append(
            ParameterSpec(
                name=parameter.name,
                kind=kind,
                lower=parameter.lower,
                upper=parameter.upper,
                default=default,
            )
        )
    return tuple(specs)


def _bank_prefix(bank: DemandTrajectoryBank, paths: int, horizon: int) -> DemandTrajectoryBank:
    return DemandTrajectoryBank(
        bank.demand_indices[:paths, :horizon],
        bank.demand_values[:paths, :horizon],
        paths,
        horizon,
        bank.seed,
        bank.policy_uniforms[:paths, :horizon],
    )


def _mode_override(config: Mapping[str, object], mode: str, name: str, default: int) -> int:
    return int(config.get(f"{mode}_{name}", config.get(name, default)))


def _optimization_config(mode: str, seed: int, explicit_budget: object = None) -> OptimizationConfig:
    if mode not in {"fast", "standard", "certify"}:
        raise ValueError(f"unsupported Stage-III fidelity: {mode!r}")
    return OptimizationConfig(
        mode=mode,
        objective="policy_cost",
        random_seed=int(seed),
        total_evaluations=None if explicit_budget is None else int(explicit_budget),
        mc_proposal=MonteCarloProposalConfig(enabled=False, routing_mode="never"),
    )


class CanonicalFamilyWorker:
    """Worker using the standard fixed-bank MC optimizer for every L.

    Stage 3 intentionally does not solve the rolling DP.  The task-level
    reference is selected explicitly as FP3 or the frozen best-baseline
    envelope, while all lead times use the same fixed-bank MC protocol.
    """

    subprocess_module = "experiments.lead_time.stage3_multi_leadtime_sr.runtime.family_worker"

    def __init__(self, *, config_path: Path, reference_cache_dir: Path | None = None, reference_mode: str = "fp3"):
        self.config_path = str(config_path)
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.cyclic_cap_enabled = bool(self.config.get("cyclic_cap_enabled", False))
        caps = {int(k): int(v) for k, v in self.config["state_caps"].items()}
        lead_times = tuple(sorted(caps))
        all_cases = cases(state_caps=caps, lead_times=lead_times)
        self.case_map = {c.name: c for c in all_cases}
        self.case_suite_signature = case_suite_signature(self.case_map.values())
        self.reference_cache_dir = Path(reference_cache_dir or os.environ.get("STAGE3_REFERENCE_CACHE", ".stage3_reference_cache"))
        self.reference_mode = str(reference_mode).lower()
        if self.reference_mode not in {"fp3", "best_baseline"}:
            raise ValueError("reference_mode must be fp3 or best_baseline")
        self.reference_definitions = _reference_definitions(self.config.get("reference_baselines"))
        if self.reference_mode == "fp3" and not any(p.id == "fp3" for p in self.reference_definitions):
            raise ValueError("fp3 reference mode requires fp3 in reference_baselines")
        catalogs = []
        for lead_time in lead_times:
            policies = [resolve_benchmark_policy(p, lead_time, caps[lead_time])
                        for p in self.reference_definitions]
            catalogs.append((lead_time, catalog_signature(_REFERENCE_PROTOCOL, policies),
                             {p.id: p.computational_spec.implementation_version for p in policies
                              if isinstance(p, ResolvedComputationalPolicy)}))
        self.reference_signature = stable_key(json.dumps(catalogs, sort_keys=True).encode())[:16]
        self.evaluation_protocol_signature = (
            f"{_REFERENCE_PROTOCOL}:{self.reference_mode}:{self.reference_signature}:"
            f"cases-{self.case_suite_signature}"
            + (":cyclic-cap-round" if self.cyclic_cap_enabled else "")
        )

    def family_domain(
        self,
        lead_times: tuple[int, ...],
        *,
        training_case_count: int,
    ) -> InventoryFamilyDomain:
        representatives = {}
        for lead_time in lead_times:
            case = next(
                (
                    row
                    for row in self.case_map.values()
                    if int(row.lead_time) == int(lead_time)
                ),
                None,
            )
            if case is None:
                raise ValueError(f"worker has no case for lead time {lead_time}")
            representatives[str(int(lead_time))] = case
        return InventoryFamilyDomain.from_cases(
            representatives,
            training_case_count=training_case_count,
            cyclic_cap=self.cyclic_cap_enabled,
        )

    def _policy_evaluator(self, policy_ast, case, ref, bank, motifs):
        """Use the lead-time array MC kernel for both ordinary and typed-cap policies.

        A cyclic-integer cap is only time-dependent in the cap node.  It does
        not require the generic composed simulator: the lead-time backend now
        supplies the explicit ``time`` feature while retaining its original
        candidate-by-path batch kernel.  This keeps cyclic evaluation on the
        same fast path as ordinary lead-time policies.
        """
        batch_size = int(self.config.get("candidate_batch_size", 64))
        from inventory.policy.ast import has_cyclic_cap
        if has_cyclic_cap(policy_ast, motifs) and not self.cyclic_cap_enabled:
            raise ValueError("cyclic cap is not enabled in the worker config")
        return MonteCarloPolicyCostEvaluator(
            policy_ast, ref, bank, motifs, horizon=bank.horizon,
            use_exact_anchor_control_variate=False, use_common_random_numbers=True,
            batch_size=batch_size,
        )

    def __call__(self, family: PolicyFamily, task: FamilyTask) -> CaseMetric:
        started = time.perf_counter()
        lead_time = int(task.context_id)
        case = self.case_map.get(task.case_id)
        if case is None:
            case = next((c for name, c in self.case_map.items() if name == f"L{lead_time}__{task.case_id}"), None)
        if case is None:
            raise KeyError(f"unknown benchmark case {task.case_id!r}")
        if lead_time != case.lead_time:
            raise ValueError("task lead time does not match case")
        specs = _specs(family)
        policy_ast, motifs = materialize_family_for_case(family, case)
        ref = rolling_reference(case)
        horizon = int(self.config.get("horizon", 1000))
        ref.horizon = horizon
        # ``task.seed`` already incorporates the canonical family content
        # hash.  Do not add family_id here: renaming an otherwise identical
        # family must preserve optimizer randomness and cache identity.
        seed = int(task.seed)
        # Stage 3 substitutes fixed-bank MC for the proposal/strict/final
        # evaluators, but retains the canonical fast/standard/certify optimizer
        # routes and budgets.  Path counts and parameter-point budgets are
        # intentionally independent settings.
        mode = str(task.fidelity).lower()
        if mode not in {"fast", "standard", "certify"}:
            raise ValueError(f"unsupported Stage-III fidelity: {task.fidelity!r}")
        proposal_paths = _mode_override(self.config, mode, "proposal_paths", 128)
        strict_paths = _mode_override(
            self.config,
            mode,
            "strict_paths",
            int(self.config.get("mc_tuning_paths", self.config.get("tuning_paths", 1000))),
        )
        evaluation_paths = _mode_override(
            self.config,
            mode,
            "evaluation_paths",
            int(self.config.get("mc_evaluation_paths", self.config.get("evaluation_paths", 8000))),
        )
        proposal_horizon = _mode_override(self.config, mode, "proposal_horizon", horizon)
        if not 0 < proposal_horizon <= horizon:
            raise ValueError("proposal_horizon must lie in [1, horizon]")
        if proposal_paths <= 0 or strict_paths <= 0 or evaluation_paths <= 0:
            raise ValueError("MC path counts must be positive")
        bank_seed = _seed("mc-bank", lead_time, case.name, task.evaluator)
        tuning_bank = DemandTrajectoryBank.create(case.demand, horizon, max(proposal_paths, strict_paths), bank_seed)
        evaluation_bank = DemandTrajectoryBank.create(case.demand, horizon, evaluation_paths, _seed("mc-evaluation-bank", lead_time, case.name, task.evaluator))
        proposal_bank = _bank_prefix(tuning_bank, proposal_paths, proposal_horizon)
        strict_bank = _bank_prefix(tuning_bank, strict_paths, horizon)
        low = self._policy_evaluator(policy_ast, case, ref, proposal_bank, motifs)
        low_scaled = ScaledMCEvaluator(low, float(horizon) / float(proposal_horizon))
        high = self._policy_evaluator(policy_ast, case, ref, strict_bank, motifs)
        evaluator = TwoFidelityMCEvaluator(high, low_scaled)
        explicit_budget = self.config.get(f"{mode}_proposal_budget", self.config.get("proposal_budget"))
        optimization_config = _optimization_config(mode, seed, explicit_budget)
        with _OPTIMIZER_LOCK:
            result = optimize_policy_constants(specs, evaluator, config=optimization_config)
        params = dict(result.best_params)
        final_evaluator = self._policy_evaluator(policy_ast, case, ref, evaluation_bank, motifs)
        cost = float(final_evaluator.evaluate_many((params,))[0].cost)
        if self.reference_mode == "fp3":
            gap_reference, reference_cache_hit = self._fp3_reference_cost(case, ref, lead_time, horizon, evaluation_bank)
            reference_type = "fp3_empirical_mc"
        else:
            gap_reference, reference_cache_hit = self._best_baseline_reference(
                case, ref, lead_time, horizon, evaluation_bank, mode
            )
            reference_type = "best_baseline_empirical_mc"
        gap = 100.0 * (cost / max(float(gap_reference), 1e-12) - 1.0)
        convergence = dict(result.convergence_report)
        return CaseMetric(task.context_id, task.case_id, gap_pct=gap, runtime_seconds=time.perf_counter() - started, metadata={"cost": cost, "gap_reference_cost": gap_reference, "constants": params, "fidelity": task.evaluator, "optimizer_mode": mode, "optimizer_route": result.route, "candidate_count": int(result.unique_parameter_points), "screen_evaluation_count": int(convergence.get("proposal_count", 0)), "tuning_evaluation_count": int(result.completed_evaluations), "strict_tuning_evaluation_count": int(convergence.get("exact_evaluations", 0)), "proposal_budget": int(convergence.get("proposal_budget_total", 0)), "strict_budget": int(convergence.get("exact_budget_total", 0)), "seed": seed, "reference_cache_hit": bool(reference_cache_hit), "reference_type": reference_type})

    def prepare_reference(self, context_id: str, case_id: str, fidelity: str = "standard") -> None:
        """Populate the reference cache without fitting a candidate family."""
        lead_time = int(context_id)
        case = self.case_map.get(case_id) or self.case_map.get(f"L{lead_time}__{case_id}")
        if case is None:
            raise KeyError(f"unknown benchmark case {case_id!r}")
        horizon = int(self.config.get("horizon", 1000))
        paths = _mode_override(self.config, fidelity, "evaluation_paths", int(self.config.get("mc_evaluation_paths", self.config.get("evaluation_paths", 8000))))
        ref = rolling_reference(case)
        ref.horizon = horizon
        bank = DemandTrajectoryBank.create(case.demand, horizon, paths, _seed("mc-evaluation-bank", lead_time, case.name, "mc"))
        if self.reference_mode == "fp3":
            self._fp3_reference_cost(case, ref, lead_time, horizon, bank)
        else:
            self._best_baseline_reference(case, ref, lead_time, horizon, bank, fidelity)

    def _best_baseline_reference(self, case, ref, lead_time, horizon, evaluation_bank, mode):
        """Fit eligible registered/computational baselines on a fixed bank."""
        self.reference_cache_dir.mkdir(parents=True, exist_ok=True)
        definitions = self.reference_definitions
        config_signature = stable_key(json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode())[:16]
        policy_signature = self.reference_signature
        key = (
            f"{_REFERENCE_PROTOCOL}|best-baseline|{case.name}|h{horizon}|"
            f"r{evaluation_bank.num_paths}|cfg{config_signature}|pol{policy_signature}|"
            f"case{case.case_signature}|seed{evaluation_bank.seed}|mode{mode}"
        )
        path = self.reference_cache_dir / (stable_key(key.encode()) + ".json")
        lock_path = path.with_suffix(".lock")
        with lock_path.open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                return float(json.loads(path.read_text())["cost"]), True
            eligible = [definition for definition in definitions if getattr(definition, "type", "") in {"registered_baseline", "computational_baseline"}]

            def fit_one(definition):
                if getattr(definition, "type", "") not in {"registered_baseline", "computational_baseline"}:
                    return None
                policy = resolve_benchmark_policy(definition, lead_time, case.n)
                adapter = None
                if isinstance(policy, ResolvedComputationalPolicy):
                    adapter = ComputationalActionAdapter(policy.computational_spec.make_provider(ref), ref)
                evaluator = MonteCarloPolicyCostEvaluator(
                    policy.ast, ref, evaluation_bank, policy.motifs,
                    horizon=horizon, action_evaluator=adapter,
                    use_exact_anchor_control_variate=False,
                    use_common_random_numbers=True,
                    batch_size=int(self.config.get("candidate_batch_size", 64)),
                )
                fit = optimize_policy_constants(
                    policy.parameter_specs,
                    TwoFidelityMCEvaluator(evaluator, evaluator),
                    config=_optimization_config(
                        "certify", _seed("best-baseline-fit", case.name, policy.id),
                        self.config.get("certify_proposal_budget", self.config.get("proposal_budget")),
                    ),
                )
                cost = float(MonteCarloPolicyCostEvaluator(
                    policy.ast, ref, evaluation_bank, policy.motifs,
                    horizon=horizon, action_evaluator=adapter,
                    use_exact_anchor_control_variate=False,
                    use_common_random_numbers=True,
                    batch_size=int(self.config.get("candidate_batch_size", 64)),
                ).evaluate(fit.best_params).cost)
                return cost, policy.id
            rows = [row for row in map(fit_one, eligible) if row is not None]
            if not rows:
                raise ValueError("best_baseline reference requires at least one eligible baseline")
            value, policy_id = min(rows, key=lambda row: (row[0], row[1]))
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"cost": value, "policy_id": policy_id, "reference_type": "best_baseline_empirical_mc"}, sort_keys=True) + "\n")
            temporary.replace(path)
            return value, False

    def _fp3_reference_cost(self, case, ref, lead_time: int, horizon: int, evaluation_bank) -> tuple[float, bool]:
        """Return the shared FP3 MC reference, computing it once per case."""
        self.reference_cache_dir.mkdir(parents=True, exist_ok=True)
        config_signature = stable_key(json.dumps(self.config, sort_keys=True, separators=(",", ":")).encode())[:16]
        policy_signature = self.reference_signature
        key = (
            f"{_REFERENCE_PROTOCOL}|L{int(lead_time)}__{case.name.split('__', 1)[-1]}"
            f"__h{horizon}__r{evaluation_bank.num_paths}__cfg{config_signature}"
            f"__pol{policy_signature}__canonical-certify-mc"
            f"__case{case.case_signature}__seed{evaluation_bank.seed}"
        )
        path = self.reference_cache_dir / (stable_key(key.encode()) + ".json")
        lock_path = path.with_suffix(".lock")
        with lock_path.open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                return float(json.loads(path.read_text())["fp3_cost"]), True
            fp3_def = next(row for row in self.reference_definitions if row.id == "fp3")
            fp3 = resolve_benchmark_policy(fp3_def, lead_time, case.n)
            if not isinstance(fp3, ResolvedComputationalPolicy):
                raise TypeError("fp3 reference must be computational")
            adapter = ComputationalActionAdapter(fp3.computational_spec.make_provider(ref), ref)
            proposal_paths = _mode_override(self.config, "certify", "proposal_paths", 128)
            strict_paths = _mode_override(self.config, "certify", "strict_paths", int(self.config.get("tuning_paths", 1000)))
            proposal_horizon = _mode_override(self.config, "certify", "proposal_horizon", horizon)
            reference_bank = DemandTrajectoryBank.create(case.demand, horizon, max(proposal_paths, strict_paths), _seed("fp3-bank", lead_time, case.name))
            low_bank = _bank_prefix(reference_bank, proposal_paths, proposal_horizon)
            high_bank = _bank_prefix(reference_bank, strict_paths, horizon)
            low = MonteCarloPolicyCostEvaluator(fp3.ast, ref, low_bank, {}, horizon=proposal_horizon, action_evaluator=adapter, use_exact_anchor_control_variate=False, use_common_random_numbers=True, batch_size=int(self.config.get("candidate_batch_size", 64)))
            high = MonteCarloPolicyCostEvaluator(fp3.ast, ref, high_bank, {}, horizon=horizon, action_evaluator=adapter, use_exact_anchor_control_variate=False, use_common_random_numbers=True, batch_size=int(self.config.get("candidate_batch_size", 64)))
            evaluator = TwoFidelityMCEvaluator(high, ScaledMCEvaluator(low, float(horizon) / float(proposal_horizon)))
            fit_seed = _seed("fp3-fit", lead_time, case.name)
            explicit_budget = self.config.get("certify_proposal_budget", self.config.get("proposal_budget"))
            result = optimize_policy_constants(
                fp3.parameter_specs,
                evaluator,
                config=_optimization_config("certify", fit_seed, explicit_budget),
            )
            final_evaluator = MonteCarloPolicyCostEvaluator(fp3.ast, ref, evaluation_bank, {}, horizon=horizon, action_evaluator=adapter, use_exact_anchor_control_variate=False, use_common_random_numbers=True, batch_size=int(self.config.get("candidate_batch_size", 64)))
            value = float(final_evaluator.evaluate(result.best_params).cost)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"key": key, "fp3_cost": value, "params": result.best_params, "seed": fit_seed, "optimizer_mode": "certify", "optimizer_route": result.route}, sort_keys=True) + "\n")
            temporary.replace(path)
            return value, False


def main() -> None:
    from sr.family.stage3_runner import run_family_worker_cli

    run_family_worker_cli(
        CanonicalFamilyWorker,
        default_reference_mode="fp3",
        reference_modes=("fp3", "best_baseline"),
        domain_for_task=lambda worker, task: worker.family_domain(
            (int(task.context_id),), training_case_count=1
        ),
    )


if __name__ == "__main__":
    main()
