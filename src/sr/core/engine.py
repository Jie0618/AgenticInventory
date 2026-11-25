"""Scenario-neutral lifecycle for asynchronous agentic search."""

from __future__ import annotations

import time
from typing import Generic, Mapping, TypeVar


ResultT = TypeVar("ResultT")


class AgenticSearchEngine(Generic[ResultT]):
    """Run the common propose-screen-evaluate-reflect search lifecycle.

    Concrete composition roots provide the scheduler, population, candidate
    operations, reporting, and all model-dependent inputs through hooks.
    """

    def run(self) -> ResultT:
        try:
            return self._run()
        except BaseException:
            self.scheduler.shutdown()
            raise

    def _run(self) -> ResultT:
        self._write_progress("search_started", iteration=0)
        self._evaluate_initial_seeds()
        self._write_progress("initial_seeds_completed", iteration=0)
        iterations_completed = int(getattr(self, "_resume_iteration", 0))
        for iteration in range(iterations_completed + 1, self.config.iterations + 1):
            iteration_started = time.perf_counter()
            stage_seconds: dict[str, float] = {}
            stage_details: dict[str, Mapping[str, float | int]] = {}

            def run_stage(name: str, function: object, *args: object, **kwargs: object) -> object:
                started = time.perf_counter()
                result = function(*args, **kwargs)  # type: ignore[operator]
                stage_seconds[name] = time.perf_counter() - started
                return result

            run_stage(
                "capacity_wait",
                self.scheduler.wait_for_capacity,
                self.config.standard_promotions_per_iteration,
                self.config.max_outstanding_tasks,
            )
            run_stage("pre_generation_poll", self.scheduler.poll)
            run_stage("pre_generation_ingest", self._ingest_completed)
            stage_details["pre_generation_ingest"] = dict(self._last_ingest_timing)
            island = (iteration - 1) % self.config.island_count
            rng = self._iteration_rng(iteration, island)
            pending_by_parent = self._pending_by_parent()
            parents = run_stage(
                "parent_selection",
                self.population.parents,
                island,
                24,
                rng,
                pending_by_parent=pending_by_parent,
            )
            raw = run_stage(
                "proposal",
                self._propose,
                iteration,
                island,
                parents,
                rng,
                pending_by_parent,
            )
            stage_details["proposal"] = dict(self._last_proposal_timing)
            screened = run_stage("screening", self._screen_candidates, raw)
            promoted = run_stage(
                "promotion_selection",
                self._select_promotions,
                screened,
                self.config.standard_promotions_per_iteration,
            )
            hashes = run_stage("submission", self._submit_promotions, promoted, iteration)
            self._write_progress(
                "iteration_submitted",
                iteration=iteration,
                island=island,
                generated=len(raw),
                screened=len(screened),
                promoted=len(hashes),
            )
            if hashes:
                run_stage(
                    "batch_wait",
                    self.scheduler.wait_for_batch,
                    hashes,
                    wait_seconds=self.config.iteration_wait_seconds,
                    idle_active_workers=self.config.iteration_idle_active_workers,
                )
            else:
                run_stage("batch_wait", self.scheduler.poll)
            run_stage("post_batch_ingest", self._ingest_completed)
            stage_details["post_batch_ingest"] = dict(self._last_ingest_timing)
            run_stage("migration", self.population.migrate, iteration)
            iterations_completed = iteration
            run_stage("reflection", self._maybe_reflect, iteration)
            run_stage("post_reflection_poll", self.scheduler.poll)
            run_stage("post_reflection_ingest", self._ingest_completed)
            stage_details["post_reflection_ingest"] = dict(self._last_ingest_timing)
            run_stage("checkpoint", self._write_checkpoint, iteration)
            iteration_seconds = time.perf_counter() - iteration_started
            self._write_progress(
                "iteration_completed",
                iteration=iteration,
                island=island,
                iteration_seconds=iteration_seconds,
                stage_seconds={key: round(value, 6) for key, value in stage_seconds.items()},
                stage_detail=stage_details,
                uninstrumented_seconds=round(
                    max(0.0, iteration_seconds - sum(stage_seconds.values())), 6
                ),
            )

        self._write_progress("final_drain_started", iteration=iterations_completed)
        self.scheduler.drain()
        self._ingest_completed()
        self._write_checkpoint(iterations_completed)
        result = self._result(iterations_completed)
        self._write_final_report(result)
        if getattr(self.config, "cyclic_cap", False):
            from runtime.artifacts import atomic_json
            atomic_json({"protocol": "cyclic-cap", "iteration": iterations_completed},
                        self.output_dir / "SEARCH_COMPLETED.json")
        self._write_progress("search_completed", iteration=iterations_completed)
        return result
