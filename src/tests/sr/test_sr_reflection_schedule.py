"""Reflection follows iteration boundaries regardless of completion counts."""
from types import SimpleNamespace

import pytest

from sr.core.models import SearchConfig
from sr.single.controller import SingleSearchController
from sr.multi.controller import MultiSearchController


@pytest.mark.parametrize("controller_type", [SingleSearchController, MultiSearchController])
def test_reflection_fires_only_at_five_iteration_boundaries(controller_type):
    controller = object.__new__(controller_type)
    controller.config = SearchConfig(reflection_iteration_interval=5)
    controller._ingested = set(range(10000))
    controller.population = SimpleNamespace(public_frontier=lambda: ())
    controller.reflection_client = object()
    controller._reflections = []
    controller._rejections = {}
    controller.case_name = "test"
    controller.motifs = {}
    controller.domain = None
    controller.scheduler = SimpleNamespace(counts=lambda: {})
    controller._reflection_representatives = lambda: ()
    calls = []
    def reflect(*args):
        calls.append(args[-2]["iteration"])
        return ()
    controller._run_while_scheduling = reflect
    for iteration in range(1, 16):
        controller._maybe_reflect(iteration)
    assert calls == [5, 10, 15]
    # Resuming at iteration 8 follows absolute iteration boundaries.
    calls.clear()
    for iteration in range(9, 16):
        controller._maybe_reflect(iteration)
    assert calls == [10, 15]




def test_default_and_invalid_iteration_schedule():
    assert SearchConfig().reflection_iteration_interval == 5
    for value in (0, -1):
        with pytest.raises(ValueError):
            SearchConfig(reflection_iteration_interval=value)


@pytest.mark.parametrize("field", ["reflection_completed_interval", "reflection_stagnation_completions"])
def test_completion_trigger_options_are_rejected(field):
    with pytest.raises(TypeError):
        SearchConfig(**{field: 500})
