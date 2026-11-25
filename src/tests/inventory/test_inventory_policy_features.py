from __future__ import annotations

import pytest

from inventory.policy.features import FeatureSpec, materialize_feature, resolve_window_rule


def test_dynamic_window_rules_resolve_for_concrete_context() -> None:
    assert resolve_window_rule("floor(L/2)", 5) == 2
    assert resolve_window_rule("ceil(L/2)", 5) == 3
    assert resolve_window_rule({"op": "floor_div", "left": {"op": "lead_time"}, "right": {"op": "const", "value": 3}}, 8) == 2


def test_dynamic_weight_rule_materializes_normalized_pipeline_weights() -> None:
    spec = FeatureSpec(
        "weighted_tail",
        "tail_weighted",
        "ceil(L/2)",
        weight_rule="linear_desc_normalized",
    )
    ast = materialize_feature(spec.name, 5, spec)
    assert [term["args"][0]["value"] for term in ast["args"]] == pytest.approx([0.5, 1.0 / 3.0, 1.0 / 6.0])
    assert [term["args"][1]["name"] for term in ast["args"]] == ["p5", "p4", "p3"]


def test_dynamic_explicit_weights_fail_with_domain_error_when_too_short() -> None:
    spec = FeatureSpec("weighted_tail", "tail_weighted", "L", (1.0,))
    with pytest.raises(ValueError, match="does not provide a weight"):
        materialize_feature(spec.name, 3, spec)
