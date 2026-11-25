"""Read the reviewed MC roster without reselecting training candidates."""
from pathlib import Path

from benchmarks.inventory_benchmark_family.policy_assets import materialize_policy_set


ROOT = Path(__file__).resolve().parents[4]


def build_mc():
    frozen = ROOT / "experiments/perishable/benchmark_campaigns/inventory_benchmark_family/policies/sets/mc_final.json"
    return materialize_policy_set(frozen)
