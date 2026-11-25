#!/usr/bin/env python3
"""Run the final perishable MC benchmark."""

from __future__ import annotations

from benchmarks.inventory_benchmark_family.evaluate_mc import main


if __name__ == "__main__":
    main(setting="perishable", model_module="experiments.perishable.benchmark_campaigns.inventory_benchmark_family.mc_model")
