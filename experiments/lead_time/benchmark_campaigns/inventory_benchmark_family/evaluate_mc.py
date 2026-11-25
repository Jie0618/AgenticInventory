#!/usr/bin/env python3
"""Run the final lead-time MC benchmark."""

from __future__ import annotations

from benchmarks.inventory_benchmark_family.evaluate_mc import main


if __name__ == "__main__":
    main(setting="lead_time", model_module="experiments.lead_time.benchmark_campaigns.inventory_benchmark_family.mc_model")
