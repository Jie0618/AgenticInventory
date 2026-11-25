#!/usr/bin/env python3
"""Write the four immutable suite manifests from the executable definitions."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.inventory_benchmark_family.suites import SETTINGS, suite_manifest


def main() -> None:
    target = Path(__file__).resolve().parent / "manifests"
    target.mkdir(parents=True, exist_ok=True)
    for setting in SETTINGS:
        path = target / f"{setting}.json"
        path.write_text(json.dumps(suite_manifest(setting), indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
