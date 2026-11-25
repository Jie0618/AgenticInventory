from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sr.inventory import run_single


def test_stage1_cli_reuses_diagnostic_samples_for_context_and_search(tmp_path: Path) -> None:
    case = SimpleNamespace(name="test_case")
    diagnostic_root = tmp_path / "diagnostics"
    diagnostic_root.mkdir()
    diagnostic = diagnostic_root / "test_case.json"
    diagnostic.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "output"
    cache_dir = tmp_path / "cache"
    reference = object()
    samples = (object(), object())
    context = object()
    bindings = object()
    result = SimpleNamespace(
        case_name="test_case",
        archive=(SimpleNamespace(gap_pct=1.0),),
        iterations_completed=0,
        elapsed_seconds=0.1,
        output_dir=output_dir,
    )

    argv = [
        "run_single",
        "--case",
        "test_case",
        "--output-dir",
        str(output_dir),
        "--diagnostic-root",
        str(diagnostic_root),
        "--dp-cache-dir",
        str(cache_dir),
        "--horizon",
        "1000",
        "--no-write-run-logs",
    ]
    with (
        patch("sys.argv", argv),
        patch.object(run_single, "load_or_create_inventory_dp", return_value=reference),
        patch.object(run_single, "diagnostic_samples", return_value=samples) as load_samples,
        patch.object(
            run_single.InventorySearchContext,
            "from_reference",
            return_value=context,
        ) as from_reference,
        patch.object(
            run_single.InventorySearchBindings,
            "build",
            return_value=bindings,
        ),
        patch.object(run_single, "SymbolicSearch") as search_type,
    ):
        search_type.return_value.run.return_value = result
        run_single.main((case,), default_diagnostic_root=diagnostic_root)

    load_samples.assert_called_once_with(diagnostic, 1000)
    from_reference.assert_called_once_with(
        case,
        1000,
        reference,
        samples=samples,
    )
    assert search_type.call_args.kwargs["samples"] is samples
