"""Keep reusable source independent of campaigns and higher domain layers."""
import ast
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2]
DOMAIN_LAYERS = ("inventory", "evaluation", "optimization", "sr")
ALLOWED = {
    package: set(DOMAIN_LAYERS[: index + 1]) | {"runtime"}
    for index, package in enumerate(DOMAIN_LAYERS)
}
ALLOWED.update(runtime={"runtime"}, reporting={"reporting", "inventory", "runtime"})
PROJECT_PACKAGES = set(ALLOWED) | {"experiments", "benchmarks", "legacy", "src"}


def imports(path):
    module = path.relative_to(SOURCE).with_suffix("").parts
    package = module[:-1]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # A level of one refers to the containing package itself.
                base = package[: len(package) - node.level + 1]
                target = ".".join((*base, *((node.module or "").split("."))))
            else:
                target = node.module or ""
            yield node.lineno, target


@pytest.mark.parametrize("package", tuple(ALLOWED))
def test_source_package_dependency_direction(package):
    violations = []
    for path in sorted((SOURCE / package).rglob("*.py")):
        for line, target in imports(path):
            root = target.split(".")[0]
            if root in PROJECT_PACKAGES and root not in ALLOWED[package]:
                violations.append(f"{path.relative_to(SOURCE)}:{line}: {target}")
    assert not violations, "Source dependency violations:\n" + "\n".join(violations)
