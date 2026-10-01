"""Shared fixtures and the suite's ordering.

Layout rule (enforced by tests/test_layout.py): `src/chairlift/<package>/<module>.py` is tested by
`tests/<package>/test_<module>.py`. Cross-cutting suites live in their own directories and carry a marker:

    tests/properties/    @property    hypothesis invariants across modules
    tests/golden/        @golden      content snapshots of whole pipelines (syrupy)
    tests/walkthroughs/  @walkthrough every debug walkthrough, asserted

Collection order follows the dependency tiers, so the first failure is the lowest broken layer:
core → data → target → features → schedule → learn → book → evaluate → diagnose → ledger → report → run → verify →
properties → golden → walkthroughs → layout.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

TIERS = [
    "core",
    "data",
    "target",
    "features",
    "schedule",
    "learn",
    "book",
    "evaluate",
    "diagnose",
    "ledger",
    "report",
    "run",
    "verify",
    "properties",
    "golden",
    "walkthroughs",
]
# within a tier, modules run in their dependency order; unlisted files follow, alphabetically
WITHIN = {
    "core": ["test_spec", "test_identity", "test_config", "test_secrets"],
    "data": ["test_store", "test_manifest"],
    "run": ["test_dag", "test_events", "test_signature", "test_live", "test_cli", "test_wiring"],
    "properties": ["test_spec_properties", "test_dag_properties"],
}
TESTS = Path(__file__).parent


def _order(item: pytest.Item) -> tuple[int, int, str]:
    rel = Path(str(item.fspath)).relative_to(TESTS)
    top = rel.parts[0] if len(rel.parts) > 1 else ""
    tier = TIERS.index(top) if top in TIERS else len(TIERS)
    names = WITHIN.get(top, [])
    stem = rel.stem
    return tier, names.index(stem) if stem in names else len(names), stem


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    items.sort(key=_order)  # stable sort: definition order is kept within a file


def pytest_collection_finish(session: pytest.Session) -> None:
    for item in session.items:
        top = Path(str(item.fspath)).relative_to(TESTS).parts[0]
        marker = {"properties": "property", "golden": "golden", "walkthroughs": "walkthrough"}.get(top)
        if marker is not None:
            item.add_marker(marker)


@pytest.fixture
def toy() -> Callable[..., object]:
    """The toy study factory: `toy(root, window=..., ic=..., seed=...)`."""
    from chairlift.verify.toy import pipeline

    return pipeline
