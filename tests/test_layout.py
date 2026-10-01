"""The suite mirrors the package: every module has a test file in the matching directory."""

from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "chairlift"
TESTS = Path(__file__).parent


def test_every_module_has_a_mirrored_test_file():
    missing = []
    for module in sorted(SRC.rglob("*.py")):
        if module.name == "__init__.py":
            continue
        rel = module.relative_to(SRC)
        expected = TESTS / rel.parent / f"test_{module.stem}.py"
        if not expected.exists():
            missing.append(f"{rel} → tests/{rel.parent / ('test_' + module.stem + '.py')}")
    assert not missing, "modules without a test file:\n  " + "\n  ".join(missing)


def test_no_test_file_is_left_at_the_top_level():
    stray = [p.name for p in TESTS.glob("test_*.py") if p.name != "test_layout.py"]
    assert not stray, f"move these into the directory of the module they test: {stray}"
