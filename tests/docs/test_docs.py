"""The guide stays true: the generated CLI reference matches the code, and every relative link resolves."""

from __future__ import annotations

import re
import runpy
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
GUIDE = REPO / "docs" / "guide"


def test_cli_reference_is_up_to_date():
    render = runpy.run_path(str(REPO / "scripts" / "gen_cli_reference.py"))["render"]
    assert (GUIDE / "08-cli-reference.md").read_text() == render(), (
        "CLI changed: run `uv run python scripts/gen_cli_reference.py` and commit the page"
    )


@pytest.mark.parametrize("page", [*sorted(GUIDE.glob("*.md")), REPO / "README.md"], ids=lambda p: p.name)
def test_relative_links_resolve(page: Path):
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", page.read_text()):
        if "://" in target:
            continue
        assert (page.parent / target).exists(), f"{page.name}: broken link {target}"


def _prices(n_tickers: int = 30, n_days: int = 120):
    import numpy as np
    import polars as pl

    rng = np.random.default_rng(3)
    rets = rng.normal(0, 0.01, (n_days, n_tickers))
    close = 100 * np.exp(np.cumsum(rets, axis=0))
    return pl.DataFrame(
        {
            "date": np.repeat(np.arange(n_days), n_tickers),
            "ticker": np.tile([f"T{i:02d}" for i in range(n_tickers)], n_days),
            "close": close.ravel(),
        }
    )


def test_the_guide_quotes_the_example_study_verbatim():
    code = (REPO / "examples" / "momentum_study.py").read_text()
    body = code[code.index("from __future__") :]
    assert body in (GUIDE / "06-writing-a-study.md").read_text()


def test_the_example_study_runs_and_is_cached(tmp_path: Path):
    example = runpy.run_path(str(REPO / "examples" / "momentum_study.py"))
    data = tmp_path / "vendor" / "daily_prices"
    data.mkdir(parents=True)
    _prices().write_parquet(data / "a.parquet")
    roots = {"vendor": tmp_path / "vendor"}
    first = example["pipeline"](tmp_path / "root").rebase(tmp_path / "root", data=roots).run()
    assert first.ran() == ["prices", "features", "book"]
    again = example["pipeline"](tmp_path / "root").rebase(tmp_path / "root", data=roots)
    report = again.run()
    assert report.ran() == []
    out = again.load("book", report)
    assert out["days"] > 50 and abs(out["ls_sharpe"]) < 10


def test_api_reference_is_current():
    render = runpy.run_path(str(REPO / "scripts" / "gen_api_reference.py"))["render"]
    assert (GUIDE / "10-api.md").read_text() == render(), (
        "public API changed: run `uv run python scripts/gen_api_reference.py` and commit the page"
    )


def test_every_public_name_imports_and_is_documented():
    import inspect

    import chairlift

    assert sorted(chairlift.__all__) == sorted(["__version__", *chairlift._EXPORTS])  # pyright: ignore[reportPrivateUsage]
    undocumented = [n for n in chairlift._EXPORTS if not inspect.getdoc(getattr(chairlift, n))]
    assert not undocumented, f"public names without a docstring: {undocumented}"
