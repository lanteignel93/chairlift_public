"""report/html.py: a run's page and the index, from the files a run wrote."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from chairlift.report.html import index_page, run_report
from chairlift.run.dag import Pipeline, Stage


def _run(root: Path) -> Path:
    p = Pipeline(
        [
            Stage("src", lambda: pl.DataFrame({"x": [1, 2]}), fingerprint=lambda: "v"),
            Stage("n", lambda src: {"n": src.height}, ("src",)),
        ],
        root,
    )
    r = p.run(meta={"name": "demo", "params": {"a": 1}}, reason="report <test>")
    return root / "runs" / f"{r.run_id}.json"


def test_a_run_page_names_status_stages_and_parameters_and_escapes_text(tmp_path: Path):
    page = run_report(_run(tmp_path / "studies" / "demo"), headline={"sharpe": 1.234})
    assert "<!doctype html>" in page and "demo" in page and "class='ok'" in page
    assert "src" in page and "2×1" in page and "+1.234" in page
    assert "report &lt;test&gt;" in page and "<test>" not in page


def test_the_index_lists_each_studys_latest_run(tmp_path: Path):
    _run(tmp_path / "studies" / "demo")
    _run(tmp_path / "studies" / "demo")
    _run(tmp_path / "studies" / "other")
    page = index_page(tmp_path / "studies")
    assert page.count("<tr class=") == 2 and "other" in page
