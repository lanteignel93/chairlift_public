"""study/build.py: a declared study builds the DAG it describes, keyed on everything it depends on."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from chairlift.book.quantile import QuantileBook
from chairlift.book.timeseries import SignalBook
from chairlift.evaluate.daily import DailyStatsSpec
from chairlift.learn.fit import FitSpec
from chairlift.learn.models import RidgeSpec
from chairlift.schedule.walkforward import WalkForward
from chairlift.study.build import CrossSectionBook, Ensemble, FeatureSet, Model, Source, Study, TimeSeriesBook
from chairlift.verify.twins import CrossSectionTwin, TimeSeriesTwin, cross_section_twin, time_series_twin

TWIN = CrossSectionTwin(names=60, dates=72)


def cs_frame(twin: pl.DataFrame) -> pl.DataFrame:
    from chairlift.features.cross_section import percentile_rank, rank_of_rows, zscore

    feats = [c for c in twin.columns if c.startswith(("f_sig", "f_noise"))]
    return twin.select(
        "root", "t0", "exit_date", "y_year", "y", *(percentile_rank(f).alias(f) for f in feats)
    ).with_columns(y_z=zscore("y"), y_rank=rank_of_rows("y"), eligible=pl.lit(True))


def paths(book: pl.DataFrame) -> pl.DataFrame:
    legs = book.filter(pl.col("side").is_not_null())
    return pl.concat(
        [legs.select("root", "t0", date="t0", pnl=pl.lit(0.0)), legs.select("root", "t0", date="exit_date", pnl="y")]
    )


def cs_study(alpha: float = 1e-2) -> Study:
    feats = FeatureSet(exclude=("root", "t0", "exit_date", "y_year", "y", "y_z", "y_rank", "eligible"))
    return Study(
        name="cs",
        sources=(
            Source("twin", cross_section_twin, spec=TWIN, fingerprint=lambda: "s"),
            Source("panel", cs_frame, inputs=("twin",)),
        ),
        index="panel",
        schedule=WalkForward(start=dt.date(2012, 1, 1), first_test_year=2015, mode="expanding", embargo=1),
        models=(
            Model("a", FitSpec(model=RidgeSpec(alpha=alpha)), "panel", feats),
            Model("b", FitSpec(model=RidgeSpec(alpha=10.0)), "panel", feats),
        ),
        ensembles=(Ensemble("ab", ("a", "b")),),
        book=CrossSectionBook(spec=QuantileBook(), frame="panel", paths=paths, per_position=True),
        stats=DailyStatsSpec(periods=12, block=3, min_per_year=6, min_per_half=3, n_boot=200),
    )


def test_the_dag_has_one_stage_per_declared_piece():
    names = [s.name for s in cs_study().stages()]
    assert names == [
        "twin", "panel", "folds", "fit_a", "fit_b", "ens_ab",
        "book_a", "paths_a", "eval_a", "book_b", "paths_b", "eval_b", "book_ab", "paths_ab", "eval_ab",
        "power", "report",
    ]  # fmt: skip


def test_a_cross_section_study_runs_and_reports(tmp_path: Path):
    p = cs_study().pipeline(tmp_path)
    r = p.run()
    rep = p.load("report", r)["books"]
    assert set(rep) == {"a", "b", "ab"} and rep["a"]["sharpe"] > 0.5  # the planted signal, through the whole study
    assert cs_study().pipeline(tmp_path).run().ran() == []


def test_changing_one_model_reruns_only_what_depends_on_it(tmp_path: Path):
    cs_study().pipeline(tmp_path).run()
    again = cs_study(alpha=0.5).pipeline(tmp_path).run()
    assert set(again.ran()) <= {
        "fit_a",
        "ens_ab",
        "book_a",
        "paths_a",
        "eval_a",
        "book_ab",
        "paths_ab",
        "eval_ab",
        "report",
    }
    assert "fit_b" not in again.ran() and "twin" not in again.ran()


def test_a_time_series_study_compares_every_book_to_its_benchmark(tmp_path: Path):
    s = Study(
        name="ts",
        sources=(Source("twin", time_series_twin, spec=TimeSeriesTwin(days=1500), fingerprint=lambda: "s"),),
        index="twin",
        schedule=WalkForward(
            start=dt.date(2012, 1, 1), first_test_year=2014, mode="expanding", embargo=1, entry="date"
        ),
        models=(
            Model(
                "r",
                FitSpec(model=RidgeSpec(intercept=True), target="y", rank_target="y", cross_section=False),
                "twin",
                FeatureSet(columns=("x_sig0", "x_noise0")),
            ),
        ),
        keys=("date",),
        book=TimeSeriesBook(
            spec=SignalBook(), frame="twin", short_or_flat=True, constants=(("hold", 1.0),), benchmark="hold"
        ),
        stats=DailyStatsSpec(n_boot=200),
    )
    p = s.pipeline(tmp_path)
    rep = p.load("report", p.run())["books"]
    assert set(rep) == {"r", "r_short_or_flat", "hold"}
    assert rep["hold"]["vs_hold"]["d_sharpe"] == 0.0 and "d_sharpe_ci" in rep["r"]["vs_hold"]
    act, ev, base = (p.load(n, p.run()) for n in ("active_r", "eval_r", "eval_baselines"))
    assert act["benchmark"] == "hold" and act["n_days"] == len(act["daily"])
    d0 = act["daily"][0]["date"]
    mine = next(r["pnl"] for r in ev["book"]["daily"] if r["date"] == d0)
    held = next(r["pnl"] for r in base["hold"]["daily"] if r["date"] == d0)
    assert act["daily"][0]["pnl"] == mine - held


def test_an_event_book_ranks_against_a_trailing_pool_and_waits_for_its_ramp(tmp_path: Path):
    base = cs_study()
    s = dataclasses.replace(
        base,
        book=CrossSectionBook(
            spec=QuantileBook(pool_days=62, min_pool=60), frame="panel", paths=paths, per_position=True, min_live=5
        ),
        evaluate=("a",),
    )
    p = s.pipeline(tmp_path)
    rep = p.run()
    bk, ev = p.load("book_a", rep), p.load("eval_a", rep)
    first_month = bk["t0"].min()
    assert bk.filter(pl.col("t0") == first_month)["pct"].null_count() == bk.filter(pl.col("t0") == first_month).height
    assert min(min(r["n_long"], r["n_short"]) for r in ev["daily"]) >= 5
    assert "oos_ic_all" in ev["ic"]  # pooled across the year's rows


def test_paths_can_read_other_stages(tmp_path: Path):
    def paths_from_panel(book: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
        assert "y_z" in panel.columns  # the panel stage's output, handed in by name
        return paths(book)

    base = cs_study()
    b = CrossSectionBook(
        spec=QuantileBook(), frame="panel", paths=paths_from_panel, paths_inputs=("panel",), per_position=True
    )
    s = dataclasses.replace(base, book=b, evaluate=("a",))
    assert next(st for st in s.stages() if st.name == "paths_a").inputs == ("book_a", "panel")
    p = s.pipeline(tmp_path)
    assert p.load("eval_a", p.run())["ls"]["n_days"] > 0


def test_a_study_without_evaluations_has_no_report_stage():
    s = dataclasses.replace(cs_study(), evaluate=())
    assert "report" not in {st.name for st in s.stages()}


def test_feature_sets_resolve_explicit_columns_or_exclusions():
    f = pl.DataFrame({"k": [1], "a": [1.0], "b": [2.0]})
    assert FeatureSet(columns=("b",)).resolve(f) == ["b"]
    assert FeatureSet(exclude=("k",)).resolve(f) == ["a", "b"]
    with pytest.raises(KeyError, match="zz"):
        FeatureSet(columns=("zz",)).resolve(f)


def test_a_fit_keeps_each_fold_s_model_and_the_report_reads_their_stability(tmp_path: Path):
    p = cs_study().pipeline(tmp_path)
    r = p.run()
    models = p.load("fit_a.models", r)
    assert [m["test_year"] for m in models] == [f["test_year"] for f in p.load("folds", r)["folds"]]
    assert all(m["model"]["kind"] == "ridge" and len(m["model"]["coef_standardised"]) == m["n_inputs"] for m in models)
    st = p.load("report", r)["models"]
    assert set(st) == {"a", "b"} and st["a"]["folds"] == len(models)


# ---- path-marked trades: a path book judged on its gain over constant size -------------------------------------------


def _path_study(alpha: float = 1e-2, drop_day: bool = False) -> Study:
    from chairlift.book.path import PathSizing
    from chairlift.study.build import PathBook
    from chairlift.verify.twins import PathTwin, path_entries, path_twin

    def trades(twin: pl.DataFrame) -> pl.DataFrame:
        return twin.select("t0", "date", "pnl", "expiry")

    def calendar(twin: pl.DataFrame) -> pl.DataFrame:
        cal = twin.select("date").unique().sort("date")
        return cal.filter(pl.col("date") != dt.date(2017, 3, 15)) if drop_day else cal

    def panel(twin: pl.DataFrame) -> pl.DataFrame:
        return path_entries(twin).with_columns(w_half=pl.lit(0.5))

    feats = FeatureSet(columns=("x_sig", "x_noise0", "x_noise1", "x_noise2"))
    fit = FitSpec(model=RidgeSpec(alpha=alpha, intercept=True), target="y", rank_target="y", cross_section=False)
    return Study(
        name="paths",
        sources=(
            Source("twin", path_twin, spec=PathTwin(days=1300), fingerprint=lambda: "s"),
            Source("trades", trades, inputs=("twin",)),
            Source("calendar", calendar, inputs=("twin",)),
            Source("panel", panel, inputs=("twin",)),
        ),
        index="panel",
        schedule=WalkForward(start=dt.date(2014, 1, 1), first_test_year=2016, embargo=21),
        keys=("t0",),
        models=(Model("ridge", fit, "panel", feats),),
        book=PathBook(
            sizing=PathSizing(rule="rank"),
            trades="trades",
            calendar="calendar",
            frame="panel",
            baselines=(("half", "w_half"),),
            flag="stopped",
        ),
        stats=DailyStatsSpec(n_boot=200),
        power=None,
    )


def test_a_path_book_sizes_on_the_signal_and_is_judged_on_its_gain(tmp_path: Path):
    from chairlift.ledger.trials import Ledger

    p = _path_study().pipeline(tmp_path)
    r = p.run()
    ev = p.load("eval_ridge", r)
    assert {"book", "constant", "half", "vs_constant", "vs_half", "active", "sizing"} <= set(ev)
    assert ev["vs_constant"]["d_sharpe"] > 0 and ev["sizing"]["oos_rank_ic"] > 0.2  # the planted signal, sized
    assert ev["book"]["n_days"] == ev["constant"]["n_days"] == ev["active"]["n_days"]  # one calendar for all
    assert abs(ev["half"]["sharpe"] - ev["constant"]["sharpe"]) < 1e-9  # half size, same Sharpe
    per = p.load("eval_ridge.trades", r)
    assert per.columns == ["t0", "pred", "w", "y", "stopped"] and abs(per["w"].mean() - 1.0) < 0.15
    rep = p.load("report", r)["books"]
    assert {"ridge", "ridge_active", "constant", "half"} <= set(rep)
    books = Ledger(tmp_path / "ledger.jsonl").books()
    assert [(b["stage"], b["family"]) for b in books] == [("eval_ridge", "gain over constant")]
    assert books[0]["headline"]["sharpe"] == ev["active"]["sharpe"]
    _path_study().pipeline(tmp_path).run()  # the same book again: not a new one
    _path_study(alpha=1.0).pipeline(tmp_path).run()  # another penalty: another book
    assert len(Ledger(tmp_path / "ledger.jsonl").books()) == 2


def test_a_path_book_refuses_a_calendar_that_misses_a_pnl_day(tmp_path: Path):
    with pytest.raises(ValueError, match="not on the trading calendar"):
        _path_study(drop_day=True).pipeline(tmp_path).run()
