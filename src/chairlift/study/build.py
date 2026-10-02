"""A study, declared: sources, a schedule, models, ensembles, a book and an evaluation. chairlift builds the DAG.

    Study(
        name="momentum",
        sources=(Source("panel", panel_fn, spec=PanelSpec(), fingerprint=fingerprint_of(DATA)),),
        index="panel",                                  # the source whose entry / exit dates the folds count on
        schedule=WalkForward(start=..., first_test_year=2016),
        models=(Model("ridge", FitSpec(model=RidgeSpec()), panel="panel", features=FeatureSet(exclude=KEYS)),),
        ensembles=(Ensemble("blend", members=("ridge", "gbm")),),
        book=CrossSectionBook(...) | TimeSeriesBook(...),
        stats=DailyStatsSpec(),
    ).pipeline(root)

Stages it builds, by name:
- every source
- `folds`
- `fit_<model>` for each model, and `ens_<ensemble>` for each ensemble
- `eval_<name>` for each model and ensemble the book evaluates (with `book_<name>` and `paths_<name>` for a
  cross-section)
- for a time series, `eval_baselines`, and `active_<name>` when the benchmark is a baseline: the book's return minus
  the benchmark's, the series to judge when the benchmark itself earns a premium
- `report`: every evaluation's headline numbers side by side
- any `extra` stages the study adds (a replication check, a diagnostic)

Each piece stays a spec or a plain function, so every stage key covers exactly what it depends on: spec hashes,
captured values and code digests (`run.dag`). The study's experiment parameters are the arguments of the function
that returns the Study, as before.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from chairlift.book.quantile import QuantileBook, quantile_book
from chairlift.book.timeseries import SignalBook, constant_book, rule_book, signal_book
from chairlift.core.spec import spec
from chairlift.evaluate.daily import (
    DailyStatsSpec,
    book_stats,
    daily_book,
    daily_stats,
    ic_by_year,
    paired_sharpe_diff,
    ts_ic_by_year,
)
from chairlift.learn.fit import FitSpec, fit_walk_forward, zscore_ensemble
from chairlift.run import events
from chairlift.run.dag import Pipeline, Stage
from chairlift.schedule.walkforward import WalkForward, check_folds, fold_table_json, folds_from_json, make_folds


@spec(name="FeatureSet", version=1)
class FeatureSet:
    """Which panel columns a model reads: `columns` if given, else every column not in `exclude`."""

    columns: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def resolve(self, frame: pl.DataFrame) -> list[str]:
        if self.columns:
            missing = [c for c in self.columns if c not in frame.columns]
            if missing:
                raise KeyError(f"feature(s) not in the panel: {missing}")
            return list(self.columns)
        return [c for c in frame.columns if c not in set(self.exclude)]


@dataclass(frozen=True)
class Source:
    """A stage with no study inputs (or with `inputs` among other sources): it reads data or derives a frame."""

    name: str
    fn: Callable[..., Any]
    spec: Any = None
    fingerprint: Callable[[], str] | None = None
    inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Model:
    """One walk-forward fit: `fit` on `features` of the source `panel`; becomes the stage `fit_<name>`."""

    name: str
    fit: FitSpec
    panel: str
    features: FeatureSet


@dataclass(frozen=True)
class Ensemble:
    """Z-scored member predictions, weighted (equal by default); becomes the stage `ens_<name>`.

    Within date when the book is a cross-section and every member was fitted cross-sectionally; otherwise by each
    fold's training-row moments (a time series, or pooled events with a handful a day). `cross_section` overrides."""

    name: str
    members: tuple[str, ...]
    weights: tuple[float, ...] = ()  # default: equal
    cross_section: bool | None = None


@dataclass(frozen=True)
class CrossSectionBook:
    """A quantile book among eligible names, marked by per-position P&L paths.

    `frame` is the source with keys, `eligible`, the target rank (for the IC) and anything the paths need.
    `paths(book)` returns (keys, date, pnl) cumulative per position; `end` cuts the daily series.
    """

    spec: QuantileBook
    frame: str
    paths: Callable[..., pl.DataFrame]
    paths_fingerprint: Callable[[], str] | None = None
    paths_inputs: tuple[str, ...] = ()  # other stages `paths` reads: paths(book, *those outputs), e.g. a price grid
    end: Any = None
    per_position: bool = False  # Sharpe of the equal-weight spread (ls_pp) instead of the unit-per-name book
    ic_target: str = "y_rank"
    frame_for: tuple[tuple[str, str], ...] = ()  # (evaluation name, frame source) where a book uses another universe
    min_live: int = 0  # the daily series starts once both sides hold this many live positions (an event book's ramp)

    def frame_of(self, name: str) -> str:
        return dict(self.frame_for).get(name, self.frame)


@dataclass(frozen=True)
class TimeSeriesBook:
    """A position through time from a prediction (or a rule), against constant and rule baselines.

    `frame` is the source with the date, the next-period return and any rule-position columns.
    """

    spec: SignalBook
    frame: str
    sign_models: tuple[str, ...] = ()  # models whose prediction already is a position (rules)
    short_or_flat: bool = False  # also evaluate each model as a short-or-flat book
    constants: tuple[tuple[str, float], ...] = (("always_long", 1.0), ("always_short", -1.0))
    rules: tuple[tuple[str, str], ...] = ()  # (book name, position column in frame)
    benchmark: str = "always_short"  # every book's paired Sharpe difference is against this one
    ic_target: str = "y"


@dataclass
class Study:
    """A whole study, declared: sources, the index the folds count on, the walk-forward schedule, models, ensembles,
    one book and its statistics, plus any extra stages. `stages()` builds the DAG; `pipeline(root)` wraps it."""

    name: str
    sources: Sequence[Source]
    index: str
    schedule: WalkForward
    models: Sequence[Model]
    book: CrossSectionBook | TimeSeriesBook
    stats: DailyStatsSpec
    ensembles: Sequence[Ensemble] = ()
    evaluate: Sequence[str] | None = None  # which models / ensembles get a book (default: all)
    keys: tuple[str, ...] = ("root", "t0")
    extra: Sequence[Stage] = field(default_factory=list[Stage])

    def pipeline(self, root: Path | str) -> Pipeline:
        return Pipeline(self.stages(), root)

    def stages(self) -> list[Stage]:
        w, keys = self.schedule, self.keys
        out: list[Stage] = [Stage(s.name, s.fn, s.inputs, spec=s.spec, fingerprint=s.fingerprint) for s in self.sources]
        out.append(Stage("folds", _folds_stage(self.index, keys), (self.index,), spec=w))
        for m in self.models:
            out.append(Stage(f"fit_{m.name}", _fit_stage(m.panel, w, keys, m.features), (m.panel, "folds"), spec=m.fit))
        for e in self.ensembles:
            weights = e.weights or tuple(1 / len(e.members) for _ in e.members)
            ins = tuple(f"fit_{x}" for x in e.members)
            fits = {m.name: m.fit for m in self.models}
            cs = (
                e.cross_section
                if e.cross_section is not None
                else isinstance(self.book, CrossSectionBook) and all(fits[x].cross_section for x in e.members)
            )
            out.append(Stage(f"ens_{e.name}", _ensemble_stage(ins, weights, w.entry, cs), ins))
        names = (
            list(self.evaluate)
            if self.evaluate is not None
            else [*(m.name for m in self.models), *(e.name for e in self.ensembles)]
        )
        pred_stage = {m.name: f"fit_{m.name}" for m in self.models} | {e.name: f"ens_{e.name}" for e in self.ensembles}
        if isinstance(self.book, CrossSectionBook):
            b = self.book
            for n in names:
                p, fr = pred_stage[n], b.frame_of(n)
                out.append(Stage(f"book_{n}", _cs_book_stage(p, fr), (p, fr), spec=b.spec))
                out.append(
                    Stage(
                        f"paths_{n}",
                        _cs_paths_stage(f"book_{n}", b.paths, b.paths_inputs),
                        (f"book_{n}", *b.paths_inputs),
                        fingerprint=b.paths_fingerprint,
                    )
                )
                ins = (f"book_{n}", f"paths_{n}", p, fr)
                pooled = b.spec.pool_days > 0
                fn = _cs_eval_stage(ins, keys, b.end, b.per_position, b.ic_target, b.min_live, pooled)
                out.append(Stage(f"eval_{n}", fn, ins, spec=self.stats))
        else:
            b = self.book
            for n in names:
                p = pred_stage[n]
                kind: Literal["scaled", "sign"] = "sign" if n in b.sign_models else b.spec.kind
                fn = _ts_eval_stage(p, b.frame, kind, b.short_or_flat, b.ic_target)
                out.append(Stage(f"eval_{n}", fn, (b.frame, p), spec=_TsEval(stats=self.stats, book=b.spec)))
            fn = _ts_baselines_stage(b.frame, b.constants, b.rules, self.schedule)
            out.append(Stage("eval_baselines", fn, (b.frame, "folds"), spec=_TsEval(stats=self.stats, book=b.spec)))
            if b.benchmark in {c for c, _ in b.constants} | {r for r, _ in b.rules}:
                for n in names:
                    fn = _ts_active_stage(f"eval_{n}", b.benchmark)
                    out.append(Stage(f"active_{n}", fn, (f"eval_{n}", "eval_baselines"), spec=self.stats))
        evals = tuple(f"eval_{n}" for n in names) + (
            ("eval_baselines",) if isinstance(self.book, TimeSeriesBook) else ()
        )
        bench = self.book.benchmark if isinstance(self.book, TimeSeriesBook) else None
        if evals:  # a study that books everything in extra stages has nothing to report here
            out.append(Stage("report", _report_stage(evals, bench), evals, spec=self.stats))
        return out + list(self.extra)


@spec(name="TimeSeriesEvaluation", version=1)
class _TsEval:
    stats: DailyStatsSpec
    book: SignalBook


# ---- stage functions: each a closure over specs and functions only, so captured-value hashing covers it -------------


def _folds_stage(index: str, keys: tuple[str, ...]) -> Callable[..., dict[str, Any]]:
    def folds(w: WalkForward, **inputs: pl.DataFrame) -> dict[str, Any]:
        frame = inputs[index]
        fs = make_folds(frame, w)
        table = check_folds(frame, fs, w, keys=keys)
        return {"folds": fold_table_json(fs), "table": table.with_columns(pl.col(pl.Date).cast(pl.String)).to_dicts()}

    return folds


def _fit_stage(panel: str, w: WalkForward, keys: tuple[str, ...], features: FeatureSet) -> Callable[..., pl.DataFrame]:
    def fit(s: FitSpec, **inputs: Any) -> pl.DataFrame:
        frame = inputs[panel]
        preds, infos = fit_walk_forward(
            frame, folds_from_json(inputs["folds"]["folds"]), w, features.resolve(frame), s, keys=keys
        )
        for i in infos:
            events.metric("test_rows", float(i["test_rows"]), fold=i["test_year"])
        return preds

    return fit


def _ensemble_stage(
    ins: tuple[str, ...], weights: tuple[float, ...], date: str, cs: bool
) -> Callable[..., pl.DataFrame]:
    def ensemble(**inputs: pl.DataFrame) -> pl.DataFrame:
        return zscore_ensemble([inputs[i] for i in ins], list(weights), date=date, cross_section=cs)

    return ensemble


def _cs_book_stage(pred: str, frame: str) -> Callable[..., pl.DataFrame]:
    def book(b: QuantileBook, **inputs: pl.DataFrame) -> pl.DataFrame:
        X = quantile_book(inputs[pred], inputs[frame], b)
        legs = X.filter(pl.col("side") == "long").group_by(b.date).len()["len"].to_numpy()
        events.metric("legs_per_side_per_day", float(legs.mean()) if len(legs) else 0.0)
        return X

    return book


def _cs_paths_stage(
    book: str, paths: Callable[..., pl.DataFrame], extra: tuple[str, ...] = ()
) -> Callable[..., pl.DataFrame]:
    def p(**inputs: Any) -> pl.DataFrame:
        return paths(inputs[book], *(inputs[e] for e in extra))

    return p


def _cs_eval_stage(
    ins: tuple[str, ...],
    keys: tuple[str, ...],
    end: Any,
    per_position: bool,
    ic_target: str,
    min_live: int = 0,
    pooled_ic: bool = False,
) -> Callable[..., dict[str, Any]]:
    book_n, paths_n, pred_n, frame_n = ins

    def evaluate(s: DailyStatsSpec, **inputs: pl.DataFrame) -> dict[str, Any]:
        bk, frame = inputs[book_n], inputs[frame_n]
        day = daily_book(bk.filter(pl.col("side").is_not_null()), inputs[paths_n], keys=keys, end=end)
        if per_position:
            day = day.with_columns(ls=pl.col("ls_pp"))
        if min_live > 0:
            first = day.filter((pl.col("n_long") >= min_live) & (pl.col("n_short") >= min_live))["date"].min()
            day = day.filter(pl.col("date") >= first) if first is not None else day.clear()
        out = book_stats(day, s)
        oos = inputs[pred_n].filter(~pl.col("is_train")).select(*keys, "pred")
        joined = frame.join(oos, on=list(keys), how="inner")
        # events a few a day: a within-date IC is noise or undefined, so read it across the year's rows
        out["ic"] = (
            ts_ic_by_year(joined, target=ic_target, date=keys[-1])
            if pooled_ic
            else ic_by_year(joined, target=ic_target, date=keys[-1])
        )
        out["daily"] = (
            day.with_columns(pl.col("date").cast(pl.String))
            .select("date", "long", "short", "ls", "n_long", "n_short")
            .to_dicts()
        )
        events.metric("sharpe", out["ls"]["sharpe"])
        return out

    return evaluate


def _ts_eval_stage(
    pred: str, frame: str, kind: Literal["scaled", "sign"], short_or_flat: bool, ic_target: str
) -> Callable[..., dict[str, Any]]:
    def evaluate(s: _TsEval, **inputs: pl.DataFrame) -> dict[str, Any]:
        P, F, b = inputs[pred], inputs[frame], s.book
        variants = {
            "book": SignalBook(
                kind=kind,
                cap=b.cap,
                cost=b.cost,
                long_only=b.long_only,
                short_only=b.short_only,
                date=b.date,
                ret=b.ret,
            )
        }
        if short_or_flat:
            variants["short_or_flat"] = SignalBook(
                kind=kind, cap=b.cap, cost=b.cost, short_only=True, date=b.date, ret=b.ret
            )
        out: dict[str, Any] = {}
        for label, bb in variants.items():
            B = signal_book(P, F, bb)
            st = daily_stats(B["pnl"].to_numpy(), B[b.date].to_list(), s.stats)
            pos = B["position"].to_numpy()
            st["share_short"] = float((pos < 0).mean())
            st["mean_abs_position"] = float(np.abs(pos).mean())
            st["daily"] = B.select(pl.col(b.date).cast(pl.String).alias("date"), "position", "pnl").to_dicts()
            out[label] = st
        out["ic"] = ts_ic_by_year(
            P.filter(~pl.col("is_train")).join(F.select(b.date, ic_target), on=b.date), target=ic_target, date=b.date
        )
        events.metric("sharpe", out["book"]["sharpe"])
        return out

    return evaluate


def _ts_active_stage(ev: str, benchmark: str) -> Callable[..., dict[str, Any]]:
    """A book's return minus the benchmark's, day by day: what a search should judge when the benchmark itself earns
    a premium (a long-only timing book is mostly the market; its own Sharpe says little about the timing)."""

    def active(s: DailyStatsSpec, **inputs: Any) -> dict[str, Any]:
        mine = {d["date"]: d["pnl"] for d in inputs[ev]["book"]["daily"]}
        base = {d["date"]: d["pnl"] for d in inputs["eval_baselines"][benchmark]["daily"]}
        dates = sorted(set(mine) & set(base))
        x = np.array([mine[d] - base[d] for d in dates])
        st = daily_stats(x, [dt.date.fromisoformat(d) for d in dates], s)
        st["benchmark"] = benchmark
        st["daily"] = [{"date": d, "pnl": float(v)} for d, v in zip(dates, x, strict=True)]
        events.metric("active_sharpe", st["sharpe"])
        return st

    return active


def _ts_baselines_stage(
    frame: str, constants: tuple[tuple[str, float], ...], rules: tuple[tuple[str, str], ...], w: WalkForward
) -> Callable[..., dict[str, Any]]:
    def baselines(s: _TsEval, **inputs: Any) -> dict[str, Any]:
        F, folds = inputs[frame], inputs["folds"]["folds"]
        import datetime as dt

        start, end = dt.date.fromisoformat(folds[0]["test_start"]), dt.date.fromisoformat(folds[-1]["test_end"])
        out: dict[str, Any] = {}
        books = {n: constant_book(F, pos, s.book, start, end) for n, pos in constants}
        books |= {n: rule_book(F, col, s.book, start, end) for n, col in rules}
        for n, B in books.items():
            st = daily_stats(B["pnl"].to_numpy(), B[s.book.date].to_list(), s.stats)
            st["share_short"] = float((B["position"].to_numpy() < 0).mean())
            st["daily"] = B.select(pl.col(s.book.date).cast(pl.String).alias("date"), "position", "pnl").to_dicts()
            out[n] = st
        return out

    return baselines


def _report_stage(evals: tuple[str, ...], benchmark: str | None) -> Callable[..., dict[str, Any]]:
    """Every book's headline in one table; with a benchmark, each book's paired Sharpe difference against it."""

    def report(s: DailyStatsSpec, **inputs: dict[str, Any]) -> dict[str, Any]:
        rows: dict[str, dict[str, Any]] = {}
        series: dict[str, dict[str, float]] = {}
        for ev in evals:
            res = inputs[ev]
            if "ls" in res:  # a cross-section evaluation
                rows[ev.removeprefix("eval_")] = _headline(res["ls"])
                series[ev.removeprefix("eval_")] = {d["date"]: d["ls"] for d in res["daily"]}
                continue
            for label, raw in res.items():
                if not isinstance(raw, dict) or "sharpe" not in raw:
                    continue
                st = cast("dict[str, Any]", raw)
                name = (
                    ev.removeprefix("eval_")
                    if label == "book"
                    else (label if ev == "eval_baselines" else f"{ev.removeprefix('eval_')}_{label}")
                )
                rows[name] = _headline(st)
                daily: list[dict[str, Any]] = st["daily"]
                series[name] = {d["date"]: d["pnl"] for d in daily}
        if benchmark and benchmark in series:
            base = series[benchmark]
            for name, pnl_by_date in series.items():
                common = sorted(set(base) & set(pnl_by_date))
                import numpy as np

                rows[name]["vs_" + benchmark] = paired_sharpe_diff(
                    np.array([base[d] for d in common]), np.array([pnl_by_date[d] for d in common]), s
                )
        return {"books": rows}

    return report


def _headline(st: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "sharpe",
        "sharpe_ci",
        "n_days",
        "max_drawdown",
        "years_pos",
        "n_years",
        "top5_share",
        "sharpe_without_top5",
    )
    return {k: st.get(k) for k in keep}
