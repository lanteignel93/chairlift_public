"""Build the gallery: figures and tables from finished runs of the slalom and VXX client studies.

    uv run --extra ml python examples/gallery/build.py

Reads only run records and stored outputs under the configured home (`chairlift config show`); imports no client
code and reads no client data. Writes PNG figures and `numbers.json` next to this file. Nothing it writes is data:
every figure and number is an aggregate of a run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from chairlift.core.config import load_config
from chairlift.data.store import ArtifactStore

HERE = Path(__file__).parent
FIG = HERE / "figures"
INK, MUTED, GRID = "#1f2430", "#7b8496", "#e6e8ee"
C = {"chairlift": "#3b6fd8", "slalom": "#e0823d", "long": "#2e9e6a", "short": "#c94f4f", "ls": "#3b6fd8"}
VXX = {"ridge": "#3b6fd8", "gbm": "#8b5cd6", "ensemble": "#2e9e6a", "always_short": "#e0823d", "always_long": "#c94f4f"}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.color": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 9,
    }
)


def records(home: Path, study: str) -> list[dict[str, Any]]:
    out = [json.loads(p.read_text()) for p in (home / "studies" / study / "runs").glob("*Z-*.json")]
    return sorted((r for r in out if r.get("status") == "ok"), key=lambda r: r["run_id"])


def latest_with(recs: list[dict[str, Any]], stages: list[str]) -> dict[str, Any]:
    for r in reversed(recs):
        if all(s in r.get("results", {}) for s in stages):
            return r
    raise LookupError(f"no finished run holds {stages}")


def load(store: ArtifactStore, rec: dict[str, Any], stage: str) -> Any:
    return store.get(rec["results"][stage]["output"])


# ---- slalom -------------------------------------------------------------------------------------------------------


def slalom(store: ArtifactStore, home: Path, ref_dir: Path | None) -> dict[str, Any]:
    rec = latest_with(records(home, "slalom"), ["replication", "eval_BG_nxy", "eval_A", "folds"])
    R = load(store, rec, "replication")
    E = load(store, rec, "eval_BG_nxy")
    folds = load(store, rec, "folds")

    day = pl.DataFrame(E["daily"]).with_columns(pl.col("date").str.to_date())
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    for col, lab in (("long", "long decile"), ("short", "short decile (held short)"), ("ls", "long − short")):
        y = day[col].to_numpy()
        ax[0].plot(
            day["date"], np.cumsum(-y if col == "short" else y), color=C[col], lw=1.4 if col == "ls" else 1.0, label=lab
        )
    ax[0].set_title("slalom candidate BG_nxy: daily book, dev 2017–2023 walk-forward")
    ax[0].set_ylabel("cumulative volpts per unit vega")
    ax[0].legend(loc="upper left")
    yrs = E["ls"]["by_year_sharpe"]
    ax[1].bar([str(y) for y in yrs], list(yrs.values()), color=C["ls"])
    ax[1].axhline(E["ls"]["sharpe"], color=INK, lw=0.8, ls="--")
    ax[1].text(0, E["ls"]["sharpe"] + 0.1, f"all years {E['ls']['sharpe']:.2f}", color=INK, fontsize=8)
    ax[1].set_title("daily L−S Sharpe by year")
    fig.tight_layout()
    fig.savefig(FIG / "slalom_book.png")
    plt.close(fig)

    # replication: our cumulative L−S over slalom's own, and the per-model prediction agreement
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
    if ref_dir is not None and (ref_dir / "daily_BG_nxy_v500_liq50.parquet").exists():
        ref = pl.read_parquet(ref_dir / "daily_BG_nxy_v500_liq50.parquet").filter(pl.col("date") <= day["date"].max())
        ax[0].plot(
            ref["date"],
            np.cumsum(ref["ls"].to_numpy()),
            color=C["slalom"],
            lw=4,
            alpha=0.45,
            label="slalom (its own code)",
        )
    ax[0].plot(
        day["date"],
        np.cumsum(day["ls"].to_numpy()),
        color=C["chairlift"],
        lw=1.1,
        label="chairlift (generic components)",
    )
    ax[0].set_title("the same book, built twice: cumulative daily L−S")
    ax[0].legend(loc="upper left")
    models = list(R["predictions"])
    diffs = [max(R["predictions"][m]["max_abs_diff"], 1e-18) for m in models]
    ax[1].barh(models, diffs, color=C["chairlift"])
    ax[1].set_xscale("log")
    ax[1].set_xlim(1e-17, 1e-12)
    ax[1].set_title("max |pred − slalom pred| (912,043 rows)")
    fig.tight_layout()
    fig.savefig(FIG / "slalom_replication.png")
    plt.close(fig)

    # the folds: training windows, the embargo, the test years
    T = pl.DataFrame(folds["table"]).with_columns(
        pl.col(c).str.to_date() for c in ("train_start", "train_exit_cut", "test_start", "test_end")
    )
    fig, ax = plt.subplots(figsize=(11, 2.6))
    for i, r in enumerate(T.iter_rows(named=True)):
        ax.barh(i, (r["train_exit_cut"] - r["train_start"]).days, left=r["train_start"], color="#c9d6f2", height=0.6)
        ax.barh(i, (r["test_end"] - r["test_start"]).days, left=r["test_start"], color=C["chairlift"], height=0.6)
        ax.text(r["test_end"], i, f"  {r['mode']}, embargo {r['gap_td']} td", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(range(T.height), [str(y) for y in T["test_year"]])
    ax.invert_yaxis()
    ax.set_title("walk-forward folds: training window (light), test year (dark); the gap is the 21-trading-day embargo")
    fig.tight_layout()
    fig.savefig(FIG / "slalom_folds.png")
    plt.close(fig)
    return {"run_id": rec["run_id"], "signature": rec["signature"], "replication": R, "folds": folds["table"]}


# ---- VXX ----------------------------------------------------------------------------------------------------------


def vxx(store: ArtifactStore, home: Path) -> dict[str, Any]:
    recs = records(home, "vxx")
    dev = next(
        r
        for r in reversed(recs)
        if r["meta"].get("params", {}) in ({}, {"horizon": 5, "cost_bp": 5.0}) and "evaluate" in r["results"]
    )
    E = load(store, dev, "evaluate")
    idx = load(store, dev, "index")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax[0].semilogy(idx["date"], idx["level"], color=INK, lw=1)
    ax[0].set_title("the short-term VIX futures index, rebuilt from VX settlements (start = 100)")
    w = idx.filter(pl.col("date").is_between(pl.date(2023, 1, 1), pl.date(2023, 6, 30)))
    ax[1].plot(w["date"], w["w1"], color=C["chairlift"], lw=1)
    ax[1].set_title("front-month weight: the daily roll")
    ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(FIG / "vxx_index.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    names = ["ridge", "gbm", "ensemble", "always_short", "always_long"]
    for n in names:
        d = pl.DataFrame(E[n]["daily"]).with_columns(pl.col("date").str.to_date())
        ax[0].plot(
            d["date"],
            np.cumsum(d["pnl"].to_numpy()),
            color=VXX[n],
            lw=1.3 if n.startswith("always") else 1.0,
            label=n.replace("_", " "),
        )
    ax[0].set_title("VXX: out-of-sample position P&L, 5 bp per unit traded, 2017–2023")
    ax[0].set_ylabel("cumulative return (sum of daily)")
    ax[0].legend(loc="upper left", ncol=3, fontsize=7)
    sh = [E[n]["sharpe"] for n in names]
    lo = [E[n]["sharpe"] - E[n]["sharpe_ci"][0] for n in names]
    hi = [E[n]["sharpe_ci"][1] - E[n]["sharpe"] for n in names]
    ax[1].barh(names, sh, xerr=[lo, hi], color=[VXX[n] for n in names], error_kw={"ecolor": MUTED, "lw": 0.8})
    ax[1].axvline(0, color=INK, lw=0.8)
    ax[1].invert_yaxis()
    ax[1].set_title("daily Sharpe, block-bootstrap CI99")
    fig.tight_layout()
    fig.savefig(FIG / "vxx_books.png")
    plt.close(fig)

    sweep_ids = [r["meta"]["sweep"]["id"] for r in recs if "sweep" in r.get("meta", {})]
    cells: list[dict[str, Any]] = []
    if sweep_ids:
        sid = sweep_ids[-1]
        for r in recs:
            if r.get("meta", {}).get("sweep", {}).get("id") == sid:
                ev = load(store, r, "evaluate")
                v = r["meta"]["sweep"]["values"]
                cells.append(
                    {
                        **v,
                        **{n: ev[n]["sharpe"] for n in ("ridge", "gbm", "ensemble", "always_short")},
                        "run_id": r["run_id"],
                    }
                )
        S = pl.DataFrame(cells).sort("horizon", "cost_bp")
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        x = np.arange(S.height)
        for k, n in enumerate(("ridge", "gbm", "ensemble", "always_short")):
            ax.bar(x + (k - 1.5) * 0.2, S[n], width=0.2, color=VXX[n], label=n.replace("_", " "))
        ax.set_xticks(x, [f"h={h}\n{c:g} bp" for h, c in zip(S["horizon"], S["cost_bp"], strict=True)])
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_title("illustrative sweep (6 cells, one command): daily Sharpe by target horizon and cost")
        ax.legend(ncol=4, loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / "vxx_sweep.png")
        plt.close(fig)
    summary = {
        n: {
            k: E[n][k]
            for k in (
                "sharpe",
                "sharpe_ci",
                "ann_return",
                "max_drawdown",
                "years_pos",
                "n_years",
                "mean_abs_position",
                "turnover_per_day",
            )
        }
        for n in names
    }
    for n in ("ridge", "gbm", "ensemble"):
        summary[n]["oos_ic"] = E[n]["ic"]["oos_ic_all"]
    return {
        "run_id": dev["run_id"],
        "signature": dev["signature"],
        "books": summary,
        "sweep": cells,
        "index_days": idx.height,
        "regime": _vxx_regime(store, recs),
    }


def _vxx_regime(store: ArtifactStore, recs: list[dict[str, Any]]) -> dict[str, Any]:
    """Always short against the futures-contango rule and LightGBM on a 1-day target (the v4 evaluation)."""

    def latest(h: int) -> dict[str, Any] | None:
        for r in reversed(recs):
            ev = r.get("results", {}).get("evaluate")
            if ev and r["meta"].get("params", {}).get("horizon") == h:
                out = store.get(ev["output"])
                if "short_if_roll_yield_gt_0.0%" in out:
                    return out
        return None

    e5, e1 = latest(5), latest(1)
    if e5 is None or e1 is None:
        return {}
    lines = {
        "always short": e5["always_short"],
        "short if F2 > F1 (futures contango)": e5["short_if_roll_yield_gt_0.0%"],
        "LightGBM, 1-day target": e1["gbm"],
    }
    colors = [VXX["always_short"], "#2e9e6a", "#8b5cd6"]
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for (name, st), c in zip(lines.items(), colors, strict=True):
        d = pl.DataFrame(st["daily"]).with_columns(pl.col("date").str.to_date())
        cum = np.cumsum(d["pnl"].to_numpy())
        label = f"{name}: Sharpe {st['sharpe']:.2f}, max DD {st['max_drawdown']:.2f}"
        ax[0].plot(d["date"], cum, color=c, lw=1.2, label=label)
        ax[1].plot(d["date"], cum - np.maximum.accumulate(cum), color=c, lw=1.0)
    ax[0].set_title("VXX: can regime conditioning beat always short? (2017–2023 OOS, 5 bp per unit traded)")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[1].set_title("drawdown (cumulative return below its running peak)")
    fig.tight_layout()
    fig.savefig(FIG / "vxx_regime.png")
    plt.close(fig)
    keep = ("sharpe", "sharpe_ci", "max_drawdown", "vs_always_short")
    return {name: {k: st[k] for k in keep} for name, st in lines.items()}


def _searches(home: Path, study: str, experiment: str | None = None) -> list[dict[str, Any]]:
    """A study's search records, oldest first; with `experiment`, only that experiment file's searches."""
    d = home / "studies" / study / "searches"
    out = [json.loads(f.read_text()) for f in sorted(d.glob("*.json"))] if d.exists() else []
    return [r for r in out if experiment is None or r["experiment"]["name"] == experiment]


def vxx_holdout(store: ArtifactStore, home: Path) -> dict[str, Any]:
    """Candidate A and always short, 2017 → 2026, with the one holdout look shaded."""
    rec = latest_with(records(home, "vxx"), ["holdout_look", "eval_baselines"])
    look = load(store, rec, "holdout_look")
    if look.get("sealed"):
        return {"sealed": True}
    E = load(store, rec, "eval_baselines")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for name, c in (("always_short", VXX["always_short"]), ("candidate_A", "#2e9e6a")):
        d = pl.DataFrame(E[name]["daily"]).with_columns(pl.col("date").str.to_date())
        ax.plot(d["date"], np.cumsum(d["pnl"].to_numpy()), color=c, lw=1.2, label=name.replace("_", " "))
    import datetime as _dt

    ax.axvspan(_dt.date(2024, 1, 1), _dt.date(2026, 8, 31), color="#e9edf5", zorder=0)
    ax.text(_dt.date(2024, 2, 1), ax.get_ylim()[1] * 0.92, "holdout: one look", color=MUTED, fontsize=8)
    ax.set_title(
        f"VXX candidate A, registered then read once: holdout Sharpe {look['candidate_A']['sharpe']:.2f} vs "
        f"{look['always_short']['sharpe']:.2f} for always short (FAIL)"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "vxx_holdout.png")
    plt.close(fig)
    return {k: look[k] for k in ("window", "candidate_A", "always_short", "vs_always_short", "gates", "pass")}


def equity(store: ArtifactStore, home: Path) -> dict[str, Any]:
    recs = records(home, "equity_ls")
    rec = next(
        r
        for r in reversed(recs)
        if "incremental" in r["results"]
        and r["meta"].get("params", {}).get("fundamentals", "published") == "published"
        and not r["meta"].get("search")
    )
    E, F, inc = load(store, rec, "eval_main"), load(store, rec, "factors"), load(store, rec, "incremental")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    d = pl.DataFrame(E["daily"]).with_columns(pl.col("date").str.to_date())
    ax[0].plot(d["date"], np.cumsum(d["ls"].to_numpy()), color=INK, lw=1.6, label="model (ensemble)")
    for (name, factor_series), c in zip(F.items(), ["#3b6fd8", "#e0823d", "#8b5cd6", "#2e9e6a"], strict=False):
        dd = sorted((k, v) for k, v in factor_series.items() if k >= str(d["date"].min()))
        ax[0].plot(
            [pl.Series([k]).str.to_date()[0] for k, _ in dd],
            np.cumsum([v for _, v in dd]),
            color=c,
            lw=1,
            label=name.replace("_", " "),
        )
    ax[0].set_title("S&P 500 decile L/S 2015–19 (gross): model vs known factors")
    ax[0].legend(loc="upper left", fontsize=8, ncol=2)
    searches = _searches(home, "equity_ls", "structure-search")
    out: dict[str, Any] = {"incremental": inc, "run_id": rec["run_id"]}
    if searches:
        srch = searches[-1]
        xs = [c["sharpe"] for c in srch["candidates"]]
        ys = [c["sharpe_without_top5"] or 0 for c in srch["candidates"]]
        ax[1].scatter(xs, ys, s=18, color="#3b6fd8")
        ax[1].axvline(srch["deflated"]["expected_max_sharpe_null"], color="#c94f4f", lw=1, ls="--")
        ax[1].text(
            srch["deflated"]["expected_max_sharpe_null"],
            min(ys),
            f" expected max\n of {srch['deflated']['n_trials']} nulls",
            color="#c94f4f",
            fontsize=7,
        )
        ax[1].set_xlabel("Sharpe")
        ax[1].set_ylabel("Sharpe without its 5 best months")
        ax[1].set_title(f"structural search: DSR {srch['deflated']['dsr']:.2f}, PBO {srch['pbo']['pbo']:.2f}")
        out["search"] = {k: srch[k] for k in ("deflated", "walk_forward_selection", "pbo")}
    tf = _searches(home, "equity_ls", "transform-search")
    if tf:
        out["transform_search"] = {k: tf[-1][k] for k in ("deflated", "walk_forward_selection", "pbo")}
    fig.tight_layout()
    fig.savefig(FIG / "equity_ls.png")
    plt.close(fig)
    return out


def spy_timing(store: ArtifactStore, home: Path) -> dict[str, Any]:
    rec = latest_with(records(home, "spy_timing"), ["report", "eval_ensemble", "eval_baselines"])
    E, B, rep = load(store, rec, "eval_ensemble"), load(store, rec, "eval_baselines"), load(store, rec, "report")
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for name, st, c in (
        ("buy and hold", B["buy_and_hold"], INK),
        ("ensemble (long-or-flat)", E["book"], "#3b6fd8"),
        ("volatility target", B["vol_target_long"], "#2e9e6a"),
        ("200-day trend", B["trend_200"], "#e0823d"),
    ):
        d = pl.DataFrame(st["daily"]).with_columns(pl.col("date").str.to_date())
        ax.plot(d["date"], np.cumsum(d["pnl"].to_numpy()), color=c, lw=1.2, label=f"{name}: Sharpe {st['sharpe']:.2f}")
    ax.set_title("SPY timing, 2014–2019 OOS: nothing beats holding")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "spy_timing.png")
    plt.close(fig)
    return {"books": rep["books"], "run_id": rec["run_id"]}


def spy_search(home: Path) -> dict[str, Any]:
    """The same 12 candidates judged twice: on the book's own return, and on its return over buy-and-hold."""
    srch = _searches(home, "spy_timing", "transform-search")
    if len(srch) < 2:
        return {}
    own, act = srch[0], srch[-1]
    by = {json.dumps(c["params"], sort_keys=True): c["sharpe"] for c in act["candidates"]}
    rows = sorted(
        ((json.dumps(c["params"], sort_keys=True), c["params"], c["sharpe"]) for c in own["candidates"]),
        key=lambda r: -r[2],
    )
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[2] for r in rows], 0.4, color="#9aa2b5", label=f"own return: DSR {own['deflated']['dsr']:.2f}")
    ax.bar(
        x + 0.2,
        [by[r[0]] for r in rows],
        0.4,
        color="#3b6fd8",
        label=f"over buy-and-hold: DSR {act['deflated']['dsr']:.2f}",
    )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x, [f"{r[1]['transform']}\n{r[1]['model']}" for r in rows], fontsize=7)
    ax.set_ylabel("Sharpe")
    ax.set_title("SPY timing search: the same candidates, judged on the right series")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "spy_search.png")
    plt.close(fig)
    return {
        "own_return": {k: own[k] for k in ("deflated", "walk_forward_selection", "pbo")},
        "active_return": {k: act[k] for k in ("deflated", "walk_forward_selection", "pbo")},
    }


def twin_search(home: Path) -> dict[str, Any]:
    srch = _searches(home, "equity_twin")
    if len(srch) < 2:
        return {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for a, s_, title in ((ax[0], srch[-2], "planted signal"), (ax[1], srch[-1], "null twin")):
        xs = sorted((c["sharpe"] for c in s_["candidates"]), reverse=True)
        a.bar(range(len(xs)), xs, color="#3b6fd8")
        a.axhline(s_["deflated"]["expected_max_sharpe_null"], color="#c94f4f", ls="--", lw=1)
        a.axhline(s_["walk_forward_selection"]["selected_oos_sharpe"], color="#2e9e6a", lw=1.5)
        a.set_title(f"{title}: DSR {s_['deflated']['dsr']:.2f} · PBO {s_['pbo']['pbo']:.2f}")
        a.set_xlabel("candidates, best first")
    ax[0].set_ylabel("Sharpe")
    note = "red dashed: expected max of the null trials · green: walk-forward selection, no hindsight"
    fig.suptitle(note, fontsize=8, color=MUTED, y=0.02)
    fig.tight_layout()
    fig.savefig(FIG / "twin_search.png")
    plt.close(fig)
    return {
        "signal": {k: srch[-2][k] for k in ("deflated", "walk_forward_selection", "pbo")},
        "null": {k: srch[-1][k] for k in ("deflated", "walk_forward_selection", "pbo")},
    }


def main() -> None:
    cfg = load_config().config
    FIG.mkdir(exist_ok=True)
    store = ArtifactStore(cfg.paths.store_dir())
    home = cfg.paths.home
    ref = cfg.data.get("slalom")
    numbers = {
        "slalom": slalom(store, home, ref / "wf" if ref else None),
        "vxx": vxx(store, home),
        "vxx_holdout": vxx_holdout(store, home),
        "equity_ls": equity(store, home),
        "spy_timing": spy_timing(store, home),
        "spy_search": spy_search(home),
        "twin_search": twin_search(home),
    }
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=1, default=str))
    print(f"wrote {len(list(FIG.glob('*.png')))} figures and numbers.json under {HERE}")


if __name__ == "__main__":
    main()
