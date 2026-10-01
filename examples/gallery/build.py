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


def main() -> None:
    cfg = load_config().config
    FIG.mkdir(exist_ok=True)
    store = ArtifactStore(cfg.paths.store_dir())
    home = cfg.paths.home
    ref = cfg.data.get("slalom")
    numbers = {"slalom": slalom(store, home, ref / "wf" if ref else None), "vxx": vxx(store, home)}
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=1, default=str))
    print(f"wrote {len(list(FIG.glob('*.png')))} figures and numbers.json under {HERE}")


if __name__ == "__main__":
    main()
