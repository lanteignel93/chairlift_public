"""Walk-forward folds with an embargo the folds assert themselves.

One fold per calendar test year. A row is a position entered at `entry` whose outcome is known at `exit`. A row trains
a fold only if its exit is at least `embargo` trading days before the fold's first test entry, so no training outcome
overlaps the test period; its entry must also be inside the training window:

- expanding: from `start` until `roll_years` calendar years are available,
- rolling: the `roll_years` calendar years before the test year.

The trading-day calendar the embargo counts on is every entry and every exit date seen in the index. Nothing here reads
a target; the same folds serve every model, and `check_folds` is the invariant every fit asserts.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

import polars as pl

from chairlift.core.spec import spec

Mode = Literal["expanding", "rolling", "expanding_then_rolling"]


def _date(v: object) -> dt.date:
    if not isinstance(v, dt.date):
        raise TypeError(f"expected a date, got {type(v).__name__}")
    return v


@spec(name="WalkForward", version=1)
class WalkForward:
    start: dt.date
    first_test_year: int
    last_test_year: int | None = None
    end: dt.date | None = None  # last entry date of the sample (a dev end); later rows are ignored
    mode: Mode = "expanding_then_rolling"
    roll_years: int = 4
    embargo: int = 21
    entry: str = "t0"
    exit: str = "exit_date"


@dataclass(frozen=True)
class Fold:
    test_year: int
    train_start: dt.date  # first allowed entry in training
    train_exit_cut: dt.date  # last allowed exit in training
    test_start: dt.date
    test_end: dt.date
    mode: str

    def train_mask(self, w: WalkForward) -> pl.Expr:
        return (pl.col(w.entry) >= self.train_start) & (pl.col(w.exit) <= self.train_exit_cut)

    def test_mask(self, w: WalkForward) -> pl.Expr:
        return (pl.col(w.entry) >= self.test_start) & (pl.col(w.entry) <= self.test_end)


def _sample(index: pl.DataFrame, w: WalkForward) -> pl.DataFrame:
    keep = pl.col(w.entry) >= w.start
    if w.end is not None:
        keep = keep & (pl.col(w.entry) <= w.end)
    return index.filter(keep)


def trading_days(index: pl.DataFrame, w: WalkForward) -> list[dt.date]:
    d = _sample(index, w)
    return sorted(set(d[w.entry].unique().to_list()) | set(d[w.exit].unique().to_list()))


def make_folds(index: pl.DataFrame, w: WalkForward) -> list[Fold]:
    """`index` needs the entry and exit columns only."""
    tdays = trading_days(index, w)
    pos = {d: i for i, d in enumerate(tdays)}
    entries = _sample(index, w)[w.entry]
    latest = _date(entries.max())
    last = w.last_test_year or latest.year
    folds: list[Fold] = []
    for year in range(w.first_test_year, last + 1):
        test_days = [
            d for d in tdays if d.year == year
        ]  # the calendar, not only entries: an exit-only day still counts
        if not test_days:
            continue
        test_start, test_end = test_days[0], test_days[-1]
        i = pos[test_start] - w.embargo
        if i < 0:
            raise ValueError(f"fold {year}: fewer than {w.embargo} trading days before its first test entry")
        cut = tdays[i]
        available = year - w.start.year
        if w.mode == "expanding" or (w.mode == "expanding_then_rolling" and available <= w.roll_years):
            train_start, mode = w.start, "expanding"
        else:
            train_start, mode = dt.date(year - w.roll_years, 1, 1), "rolling"
        folds.append(Fold(year, max(train_start, w.start), cut, test_start, test_end, mode))
    return folds


def check_folds(index: pl.DataFrame, folds: list[Fold], w: WalkForward, keys: tuple[str, ...] = ()) -> pl.DataFrame:
    """Assert every fold's invariants and return the fold table.

    - the embargo: at least `w.embargo` trading days between the last training exit and the first test entry
    - no training entry inside the test period; the test is exactly the fold's test year
    - training and test rows disjoint on `keys` (when given)
    """
    tdays = trading_days(index, w)
    pos = {d: i for i, d in enumerate(tdays)}
    sample = _sample(index, w)
    rows: list[dict[str, object]] = []
    for f in folds:
        tr = sample.filter(f.train_mask(w))
        te = sample.filter(f.test_mask(w))
        if tr.height == 0 or te.height == 0:
            raise AssertionError(f"fold {f.test_year}: {tr.height} training and {te.height} test rows")
        last_exit, first_test = _date(tr[w.exit].max()), _date(te[w.entry].min())
        gap = pos[first_test] - pos[last_exit]
        if gap < w.embargo:
            raise AssertionError(f"fold {f.test_year}: embargo {gap} < {w.embargo} trading days")
        if not _date(tr[w.entry].max()) < first_test:
            raise AssertionError(f"fold {f.test_year}: a training entry falls inside the test period")
        if {d.year for d in te[w.entry].unique().to_list()} != {f.test_year}:
            raise AssertionError(f"fold {f.test_year}: test rows outside the test year")
        if keys and tr.join(te, on=list(keys), how="inner").height:
            raise AssertionError(f"fold {f.test_year}: training and test rows overlap on {keys}")
        rows.append(
            {
                "test_year": f.test_year,
                "mode": f.mode,
                "train_start": f.train_start,
                "train_exit_cut": f.train_exit_cut,
                "test_start": f.test_start,
                "test_end": f.test_end,
                "train_rows": tr.height,
                "test_rows": te.height,
                "gap_td": gap,
            }
        )
    return pl.DataFrame(rows)


def fold_table_json(folds: list[Fold]) -> list[dict[str, object]]:
    """The folds as plain JSON (a stage output)."""
    return [
        {
            "test_year": f.test_year,
            "train_start": f.train_start.isoformat(),
            "train_exit_cut": f.train_exit_cut.isoformat(),
            "test_start": f.test_start.isoformat(),
            "test_end": f.test_end.isoformat(),
            "mode": f.mode,
        }
        for f in folds
    ]


def folds_from_json(rows: list[dict[str, object]]) -> list[Fold]:
    return [
        Fold(
            int(str(r["test_year"])),
            dt.date.fromisoformat(str(r["train_start"])),
            dt.date.fromisoformat(str(r["train_exit_cut"])),
            dt.date.fromisoformat(str(r["test_start"])),
            dt.date.fromisoformat(str(r["test_end"])),
            str(r["mode"]),
        )
        for r in rows
    ]
