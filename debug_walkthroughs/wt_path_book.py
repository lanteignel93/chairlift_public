"""Walkthrough: a path book, from predictions to its gain over constant size, on numbers small enough to check by hand.

    uv run python debug_walkthroughs/wt_path_book.py          # asserted run
    uv run python debug_walkthroughs/wt_path_book.py --pdb    # step through

Three trades, one entered a day, each marked daily until it exits. One fold: the model saw three training
predictions (1, 2, 3) and now predicts the three test entries 2.5, 0.5 and 4.0. The walkthrough sizes them with the
rank rule, marks the sized book and the constant book on a five-day calendar, and checks every number:
- weights: 2.5 beats two of {1, 2, 3} → 2 × 2/3; 0.5 beats none of {1, 2, 3, 2.5} → 0; 4.0 beats all five → 2
- the first mark of a trade is its P&L from entry to that mark (a cost paid at entry lands on the entry day)
- the book is zero on a calendar day with nothing live, and a P&L day missing from the calendar is refused
- the active series is the sized book minus the constant book, day by day

Under --pdb, useful breakpoints:
  chairlift/book/path.py   sizes        `ref` is the sorted reference set; watch it grow by each test prediction
  chairlift/book/path.py   daily_pnl    `t` after the step column: each row's change × unit × weight × scale
  chairlift/book/path.py   on_calendar  `missing` must be empty; `cal` is the window [lo, hi]
What to expect:
- weights 1.333…, 0, 2
- sized book by day: −0.667, +2, +1.333, +6, 0; constant book: −0.5, +1.5, −1, +3, 0
"""

from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import polars as pl

from chairlift.book.path import PathSizing, daily_pnl, on_calendar, sizes

D = [dt.date(2024, 3, 4) + dt.timedelta(days=i) for i in range(5)]  # Monday to Friday


def main() -> None:
    preds = pl.DataFrame(
        {
            "t0": [D[0], D[1], D[2], D[0], D[1], D[2]],
            "pred": [1.0, 2.0, 3.0, 2.5, 0.5, 4.0],
            "fold": [2024] * 6,
            "is_train": [True, True, True, False, False, False],
        }
    )
    # trade A (entered Monday): −0.5 at entry (the cost), then +1.5, +1.0 → held to Wednesday
    # trade B (entered Tuesday): 0, then −2 → stopped Wednesday
    # trade C (entered Wednesday): 0, then +3 Thursday → expires Thursday
    trades = pl.DataFrame(
        {
            "t0": [D[0], D[0], D[0], D[1], D[1], D[2], D[2]],
            "date": [D[0], D[1], D[2], D[1], D[2], D[2], D[3]],
            "pnl": [-0.5, 1.0, 2.0, 0.0, -2.0, 0.0, 3.0],
        }
    )
    s = PathSizing(rule="rank")
    w = sizes(preds, s)
    print(w)
    assert np.allclose(w["w"].to_numpy(), [2 * 2 / 3, 0.0, 2.0])

    cal = pl.DataFrame({"date": D})
    lo, hi = D[0], D[4]
    sized = on_calendar(daily_pnl(trades, w, s), cal, lo, hi)
    const = on_calendar(daily_pnl(trades, w.with_columns(w=pl.lit(1.0)), s), cal, lo, hi)
    print(sized.join(const, on="date", suffix="_constant"))
    assert np.allclose(sized["pnl"].to_numpy(), [-0.5 * 4 / 3, 1.5 * 4 / 3, 1.0 * 4 / 3 - 2 * 0, 3.0 * 2, 0.0])
    assert np.allclose(const["pnl"].to_numpy(), [-0.5, 1.5, 1.0 - 2.0, 3.0, 0.0])
    active = sized["pnl"].to_numpy() - const["pnl"].to_numpy()
    assert np.isclose(active.sum(), sized["pnl"].to_numpy().sum() - const["pnl"].to_numpy().sum())

    try:
        on_calendar(daily_pnl(trades, w, s), cal.filter(pl.col("date") != D[3]), lo, hi)
    except ValueError as exc:
        print("refused:", exc)
    else:
        raise AssertionError("a P&L day missing from the calendar must be refused")
    print("walkthrough passed")


if __name__ == "__main__":
    if "--pdb" in sys.argv:
        import pdb

        pdb.set_trace()  # `s` into main(), then `b chairlift/book/path.py:<sizes line>`
    main()
