# 6. Writing a study

A study is a Python **factory** in its own repository: a function that takes a run root and keyword parameters and
returns a `Pipeline`. The factory's keyword parameters, with their types and defaults, are the study's whole public
surface: they are what `--set` and experiment files set ([7](07-experiments.md)). Today a study wires its stages by
hand. The study protocols (universe, target, feature families, walk-forward schedule, book, evaluation), which will
replace most of that wiring, are milestone 1.

## A complete small study

`examples/momentum_study.py` in this repository: cross-sectional momentum on daily closes, read through a data
alias. `tests/test_docs.py` runs it on synthetic prices, and checks that this page quotes it verbatim.

```python
from __future__ import annotations

from pathlib import Path

import polars as pl

from chairlift.core.spec import spec
from chairlift.data.refs import DataRef, fingerprint_of
from chairlift.run import events
from chairlift.run.dag import Pipeline, Stage

STUDY_NAME = "momentum"  # runs go to <home>/studies/momentum/

PRICES = DataRef(alias="vendor", relpath="daily_prices")  # resolved through [data] vendor = "..."


@spec(name="MomentumFeature", version=1)
class FeatureSpec:
    lookback: int = 20
    skip: int = 1


@spec(name="MomentumBook", version=1)
class BookSpec:
    quantile: float = 0.1


def prices() -> pl.DataFrame:
    return pl.read_parquet(PRICES.path() / "*.parquet")  # columns: date, ticker, close


def features(s: FeatureSpec, prices: pl.DataFrame) -> pl.DataFrame:
    close = pl.col("close")
    return (
        prices.sort("ticker", "date")
        .with_columns(
            mom=(close.shift(s.skip) / close.shift(s.lookback) - 1).over("ticker"),
            fwd=(close.shift(-1) / close - 1).over("ticker"),
        )
        .drop_nulls(["mom", "fwd"])
    )


def book(s: BookSpec, features: pl.DataFrame) -> dict[str, float]:
    events.progress(0, 2, "ranking")  # shows in the live view; never part of a key
    ranked = features.with_columns(pct=(pl.col("mom").rank() / pl.len()).over("date"))
    long = pl.col("fwd").filter(pl.col("pct") > 1 - s.quantile).mean()
    short = pl.col("fwd").filter(pl.col("pct") <= s.quantile).mean()
    daily = ranked.group_by("date").agg((long - short).alias("ls")).drop_nulls().sort("date")["ls"]
    events.progress(1, 2, "scoring")
    mean, sd = float(daily.mean()), float(daily.std())  # pyright: ignore[reportArgumentType]
    sharpe = mean / sd * 252**0.5 if sd > 0 else 0.0
    events.metric("ls_sharpe", sharpe)
    events.progress(2, 2, f"{daily.len()} days")
    return {"days": daily.len(), "ls_mean_bp": round(mean * 1e4, 3), "ls_sharpe": round(sharpe, 3)}


def pipeline(root: Path, *, lookback: int = 20, skip: int = 1, quantile: float = 0.1) -> Pipeline:
    return Pipeline(
        [
            Stage("prices", prices, fingerprint=fingerprint_of(PRICES)),
            Stage("features", features, ("prices",), FeatureSpec(lookback=lookback, skip=skip)),
            Stage("book", book, ("features",), BookSpec(quantile=quantile)),
        ],
        root,
    )
```

```bash
# with [data] vendor = "/path/to/vendor" in the site config, and daily_prices/*.parquet under it
uv run chairlift run examples/momentum_study.py:pipeline --set lookback=60 --reason "longer lookback"
```

In a study repository the factory would be a module, `my_study.study:pipeline`, rather than a file.

## Stages

`Stage(name, fn, inputs=(), spec=None, version=1, fingerprint=None)`

- **`fn`** is called as `fn(spec, **inputs)` when the stage has a spec, and `fn(**inputs)` otherwise. Inputs are
  passed by stage name, so the parameter names must match the input stage names.
- **Outputs** are a polars `DataFrame`, stored as parquet, or plain JSON: dicts, lists, strings, numbers, booleans,
  None. Tuples come back as lists, and NaN or infinity is refused. Anything else fails the stage with a clear error.
- **A stage must be a pure function of its spec and inputs.** Anything else it reads (the clock, an environment
  variable, a file not declared as data) makes the cache lie.
- **A stage with no inputs reads the outside world**, so it needs a `fingerprint`, a zero-argument callable returning
  a string. Without one it runs every time: chairlift never assumes external data is unchanged. For data on disk use
  `fingerprint_of(ref, ...)`. For data generated from the spec alone, as in the toy, a constant such as `lambda:
  "synthetic"` is right, because the spec already carries the identity.
- **`version`**: bump it whenever the stage computes something different without its spec changing (a bug fix, a new
  formula). Every cached result built from the old meaning is rebuilt. Never change what a released stage computes
  without bumping it.

## Specs

```python
@spec(name="MomentumFeature", version=1)
class FeatureSpec:
    lookback: int = 20
```

A spec is a frozen, keyword-only dataclass with an explicit identity. Its hash is the sha256 of a canonical JSON form,
which is stable across runs, processes and machines:
- `1` and `1.0` hash differently, on purpose.
- `-0.0` equals `0.0`.
- NaN and infinity are refused.
- Lists and dicts are frozen on construction.
- Nested specs, enums, dates and sets are allowed.
- A plain dataclass is refused; declare it with `@spec`.

`name` keeps the identity stable if you rename the class. Bump `version` when the meaning of the fields changes.

## Data references

`DataRef(alias="vendor", relpath="daily_prices")` names data without a path. Inside a run, `ref.path()` resolves the
alias through the machine's `[data]` table ([2](02-configure.md)). It is an error, naming the configured aliases, if
the alias is unknown or the path does not exist. Outside a run, nothing is bound and `path()` says so.

`fingerprint_of(*refs)` hashes the content of every reference in order ([5](05-reproducibility.md)), so the
signature follows the data, not where it is. A `DataRef` is itself a spec, so it can sit inside another spec: its
identity is the alias and relative path, never the resolved directory.

## Progress, metrics, warnings

Inside a stage, `chairlift.run.events` writes to the current run's event stream, tagged with the stage name:

```python
events.progress(done, total, "fold 3 of 10")   # the live view's bar and detail columns
events.metric("oos_ic", 0.031, fold=3)          # name, value, any dimensions; `compare` diffs the last values
events.warning("212 rows dropped: iv < 7%")     # shown in yellow; does not fail the stage
```

All three are no-ops outside a run, so the same function works in a notebook or a test. They never enter a key or an
output, so instrumenting a stage never invalidates its cache.

## Naming and layout

`STUDY_NAME` in the factory's module sets the directory under `<home>/studies/`. Without it, the module name is used
(plus `-<factory>` when the factory is not called `pipeline`), and `--name` overrides both. A study repository looks
like:

```
my-study/
├── pyproject.toml            depends on chairlift (pinned in uv.lock)
├── src/my_study/study.py     the factory and its stages
├── experiments/*.toml        the runs you actually make (7)
└── tests/test_study.py       run the pipeline on a tmp root, twice; assert the second run is all hits
```

## Testing a study

```python
def test_study_runs_and_is_cached(tmp_path):
    data = tmp_path / "vendor" / "daily_prices"
    data.mkdir(parents=True)
    make_tiny_prices().write_parquet(data / "a.parquet")
    pipe = pipeline(tmp_path / "root").rebase(tmp_path / "root", data={"vendor": tmp_path / "vendor"})
    assert pipe.run().ran() == ["prices", "features", "book"]
    assert pipe.run().ran() == []                                  # every key unchanged: all hits
```

`Pipeline(stages, root, store=None, *, data=None, cache=None)`:
- `rebase(root, store, data=..., cache=...)` returns the same stages over another layout. The CLI does exactly this
  with the configured paths.
- `plan()`, `signature()` and `run(targets, reason, meta, listeners)` are the same operations the CLI exposes.
- `pipe.load(name, report)` returns a stage's stored output.

Next: [experiments and sweeps](07-experiments.md).
