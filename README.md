# chairlift

> [!NOTE]
> **This is the public copy of chairlift's development history.** Development continues in a private repository;
> machine-specific paths and internal links in `plans/` and `docs/` were generalized for this copy. To request access
> to the private repository, please email laurent.lanteigne@gmail.com.

A machine-learning research pipeline in which the research protocol is code. A study declares its data, folds,
models and book; chairlift runs it so that the usual ways a backtest lies are either impossible or recorded:

- **In-fold or it is not a number.** Selection, transforms, ensembling and search are fitted on training rows only;
  every fold re-checks its own embargo before it fits.
- **The holdout does not exist until it is opened.** Sealed rows are dropped at the loader. Opening takes a reason,
  happens once, is written to the study's ledger, and re-keys everything behind it.
- **Every run is a trial.** The ledger counts distinct experiments, so the best one is judged by its deflated
  Sharpe; a search is judged by walk-forward selection (what it would have chosen at the time) and by PBO.
- **Same inputs, same bytes.** Stages are keyed by spec, captured values, the source of the code they reach and the
  content of their data. `chairlift rerun` rebuilds a recorded run and says which stage differs, and why.

```python
# a sketch of examples/studies/equity_ls.py, which runs as is on a synthetic twin
from chairlift import CrossSectionBook, DailyStatsSpec, Ensemble, FeatureSet, FitSpec, Interactions, LightGBMSpec
from chairlift import Model, QuantileBook, RidgeSpec, Source, Study, WalkForward

def study(*, alpha: float = 1e-2, q: float = 0.1, top: int = 0) -> Study:   # keyword-only knobs: what a search varies
    tf = (Interactions(top=top),) if top else ()                          # fitted in each fold, on training rows
    inputs = FeatureSet(exclude=KEYS)
    return Study(
        name="equity_ls",
        sources=(Source("twin", twin, spec=TwinSpec(), fingerprint=lambda: "synthetic"),
                 Source("panel", panel, inputs=("twin",)),
                 Source("frame", frame, inputs=("twin",))),
        index="panel",                                                      # the folds count on its dates
        schedule=WalkForward(start=dt.date(2012, 1, 1), first_test_year=2016, mode="expanding", embargo=1),
        models=(Model("ridge", FitSpec(model=RidgeSpec(alpha=alpha), transforms=tf), "panel", inputs),
                Model("gbm", FitSpec(model=LightGBMSpec(params=GBM, seeds=(0, 1, 2)), transforms=tf), "panel", inputs)),
        ensembles=(Ensemble("ensemble", ("ridge", "gbm")),),
        book=CrossSectionBook(spec=QuantileBook(q=q), frame="frame", paths=one_period_paths),
        stats=DailyStatsSpec(periods=12, block=3),
    )
```

```bash
uv sync --extra ml
uv run chairlift run examples/studies/equity_ls.py:study             # a synthetic twin with a planted signal
uv run chairlift search examples/experiments/equity-search.toml      # 16 candidates, judged without hindsight
uv run chairlift ledger --study equity_twin                          # trials, deflated Sharpe, holdout state
uv run chairlift report --index                                      # HTML run reports
```

## Clients

chairlift is developed against real studies of very different shapes. None needed a change to the core to run, and
most of them are honest negatives: the pipeline's job is to make that verdict cheap and believable.

| client | shape | verdict |
|---|---|---|
| slalom | ~800 single-name option straddles a day, cross-sectional ML | dev 2.16, holdout 0.45: failed (published post-mortem) |
| VXX | one instrument, regime timing of short VIX futures | holdout 0.24 vs 0.50 always-short: failed |
| equity L/S | S&P 500, monthly, fundamentals as filed | dev null (−0.51); search DSR 0.39, PBO 0.42 |
| SPY timing | one instrument, macro with publication lags | dev null (0.95 vs buy-and-hold 0.90) |

The [gallery](examples/gallery/README.md) shows each one's figures, and the search judgment on synthetic twins with
and without a planted signal (DSR 1.00 / PBO 0.01 against 0.45 / 0.56).

## Documentation

- [User guide](docs/guide/README.md): install, configure, first run, monitoring, reproducibility, writing a study,
  experiments, the CLI reference, studies and search, the API reference.
- [How to read the outputs](docs/how-to-read.md) and [decisions](docs/decisions.md).
- [CHANGELOG](CHANGELOG.md).

Status: 0.1.0, alpha. The names in `chairlift.__all__` are the public API; everything else may change.
