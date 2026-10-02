# 9. Declaring a study, searching it, and keeping score

Pages 1–8 cover the runner. This page covers what runs on it:
- a declared `Study`
- the models and books it composes
- the ledger and the holdout gate that keep a study honest
- `search`, which tries many experiments and judges itself
- the synthetic twins that check all of it

Every example here is in `examples/` and runs without data.

## A study is a declaration

```python
from chairlift.study.build import CrossSectionBook, Ensemble, FeatureSet, Model, Source, Study

def study(*, alpha: float = 1e-2, q: float = 0.1) -> Study:      # keyword-only: these are the experiment's knobs
    return Study(
        name="equity_ls",
        sources=(Source("twin", cross_section_twin, spec=CrossSectionTwin(), fingerprint=lambda: "synthetic"),
                 Source("panel", panel, spec=PanelSpec(), inputs=("twin",)),
                 Source("frame", frame, inputs=("twin",))),
        index="panel",                                            # the folds count on its entry / exit dates
        schedule=WalkForward(start=dt.date(2012, 1, 1), first_test_year=2016, mode="expanding", embargo=1),
        models=(Model("ridge", FitSpec(model=RidgeSpec(alpha=alpha)), "panel", FeatureSet(exclude=KEYS)),
                Model("gbm", FitSpec(model=LightGBMSpec(params=GBM, seeds=(0, 1, 2))), "panel", FeatureSet(exclude=KEYS))),
        ensembles=(Ensemble("ensemble", ("ridge", "gbm")),),
        book=CrossSectionBook(spec=QuantileBook(q=q), frame="frame", paths=one_period_paths, per_position=True),
        stats=DailyStatsSpec(periods=12, block=3),
    )
```

`chairlift run examples/studies/equity_ls.py:study` builds this DAG:
- `twin`, `panel`, `frame`
- `folds`
- `fit_ridge`, `fit_gbm`, `ens_ensemble`
- `book_*`, `paths_*`, `eval_*`
- `report`, with every book's headline side by side

Each stage key covers its spec, every value its function captures, and the source of every study and chairlift
function it reaches ([5](05-reproducibility.md)). Changing `alpha` re-runs `fit_ridge` and what depends on it, and
nothing else.

| piece | what it is | options |
|---|---|---|
| `Source` | a frame from data or from other sources | `spec`, `fingerprint` (needed when it reads the outside world), `inputs` |
| `Model` | a `FitSpec` on a panel's columns | `FeatureSet(columns=…)` or `FeatureSet(exclude=…)` |
| `FitSpec` | what is fitted, per fold, on training rows only | `RidgeSpec(alpha, intercept)`, `LightGBMSpec(params, seeds, deterministic)`, `RuleSpec`; `selection=ClusterSelection()`; `cross_section=False` for one row per date |
| `Ensemble` | z-scored members, weighted | within date for a cross-section; by training-fold moments for a time series |
| `CrossSectionBook` | long the top `q`, short the bottom `q`, among eligible names | `paths(book)` gives each position's P&L path; `frame_for` puts a book on another universe; `per_position` |
| `TimeSeriesBook` | a position through time | `sign_models` (rules), `short_or_flat`, `constants`, `rules` (position columns), `benchmark` (every book's paired Sharpe difference) |
| `DailyStatsSpec` | the statistics | `periods` (252 daily, 12 monthly), block bootstrap `block`, `n_boot`, `seed` |

`FitSpec(transforms=…)` adds in-fold input transforms, fitted on each fold's training rows after selection and the
null fill, in order:
- `Winsorize(q)` clips each input at its training q and 1 − q quantiles
- `Interactions(top)` adds the pairwise products of the `top` inputs most correlated with the training target
- `PCA(k, keep_inputs)` replaces or extends the inputs with k principal components

A transform's choices (quantiles, chosen pairs, loadings) are as unknown to the test year as the model's weights. A
fold's `info` records them, with the model's final input names.

`RuleSpec` searches, inside each fold, one input × one training quantile × one side by the position's own
training Sharpe. "Always in" is one of its candidates. It is the model to use when the question is "when should the
position be on", not "what will the return be".

Folds run in parallel when `[compute] workers` > 1 (spawned processes). The result is identical, bit for bit, to
the serial run.

## The ledger: every experiment is a trial

Every run is charged to `studies/<name>/ledger.jsonl`. A new signature is a new trial; a re-run of the same
signature is not. When the study module declares a `HEADLINE`, the trial records it:

```python
HEADLINE = {"stage": "eval_ensemble", "sharpe": "ls.sharpe", "n_obs": "ls.n_days", "periods": 12, "daily": "daily", "value": "ls"}
```

```bash
chairlift ledger --study vxx
```

The output lists every trial, then the best Sharpe, the expected maximum of that many null trials, and the
**deflated Sharpe probability** (Bailey and López de Prado). That probability says how likely the best is to be real,
given how many were tried. Treat a best trial below about 0.95 as luck until shown otherwise.

## The holdout gate: one look, on the record

A source reads its rows through `guard(frame, Holdout(start=…, column=…))` and declares
`fingerprint=with_holdout(fp)`. Until the ledger records the opening, rows from `start` on do not exist for the
study.

```bash
chairlift holdout open --study vxx --reason "candidate A frozen in REGISTRATION.md; reading 2024-01 → 2026-08 once"
```

Opening needs a reason. It records who opened it, when, and how many trials came before. A second opening is
refused: looking again is a new registration, not a second look. Opening changes the gate's fingerprint, so every
stage behind it re-keys: no result built on sealed data is ever mixed up with one built on open data.

Gate on the *outcome* date, not the entry date. A row entered on 2023-12-28 whose 5-day target ends in January is a
holdout row (VXX's panel seals on `outcome_end`).

## Search: many experiments, judged without hindsight

```toml
# examples/experiments/equity-search.toml
study  = "../studies/equity_ls.py:study"
[search]
sampler = "random"          # or "grid"
budget  = 16
seed    = 1
[search.params]
alpha  = { low = 1e-4, high = 10.0, log = true }
q      = [0.05, 0.1, 0.2]
leaves = { low = 4, high = 32, int = true }
# objective = { stage = "...", sharpe = "...", n_obs = "...", daily = "...", value = "..." }   # default: HEADLINE
```

```bash
chairlift search examples/experiments/equity-search.toml --dry-run    # the candidates
chairlift search examples/experiments/equity-search.toml
```

Options:
- `--jobs N` runs N candidates at once, in spawned processes. The machine's fold workers are split between them,
  and appends to the shared manifest, ledger and stat cache are locked.
- `[search.complexity]`, a table of parameter → `{value = cost}`, is a prior. Walk-forward selection ranks
  candidates by past Sharpe minus their cost, so a more complex candidate has to beat the simpler ones by its cost.
  Example: `model = { gbm = 0.2, ensemble = 0.1 }`.
- `[search] objective` judges another book than the study's ledger headline.
- Candidates that build the same pipeline count once. A knob that does not reach it (a transform's size with no
  transform) would otherwise add an identical trial and inflate the deflation's N. `--dry-run` marks them `= i`.

Every candidate is an ordinary run, charged to the ledger. The search is then judged in three ways:
- **Deflated Sharpe probability of the best candidate.**
- **Walk-forward selection:** each year, the candidate with the best record up to then is chosen, and the chosen
  years are stitched together. That is what the search would actually have delivered. The gap to the best in
  hindsight is the search's optimism.
- **PBO**, the probability of backtest overfitting: over symmetric splits of time into halves, how often the
  in-sample winner lands below the median out of sample. 0.5 is a coin flip.

| on the equity twin | best | expected max of 16 nulls | DSR | walk-forward selection | PBO |
|---|---|---|---|---|---|
| planted signal | 2.12 | 0.79 | 1.000 | 1.24 | 0.008 |
| null (`equity-search-null.toml`) | 0.34 | 0.39 | 0.45 | −0.12 | 0.56 |

On the signal twin, the search finds it and its judgment says so. On the null twin, the same search reports nothing:
the best candidate is what 16 coin flips would produce. The record goes to `studies/<name>/searches/<id>.json`.

## Twins: data with a known answer

`chairlift.verify.twins` generates seeded data with planted signals:
- `cross_section_twin`: names × dates, market and sector factors, signals of a chosen IC, noise features, and a
  fundamental published `release_lag` dates late
- `time_series_twin`: one instrument, a drift, persistent predictive states and noise states

Use a twin first, on any new study shape. The twin tests check that the planted IC is recovered out of sample and that
the null twin finds nothing. They also check that the release-lag leak (`fundamentals="peek"`) inflates the IC by
about the fundamental's share. That is how a look-ahead bug shows up before it reaches real data.

## Running and watching unattended

```bash
chairlift systemd install                      # once: the OnFailure alert unit
chairlift submit examples/experiments/equity-search.toml --watchdog 45min
chairlift report                               # the newest run, as a self-contained HTML page
chairlift report --index                       # every study's latest run
chairlift alerts check                         # failed / died / stalled runs, delivered once; exit 1 if new
```

`submit` runs the study as a transient systemd user unit:
- **Watchdog:** the heartbeat is sent only while the run's events advance, so a hung stage stops it. systemd then
  kills the unit, and `OnFailure` raises the alert.
- **Alerts:** they go to the `[alerts] sinks`: `jsonl` (`<home>/alerts.jsonl`), `stdout`, or `webhook`. The webhook
  URL comes from the secret `webhook`, never from config.
- **Stalls:** a run is stalled after `[alerts] stall_minutes` without an event.
- **Timers:** put `chairlift alerts check` on a systemd timer or cron; its exit code says whether anything new
  happened.

## Factor attribution: does a book add anything?

`chairlift.evaluate.attribution.factor_attribution(series, factors, periods, lags)` regresses a book's returns on
factor books, with Newey-West errors. It reports α per period and annualised, α's t-statistic, the betas, R², and
the information ratio of the unexplained part. The equity client builds its four factor books (momentum, value,
size, low volatility) as single-feature decile books and runs the attribution as an extra stage.

