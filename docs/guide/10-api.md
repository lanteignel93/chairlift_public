# 10. API reference

Generated from `chairlift.__all__` by `scripts/gen_api_reference.py`; do not edit by hand. Every name below imports
from the top level (`from chairlift import Study`) and is part of the public API: it changes only with a version
bump and a CHANGELOG entry. Anything else is internal, even when importable.


## `chairlift.study`

- **`Study`** (class, `chairlift.study.build`) `(…)`  
  A whole study, declared: sources, the index the folds count on, the walk-forward schedule, models, ensembles, one book and its statistics, plus any extra stages. `stages()` builds the DAG; `pipeline(root)` wraps it.
- **`Source`** (class, `chairlift.study.build`) `(…)`  
  A stage with no study inputs (or with `inputs` among other sources): it reads data or derives a frame.
- **`Model`** (class, `chairlift.study.build`) `(name: 'str', fit: 'FitSpec', panel: 'str', features: 'FeatureSet') -> None`  
  One walk-forward fit: `fit` on `features` of the source `panel`; becomes the stage `fit_<name>`.
- **`FeatureSet`** (class, `chairlift.study.build`) `(*, columns: 'tuple[str, ...]' = (), exclude: 'tuple[str, ...]' = ()) -> None`  
  Which panel columns a model reads: `columns` if given, else every column not in `exclude`.
- **`Ensemble`** (class, `chairlift.study.build`) `(…)`  
  Z-scored member predictions, weighted (equal by default); becomes the stage `ens_<name>`.
- **`CrossSectionBook`** (class, `chairlift.study.build`) `(…)`  
  A quantile book among eligible names, marked by per-position P&L paths.
- **`TimeSeriesBook`** (class, `chairlift.study.build`) `(…)`  
  A position through time from a prediction (or a rule), against constant and rule baselines.

## `chairlift.run`

- **`Stage`** (class, `chairlift.run.dag`) `(…)`  
  One node: `fn(spec?, *inputs)` with its identity (spec hash, version, captured values, code digest, input keys, and for a source its data fingerprint). A change to any of them re-keys the stage and everything downstream.
- **`Pipeline`** (class, `chairlift.run.dag`) `(…)`  
  A DAG of stages over a content-addressed store: `plan` says what would rebuild, `run` builds what is missing and records the run (signature, manifest, ledger trial, events), `load` reads a stage's output.

## `chairlift.core`

- **`spec`** (function, `chairlift.core.spec`) `(*, name: 'str | None' = None, version: 'int' = 1) -> 'Callable[[type[T]], type[T]]'`  
  Declare a frozen, keyword-only spec dataclass with an explicit identity.
- **`added`** (function, `chairlift.core.spec`) `(default: 'Any') -> 'Any'`  
  A field added to an existing spec without changing its meaning: while it holds `default` it is left out of the canonical JSON, so every spec written before the field existed keeps its hash (and its cached results).
- **`spec_hash`** (function, `chairlift.core.spec`) `(obj: 'Any') -> 'str'`  
  sha256 of the canonical JSON: the value's identity in the manifest and the ledger.

## `chairlift.data`

- **`DataRef`** (class, `chairlift.data.refs`) `(*, alias: 'str', relpath: 'str' = '') -> None`  
  DataRef(*, alias: 'str', relpath: 'str' = '')
- **`fingerprint_of`** (function, `chairlift.data.refs`) `(*refs: 'DataRef') -> 'Callable[[], str]'`  
  A source stage's fingerprint: the content of what it reads. Aliases are names, so they are left out.
- **`content_hash`** (function, `chairlift.data.refs`) `(path: 'Path', cache: 'StatCache | None' = None) -> 'str'`  
  `b3f:<hex>` for a file, `b3d:<hex>` for a directory; independent of where the data lives.

## `chairlift.schedule`

- **`WalkForward`** (class, `chairlift.schedule.walkforward`) `(…)`  
  WalkForward(*, start: 'dt.date', first_test_year: 'int', last_test_year: 'int | None' = None, end: 'dt.date | None' = None, mode: 'Mode' = 'expanding_then_rolling', roll_years: 'int' = 4, embargo: 'int' = 21, entry: 'str' = 't0', exit: 'str' = 'exit_date')
- **`make_folds`** (function, `chairlift.schedule.walkforward`) `(index: 'pl.DataFrame', w: 'WalkForward') -> 'list[Fold]'`  
  `index` needs the entry and exit columns only.

## `chairlift.learn`

- **`FitSpec`** (class, `chairlift.learn.fit`) `(…)`  
  FitSpec(*, model: 'ModelSpec', target: 'str' = 'y_z', rank_target: 'str' = 'y_rank', fill: 'float' = 0.5, selection: 'ClusterSelection | None' = None, keep_train: 'bool' = True, cross_section: 'bool' = True, allow_constant: 'bool' = False, transforms: 'tuple[Transform, ...]' = ())
- **`fit_walk_forward`** (function, `chairlift.learn.fit`) `(…)`  
  Every fold, in parallel when the run's compute allows (`compute.workers`); results are identical either way, because each fold sees only its own rows and they are assembled in fold order.
- **`zscore_ensemble`** (function, `chairlift.learn.fit`) `(…)`  
  Weighted average of each member's standardised prediction; nothing is refit.
- **`RidgeSpec`** (class, `chairlift.learn.models`) `(*, alpha: 'float' = 0.001, intercept: 'bool' = False) -> None`  
  RidgeSpec(*, alpha: 'float' = 0.001, intercept: 'bool' = False)
- **`LightGBMSpec`** (class, `chairlift.learn.models`) `(…)`  
  LightGBMSpec(*, params: 'Mapping[str, Any]', seeds: 'tuple[int, ...]' = (0,), threads: 'int' = 16, deterministic: 'bool' = True)
- **`RuleSpec`** (class, `chairlift.learn.models`) `(…)`  
  The target `y` handed to fit() must be the return the position earns (e.g. the next period's return).
- **`ClusterSelection`** (class, `chairlift.learn.selection`) `(…)`  
  ClusterSelection(*, cut: 'float' = 0.6, min_cluster: 'int' = 3, tie_tol: 'float' = 0.1, singleton_ir: 'float' = 2.0, singleton_cons: 'float' = 0.75, sample: 'int' = 300000, seed: 'int' = 0, fill: 'float' = 0.5)
- **`Winsorize`** (class, `chairlift.learn.transforms`) `(*, q: 'float' = 0.01) -> None`  
  Winsorize(*, q: 'float' = 0.01)
- **`Interactions`** (class, `chairlift.learn.transforms`) `(*, top: 'int' = 4) -> None`  
  Interactions(*, top: 'int' = 4)
- **`PCA`** (class, `chairlift.learn.transforms`) `(*, k: 'int' = 5, keep_inputs: 'bool' = False) -> None`  
  PCA(*, k: 'int' = 5, keep_inputs: 'bool' = False)

## `chairlift.book`

- **`QuantileBook`** (class, `chairlift.book.quantile`) `(*, q: 'float' = 0.1, date: 'str' = 't0') -> None`  
  QuantileBook(*, q: 'float' = 0.1, date: 'str' = 't0')
- **`SignalBook`** (class, `chairlift.book.timeseries`) `(…)`  
  SignalBook(*, kind: "Literal['scaled', 'sign']" = 'scaled', cap: 'float' = 1.0, cost: 'float' = 0.0, long_only: 'bool' = False, short_only: 'bool' = False, date: 'str' = 'date', ret: 'str' = 'r_next')

## `chairlift.evaluate`

- **`DailyStatsSpec`** (class, `chairlift.evaluate.daily`) `(…)`  
  DailyStatsSpec(*, n_boot: 'int' = 2000, seed: 'int' = 0, block: 'int' = 21, alpha: 'float' = 0.01, periods: 'int' = 252, min_per_year: 'int' = 60, min_per_half: 'int' = 40)
- **`daily_stats`** (function, `chairlift.evaluate.daily`) `(x: 'np.ndarray', dates: 'Sequence[dt.date]', s: 'DailyStatsSpec') -> 'dict[str, Any]'`  
  A return series judged: Sharpe and mean with block-bootstrap CIs, p(mean ≤ 0), Sharpe by year and half-year (with the counts of positive ones), drawdown, and how much the best five periods carry (`top5_share`, `sharpe_without_top5`).
- **`paired_sharpe_diff`** (function, `chairlift.evaluate.daily`) `(x1: 'np.ndarray', x2: 'np.ndarray', s: 'DailyStatsSpec') -> 'dict[str, Any]'`  
  Sharpe(x2) − Sharpe(x1) on the same days, with a paired moving-block bootstrap: the same blocks are drawn for both series, so the common component cancels and the interval is that of the difference itself.
- **`deflated_sharpe`** (function, `chairlift.evaluate.deflated`) `(…)`  
  `sharpes`: every trial's annualised Sharpe. Returns the best trial's DSR and the pieces behind it.
- **`factor_attribution`** (function, `chairlift.evaluate.attribution`) `(…)`  
  Regress `series` on factor return series (dates in common): α per period and annualised with its Newey-West t-statistic, betas and their t-statistics, R², the factors' correlation, and the Sharpe of α + residual.

## `chairlift.ledger`

- **`Holdout`** (class, `chairlift.ledger.holdout`) `(*, start: 'dt.date', column: 'str' = 't0') -> None`  
  Holdout(*, start: 'dt.date', column: 'str' = 't0')
- **`guard`** (function, `chairlift.ledger.holdout`) `(frame: 'pl.DataFrame', h: 'Holdout') -> 'pl.DataFrame'`  
  The frame without its holdout rows, unless the ledger records the opening.
- **`with_holdout`** (function, `chairlift.ledger.holdout`) `(fingerprint: 'Callable[[], str]') -> 'Callable[[], str]'`  
  A source fingerprint that also changes when the holdout is opened.
- **`Ledger`** (class, `chairlift.ledger.trials`) `(path: 'Path | str') -> 'None'`  
  A study's trial ledger (JSON lines): one trial per distinct run signature, with its headline, and the holdout opening, which can be recorded once.

## `chairlift.search`

- **`SearchSpace`** (class, `chairlift.search.space`) `(…)`  
  SearchSpace(params: 'dict[str, tuple[Any, ...] | Range]' = <factory>, sampler: 'str' = 'random', budget: 'int' = 20, seed: 'int' = 0, complexity: 'dict[str, dict[str, float]]' = <factory>)
- **`pbo`** (function, `chairlift.search.select`) `(…)`  
  Probability of backtest overfitting (CSCV, Bailey et al. 2015): over every split of `blocks` time blocks into halves, the share of splits in which the in-sample best candidate ranks at or below the median out of sample.
- **`walk_forward_selection`** (function, `chairlift.search.select`) `(…)`  
  Each year, the candidate with the best past Sharpe net of its complexity cost; the stitched series.
