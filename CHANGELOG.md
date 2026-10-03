# Changelog

The public API is `chairlift.__all__` ([docs/guide/10-api.md](docs/guide/10-api.md)). Until 1.0 a minor version may
change it; every change is listed here.

## 0.2.0 (2026-10-03)

A review after five real studies, every one null or failed out of sample, found a pipeline that keeps results
honest but could not tell "no edge" from "cannot see". This release adds the means to tell them apart, and models
for weak signals.

- **Power:** `evaluate.power`: `fama_macbeth_ic` (mean per-date IC, Newey-West t), `mde_ic`, `mde_sharpe`,
  `power_report`. Every study gets a `power` stage (`Study(power=PowerSpec(...))`, `None` to turn it off), shown in
  its report. Cross-section evaluations report `ic_fm`.
- **Realistic twins and the power function:**
  - `CrossSectionTwin(style_vol, idio_df)` adds date-to-date factor returns on every feature, and Student-t
    idiosyncratic returns.
  - `verify.power.detection_rates` / `power_curve` measure the power on them; `examples/power/power_curves.py` runs
    the grid.
  - The default twin is unchanged byte for byte.
- **Weak-signal models:** `CompositeSpec` (sign-prior equal weights, optionally learned signs); `RidgeSpec(prior)`
  (shrink toward a prior direction). Estimators receive their input names.
- `CompositeSpec` with no signed input predicts a constant (no evidence, no tilt) instead of raising.
- **Neutralization:** `features.cross_section.residualize` (per-date OLS on group dummies and exposures).
- **Fixes:**
  - LightGBM keeps a study's own `objective` (it was overwritten with L2 regression).
  - Training rows without a target are dropped from the fit (they were trained as 0).
  - The DSR uses the best candidate's skewness and kurtosis; ledger trials now record them.
  - In-fold transforms fit only on rows with a target.
- **Behaviour changes:**
  - `RuleSpec(margin="deflated")` is the default: a rule must beat "always in" by the expected maximum Sharpe of its
    noise candidates.
  - `WalkForward(mode="expanding")` is the default (was `expanding_then_rolling`); slalom's client now states its
    rolling design explicitly.

## 0.1.2 (2026-10-02)

- **Refit seam in trailing pools fixed:** `QuantileBook(pool_scale="train")`, the default, standardises each prediction
  by its own fold's training-row moments before pooling. A pool spanning a refit no longer compares two models'
  prediction scales. Within a fold this changes no rank. On the PEAD client: ridge 0.33 → 0.30, LightGBM 0.20 →
  0.18, ensemble 0.28 → 0.29, all inside their CIs. `pool_scale="raw"` keeps 0.1.1's behaviour.
- `ty` (Astral's type checker) passes. Polars reductions in typed code go through numpy or lists, where polars types
  every scalar as a union of all literal types.

## 0.1.1 (2026-10-02)

Found by the fifth client (event-driven, pooled events):
- **Identity fix:** a module-level value a stage reads (`START = date(...)`, a parameter table) is now part of its
  code digest. Before, editing one re-keyed nothing and reused stale results. Values that cannot be canonical
  (frames, locks) are skipped.
- Digest keys no longer include line numbers: moving a function in its file does not re-key.
- Every code digest changes with this release, so the first run after upgrading rebuilds every stage.
- `chairlift show` no longer charges a ledger trial (`Pipeline.run(charge=False)`).
- `Ensemble.cross_section`: by default it follows the members. Pooled fits (`FitSpec(cross_section=False)`) are
  z-scored by training moments even under a cross-section book.
- LightGBM's numpy feature-name warning is silenced at predict.
- **Event books:** `QuantileBook(pool_days, min_pool)` ranks each row against the trailing pool of earlier rows, not
  within its date. `CrossSectionBook(min_live)` starts the daily series once both sides hold that many positions. A
  pool book reads its OOS IC across each year's rows. A study with no evaluations has no `report` stage. `CrossSectionBook(paths_inputs)`: a
  paths function may read other stages (`paths(book, *outputs)`), e.g. a price grid.

## 0.1.0 (2026-10-02)

First versioned release: the runner, the research layer and the operations layer, run by four real clients without
core changes.

- **Runner:** content-addressed store and manifest; stage keys over spec, captured values, the source of reached code
  (independent of the hash seed and of the loader) and data fingerprints; run signatures and records; `rerun`,
  `compare`, the environment hash; parquet outputs rechunked so serial and parallel runs write identical bytes.
- **Configuration:** layered site config with provenance, profiles and host maps, secrets, data aliases.
- **Research layer:** `Study` declarations; walk-forward folds that check their embargo, fitted in parallel;
  ridge, LightGBM (deterministic), regime rules; in-fold cluster selection; in-fold transforms (`Winsorize`,
  `Interactions`, `PCA`); z-score ensembles; quantile and signal books; daily-book statistics with block-bootstrap
  CIs and concentration checks; paired Sharpe differences; factor attribution with Newey-West errors.
- **Keeping score:** the trial ledger (one trial per signature, with its headline), deflated Sharpe, the holdout gate
  (`guard`, `with_holdout`, `chairlift holdout open`).
- **Search:** grid and seeded random samplers, ranges, complexity priors, an objective override, `--jobs`; judged by
  deflated Sharpe, walk-forward selection and PBO; candidates with identical signatures count once.
- **Operations:** the event stream and live view, `watch`, `status`; HTML reports; `submit` under systemd with a
  progress-tied watchdog; alerts (failed, died, stalled) to jsonl, stdout or a webhook.
- **Judging time series:** `active_<book>` stages (book minus benchmark, day by day) whenever a time-series study's
  benchmark is one of its baselines, so a search can judge timing rather than the premium the benchmark earns.
- **Trials:** with a HEADLINE, a trial is keyed on the headline stage's key, not the whole run signature. Adding a
  report or diagnostic stage no longer charges the same experiment again. Ledgers written before this change carry no
  trial key, so an unchanged experiment whose signature moved is charged once more, a single time.
- **Specs:** `added(default)` declares a field added to an existing spec without changing its hash.
- **Packaging:** typed wheel, a lazy top-level API, generated CLI and API references checked by the tests.
