# Changelog

The public API is `chairlift.__all__` ([docs/guide/10-api.md](docs/guide/10-api.md)). Until 1.0 a minor version may
change it; every change is listed here.

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
