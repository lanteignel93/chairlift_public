# chairlift

A machine-learning research pipeline in which the research protocol is code. A study declares its data, target,
feature families and portfolio rules; chairlift supplies point-in-time data access, walk-forward folds that assert
their own embargo, fold-local fitting, evaluation with block-bootstrap intervals and multiple-testing accounting, a
holdout the loader refuses to read, and a manifest that makes every rebuild reproducible and explainable.

- **User guide: [guide/](guide/README.md)**: setup → running → monitoring → reproducing → writing a study
- Decisions: [decisions.md](decisions.md)
- How to read the outputs: [how-to-read.md](how-to-read.md)
- Plans: [../plans/](../plans/)

## Clients

| client | shape | role |
|---|---|---|
| slalom | cross-sectional: ~60–790 single names a day, hedged 1m straddle P&L per unit vega | golden reference: milestone 1 must reproduce its dev numbers card and its holdout verdict |
| VXX | time series: one instrument, position sizing | modularity test: must run with zero core edits |
| Equity L/S | cross-section of index members, monthly, forward 21d return | fundamentals with release lags, factor neutralization, incremental mode against known factors |
| SPY timing | one instrument, daily, 1-day excess return | macro inputs with release lags, overlapping horizons (HAC), incremental mode against an incumbent |
| Intraday futures (later) | event-time, sessions | session boundaries, row volume, the full feature-search funnel |

Each study first runs on a synthetic twin with a planted signal (Laurent, 2026-09-30: test the pipeline's
versatility on several different strategies once the setup is settled, and improve it from what breaks).

## Status

Milestone 0 in progress. Landed:
- spec identity, code identity, the content store, and the manifest (one file per host in a shared store)
- the DAG runner with plan keys and run signatures, and run records with an environment hash
- layered site configuration with provenance, and secrets
- data references with blake3 content fingerprints and a stat cache
- experiment files with typed parameters, and sweeps
- `rerun` and `compare`
- the event stream and the animated live view
- the CLI: `run`, `plan`, `show`, `signature`, `sweep`, `log`, `runs`, `rerun`, `compare`, `watch`, `status`,
  `config`

M1 core (2026-10-01):
- walk-forward folds, cross-section and time-series transforms
- ridge and LightGBM, in-fold cluster selection, the z-score ensemble
- quantile and signal books, and daily-book statistics

Both required clients run on it. slalom's candidate reproduces bit for bit; VXX ran with three small,
recorded changes. See `examples/gallery/`.

240 tests mirroring `src/`, three walkthroughs, and a user guide whose CLI reference is generated and checked.
