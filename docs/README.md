# chairlift

A machine-learning research pipeline in which the research protocol is code. A study declares its data, target,
feature families and portfolio rules; chairlift supplies point-in-time data access, walk-forward folds that assert
their own embargo, fold-local fitting, evaluation with block-bootstrap intervals and multiple-testing accounting, a
holdout the loader refuses to read, and a manifest that makes every rebuild reproducible and explainable.

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

Milestone 0 in progress. Landed: spec identity, code identity, the content store and manifest, the DAG runner with
plan keys and run signatures, run records, layered site configuration with provenance and secrets, the event stream
and the animated live view, and the CLI (`run`, `plan`, `show`, `log`, `signature`, `runs`, `watch`, `status`,
`config`). 129 tests mirroring `src/`, two walkthroughs.
