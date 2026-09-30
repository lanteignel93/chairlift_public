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

## Status

Milestone 0 in progress. Landed: spec identity (`core/spec.py`), code identity (`core/identity.py`), the artifact
store and manifest (`data/`), and the DAG runner (`run/dag.py`), with tests and `debug_walkthroughs/wt_manifest.py`.
