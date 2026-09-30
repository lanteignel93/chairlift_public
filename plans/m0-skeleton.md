# M0 Skeleton

**Status:** in-flight
**Prepared:** 2026-09-30
**Owner:** Laurent Lanteigne
**Buy-in:** Laurent — kickoff calls taken 2026-09-30 (docs/decisions.md)

---

## Problem / motivation

Every later milestone stands on a handful of foundations that are expensive to change once code depends on them:
- spec identity
- the artifact store and manifest
- the stage runner
- the core types that carry time and grouping (`AsOf`, `Grouping`, `Session`)
- the property-test harness that proves the leakage guards work

slalom's history shows what happens without them:
- three rebuild scripts
- a notebook-hash file
- a resume-safety trap that silently reused stale skew exports
- a leak test that existed for one family only
- an unguarded derived target that reached 1e82

M0 builds these foundations with no client code, so that M1 (slalom through chairlift) and M2 (VXX) plug into interfaces that already hold under test.

The Feature Creature v2 audit (2026-09-30) added three foundations that must be present from the start rather than bolted on later:
- **code identity and immutable transforms:** stored artifacts that resolve code at load time rewrite history when a helper is fixed in place
- **a data contract with release lags and lineage:** forward columns excluded by construction, not by hand
- **a `uses_target` flag on transforms:** target-encoding transforms leak into the downstream fit

## Proposed approach

Build bottom-up through the tiers, each increment landing with its test and, for logic a human should step through, a walkthrough. Order:
- identity and storage first (everything caches through them)
- then the types every protocol signature uses
- then the protocols themselves as `typing.Protocol` classes with no implementation beyond a toy one
- then the property-test harness, applied to the toy implementations
- then a toy study end to end through the runner

Nothing in M0 reads real data. The toy study is synthetic, seeded, and small enough to verify by hand. It has a planted signal and a null variant, so the calibration harness has something to calibrate against in M1.

**Why protocols now and implementations later:** the protocol signatures are where the two clients' differences must be absorbed. Writing them against a toy cross-sectional study *and* a toy time-series study in M0 (both synthetic) is the cheapest place to find that a signature assumes a cross-section. For example, a `Grouping` that cannot be empty, or a `Transform.fit` that needs a date column.

## Scope in v1 / out-of-scope

**In v1 (M0):**
- `core/spec.py`, `core/identity.py`, `data/store.py`, `data/manifest.py`, `run/dag.py` — **landed 2026-09-30**
- `core/types.py`: `AsOf`, `Grouping` (possibly empty), `Session` (episode boundaries), `Frame` alias, and the column-naming conventions (entity, t0, exit_date, known_at)
- Protocols: `Source` (with `release_lag`, `lineage`, `fingerprint`), `Universe`, `Target`, `FeatureFamily` (`version`, `lookback`), `Transform` (`causal_within_group`, `uses_target`), `Model`, `BookBuilder`, `Evaluator`, `Gate`, `Incumbent`, `Residualize`
- `Schedule` spec and fold construction with the embargo on exit dates, plus fold invariants (a port of the slalom `folds.check_folds` logic, rewritten)
- Property-test harness in `verify/`:
  - a truncation test for any `FeatureFamily` and for the target
  - a fit-on-train spy for any `Transform`
  - a planted-leak test for any `Source` set
  - each harness proven by a deliberately broken implementation that it must catch
- A toy study, cross-sectional and time-series variants, synthetic and seeded, run end to end through the runner
- A golden reproduction test: a pinned toy run's outputs compared on contents after a reinstall
- CI green on 3.12 and 3.13

**Out of scope:**
- Any real data or client code (M1/M2).
- The selection funnel, search, calibration runs at scale, the spanning test, reports beyond the manifest (M1+).
- A CLI beyond what the tests need.

## Stakeholders & buy-in

- Laurent: the owner. Walkthrough review is the joint verification step at each checkpoint.

## Preconditions

- Kickoff calls: **met** (2026-09-30).
- Private GitHub repository: **pending**, created at the first push.

## Implementation sequence

1. **Identity, store, manifest, runner** — done (27 tests, `wt_manifest.py`).
2. **Core types.** `AsOf`, `Grouping`, `Session`, the column conventions. Test that an empty grouping degrades every within-group operation to its time-only form.
3. **Protocols** as `typing.Protocol` with docstrings stating each contract. A toy implementation of each lives in `verify/toy.py`.
4. **Schedule and folds.** The spec (first test, refit cadence, window, embargo, weight half-life) and fold tables. Invariants: no training exit ≥ test start − embargo, disjointness, expanding/rolling semantics. Walkthrough `wt_folds.py`.
5. **Property-test harness.** Truncation, fit-on-train and planted leak, each with a broken twin it must catch. Walkthrough `wt_truncation.py`.
6. **Toy study end to end**, in both a cross-sectional and a time-series variant, through the runner; manifest re-run rules exercised on it.
7. **Golden reproduction test** on the pinned toy run.
8. **Push, CI, closeout.**

## Verification criteria

- `uv run pytest` passes on 3.12 and 3.13 in CI, walkthroughs included.
- Each harness test fails against its deliberately broken twin:
  - a feature family that peeks one day ahead
  - a transform that fits on test rows
  - a source that leaks a forward column
- The toy study's rebuild after a feature-spec change re-runs exactly the downstream stages (the design page's three scenarios).
- The golden test reproduces the pinned run's outputs on contents after `uv sync --reinstall`.
- The time-series toy runs through the same protocols without a single `if grouping` branch in core code.

## Open questions

- Whether `Frame` is always a polars `DataFrame` at protocol boundaries, or `LazyFrame` is allowed for sources. Lean: `LazyFrame` for sources only.
- Where the embargo is measured when a target has no exit date (a pure time-series target with horizon h): exit = t0 + h trading days, declared by the target.

## Related plans / docs / PRs

- `plans/speculative/chairlift-automated-strategy-search.md`
- `docs/decisions.md`

## Closeout / as-built

*Fill at completion.*
