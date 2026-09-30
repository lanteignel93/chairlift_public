# Decisions

ADR-lite log: one dated entry per non-obvious choice, with the reason and what was rejected. Consult before
re-litigating a choice; append, never rewrite — supersede with a new entry.

## 2026-09-30 — Kickoff calls (Laurent)

- **Repository:** `lanteignel93/chairlift` on the personal GitHub, private until milestone 1 passes (the reference study
  reproduced through the package); the core goes public then. Study clients live in their own private repositories.
  Rejected: public from the first commit (early design churn in the open); a self-hosted GitLab (the package is meant to
  be portable and to hold no firm data or vendor code).
- **Toolchain:** uv, Python ≥ 3.12, ruff, pytest, CI on push. Rejected: reusing the existing 3.10 research venv (ties a
  portable package to a shared firm environment and an older interpreter).
- **Two required clients:** slalom (cross-sectional, closed after a failed holdout) and VXX (time series). Any core
  change either needs is recorded as a finding about the abstraction.
- **Orchestration:** a small in-house DAG runner over a content-addressed store. Research needs determinism and
  provenance more than scheduling. Rejected for the core: Kedro, Prefect, Dagster, Hamilton; any of them can wrap the
  runner later for production scheduling.
- **Spec format:** frozen, keyword-only dataclasses (`@spec(name, version)`) hashed as canonical JSON. Rejected: YAML
  (loses types, invites stringly-typed options).
- **Port or rewrite:** port the numerical kernels (target engine, surface, feature families) verbatim under golden
  tests; rewrite orchestration against the protocols. Rejected: a full rewrite, which would re-derive the data-repair
  and accounting bugs already found and fixed.
- **Ledger:** every walk-forward evaluation on development appends a look automatically, and every scored grid cell is
  charged as a trial. Rejected: an opt-in ledger.
- **Calibration defaults:** at most 5% false-surviving families on 200 shifted targets and at least 80% recovery of a
  planted IC; a study may tighten these and may not loosen them.
- **Search:** Optuna only as an optional `[search]` extra, later; random search and successive halving live in core.

## 2026-09-30 — Spec identity is the declared name and version, not the module path

`@spec(name=..., version=...)` sets the identity. Moving a class between modules does not change its hash; changing
what it means without changing its fields requires a version bump, which does. Containers are frozen at construction
(list → tuple, dict → read-only mapping, set → frozenset) so an object cannot drift from its hash. `int` and `float`
hash differently (`alpha=1` and `alpha=1.0` are different parameters to be explicit about); `-0.0` and `0.0` hash
the same; NaN and infinities are rejected. Rejected: pickling or `repr` for identity (unstable across versions),
module-qualified names (renames would invalidate every cache).

## 2026-09-30 — Stage keys carry the declared version; the commit is provenance only

A stage's cache key hashes (name, version, spec hash, fingerprint, input references). The git commit and package
version are written on every manifest record but are not in the key. Keying on the commit would rebuild everything
after every commit, including documentation edits; not keying on anything would let an undeclared code change reuse
stale outputs. The version is the author's declaration that a stage computes something different, and the golden
reproduction test (milestone 0) catches the changes nobody declared.

## 2026-09-30 — Source stages must fingerprint what they read, or they always run

A stage with no inputs reads the outside world. It is cacheable only if it supplies a fingerprint of what it reads
(file hashes, a vendor snapshot id, a query plus as-of date); without one it re-runs on every build. Its output is
still content-addressed, so a refresh that produces identical bytes leaves every downstream key unchanged. Rejected:
treating an unchanged spec as proof the data is unchanged — the silent-reuse failure the manifest exists to remove.

## 2026-09-30 — Artifacts: parquet for frames, plain sorted JSON for everything else

Frames are written as zstd parquet with statistics off and hashed on those bytes; other outputs must be plain JSON and
are written with sorted keys. A polars upgrade may change parquet bytes for identical data — that is a cache miss,
the safe direction. Cross-version equality is tested on contents (golden tests), never on hashes. The manifest is an
append-only JSON-lines file: a later record for a key supersedes an earlier one, and a record whose object is missing
from the store is a miss, never a skip.

## 2026-09-30 — CLI on click; studies are `module:factory` references

`chairlift run | plan | show | log` (`run/cli.py`, entry point `chairlift`). A study is any callable
`factory(root, **params) -> Pipeline`, referenced as `package.module:factory` or `path/to/file.py:factory`; `--set
name=value` passes parameters (JSON-parsed when possible). The CLI holds no pipeline logic: `plan` and `run` share one
decision rule (`Pipeline._decide`), so a dry run cannot disagree with a run. `plan` reports a stage downstream of a
re-running stage as `upstream` rather than guessing, because content addressing may turn it into a hit. Chosen: click
(Laurent's request; mature, composable groups, `CliRunner` for tests). Rejected: argparse (verbose for grouped
commands), a study registry or config-file discovery (a reference is explicit and works for studies living in other
repositories).

## 2026-09-30 — Tooling batch: strict types, property and golden tests, prek

- **Types:** basedpyright strict on `src/` (standard on tests and walkthroughs) is the blocking checker; ty (Astral,
  beta) runs in CI as a non-blocking job. Strict mode found that type checkers did not see `@spec` classes as
  dataclasses (constructors untyped); `spec` is now declared with PEP 681 `dataclass_transform(kw_only_default=True,
  frozen_default=True)`, so every spec constructor is fully typed. Rejected: mypy (no plugin we need).
- **Property tests (hypothesis):** spec hashing (mapping-order invariance, determinism, freezing, single-value
  sensitivity) and the rebuild rule on random DAGs (a version bump re-runs exactly its descendants when outputs carry
  their inputs; a third build reuses everything).
- **Golden test (syrupy):** the toy study's outputs are snapshotted by content in `tests/__snapshots__/`; a change is
  either intended (`--snapshot-update`, reason in the commit) or a silent change in what the pipeline computes.
- **Build backend:** `uv_build` (uv's own, stable for pure-Python). polars pinned `<2` until the 2.0 release is
  assessed. Dependency groups (PEP 735) for dev tools.
- **Hooks:** prek (Rust drop-in for pre-commit, same config) runs ruff, basedpyright and typos on commit and pytest on
  push. Replaces the bash hook. Dependabot watches uv and GitHub Actions weekly.
- **CLI output:** rich tables for `run`, `plan`, `log`; colour only on a terminal.
- **Coverage:** branch coverage on every test run. pytest-xdist installed but opt-in (`-n 4`): `-n auto` on this
  256-core box made the suite slower (31 s vs 1 s).
- **Deferred:** PEP 723 inline metadata for walkthroughs (they import the local package, which inline metadata cannot
  express without a path source); dataframely frame contracts arrive with the first protocol that returns a frame.
