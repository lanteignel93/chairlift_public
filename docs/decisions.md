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

## 2026-09-30 — Site configuration: one home per machine, layered TOML with provenance

Implemented `core/config.py` (plans/m0-config.md steps 1–3). Laurent: all data in one directory, studies named from
config, so strategies can share work. Layout: `<paths.home>/store/` (the content store and its manifest, shared by
every study on the machine: identical stages are reused across studies), `<paths.home>/studies/<name>/` (run
records, later events and reports), `<paths.home>/reports/`, `<paths.home>/cache/`; each subpath can be moved
alone. Layers: defaults (XDG data dir), system, user, the nearest `chairlift.toml`, `CHAIRLIFT_CONFIG`, the profile
(explicit > `CHAIRLIFT_PROFILE` > host map), `CHAIRLIFT_<SECTION>__<KEY>` env, CLI. Every leaf records the layer
that set it. Paths expand `~` and `${VAR}`/`$VAR` (undefined is an error) and resolve relative to the file that set
them. Unknown keys and wrong types fail with the dotted key and the layer. Validation: msgspec `convert` into frozen
dataclasses. Secrets live in `core/secrets.py`: env `CHAIRLIFT_SECRET_*` or a 0600 `secrets.toml`, masked repr, refuse
to pickle or serialize, never part of `Config`. The study name is the module's `STUDY_NAME` or `--name`; `--root`
keeps the old self-contained layout. Store and manifest no longer create directories on construction (a factory's
throwaway pipeline left empty folders). Rejected: pydantic-settings (second model system, its own CLI parser),
Hydra/OmegaConf (maintenance-only, YAML, takes over `main`), dynaconf (untyped). Risk: concurrent appends to one shared
manifest are atomic for single short lines on a local filesystem, not guaranteed on NFS; revisit (per-host manifest
or a lock) before two machines write one store.

## 2026-09-30 — Tests mirror the package, ordered by tier

`src/chairlift/<pkg>/<module>.py` is tested by `tests/<pkg>/test_<module>.py`; `tests/test_layout.py` fails when a
module has none. Cross-cutting suites: `tests/properties/` (hypothesis), `tests/golden/` (syrupy), `tests/walkthroughs/`
(each carries a marker). `tests/conftest.py` orders collection by dependency tier (core → data → … → run → verify →
properties → golden → walkthroughs → layout) and by dependency within a tier, so the first failure is the lowest
broken layer. `--import-mode=importlib` and `--strict-markers`.

## 2026-09-30 — Event stream and live view (observability E1 + E2)

`run/events.py`: one file per run, `<root>/runs/<run_id>.events.jsonl`, single writer, every line carries `kind`,
`schema`, `run_id`, `seq`, `ts`. Kinds: run_started, stage_started, progress, metric, warning, stage_finished (hit /
ran / failed, seconds, CPU seconds, peak RSS, rows × columns, output ref, error), run_finished. Stages report through
`events.progress / metric / warning`, bound by a context variable, so they stay pure and the calls are no-ops outside a
run. In-process listeners are called defensively: an exception in one is counted and swallowed. Readers keep a byte
offset, leave a partial last line for the next read and skip corrupt lines. Events never enter keys, signatures or
outputs (tested). `run/live.py`: a pure reducer (events → RunState) plus a rich renderer, used by `chairlift run`
(animated on a terminal, `--live/--no-live`) and `chairlift watch` (the same view from the file, any terminal). `chairlift
status` exits 0 ok, 1 failed or died (a run left `running` whose process is gone on this host), 2 running.

Found while building it: run ids were second-resolution, so two runs of one experiment in the same second shared an
id and appended to one event file. Run ids are now `<UTC time to the microsecond>Z-<signature[:12]>` with a collision
suffix; regression test added.

## 2026-10-01 — Config completion: per-host manifests, data references, experiments, rerun / compare (m0-config 4–7)

- **Per-host manifest.** A shared store writes `manifest.<host>.jsonl` and reads every `manifest*.jsonl`, so each
  file has one writer and two machines on NFS never interleave appends. `built_at` has microseconds and records carry
  `host`; when one key appears in two files, the later build wins. A `--root` directory keeps one `manifest.jsonl`.
  This closes the NFS risk noted on 2026-09-30.
- **The manifest re-reads on a miss.** Found building `sweep`: every cell's pipeline is built (and validated) before
  any runs, and each opened its manifest as a snapshot, so later cells never saw earlier cells' stages and nothing was
  reused. `lookup` now folds in complete lines appended since the last read (per-file byte offsets; a line being
  written is left for the next read). The same fix covers two processes or hosts sharing a store.
- **`DataRef(alias=, relpath=)`** is a `@spec` (keyword-only like every spec), so its identity is the alias and
  relative path, never a directory. The Pipeline binds its data roots (from `[data]`) and stat cache while it
  fingerprints and while stages run; `ref.path()` outside a run is an error. `fingerprint_of(*refs)` hashes content:
  blake3 per file (`b3f:`), a Merkle hash over sorted (relpath, hash) per directory (`b3d:`), dot-files skipped;
  aliases are left out, so renaming an alias moves nothing. Stat cache: (path, size, mtime_ns, inode) → hash, one
  append-only JSON-lines file per host under `paths.cache` (sqlite rejected: locking over NFS). `--root` runs use
  `<root>/cache` and still resolve data through the machine's config.
- **Experiment files** (`run/experiment.py`): `study`, `name`, `reason`, `study_name`, `[params]`, `[run] targets`,
  `[sweep]`; unknown keys are errors. Parameters from the file and `--set` are validated against the factory's
  signature (`get_type_hints` + msgspec `convert`, strict) before anything is built: unknown, missing and mistyped
  parameters exit 2 having written nothing. The run record keeps the file's full text and sha256. `sweep` validates
  every cell first, runs them in order over one store, and records `meta.sweep = {id, cell, cells, values}`; a sweep
  file passed to `run` is refused. The trial-ledger charge is deferred to milestone 1.
- **Environment hash** = sha256 of the Python version and implementation, the machine architecture, every installed
  distribution with its version, and the code identity. The nearest `uv.lock` hash is recorded beside it, not hashed:
  the installed set is what ran, and a study run outside its repository has no lock.
- **`rerun RUN`** rebuilds from the record alone (factory reference, validated params, name, targets) in a fresh
  directory, so every stage really runs. It fails with exit 1 before running if the signature moved (and names the
  stages whose plan keys moved), fails with exit 1 if any output reference differs, and only warns on an environment
  change. **`compare A B`** diffs params, per-stage spec hashes, data fingerprints, outputs, the environment
  (distributions that differ), and the last value of every metric from the event files.
- Found while writing the guide against real output: `show` ran without metadata, so `status`, `watch` and `runs`
  showed `-` for the study; and rich markup ate `[fold=1]` in `compare` (and would eat any `[word]` in a reason). Both
  fixed, with tests.
- **User guide** in `docs/guide/` (repo-first; Laurent 2026-09-30: documents from setup to running to monitoring),
  written against shipped behaviour with real command output; planned features are labelled as planned. The CLI
  reference is generated from click (`scripts/gen_cli_reference.py`), and `tests/docs/` fails when it is stale, when a
  relative link breaks, or when the guide's example study differs from `examples/momentum_study.py` (which it also
  runs).

## 2026-10-01 — M1 core, and the two required clients run on it

Laurent: run slalom first against its full recorded result, then VXX, then a gallery of example outputs.

- **Generic components** (each with tests mirroring it):
  - `schedule/walkforward` (folds asserting their embargo)
  - `features/cross_section` and `features/time_series` (causal transforms)
  - `learn/models` (closed-form ridge with penalty α·n; LightGBM averaged over seeds)
  - `learn/selection` (in-fold clusters, best member by |IC| × consistency)
  - `learn/fit` (the walk-forward fit and the ensemble)
  - `book/quantile`, `book/timeseries`
  - `evaluate/daily` (overlapping-book P&L, moving-block bootstrap)
  - Studies stay in their own repositories: `slalom_trade_private/chairlift_client`, `vxx_trade/chairlift_client`.
- **slalom reproduces.** Out-of-sample predictions of A, B_nxy (in-fold clustering and selection included), G_nxy
  (LightGBM DART, five seeds) and the BG_nxy ensemble match slalom's within 1e-13 on all 912,043 rows. The candidate
  book's daily L−S Sharpe, its CI99, both legs and 7/7 years are bit-identical (2.160989443263191). LightGBM 4.6.0 is
  locked to match slalom's; 4.7 was not tried.
- **Found by the comparison**: slalom's option-path engine breaks equal-gap expiry ties by row order (SWK 2021-09-08:
  23 vs 37 days to a 30-day target), so its own A-book Sharpe varies by about 3e-4 from run to run. Recorded in the
  client README; slalom's frozen engine is left unchanged.
- **The modularity test changed three things**; VXX (one row per date) needed:
  1. `FitSpec.cross_section = False`. The degenerate-prediction check and the in-sample IC were within-date; with
     one row per date they read across time. A cross-sectional fit on time-series data now fails with a message
     naming the switch.
  2. The ensemble's standardisation. Within-date z-scores are meaningless with one row, and z-scoring over the test
     period would leak its moments, so time-series members are scaled by their training-fold mean and sd.
  3. A time-series IC (`ts_ic_by_year`) beside the cross-sectional one.

  Folds, models, selection, books and statistics needed no change. Adding the field re-keyed slalom's fits once
  (spec identity working as designed).
- **VXX's first look is negative**: IC ≈ 0 out of sample, Sharpe ≈ 0 against standing short at 0.50 [−0.53, 1.68].
  Not tuned on dev.
- **CLI findings from real use:**
  - `watch` and `status` take `--run` (they only followed the newest run, so an inspection run hid the long one).
  - `compare` takes a reused stage's metrics from the run that built its output; a hit emits no metric, so a run
    that reused work looked as if it had none.
- **LightGBM on several threads is not bitwise reproducible**: a second rebuild moved 1.2M of 3.7M G_nxy predictions
  by at most 1.4e-14. The book and its statistics did not move, but `rerun` would report the stage as DIFFERS and
  content addressing re-runs everything downstream. LightGBM's `deterministic=True` is the candidate fix; deferred
  until it is shown to keep the slalom match.
- **Gallery** in `examples/gallery/`: figures, tables and CLI transcripts from these runs, regenerated by `build.py`
  from run records. No data.

## 2026-10-01 (later) — identity holes closed, the ledger, rules, twins

Two bugs of one class, both silent stale outputs, both found by using chairlift on real studies:
1. A stage lambda capturing a factory parameter (`fundamentals`) was not keyed on it: `fundamentals=peek` was
   served the published panel.
2. A same-module helper (`rule_choices`) was edited without a version bump: the edited stage was served its old output.

The key now covers both:
- **captured values:** closure cells and `functools.partial` arguments, canonicalised; an unhashable capture is an
  error at build; `ignore=` names deliberate exclusions
- **code digest:** the stage function's source plus every same-module function it reaches by name, transitively

`version` remains for changes outside the study's module (chairlift, libraries) and for intent. Every stage re-keyed
once.

- **Determinism:** LightGBMSpec.deterministic (default) sets LightGBM's deterministic + col-wise. Run-to-run
  bit-identical, and the slalom match is kept (candidate unchanged, G_nxy within 1.4e-14).
- **Ledger** (`studies/<name>/ledger.jsonl`):
  - one trial per signature, charged by `Pipeline.run`, with the headline read from the study's HEADLINE
  - `chairlift ledger` reports the deflated Sharpe of the best trial
  - `chairlift holdout open` records the one look (a reason is required, a second opening is refused)
  - `guard()` / `with_holdout()` hide holdout rows, and opening re-keys every stage behind the gate
- **Rule learner** (`RuleSpec`): a regime rule chosen in-fold by the position's own Sharpe. On VXX it found the
  futures-slope rule from 2019 on, but not before the 2018 spike its training had not seen. Laurent: "if chairlift
  can't find that on its own it needs work." The answer here is partial: the search finds the rule once history
  contains the event.
- **Frequency:** DailyStatsSpec has periods / min_per_year / min_per_half (monthly books).
- **Twins** (`verify/twins.py`): planted-signal cross-section and time-series data. `examples/studies/equity_ls`
  recovers the planted IC (0.050 OOS), shows the release-lag leak (+0.011 IC) and finds nothing on the null; the
  SPY null twin does not beat buy-and-hold.
- **Store** errors name the non-JSON path; NaN correlations become null.

## 2026-10-02 — Study, search v2, observability; four clients run; the VXX holdout look

- **Study** (`study/build.py`): a declared study builds its DAG.
  - Pieces: sources, folds, `fit_*` per model, `ens_*`, books, paths, `eval_*`, `report`, plus extra stages.
  - Factories may return a Study and take keyword-only knobs, so parameter validation stays typed.
  - The four clients (slalom, VXX, equity L/S, SPY timing) and the two twin examples are declared this way.
  - slalom's candidate stayed bit-identical through the port.
- **Determinism and identity, end to end.** `chairlift rerun` reproduces all 18 slalom stages bit for bit. Bugs found
  on the way, all silent and all fixed:
  1. the code digest depended on set order (PYTHONHASHSEED), through dataclass-generated methods that share one code
     object
  2. the code digest did not follow user code captured in closures
  3. parallel-fold frames encoded to different parquet bytes; the store now rechunks
  4. the slalom engine broke an expiry tie by row order; that is a client fix, and its source tree is now part of
     the paths fingerprint
- **Parallel folds** (`compute.workers`, spawned) and **parallel search candidates** (`--jobs`). Appends to shared
  JSON-lines files take an fcntl lock (`data/locking.append_line`).
- **Search v2:** `[search.complexity]` priors in walk-forward selection; an objective override; structural knobs
  (feature family, model) as study parameters.
  - The twins validate the judgment: planted signal DSR 1.00 / PBO 0.01; null DSR 0.45 / PBO 0.56.
- **Observability:**
  - `chairlift report`: an HTML page per run, and an index
  - `chairlift submit`: a systemd transient unit; the watchdog heartbeat is sent only while events advance;
    OnFailure goes to `chairlift alerts unit`
  - `chairlift alerts check`: failed, died and stalled runs, delivered once to jsonl / stdout / webhook
- **Factor attribution** (`evaluate/attribution.py`): α after factor books, with Newey-West errors.
- **The clients' results:**
  - **VXX:** candidate A was registered, then the holdout opened once (2026-10-02, after 17 trials). FAIL (G2, G3):
    0.24 vs 0.50 for always short. The contango switch sits out V-shaped snap-backs.
  - **Equity L/S:** the S&P 500 point-in-time, SEC fundamentals as filed, CIKs mapped by ticker then by name (737 of
    876 members). Dev has no edge (−0.51; α t −1.2). A structural search finds nothing (DSR 0.39, PBO 0.42). The
    release-lag leak test moves nothing because fundamentals carry no one-month IC here.
  - **SPY timing:** macro with publication lags, long-or-flat. 0.95 against buy-and-hold's 0.90: no edge.
  - Holdouts for equity and SPY stay sealed; no candidate qualified.
- **Names:** the twin examples are `equity_twin` / `spy_twin`, so they do not share directories or ledgers with the
  real clients.

## 2026-10-02 (evening) — 0.1.0: a public API, in-fold transforms, and two judging fixes found by real searches

- **Public API = `chairlift.__all__`:** 42 names, loaded lazily from the top level so `import chairlift` and the
  CLI stay fast. A literal `__all__`, checked against `_EXPORTS` by the tests, so type checkers see it.
  `docs/guide/10-api.md` is generated from it and checked like the CLI reference. Every public name needs a docstring.
- **Version 0.1.0, CHANGELOG, `py.typed`.** A CI job builds the wheel, installs it in a clean environment and runs the
  toy study from it. The empty `diagnose` and `target` packages are removed: a public package should not ship
  promises.
- **Site detail:** a test greps every tracked file for machine paths, host names and firm or vendor names. The plans
  were scrubbed to pass it. The public copy's history was generalized the same way.
- **In-fold transforms** (`FitSpec.transforms`: `Winsorize`, `Interactions`, `PCA`). Fitted on each fold's training
  rows, after selection and the fill. Each records its choices in the fold info.
  - Rejected: transforms as panel stages. They would be fitted on all rows, so a training-quantile clip or a PCA
    loading would see the test year.
- **`added(default)` for spec evolution:** a field added at its default is left out of the canonical JSON, so old
  specs keep their hashes. Without it, adding `transforms` would have re-keyed every fit in every store.
- **Search counts identical candidates once.** Grid knobs that do not reach the pipeline (a transform's size with no
  transform) would have added identical trials and inflated N in the deflation.
- **Judging timing books on active return.** SPY's transform search, judged on the book's own Sharpe, gave DSR 0.93
  and walk-forward selection 0.86: a "find" that was just being long SPY. Judged on the return over buy-and-hold, it
  gives DSR 0.06 and every candidate negative.
  - Hence `active_<book>` stages for time-series studies whose benchmark is a baseline. A long-only or short-only
    study should name one as its HEADLINE.
  - PBO stayed low (0.19) through all of it, which is the lesson: PBO tests the choice of candidate, not the edge.
- **Trial identity = the headline stage's key** (else the signature). Adding the `active_*` stages changed every SPY
  run signature without changing a single judged number; under the old rule that would have charged 12 trials twice.
- **Results:** in-fold transforms find nothing on equity L/S (14 distinct: best 0.08, DSR 0.19, walk-forward
  selection −0.63) or SPY (DSR 0.06 on active return). Both stay dev nulls, with sealed holdouts.

