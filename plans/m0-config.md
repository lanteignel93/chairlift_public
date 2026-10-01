# M0 Config: machine-independent settings, experiment files, and run identity

**Status:** actionable
**Prepared:** 2026-09-30
**Owner:** Laurent Lanteigne
**Buy-in:** Laurent — asked for centralized configs, no machine-specific paths, hashed inputs per run (2026-09-30)

---

## Problem / motivation

chairlift has to run on more than one machine, so paths cannot be baked in, and every run must carry a signature that
identifies exactly what went into it. Run signatures and run records landed on 2026-09-30 (`run/dag.py`). What is
missing is the configuration side:
- where the files live on this machine
- which data a study reads
- what an experiment varies
- what is secret

The design goal is **versatile without being overwhelming**. The test of it: a user touches at most three files, and
a new study runs with zero configuration on a fresh machine.

## Proposed approach: four kinds of input, separated by one question

The question is: **does it change the result?** It splits every setting into four kinds, and each kind gets one home.

| kind | what it holds | changes the result? | where it lives | hashed into the signature |
|---|---|---|---|---|
| **Site** | paths, data roots, compute limits, systemd defaults, report and alert settings | no | `chairlift.toml`, layered | no; recorded with provenance |
| **Study** | universe, target, families, transforms, schedule, models, book, evaluation, gates, holdout start | yes | Python, in the study repository (the factory) | yes, through every stage's spec |
| **Experiment** | the knobs one study exposes and a run varies: windows, alphas, cutoffs, which targets to build, sweeps | yes | `experiments/<name>.toml` in the study repo, or `--set` | yes, through the specs the parameters build |
| **Secrets** | webhook URLs, credentials | no | environment, or `~/.config/chairlift/secrets.toml` (mode 0600) | never; never loaded into the config object |

Two consequences carry the design:
1. The same experiment has the same signature on any machine. Paths and limits are site settings, and data is
   identified by its content fingerprint, never by its path.
2. Changing a site setting can never silently change a result: by construction it is outside every stage's spec.

### Site config: layered, typed, explainable

One file shape, merged from several places, low to high precedence:

1. built-in defaults: XDG directories via platformdirs, so zero configuration works
2. `/etc/xdg/chairlift/config.toml`: system
3. `~/.config/chairlift/config.toml`: user
4. `./chairlift.toml`: project
5. the active profile's table: `[profile.<name>]`, chosen by `CHAIRLIFT_PROFILE`, else by the host map
6. environment variables: `CHAIRLIFT_PATHS__RUNS=/fast/runs`
7. CLI flags (`--root`, `--profile`), which default to "not given" so a flag never overrides a file by accident

```toml
# ~/.config/chairlift/config.toml
[paths]
runs    = "~/chairlift/runs"     # run roots, one directory per run
store   = "~/chairlift/store"    # one content-addressed store shared by every run on this machine
reports = "~/chairlift/reports"
cache   = "~/.cache/chairlift"

[data]                                           # aliases studies use instead of paths
vendor   = "${DATA_ROOT}/vendor"
holdings = "${DATA_ROOT}/holdings"

[compute]
workers = 8
polars_threads = 16
memory_default = "48G"

[systemd]
watchdog_default = "45min"
on_failure = true

[alerts]
sinks = ["page", "jsonl"]                        # "email", "webhook" need a secret of the same name

[profile.laptop]                                  # overrides applied when this profile is active
paths.runs = "~/chairlift/runs"
data.vendor = "~/data/vendor-sample"
compute.workers = 4

[profile.hosts]                                   # host → profile, so the right profile applies without an env var
"research-box" = "box1"
"laurent-xps" = "laptop"
```

Rules:
- Paths expand `~` and `${VAR}` after merging, in path-typed fields only. An undefined variable is an error, not an
  empty string.
- Every resolved key records which layer set it. `chairlift config show --explain` prints `paths.runs =
  /fast/runs (env CHAIRLIFT_PATHS__RUNS)`, the debuggability that framework libraries do not give.
- `chairlift config check` validates every layer and exits non-zero on the first problem, with the TOML path
  (`$.compute.workers: expected int, got "eight"`).
- `chairlift config init` writes a commented starter file.

### Data references: aliases, so studies never contain paths

A study reads `DataRef("vendor", "option_close_t5")`. The site config resolves the alias to a directory on this
machine. The fingerprint is computed on content, DVC-style: a cache from (path, size, mtime_ns, inode) to a blake3
hash of the bytes, an optimization and never the source of truth, with a Merkle hash over sorted (relpath, hash) for
a directory. The same data on two machines therefore gets the same fingerprint, and the same signature.

### Study config: Python, because a study is code

Families, targets and books are functions, so a study is a Python factory in its own repository returning the
`Study` object of the design page. This keeps it typed, composable and debuggable with breakpoints. The factory
declares its experiment parameters as keyword arguments with types and defaults; those parameters are the study's
entire public surface. Gates, holdout start and calibration thresholds are part of the spec. Once a study is
registered (ledger), a change to them is an amendment.

### Experiment files: the daily surface

```toml
# experiments/window-sweep.toml (in the study repository)
study  = "slalom_client.study:pipeline"
reason = "does a longer feature window help the surface family?"

[params]               # typed against the factory's signature; an unknown name or a wrong type is an error
window = 5
alpha  = 1e-3

[run]
targets = ["evaluate", "report"]

[sweep]                # optional: the cross product becomes one run per cell, each with its own signature
window = [3, 5, 10]    # every cell is charged to the study's trial ledger
```

- `chairlift run experiments/window-sweep.toml` (an experiment file, or `STUDY --set` as today).
- `chairlift sweep experiments/window-sweep.toml` expands the cells and runs them in order, sharing the store, so
  unchanged stages are reused across cells.
- Parameters are validated by converting against the factory's annotations (msgspec `convert`, strict), before
  anything runs.

### Run identity: three hashes, each with one job

| hash | covers | used for |
|---|---|---|
| **stage key** | name, version, spec, fingerprint, input *output* refs | cache reuse (exists) |
| **signature** | every needed stage's plan key: specs, versions, data fingerprints | "is this the same experiment": the same on any machine (exists) |
| **environment hash** | `uv.lock` hash, Python version, chairlift commit and dirty flag | "can I expect identical bytes": recorded with every run, compared by `rerun` |

A run id is `<UTC timestamp>-<signature[:12]>`. The record under `runs/<run_id>.json` holds:
- the resolved site config with its provenance
- the experiment file's contents
- every spec, fingerprint and plan key
- the code and environment identity
- the outcome

Three commands use it:
- `chairlift rerun RUN` reproduces a run from its record alone. It warns when the environment hash differs, and
  fails when the signature does.
- `chairlift compare RUN_A RUN_B` diffs specs, parameters, fingerprints and metrics.
- `chairlift runs` lists runs.

### Libraries (checked 2026-09-30)

| job | choice | why | rejected |
|---|---|---|---|
| read TOML | stdlib `tomllib` | no dependency | — |
| write TOML | `tomli-w` 1.2 | small and finished | — |
| validate | `msgspec` 0.22 (`convert`, strict) | decodes into frozen dataclasses, errors carry the field path, fast | pydantic-settings (a second model system and its own CLI parser, which clashes with click) |
| layering | about 100 lines in `core/config.py` | provenance per key, exact precedence | Hydra/OmegaConf (maintenance-only, YAML-first, takes over `main`), dynaconf (untyped), typed-settings (single maintainer; the strongest alternative) |
| XDG paths | `platformdirs` 4.12 | correct per OS | — |
| data hashing | `blake3` 1.0.10 | cryptographic, multithreaded, mmap | xxhash (not for signatures) |

Pitfalls this design must handle:
- **TOML has no null:** optional means an absent key.
- **int vs float:** `1` and `1.0` are different, and the spec hash keeps them different on purpose.
- **datetimes:** local datetimes are rejected; dates are allowed.
- **env vars are strings:** coerce them through the schema.
- **hostname profiles break in containers and CI:** `CHAIRLIFT_PROFILE` wins over the host map.

## Scope in v1 / out-of-scope

**In v1:**
- `core/config.py`: schema, layering, provenance, path expansion
- data aliases and the `DataRef` fingerprint helper
- the secrets loader
- `chairlift config show / check / init`
- experiment files and `--set` validated against factory signatures
- `chairlift sweep`
- the environment hash in run records
- `rerun` and `compare`
- every path in chairlift resolved from config, so `--root` becomes optional

**Out of scope:** remote stores (S3 and the like); per-user access control; search spaces (in the
automated-search plan, as a `[search]` section of the same experiment file).

## Implementation sequence

1. **Schema and layering.** `Config` dataclasses, merge with provenance, profiles, env, path expansion. Tests:
   - precedence per layer
   - an unknown key is an error with its path
   - an undefined `${VAR}` is an error
   - a host-map profile, with the env override winning
2. **Wiring.** CLI defaults come from config: `--root` becomes the configured `paths.runs/<study>/<signature>`; the
   shared store comes from `paths.store`. Tests: a run on two configured "machines" (two tmp configs) gives one
   signature, and different roots.
3. **Secrets.** A separate loader with the 0600 check. Test: no secret field name ever appears in a run record or a
   hash input.
4. **`DataRef` and fingerprints.** The blake3 file and directory hash with the stat cache. Tests: the same bytes at
   two paths give one fingerprint; one changed byte changes it; the cache is invalidated by mtime.
5. **Experiment files and `sweep`.** Parameter validation against the factory signature; the sweep cross product.
   Tests: a wrong type fails before any stage runs; three cells give three signatures and shared stages are reused.
6. **Environment hash, `rerun`, `compare`.** Test: `rerun` of a toy run reproduces the golden outputs.
7. **`config show / check / init`**, the docs page, and a walkthrough `wt_config.py` stepping through the merge.

## Verification criteria

- The toy study runs on a fresh machine with no config file (defaults under XDG directories).
- The same experiment on two simulated machines (different `paths`, the same data copied to different places) has
  one signature.
- No machine-specific path exists in `src/`.
- `chairlift config show --explain` names the layer of every key.
- A secret set in the environment is absent from every run record and every hashed structure.
- `chairlift rerun` reproduces a recorded toy run's outputs bit for bit.

## Open questions

- Whether the shared store should also be shared across machines (NFS), or kept per machine with runs synchronized.
  Lean: per machine by default, with NFS possible by configuration.
- Whether experiment files may live outside a study repository (a central experiments directory). Lean: allow both;
  the study reference in the file is enough.

## Related plans / docs / PRs

- `plans/m0-skeleton.md`
- `plans/speculative/chairlift-observability.md`
- `plans/speculative/chairlift-automated-strategy-search.md`

## Closeout / as-built

*Fill at completion.*
