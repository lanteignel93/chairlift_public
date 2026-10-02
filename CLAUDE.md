# Claude Code Context — chairlift

A portable machine-learning research pipeline in which the research protocol is code. Owner: Laurent. Two required
clients: slalom (cross-sectional, the golden reference) and VXX (time series, the modularity test). Read
`docs/README.md`, `docs/decisions.md` and the active plan in `plans/` before changing a design choice.

## Hard rules

- **No data and no vendor code in this repository.** Vendor names, paths, credentials and study-specific adapters
  live in the client repositories. The core must be publishable.
- **Imports point toward `core`.** `core` imports nothing else in chairlift; `learn` never imports `data` (a model sees
  frames, never a store); chairlift never imports a client.
- **Specs are `@spec(name=..., version=...)` frozen dataclasses.** Bump `version` whenever a spec or a stage computes
  something different without its fields changing. Never edit a released transform or family in place: fixes ship
  under a new name or version.
- **Stages are pure functions of their spec and inputs.** A stage with no inputs must supply a `fingerprint` or it runs
  every time. Nothing is skipped because a file "looks done".
- **Tests mirror the package:** `tests/<pkg>/test_<module>.py` for every `src/chairlift/<pkg>/<module>.py` (enforced by
  `tests/test_layout.py`); add the file to `WITHIN` in `tests/conftest.py` so it runs in dependency order.
- **Paths come from the site config** (`chairlift config show --explain`); never hard-code a machine path. Studies read
  data through `DataRef(alias=..., relpath=...)` and fingerprint it with `fingerprint_of`; never a path in a study.
- **Every increment lands with its test.** Important logic units get a `debug_walkthroughs/wt_<topic>.py` (seeded,
  tiny, asserted, `--pdb`), and `tests/test_walkthroughs.py` runs every walkthrough.
- **In-fold or it is not a number:** transforms, selection, search and stacking fit on training rows only.
- **Declare studies with `chairlift.study.build.Study`**; a factory `study(*, knobs) -> Study`. Anything a stage
  depends on is its spec, its inputs, a captured value, or code the digest can see. Code reached through a module
  imported at run time (a client's own engine) goes into a fingerprint explicitly.
- **No number is written down before `chairlift rerun` reproduces its run bit for bit.** Caching hides a client
  stage that is not a pure function (an unordered `unique(keep="first")`, ties in a sort): two equity/PEAD bugs of
  that kind moved published Sharpes by 0.05 until a rerun caught them.
- **Holdouts open once, through the ledger, on Laurent's go.** Gates are written down before the look.

## Commands

```bash
uv sync                                   # env from uv.lock (Python 3.12)
uv run pytest                             # full suite, walkthroughs included
uv run ruff check . && uv run ruff format --check .
uv run basedpyright                       # strict on src/ (blocking); `uv run ty check` is advisory
uv run prek run --all-files               # every hook; `uv run prek install` once per clone
uv run pytest --snapshot-update           # only when a golden change is intended — say why in the commit
uv run python debug_walkthroughs/wt_manifest.py --pdb
uv run chairlift run  chairlift.verify.toy:pipeline --root runs/toy --reason "why"
uv run chairlift plan chairlift.verify.toy:pipeline --root runs/toy --set window=5   # dry run
uv run chairlift show chairlift.verify.toy:pipeline report --root runs/toy
uv run chairlift log  --root runs/toy --stage fit
uv run chairlift run  chairlift.verify.toy:pipeline --set pace=0.4   # animated on a terminal; studies/<name> under the configured home
uv run chairlift watch --study toy                                   # follow the latest run from its event file
uv run chairlift status --study toy                                  # exit 0 ok · 1 failed/died · 2 running
uv run chairlift config show --explain
uv run chairlift run   examples/experiments/toy-window.toml             # an experiment file (study, params, targets)
uv run chairlift sweep examples/experiments/toy-sweep.toml --dry-run    # cells and signatures; drop --dry-run to run
uv run chairlift rerun RUN                                              # rebuild from the record in a fresh dir; byte-compare
uv run chairlift compare RUN_A RUN_B                                    # params, specs, data, outputs, env, metrics
uv run python scripts/gen_cli_reference.py                              # after any CLI change (tests/docs checks it)
uv run chairlift search examples/experiments/equity-search.toml --jobs 8   # candidates judged: DSR, walk-forward selection, PBO
uv run chairlift ledger --study vxx                                     # trials, deflated Sharpe, holdout state
uv run chairlift holdout open --study NAME --reason "..."              # ONE look, irreversible: only on Laurent's explicit go
uv run chairlift report --index ; uv run chairlift alerts check         # HTML reports; failed / died / stalled runs
uv run chairlift submit EXP.toml --watchdog 45min                       # systemd transient unit (chairlift systemd install once)
```

User guide: `docs/guide/` (repo-first; describes shipped behaviour only, with real command output). When a command's
behaviour or output changes, update the page that shows it; `tests/docs/` checks the CLI reference, the links, and
that the guide quotes `examples/momentum_study.py` verbatim.

Set `UV_LINK_MODE=copy` on this box (the uv cache is on another filesystem).

## Layout

`src/chairlift/{core,data,features,schedule,learn,book,evaluate,ledger,search,study,report,run,verify}` — tiers as
on the design page. `tests/`, `debug_walkthroughs/`, `docs/` (README, decisions, how-to-read), `notebooks/`,
`plans/{speculative,complete,archived}` (active plans at the root).
