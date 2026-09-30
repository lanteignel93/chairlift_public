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
- **Every increment lands with its test.** Important logic units get a `debug_walkthroughs/wt_<topic>.py` (seeded,
  tiny, asserted, `--pdb`), and `tests/test_walkthroughs.py` runs every walkthrough.
- **In-fold or it is not a number:** transforms, selection, search and stacking fit on training rows only.

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
```

Set `UV_LINK_MODE=copy` on this box (the uv cache is on another filesystem).

## Layout

`src/chairlift/{core,data,target,features,schedule,learn,book,evaluate,diagnose,ledger,report,run,verify}` — tiers as
on the design page. `tests/`, `debug_walkthroughs/`, `docs/` (README, decisions, how-to-read), `notebooks/`,
`plans/{speculative,complete,archived}` (active plans at the root).
