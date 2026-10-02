# chairlift user guide

This guide covers chairlift from a fresh machine to a finished, reproducible run: installation, configuration, the
first run, monitoring, identity and reproduction, writing a study, and experiments. Every command shown here is real
and runs against the shipped toy study (`chairlift.verify.toy:pipeline`), so you can follow along without data.

## The model in one page

Four kinds of input, separated by one question: **does it change the result?**

| kind | what it holds | where it lives | changes the result? |
|---|---|---|---|
| **study** | the stages: data, features, folds, model, evaluation | Python, in the study's own repository (a *factory*) | yes: hashed through every stage's spec |
| **experiment** | the knobs one run sets or sweeps | `experiments/<name>.toml`, or `--set name=value` | yes: hashed through the specs they build |
| **site** | where files live on this machine, data roots, limits | `chairlift.toml` / `~/.config/chairlift/config.toml`, layered | no: recorded with provenance, never hashed |
| **secrets** | webhook URLs, credentials | environment or a mode-0600 `secrets.toml` | no: never recorded, never hashed |

A **study** is a directed graph of stages. Each stage is a pure function of its inputs and its spec. chairlift keys
every stage by what went into it, stores every output once under the hash of its bytes, and reuses an output whenever
the key is unchanged. A **run** builds the stages you ask for, writes a **run record** (everything needed to rebuild
it) and an **event stream** (what happened, as it happened), and is named by a **signature** that is the same on any
machine for the same study, parameters, code versions and data.

```
<paths.home>/                         one directory per machine (or shared over NFS)
├── store/                            content store + manifest, shared by every study
│   ├── objects/ab/ab12….parquet      every output, once, named by its sha256
│   └── manifest.<host>.jsonl         stage key → output, one append-only file per host
├── studies/<name>/runs/              per study: <run_id>.json + <run_id>.events.jsonl
├── reports/                          (planned) the cross-study index
└── cache/                            data fingerprint stat cache, per host
```

## Reading order

| page | what you get |
|---|---|
| [1. Install](01-install.md) | a working `chairlift` command and a green test suite |
| [2. Configure a machine](02-configure.md) | where things live, data aliases, profiles, secrets; two machines sharing work |
| [3. First run](03-first-run.md) | run, rerun, plan, show, and what is on disk afterwards |
| [4. Monitoring](04-monitoring.md) | the live view, `watch` from another terminal, `status` for scripts, the event file |
| [5. Identity and reproducibility](05-reproducibility.md) | stage keys, signatures, environment hashes, `rerun`, `compare`, the manifest |
| [6. Writing a study](06-writing-a-study.md) | factories, stages, specs, data references, progress and metrics |
| [7. Experiments and sweeps](07-experiments.md) | experiment files, typed parameters, `sweep` |
| [8. CLI reference](08-cli-reference.md) | every command and option, generated from the code |
| [9. Studies, search and keeping score](09-studies-and-search.md) | declared studies, models and books, the ledger, the holdout gate, search, twins, unattended runs |
| [10. API reference](10-api.md) | every public name, generated from `chairlift.__all__` |

For the meaning of each output file field by field, see [how to read the outputs](../how-to-read.md); for why things
are the way they are, [decisions](../decisions.md).

## What exists today and what does not

Built and tested:
- **The runner:** the content store and manifest, the DAG runner, signatures and run records; layered site
  configuration with provenance, secrets, data references with content fingerprints; experiment files and sweeps;
  the environment hash, `rerun` and `compare`.
- **Monitoring:** the event stream, the live view, `watch` and `status`; static HTML reports; `submit` under
  systemd with a progress-tied watchdog; alerts.
- **The research layer ([9](09-studies-and-search.md)):** declared studies; walk-forward folds that assert their
  embargo, parallel per fold; ridge, LightGBM, regime rules, in-fold cluster selection and in-fold transforms (winsorize, interactions, PCA); ensembles; quantile and
  signal books; daily-book statistics with block-bootstrap intervals and concentration checks; the trial ledger,
  deflated Sharpe and the holdout gate; search with walk-forward selection, PBO, complexity priors and one trial per
  distinct signature; factor attribution; synthetic twins.
- **Packaging:** a typed wheel (`py.typed`) with a lazy top-level API, installed and smoke-tested in a clean
  environment on every push.

Not built yet: point-in-time universe helpers, gates that block a holdout opening automatically, and search over
model families beyond a study's own knobs.
