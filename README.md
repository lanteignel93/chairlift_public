# chairlift

A machine-learning research pipeline in which the research protocol is code: point-in-time data, a target engine with
declared units, causal features, walk-forward folds that assert their own embargo, a holdout the loader refuses to
read, and a manifest and ledger that make every rebuild and every look accountable.

Status: early development (milestone 0). Start with the [user guide](docs/guide/README.md): install, configure,
first run, monitoring, reproducibility, writing a study, experiments, and the CLI reference. Design notes:
[docs/README.md](docs/README.md) and [docs/decisions.md](docs/decisions.md).

```bash
uv sync
uv run pytest
uv run chairlift run chairlift.verify.toy:pipeline                     # under the configured home; animated on a terminal
uv run chairlift plan chairlift.verify.toy:pipeline --set window=5     # what would rebuild, without running
uv run chairlift sweep examples/experiments/toy-sweep.toml --dry-run   # six cells, six signatures
```
