# 7. Experiments and sweeps

`--set` is fine for poking at a study. The runs you actually make, and want to remember, belong in experiment files.
An experiment file is small, lives in the study repository under `experiments/`, is reviewed like code, and is copied
in full into the record of every run it makes.

## An experiment file

```toml
# examples/experiments/toy-window.toml
study  = "chairlift.verify.toy:pipeline"
reason = "does a longer feature window change the out-of-sample IC?"

[params]
window = 5

[run]
targets = ["evaluate"]
```

```bash
chairlift run examples/experiments/toy-window.toml          # also: plan, signature, show
chairlift run examples/experiments/toy-window.toml --set window=7   # --set overrides the file
```

| key | meaning | default |
|---|---|---|
| `study` | `module:factory`, or `path/to/file.py:factory` (relative to the experiment file) | required |
| `name` | the experiment's name in records | the file's stem |
| `reason` | recorded on the run and on every stage that runs; `--reason` overrides it | `""` |
| `study_name` | the study directory under `<home>/studies/` | the module's `STUDY_NAME` |
| `[params]` | keyword parameters of the factory | the factory's defaults |
| `[run] targets` | build only these stages and their ancestors; `--target` overrides it | every stage |
| `[sweep]` | parameter → list of values; see below | none |

Any other key is an error. Because the experiment names the study, an experiment file can live anywhere, inside the
study repository or in a central directory.

## Parameters are typed

Every parameter is checked against the factory's signature, from both the file and `--set`, before anything is built
or run. Conversion is strict (msgspec): an int given for a `float` parameter becomes a float, and a string never
becomes a number. A list given for a `tuple[float, ...]` parameter becomes a tuple.

```bash
$ chairlift run bad.toml                 # window = "5"
Error: parameter window='5': Expected `int`, got `str`
$ chairlift run bad2.toml                # lookback = 3
Error: unknown parameter(s) lookback; the factory takes window, ic, seed, pace
```

A missing required keyword parameter (one without a default) is also an error. All three exit with status 2, and
nothing is written.

TOML types to keep in mind:
- There is no null: leave a key out to get the factory's default.
- `1` and `1.0` are different values. The factory's annotation decides, and specs hash them differently on purpose.
- Dates (`2024-01-02`) are fine. Local datetimes are better avoided.

## Sweeps

```toml
# examples/experiments/toy-sweep.toml
study  = "chairlift.verify.toy:pipeline"
reason = "how the recovered IC degrades with window and planted signal"

[run]
targets = ["report"]

[sweep]
window = [1, 3, 5]
ic     = [0.0, 0.1]
```

The cells are the cross product, in file order, with the last key varying fastest. A key may be in `[params]` or in
`[sweep]`, not both. Look before you run:

```bash
$ chairlift sweep examples/experiments/toy-sweep.toml --dry-run
cell  window  ic   signature
0     1       0.0  7db7427841d2
1     1       0.1  1e711d30d079
2     3       0.0  1ccb103cad07
3     3       0.1  d47c956dd960
4     5       0.0  f310488c4fd7
5     5       0.1  5c01b779bf2c

6 cell(s) planned · toy-sweep · sweep 20261001T133319Z-98635e55
```

```bash
$ chairlift sweep examples/experiments/toy-sweep.toml
cell  window  ic   signature     ran  reused  run_id
0     1       0.0  7db7427841d2  7    0       20261001T133319.882435Z-7db7427841d2
1     1       0.1  1e711d30d079  7    0       20261001T133320.090419Z-1e711d30d079
2     3       0.0  1ccb103cad07  5    2       20261001T133320.197277Z-1ccb103cad07
3     3       0.1  d47c956dd960  5    2       20261001T133320.296619Z-d47c956dd960
4     5       0.0  f310488c4fd7  5    2       20261001T133320.380141Z-f310488c4fd7
5     5       0.1  5c01b779bf2c  5    2       20261001T133320.471825Z-5c01b779bf2c

6 cell(s) ran · toy-sweep · sweep 20261001T133319Z-98635e55
```

What happened:
- Every cell is validated before the first one runs, so a typo in the last cell costs nothing.
- The cells run in order and share one store. Each cell reuses whatever an earlier cell already built: from cell 2 on,
  `sources` (keyed by `ic`) and `folds` are hits.
- Each cell is an ordinary run, with its own signature and its own record. The record carries
  `meta.sweep = {id, cell, cells, values}`, so you can collect a sweep's runs from `chairlift runs` or
  `runs/*.json`.
- Running the same sweep again runs nothing, and `rerun` and `compare` work on any cell.

The sweep id is `<UTC time>-<sha256 of the experiment file>[:8]`.

Every cell is a trial. When the trial ledger lands (milestone 1), sweep cells will be charged to the study's ledger,
and the deflated statistics will account for them. Until then, keep sweeps honest by keeping the experiment files:
the record of what was tried is the file plus its runs.

## Where to go from here

- [5. Identity and reproducibility](05-reproducibility.md): `compare` two cells, `rerun` the one you will report.
- [8. CLI reference](08-cli-reference.md): every option of `run` and `sweep`.
