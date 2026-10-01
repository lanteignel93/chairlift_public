# 3. First run

The shipped toy study (`chairlift.verify.toy`) is a synthetic cross-section: 50 entities over 300 days, one feature
`x`, and a target `y = ic·x + noise`, so the planted correlation is `ic = 0.1`. It has seven stages:

```
sources ──► features ──► dataset ──► fit ──► evaluate ──► report
   │                        ▲         ▲
   ├────────────────────────┘         │
   └──► folds ────────────────────────┘
```

`fit` is a least-squares slope per walk-forward fold, fitted on training rows only; `evaluate` is the out-of-sample
daily rank IC.

## Run it

```bash
$ chairlift run chairlift.verify.toy:pipeline --reason "first run"
stage     status  key           seconds  reason
sources   ran     ef2a8e15ad9a  0.04     no record for this key (new, or an input / spec / version changed)
features  ran     8934766479b5  0.06     no record for this key (new, or an input / spec / version changed)
folds     ran     7393e9fd164f  0.00     no record for this key (new, or an input / spec / version changed)
dataset   ran     739d14d85e0c  0.05     no record for this key (new, or an input / spec / version changed)
fit       ran     e7b2cfe46d2a  0.01     no record for this key (new, or an input / spec / version changed)
evaluate  ran     561f7e315fc5  0.02     no record for this key (new, or an input / spec / version changed)
report    ran     2121b28d2b1b  0.00     no record for this key (new, or an input / spec / version changed)

7 ran, 0 reused · toy · run 20261001T132639.297847Z-1e711d30d079 · ~/.local/share/chairlift/studies/toy
```

On a terminal the same run is animated instead ([4](04-monitoring.md)). This page shows the plain output you get
when the output is piped, or with `--no-live`.

- `toy` is the study name: the module's `STUDY_NAME`, or `--name`.
- The run id is `<UTC time to the microsecond>Z-<signature[:12]>`.
- `--reason` is recorded in the run record and on every stage that runs. Use it; `chairlift log` is much more useful
  when every build says why.

## Run it again

```bash
$ chairlift run chairlift.verify.toy:pipeline
stage     status  key           seconds  reason
sources   hit     ef2a8e15ad9a           inputs, spec and version unchanged
...
report    hit     2121b28d2b1b           inputs, spec and version unchanged

0 ran, 7 reused · toy · run 20261001T132639.965711Z-1e711d30d079 · ~/.local/share/chairlift/studies/toy
```

The keys are identical, so nothing runs. The run is still recorded, under the same signature.

## Change a parameter, but look first

```bash
$ chairlift plan chairlift.verify.toy:pipeline --set window=5
stage     status    key           reason
sources   hit       ef2a8e15ad9a  inputs, spec and version unchanged
features  run       0dff0b4b7856  no record for this key (new, or an input / spec / version changed)
folds     hit       7393e9fd164f  inputs, spec and version unchanged
dataset   upstream                waits on re-running input(s): features
fit       upstream                waits on re-running input(s): dataset
evaluate  upstream                waits on re-running input(s): fit
report    upstream                waits on re-running input(s): evaluate
```

`plan` runs nothing and writes nothing.
- `hit` means the stage will be reused.
- `run` means it will execute.
- `upstream` means an input re-runs first. If that input produces the same bytes, this stage becomes a `hit`;
  otherwise it runs. The plan cannot know which before the input has run.

`window` changes only `FeatureSpec`, so `sources` and `folds` are reused.

`--set name=value` values are parsed as JSON when they can be, so `5` is an int and `0.1` a float. They are checked
against the factory's annotations before anything is built: `--set window=five` and `--set lookback=5` both fail at
once, naming the parameter.

## Look at an output

```bash
$ chairlift show chairlift.verify.toy:pipeline report
{
 "line": "OOS IC +0.1072 (t = +10.07) over 150 days"
}
$ chairlift show chairlift.verify.toy:pipeline fit --rows 3     # frames print as a table
```

`show` builds the stage and its ancestors if needed, which here are all hits. It records a run like any other.

## What is on disk

```
~/.local/share/chairlift/
├── store/
│   ├── manifest.<host>.jsonl              one line per stage build: key, output, inputs, code, reason, host
│   └── objects/1e/1e84f4cb….parquet       frames as parquet (zstd), everything else as sorted JSON
└── studies/toy/runs/
    ├── 20261001T132639.297847Z-1e711d30d079.json            the run record
    └── 20261001T132639.297847Z-1e711d30d079.events.jsonl    the event stream
```

Three commands read these files:
- `chairlift runs` lists the runs of every study, newest last.
- `chairlift runs show <run id or signature prefix>` prints one record.
- `chairlift log` reads the manifest, most recent builds last:

```bash
$ chairlift log --last 3
built_at                          host         stage     v  output                 code                            reason
2026-10-01T13:26:39.528316+00:00  box          fit       1  parquet:5812ff995824…  chairlift 0.0.1 @ c3e699f84e29  first run
2026-10-01T13:26:39.544839+00:00  box          evaluate  1  json:bb824f5754fcc59…  chairlift 0.0.1 @ c3e699f84e29  first run
2026-10-01T13:26:39.545963+00:00  box          report    1  json:bba7de2bfcccae2…  chairlift 0.0.1 @ c3e699f84e29  first run
```

## A self-contained directory

`--root DIR` puts the store, the manifest and the runs all inside `DIR`, shared with nothing. Use it for a scratch
experiment, a test, or a directory you will tar up and move:

```bash
chairlift run chairlift.verify.toy:pipeline --root /tmp/toy-scratch
chairlift log --root /tmp/toy-scratch
```

Next: [watching runs](04-monitoring.md).
