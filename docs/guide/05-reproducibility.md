# 5. Identity and reproducibility

A research result is only as good as your ability to say exactly what produced it, and to produce it again.
chairlift answers both with three hashes, each with one job, and two commands that use them.

## Three hashes

| hash | covers | answers | where |
|---|---|---|---|
| **stage key** | stage name, version, spec hash, data fingerprint, and the *output references* of its inputs | "can this stage's stored output be reused?" | manifest, run record, `plan` |
| **signature** | every needed stage's *plan key*: names, versions, specs, data fingerprints, the graph | "is this the same experiment?": the same on any machine | run id, `signature`, `runs` |
| **environment hash** | Python version and implementation, machine architecture, every installed distribution and version, the code identity | "should the bytes come out identical?" | run record, `rerun`, `compare` |

The stage key and the plan key differ in one way. A stage key uses the bytes its inputs actually produced, so it is
known only after upstream has run, and it decides reuse. A plan key uses its inputs' plan keys instead: a Merkle hash
of the pipeline's definition, known before anything runs. The signature hashes the plan keys, so you can compute it
without building anything:

```bash
$ chairlift signature chairlift.verify.toy:pipeline
stage     plan key
sources   c3b926f0189ce098
features  ea7009988f8f664e
...
signature 1e711d30d079731ee3f1c16fd56ce0886ed4a4469dc5d342f90324f8800fbeb3
```

### What moves which hash

| change | stage key | signature | environment hash |
|---|---|---|---|
| a parameter that feeds a spec (`--set window=5`) | that stage and everything downstream that produces new bytes | yes | no |
| a byte of input data | the source stage, and downstream | yes | no |
| the same data at another path, or on another machine | no | no | no |
| bumping a stage's `version` | that stage and downstream | yes | no |
| `paths`, `[compute]`, profiles, any site setting | no | no | no |
| a secret | no | no | no |
| upgrading polars, or a new commit of chairlift or the study | no (but see below) | no | yes |

A code change does not move a key by itself. **If a change makes a stage compute something different, bump the
stage's `version`** (or its spec's). That is the contract ([6](06-writing-a-study.md)). The environment hash and
`rerun` exist to catch the cases where that contract was broken, or where a library changed under you.

Content addressing stops needless work: if a re-run stage produces the same bytes as before, every stage downstream
keeps its key and is a `hit`.

## The run record

`studies/<name>/runs/<run_id>.json` is written when the run starts (`status: running`) and rewritten atomically when
it ends. It holds everything needed to rebuild the run, and nothing secret:

| field | content |
|---|---|
| `run_id`, `signature`, `started_at`, `finished_at`, `status`, `error` | identity and outcome |
| `host`, `pid` | where it ran (used to detect a dead run) |
| `targets`, `reason` | what was asked for, and why |
| `meta.study`, `meta.name`, `meta.params` | the factory reference, the study name, the validated parameters |
| `meta.experiment` | the experiment file's name, path, sha256 and full text, when run from one |
| `meta.sweep` | sweep id, cell index, cell count, cell values, when run by `sweep` |
| `meta.config` | the resolved site configuration with the layer of every value |
| `stages.<name>` | version, the canonical spec, spec hash, data fingerprint, inputs, plan key |
| `results.<name>` | `hit` or `ran`, key, output reference, seconds |
| `code`, `environment` | code identity; environment hash, Python, platform, distributions, the nearest `uv.lock` hash |
| `data` | where each data alias pointed on this machine (recorded, never hashed) |

## Rerun: prove a result reproduces

```bash
$ chairlift rerun 20261001T133024
stage     recorded              reproduced
sources   parquet:ea52068938dd  parquet:ea52068938dd  identical
features  parquet:d830cfd89f63  parquet:d830cfd89f63  identical
folds     json:6207dd9d3da62ea  json:6207dd9d3da62ea  identical
dataset   parquet:e16d3f8b1273  parquet:e16d3f8b1273  identical
fit       parquet:fc705ef7bbf9  parquet:fc705ef7bbf9  identical
evaluate  json:27f4922156bca5d  json:27f4922156bca5d  identical
report    json:17e869f056d4076  json:17e869f056d4076  identical

20261001T133024.919610Z-ccecf764569f: reproduced bit for bit
```

`RUN` is a run id or signature prefix; the newest match wins. `rerun` rebuilds the run from its record alone, in a
fresh temporary directory, so nothing is reused and every stage really runs. It then compares every output reference
with the recorded one.

| outcome | exit | meaning |
|---|---|---|
| every output identical | 0 | reproduced bit for bit |
| the signature differs | 1, before anything runs | the study code, the parameters or the data changed since; it names the stages whose plan keys moved |
| an output differs | 1 | the stage is not deterministic, or it changed without a version bump |
| the environment hash differs | warning only | identical bytes are likely but not promised; read `compare` to see what moved |

Options:
- `--into DIR` rebuilds in `DIR`, which must be empty, and keeps it.
- `--keep` keeps the temporary directory.
- `--json` prints the per-stage comparison.

A rerun records its own run in the fresh directory, with `meta.rerun_of`, and leaves the original study directory
untouched.

## Compare: see what differs between two runs

```bash
$ chairlift compare 20261001T133024 20261001T133041
        key                   20261001T133024.919610Z-ccecf764569f  20261001T133041.363328Z-1e711d30d079
run     signature             ccecf764569f                          1e711d30d079
run     study                 toy                                   toy
params  window                4                                     1
spec    features              b41d36d11415                          82ba4e0bd2c6
output  dataset               parquet:e16d3f8b1273                  parquet:7e7c01ebb1a2
...
metric  evaluate.oos_ic_mean  0.057819                              0.107164
metric  fit.slope[fold=1]     0.108197                              0.0918543
...
13 of 26 rows differ (equal rows hidden; --all shows them)
```

Sections:
- `params`: the validated parameters.
- `spec`: per stage.
- `data`: per source fingerprint.
- `output`: per stage.
- `env`: hash, Python, code, and every distribution that differs.
- `metric`: the last value of every metric, from the event files.

Here one parameter moved one spec, `sources` and `folds` kept their outputs, and everything downstream of `features`
changed, along with the IC. `--json` gives every row with a `same` flag.

## The manifest

`store/manifest.<host>.jsonl` holds one line per stage build and is never rewritten:
- `stage`, `key`, `version`, `spec_hash`, `inputs`, `output`
- `code`
- `reason` (the caller's `--reason`, or the runner's)
- `built_at`, `host`

When two lines share a key, the later one supersedes. That happens only when a stored object went missing and was
rebuilt. `chairlift log --stage fit` is the history of one stage. To find out why something rebuilt, find its new key
and check which part moved: the spec hash, an input reference, or the version.

## Data fingerprints

A source stage that reads data declares `fingerprint=fingerprint_of(DataRef(...))` ([6](06-writing-a-study.md)).
- A file's fingerprint is `b3f:` plus the blake3 hash of its bytes.
- A directory's is `b3d:` plus a Merkle hash over its sorted (relative path, file hash) pairs. Dot-files are skipped.

The same bytes at any path, on any machine, give the same fingerprint. A rename inside a directory, a new file, or one
changed byte gives a different one.

Hashing every byte on every `plan` would be slow on large data. A stat cache in `paths.cache`, one append-only file
per host, maps (path, size, mtime_ns, inode) to the hash, so unchanged files are not re-read. Any change to that
tuple re-hashes the file. The cache is an optimization, never the source of truth: delete it and the next plan
re-hashes and gets the same answers.

## Limits

- Bit-identical outputs need deterministic stages. A stage with an unseeded RNG, or one whose threaded reductions are
  order-dependent, reproduces as `DIFFERS`. That is the right answer: fix the seed, or round before storing.
- Parquet bytes depend on the polars version. After an upgrade, outputs can differ in bytes while being equal in
  content. `rerun` warns that the environment changed, and golden tests compare contents, not hashes.
- A Python value bound into a stage with `functools.partial` is not part of the key. Use that only for settings that
  cannot change the result, like the toy's `pace`.

Next: [writing a study](06-writing-a-study.md).
