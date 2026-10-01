# How to read chairlift's outputs

## The manifest (`<home>/store/manifest.<host>.jsonl`, or `<root>/manifest.jsonl` for a `--root` run)

One JSON object per stage build, appended in build order; nothing is ever rewritten. A shared store has one file per
host (each file has one writer); lookups read them all.

| field | meaning |
|---|---|
| `stage` | stage name |
| `key` | sha256 of (name, version, spec hash, fingerprint, input refs): the cache key |
| `version` | the stage's declared version; a bump means "computes something different" |
| `spec_hash` | sha256 of the stage spec's canonical JSON (`null` for a stage without a spec) |
| `inputs` | input stage name → artifact ref used for this build |
| `output` | artifact ref `parquet:<sha256>` or `json:<sha256>` in `<root>/store/objects/` |
| `code` | `chairlift <version> @ <commit>[+dirty]`: provenance, not part of the key |
| `reason` | why the stage ran: the caller's reason, or the runner's (new key, missing object, volatile source) |
| `built_at` | UTC timestamp (microseconds; orders builds across hosts) |
| `host` | the machine that built it (absent in records from before 2026-10-01) |

Reading a rebuild: find the stages whose key changed, and for each, which of its parts changed — a spec hash means a
parameter moved, an input ref means something upstream produced different bytes, a version means the code declared
a change. A stage with the same key as before and status `hit` did not run.

## Run reports (`RunReport.status()`)

`hit`: the manifest held the key and the store held the object; the stage function was not called. `ran`: the stage
executed; `StageResult.reason` says why.

## `chairlift plan`

`hit`: would be reused. `run`: will execute (the reason says why). `upstream`: an input will re-run first; if it
produces the same bytes this stage becomes a hit, otherwise it runs — the plan cannot know before the input runs.
`chairlift run` prints the same columns plus the seconds each executed stage took.

## The event stream (`<root>/runs/<run_id>.events.jsonl`)

One JSON object per line, in order (`seq`). `stage_finished.status`: `hit` reused, `ran` executed, `failed` raised
(its `error` says what). `progress` carries done / total / message from inside a stage; `metric` a name, a value and
dimensions such as `fold`. `peak_rss_mb` is the process's peak resident memory so far (monotone over a run).

## The live view (`chairlift run` on a terminal, `chairlift watch`)

○ pending · spinner running · ✓ ran · · reused · ✗ failed. `shape` is rows × columns of a frame output. The progress
bar and detail come from the stage's last `progress` call; the metrics table shows the last six metrics reported.
