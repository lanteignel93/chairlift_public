# How to read chairlift's outputs

## The manifest (`<root>/manifest.jsonl`)

One JSON object per stage build, appended in build order; nothing is ever rewritten.

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
| `built_at` | UTC timestamp |

Reading a rebuild: find the stages whose key changed, and for each, which of its parts changed — a spec hash means a
parameter moved, an input ref means something upstream produced different bytes, a version means the code declared
a change. A stage with the same key as before and status `hit` did not run.

## Run reports (`RunReport.status()`)

`hit`: the manifest held the key and the store held the object; the stage function was not called. `ran`: the stage
executed; `StageResult.reason` says why.
