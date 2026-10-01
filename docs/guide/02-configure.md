# 2. Configure a machine

Site configuration answers one question: *where do things live on this machine?* None of it changes a result. It is
recorded with every run, with the layer that set each value, and it is never hashed. The same experiment therefore
has the same signature on your laptop, on a server and in CI.

## Zero configuration

With no file at all, chairlift works. Everything goes under the XDG data directory:

```bash
$ chairlift config show
key                       value
alerts.sinks              ['page', 'jsonl']
compute.memory_default    None
compute.polars_threads    None
compute.workers           1
data                      {}
paths.cache               None
paths.home                /home/you/.local/share/chairlift
paths.reports             None
paths.store               None
paths.studies             None
systemd.on_failure        True
systemd.watchdog_default  1h

profile - · store /home/you/.local/share/chairlift/store · studies /home/you/.local/share/chairlift/studies
files: none (defaults)
```

`None` for `paths.store`, `paths.studies`, `paths.reports` and `paths.cache` means "the default place under
`paths.home`": `<home>/store`, `<home>/studies`, `<home>/reports`, `<home>/cache`.

## Start a file

```bash
chairlift config init              # writes ~/.config/chairlift/config.toml (commented starter)
chairlift config init --project    # writes ./chairlift.toml instead
```

A typical user file:

```toml
# ~/.config/chairlift/config.toml
[paths]
home = "/data/you/chairlift"           # everything lives under here
# store = "/fast/chairlift-store"      # move one piece alone if it needs a faster disk

[data]                                  # aliases; studies read DataRef(alias="vendor", ...), never a path
vendor   = "/mnt/vendor/option_close"
holdings = "${DATA_ROOT}/holdings"

[profile.laptop]                        # overrides applied when this profile is active
paths.home  = "~/chairlift"
data.vendor = "~/data/vendor-sample"

[profile.hosts]                         # hostname → profile, so the right one applies with no env var
"my-laptop" = "laptop"
```

## Layers and precedence

Lowest to highest. A higher layer overrides a lower one key by key, never table by table.

| layer | source |
|---|---|
| default | built-in values (above) |
| system | `/etc/xdg/chairlift/config.toml` |
| user | `~/.config/chairlift/config.toml` |
| project | the nearest `chairlift.toml` in the working directory or a parent |
| `CHAIRLIFT_CONFIG` | a file named by that variable (an error if it does not exist) |
| profile | `[profile.<name>]`: the `--profile` flag, else `CHAIRLIFT_PROFILE`, else `[profile.hosts]` |
| env | `CHAIRLIFT_<SECTION>__<KEY>=value`, for example `CHAIRLIFT_PATHS__STORE=/fast/store` |

Rules:
- Environment values are parsed as TOML when they can be (`4` is an int, `true` a bool), and otherwise kept as
  strings.
- Path values (`[paths]` and `[data]`) expand `~`, `$VAR` and `${VAR}` after merging. An undefined variable is an
  error naming the key and the layer, never an empty string.
- A relative path is relative to the file that set it, so a project `chairlift.toml` can say `home = "./.chairlift"`.
- An unknown key is an error naming the key and the layer that set it. A misspelt setting never silently does
  nothing.
- In containers and CI the hostname is meaningless: set `CHAIRLIFT_PROFILE`, which wins over the host map.

## See where every value came from

```bash
$ chairlift config show --explain
key                       value                      layer
paths.home                /data/you/chairlift        user /home/you/.config/chairlift/config.toml
paths.store               /fast/store                env
compute.workers           4                          profile big
...
```

`chairlift config check` validates every layer and exits non-zero on the first problem, with its key and layer. Use it
in provisioning scripts. `chairlift config show --json` prints the record that every run stores.

## Data aliases

Studies never contain paths. A study names its data `DataRef(alias="vendor", relpath="option_close")`; the `[data]`
table maps `vendor` to a directory on this machine. The data is identified by a blake3 hash of its content, not by its
path ([5](05-reproducibility.md)), so moving it, or copying it to another machine, does not change any signature.
Changing a byte does. Data aliases are resolved for `--root` runs too: the data lives on the machine whichever layout
the run uses.

## Secrets

Secrets never enter the configuration object, a run record or a hash. They come from either:
- environment variables `CHAIRLIFT_SECRET_<NAME>`, or
- `~/.config/chairlift/secrets.toml` (`name = "value"` lines), which must be mode 0600; chairlift refuses to read it
  otherwise and prints the `chmod` to run.

A loaded `Secrets` object prints as `Secrets(webhook=***)` and refuses to be pickled or serialized. Nothing in
chairlift reads secrets yet. The planned alert sinks (`email`, `webhook`) will need a secret of the same name.

## Two machines

The signature never depends on paths, so two machines can work on the same studies in either of two ways.

**Separate homes** (the default). Each machine has its own `paths.home`. Results are reproducible across machines
(same signature, same output hashes), but stages are not shared: each machine builds its own store.

**One shared home over NFS.** Point `paths.home` (or only `paths.store`) on both machines at the same network
directory. Every host appends to its own `store/manifest.<host>.jsonl` and reads all of them, so two machines never
interleave writes in one file. A stage built on one machine is a `hit` on the other. Store objects are written to a
temporary name and renamed into place, so a reader never sees half an object. Keep `paths.cache` local if NFS
latency on small appends bothers you; the cache is only an optimization.

```toml
# on both machines
[paths]
home  = "/nfs/research/chairlift"
cache = "~/.cache/chairlift"            # local; per-host files either way
[data]
vendor = "/nfs/vendor/option_close"     # or a local copy: same bytes, same fingerprint
```

Next: [the first run](03-first-run.md).
