# 8. CLI reference

Generated from the click definitions by `scripts/gen_cli_reference.py`; do not edit by hand (`tests/test_docs.py`
fails when this page and the code disagree). Every command also answers `--help`.

Conventions shared by most commands:
- `STUDY` is `module:factory`, `path/to/file.py:factory`, or an experiment file `*.toml` ([7](07-experiments.md)).
- Without `--root`, a study runs under `<paths.home>/studies/<name>/` and shares `<paths.home>/store/`
  ([2](02-configure.md)); `--root DIR` makes one self-contained directory instead.
- `--json` prints machine-readable output; everything else is for people.

## Commands

- [`chairlift compare`](#chairlift-compare)
- [`chairlift config`](#chairlift-config)
- [`chairlift config check`](#chairlift-config-check)
- [`chairlift config init`](#chairlift-config-init)
- [`chairlift config show`](#chairlift-config-show)
- [`chairlift log`](#chairlift-log)
- [`chairlift plan`](#chairlift-plan)
- [`chairlift rerun`](#chairlift-rerun)
- [`chairlift run`](#chairlift-run)
- [`chairlift runs`](#chairlift-runs)
- [`chairlift runs show`](#chairlift-runs-show)
- [`chairlift show`](#chairlift-show)
- [`chairlift signature`](#chairlift-signature)
- [`chairlift status`](#chairlift-status)
- [`chairlift sweep`](#chairlift-sweep)
- [`chairlift watch`](#chairlift-watch)

<a id="chairlift"></a>

## `chairlift`

```
chairlift [OPTIONS] COMMAND [ARGS]...
```

chairlift: a research pipeline in which the protocol is code.

| option | meaning |
|---|---|
| `--version` | Show the version and exit. |
| `--profile TEXT` | Site configuration profile (overrides CHAIRLIFT_PROFILE and the host map). |

<a id="chairlift-compare"></a>

## `chairlift compare`

```
chairlift compare [OPTIONS] RUN_A RUN_B
```

Diff two runs: parameters, per-stage specs and data fingerprints, outputs, environment, last metrics.

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--all` | Show equal rows too. |
| `--json` | Machine-readable output. |

<a id="chairlift-config"></a>

## `chairlift config`

```
chairlift config [OPTIONS] COMMAND [ARGS]...
```

Site configuration: show, check, or start a file.

<a id="chairlift-config-check"></a>

## `chairlift config check`

```
chairlift config check [OPTIONS]
```

Validate every layer; exit non-zero on the first problem.

<a id="chairlift-config-init"></a>

## `chairlift config init`

```
chairlift config init [OPTIONS]
```

Write a commented starter configuration file.

| option | meaning |
|---|---|
| `--project` | Write ./chairlift.toml instead of the user file. |
| `--force` | Overwrite an existing file. |

<a id="chairlift-config-show"></a>

## `chairlift config show`

```
chairlift config show [OPTIONS]
```

Print the resolved site configuration for this machine.

| option | meaning |
|---|---|
| `--explain` | Show which layer set each value. |
| `--json` | Machine-readable output. |

<a id="chairlift-log"></a>

## `chairlift log`

```
chairlift log [OPTIONS]
```

Read the manifest: what was built, when, from which inputs and code, and why (default: the shared store's).

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--stage TEXT` | Only this stage's records. |
| `--last INTEGER` | Most recent records to show.  [default: 20] |
| `--json` | Machine-readable output. |

<a id="chairlift-plan"></a>

## `chairlift plan`

```
chairlift plan [OPTIONS] STUDY
```

Show what `run` would do, without running anything.

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--name TEXT` | Study name (default: the module's STUDY_NAME). |
| `--set NAME=VALUE` | Factory parameter (repeatable). |
| `--target TEXT` | Run only these stages and their ancestors. |
| `--json` | Machine-readable output. |

<a id="chairlift-rerun"></a>

## `chairlift rerun`

```
chairlift rerun [OPTIONS] RUN
```

Rebuild a recorded run from its record alone, from scratch, and check every output is byte-identical.

Exit 0 when every stage reproduces; 1 when an output differs or the signature does (the study code, the
parameters or the data changed since). A different environment hash is a warning: identical bytes are then
likely but not promised.

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--study TEXT` | Look for RUN in this study only. |
| `--into PATH` | Rebuild here (default: a temp dir). |
| `--keep` | Keep the rebuilt directory. |
| `--json` | Machine-readable output. |

<a id="chairlift-run"></a>

## `chairlift run`

```
chairlift run [OPTIONS] STUDY
```

Run a study's pipeline or an experiment file; reuse every stage whose inputs, spec and version are unchanged.

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--name TEXT` | Study name (default: the module's STUDY_NAME). |
| `--set NAME=VALUE` | Factory parameter (repeatable). |
| `--target TEXT` | Run only these stages and their ancestors. |
| `--reason TEXT` | Why this build; recorded on every stage that runs. |
| `--live / --no-live` | Animate the run (default: on a terminal). |
| `--json` | Machine-readable output. |

<a id="chairlift-runs"></a>

## `chairlift runs`

```
chairlift runs [OPTIONS] [COMMAND] [ARGS]...
```

List recorded runs across studies (newest last); `runs show RUN` prints one record.

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--study TEXT` | Only this study's runs. |

<a id="chairlift-runs-show"></a>

## `chairlift runs show`

```
chairlift runs show [OPTIONS] RUN
```

Print a run record by run id or signature prefix (the newest match).

<a id="chairlift-show"></a>

## `chairlift show`

```
chairlift show [OPTIONS] STUDY STAGE
```

Print a stage's stored output (building it and its ancestors first if needed).

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--name TEXT` | Study name (default: the module's STUDY_NAME). |
| `--set NAME=VALUE` | Factory parameter (repeatable). |
| `--rows INTEGER` | Rows to print for a frame.  [default: 10] |

<a id="chairlift-signature"></a>

## `chairlift signature`

```
chairlift signature [OPTIONS] STUDY
```

Print the run signature (inputs-only identity) and each stage's plan key, without running.

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--name TEXT` | Study name (default: the module's STUDY_NAME). |
| `--set NAME=VALUE` | Factory parameter (repeatable). |
| `--target TEXT` | Run only these stages and their ancestors. |
| `--json` | Machine-readable output. |

<a id="chairlift-status"></a>

## `chairlift status`

```
chairlift status [OPTIONS]
```

The latest run's state. Exit 0 ok, 1 failed or died, 2 still running: for scripts and timers.

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--study TEXT` | Study name (default: the most recent run of any study). |
| `--run TEXT` | A run id prefix (default: the newest run). |
| `--json` | Machine-readable output. |

<a id="chairlift-sweep"></a>

## `chairlift sweep`

```
chairlift sweep [OPTIONS] EXPERIMENT
```

Run every cell of an experiment's [sweep], in order, sharing one store: unchanged stages are reused.

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--name TEXT` | Study name (default: the module's STUDY_NAME). |
| `--set NAME=VALUE` | Factory parameter (repeatable). |
| `--target TEXT` | Run only these stages and their ancestors. |
| `--reason TEXT` | Why this sweep (default: the file's reason). |
| `--dry-run` | List the cells and their signatures; run nothing. |
| `--json` | Machine-readable output. |

<a id="chairlift-watch"></a>

## `chairlift watch`

```
chairlift watch [OPTIONS]
```

Follow a run from its event file: the same live view as `run`, from any terminal.

| option | meaning |
|---|---|
| `--root PATH` | A self-contained run directory (own store). Default: &lt;paths.home&gt;/studies/&lt;name&gt;, shared store. |
| `--study TEXT` | Study name (default: the most recent run of any study). |
| `--run TEXT` | A run id prefix (default: the newest run). |
| `--interval FLOAT` | Seconds between reads of the event file.  [default: 0.25] |
| `--once` | Print the current state and exit, even if the run is still going. |
