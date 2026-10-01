# 4. Monitoring

Every run writes an append-only event stream, `studies/<name>/runs/<run_id>.events.jsonl`, as it goes. Everything
on this page reads that one file, so a run looks the same whether you started it in this terminal, in tmux, or under
systemd. Watching a run never touches the run: a broken observer cannot fail it, and nothing an observer does enters a
key or an output.

## The live view

On a terminal, `chairlift run` animates:

```bash
chairlift run chairlift.verify.toy:pipeline --set pace=0.4    # pace slows the toy fit so there is something to see
```

```
chairlift toy  running     3.2s  run 20261001T133024.919610Z-ccecf764569f
   stage     status   seconds  shape    progress              detail
✓  sources   ran      0.09     15000×4
✓  features  ran      0.06     15000×3
✓  folds     ran      0.00
✓  dataset   ran      0.04     15000×4
⠼  fit       running                    ━━━━━━━━━━━━━         2/3 fold 3 of 3: train ≤ t244, test t250–t299
○  evaluate  pending
○  report    pending
latest metrics
stage  metric  value    where
fit    slope   +0.1082  fold=1
fit    slope   +0.1007  fold=2
```

Reading it:
- `○ pending`: not reached yet. A spinner: running now. `✓ ran`: executed. `· hit`: reused. `✗ failed`: it
  raised, and `detail` holds the error.
- `shape` is rows×columns for a frame output.
- `progress` and `detail` come from inside the stage: a stage reports them with `events.progress(...)`
  ([6](06-writing-a-study.md)).
- `latest metrics` shows the last six values stages reported with `events.metric(...)`, with their dimensions.
  Warnings (`events.warning(...)`) appear below it in yellow.

`--live` forces the animation and `--no-live` turns it off. It is off automatically when the output is not a
terminal, and under `--json`.

## Follow a run from anywhere

```bash
chairlift watch --study toy          # the newest run of `toy`; follows it until it finishes
chairlift watch                      # the newest run of any study
chairlift watch --study toy --once   # print the current state and exit
chairlift watch --root /tmp/scratch  # a self-contained directory
```

`watch` replays the event file from the start, then follows it. When the output is not a terminal, or the run has
already finished, it prints once and exits. If the run's process disappears while the run is still marked running
(killed, out of memory, machine restarted), `watch` says so instead of waiting forever. That check works for runs on
the same host only: a run on another host is shown as running for as long as it last said so.

## Status for scripts and timers

```bash
$ chairlift status --study toy
toy  20261001T132640.724571Z-1e711d30d079  ok
$ echo $?
0
$ chairlift status --study toy --json
{"run_id": "20261001T132640.724571Z-1e711d30d079", "status": "ok", "study": "toy", "host": "box"}
```

| exit | status | meaning |
|---|---|---|
| 0 | `ok` | the latest run finished |
| 1 | `failed` | the latest run raised; `chairlift runs show <run id>` has the error |
| 1 | `died` | marked running, but its process is gone (same host only) |
| 2 | `running` | still going |

A minimal check for cron or a systemd timer:

```bash
#!/usr/bin/env bash
chairlift status --study slalom --json > /tmp/slalom-status.json
case $? in
  0) ;;                                                      # fine
  2) ;;                                                      # still running
  *) notify-send "slalom: $(jq -r .status /tmp/slalom-status.json)" ;;
esac
```

## Running detached

Until `chairlift submit` exists, any process supervisor works, because monitoring reads only files:

```bash
# tmux
tmux new -d -s toy 'uv run chairlift run chairlift.verify.toy:pipeline --reason nightly'

# systemd, transient user unit: survives logout (with lingering), logs to the journal
systemd-run --user --unit=chairlift-toy --collect \
  --working-directory="$HOME/src/my-study" \
  uv run chairlift run my_study.study:pipeline --reason nightly --no-live
journalctl --user -u chairlift-toy -f     # stdout and stderr
chairlift watch --study my-study          # the run itself
```

## The event file

One JSON object per line, written in order by the run's single writer. Every line carries `kind`, `schema`,
`run_id`, `seq` and `ts` (UTC).

| kind | extra fields |
|---|---|
| `run_started` | `study`, `signature`, `stages` (the stages this run needs, in order), `reason` |
| `stage_started` | `stage`, `key`, `will` (`reuse` or `run`), `reason` |
| `progress` | `stage`, `done`, `total`, `message` |
| `metric` | `stage`, `name`, `value`, `dims` (for example `{"fold": 3}`) |
| `warning` | `stage`, `message` |
| `stage_finished` | `stage`, `status` (`hit` / `ran` / `failed`), `key`, `output`, `seconds`, `cpu_seconds`, `peak_rss_mb`, `rows`, `cols`, `error` |
| `run_finished` | `status` (`ok` / `failed`), `seconds`, `ran`, `reused`, `error` |

The reader (`chairlift.run.events.read_events(path, offset)`) returns complete lines only. It leaves a line that is
still being written for the next read, and skips a corrupt line instead of failing, so any tool can tail the file
safely. `peak_rss_mb` is the process's peak resident memory so far, so it never decreases over a run.

## Planned

These are designed (`plans/speculative/chairlift-observability.md`) and not built:
- static HTML run reports
- `chairlift submit`: a systemd unit per run, with `sd_notify` watchdog heartbeats and an `OnFailure=` hook
- alert rules routed to the sinks in `[alerts]`

The `[systemd]` and `[alerts]` settings are already parsed and recorded with every run.

Next: [identity and reproducibility](05-reproducibility.md).
