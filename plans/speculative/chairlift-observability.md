# chairlift observability: eyes on a run while it builds, and after

**Status:** speculative — E1 (event stream) and E2 (live view, watch, status) landed 2026-09-30 (d6a49af)
**Prepared:** 2026-09-30
**Owner:** Laurent Lanteigne
**Buy-in:** <fill when promoting to actionable>

---

## Problem / motivation

Laurent wants to see what a model build is doing while it runs, and to get an automated account of what happened
afterwards, with alerting in the devops style he now runs (lookout, systemd user units and timers).

Today chairlift has half of this. The manifest records what was *built*, but nothing records what *happened*:
- a stage that failed
- a run that hung
- a stage that took ten times longer than last week
- a fold that produced a surprising IC
- a gate that failed

slalom's runs were multi-hour walk-forwards watched through tmux and log tails. Two of its failures would have been
seen early with the right signal:
- the polars fork deadlock: every worker at 0% CPU forever
- the resume-safety trap: a stage silently reused stale exports

The lookout pattern already works on this infrastructure, and chairlift should fit it rather than add a second
system:
- collectors append plain files to a shared tree
- a stdlib renderer builds self-contained static HTML, refreshed every minute
- alerts appear on the pages with lookout's severities (critical / serious / warning)
- systemd user units and timers run everything; there is no server process

## Proposed approach

Five layers. Each depends only on the ones before it.

### 1. An event stream (the foundation)

The runner appends structured events to `<root>/events.jsonl`, one JSON object per line, each stamped with `run_id`,
`ts`, `host` and `pid`. Events:

| event | payload |
|---|---|
| `run_started` | study reference, parameters, stage list, code identity, argv, systemd unit (if any) |
| `stage_started` / `stage_finished` | status (hit / ran / failed), seconds, CPU seconds, peak RSS, output ref, rows × columns for frames, the reason |
| `progress` | stage, done / total, message (for example "fold 3 of 7") |
| `metric` | stage, name, value, optional fold and year (out-of-sample IC per fold, keep ratio, bootstrap interval) |
| `gate` | name, value, threshold, pass |
| `warning` | stage, message (for example a volatile source, a stage with a missing object) |
| `run_finished` | ok / failed / gate-failed, duration, error type and message, traceback digest |

Stages stay pure. A stage that wants to report progress or a metric calls `chairlift.run.events.progress(...)` or
`metric(...)`. These write through a `contextvars` sink the runner sets for the stage's duration, and do nothing
outside a run. Events never enter cache keys: the manifest says what was built, and the events say what happened.

A failed stage is recorded before the exception propagates. That is the difference between "the run died" and "fit
failed on fold 5 after 41 minutes with a MemoryError".

### 2. A live view in the terminal

`chairlift watch --root R` tails `events.jsonl` with a rich `Live` display:
- the stage table (status, elapsed, a progress bar for stages reporting progress)
- the last metrics
- any warnings

It works for a run in another terminal, in tmux, or under a systemd unit, because it only reads the file. `chairlift
status --root R` is the one-shot version and exits with the run's state (0 ok, 1 failed, 2 running), so scripts and
timers can poll it.

### 3. Run reports as static HTML, lookout style

`chairlift report --root R [--out DIR]` renders one self-contained page per run, stdlib only, with inline SVG and no
JavaScript dependencies. Sections:
- **header:** study, parameters, code identity, host, duration, verdict
- **stage timeline:** a Gantt of ran against reused, so a rebuild's blast radius is visible at a glance
- **stage table:** rows, seconds, CPU, peak RSS
- **metric panels:** per-fold IC and similar, and gate results
- **alerts**
- **provenance:** keys, inputs, output refs
- **diff against the previous run of the same study:** a numbers-card-style diff with the stated reason

An `index.html` lists runs across studies with status badges.

Pages are written atomically (temporary file plus rename), exactly like lookout. The output tree can live on the
shared NFS so any box's runs show up, served by a `chairlift-web.service`, or linked from the lookout index (see the
open questions). Each run page links to the lookout host/day page for that host and date: lookout's collector
already samples chairlift's processes, so resource context comes for free.

A timer (`chairlift-report.timer`, minutely while runs are active) keeps pages current. Pages of in-flight runs carry
a meta refresh, the same device lookout uses.

### 4. systemd integration

- **`chairlift submit STUDY ...`** wraps `systemd-run --user --unit=chairlift-<study>-<run_id> --collect` with resource
  properties (`MemoryMax`, `CPUQuota`, `Nice`) around `chairlift run`. It gives:
  - journald logs: `journalctl --user -u 'chairlift-*'`
  - cgroup accounting: `MemoryPeak` and `CPUUsageNSec` are read with `systemctl show` and written into
    `run_finished`
  - clean kill and status
  - survival across SSH disconnects without tmux
- **Watchdog.** When `NOTIFY_SOCKET` is set, the runner sends `sd_notify` heartbeats (stdlib socket, no dependency) at
  every stage boundary and progress event. The unit runs with `WatchdogSec` set per study, so a hung run (the fork
  deadlock class) is killed and marked failed instead of sitting at 0% CPU until someone notices.
- **Failure hooks.** Units carry `OnFailure=chairlift-alert@%n.service`, which renders the report and fires the alert
  sinks.
- **Scheduled refits (milestone 3).** `chairlift-refit@<study>.service` plus `.timer`, templated, in `systemd/` in the
  repo, installed by a script like lookout's `install_collector.sh`.
- **Exit codes that systemd and timers can act on:** 0 ok, 1 stage failure, 3 gate failure (a result, not a crash), 4
  watchdog.

### 5. Alert rules and sinks

Rules are evaluated at `run_finished` (and continuously by `watch`), using lookout's severities:

| rule | severity |
|---|---|
| run failed, watchdog timeout | critical |
| a registered gate failed; the golden snapshot changed without a stated reason | serious |
| a stage took more than 3× its median over the last 10 runs of the same key shape | warning |
| peak RSS above 80% of the unit's `MemoryMax` (or of MemTotal) | warning |
| a source stage without a fingerprint (always runs) | info |
| milestone 3, live: rolling IC outside its HAC band, an input fingerprint unchanged for N days (stale feed), decay per input past its death rule | serious |

Sinks:
- (a) the report page and index badges, always
- (b) an append-only `alerts.jsonl` in the shared tree, which a lookout panel can read
- (c) opt-in push: local `sendmail`, or a webhook such as Slack

Push is off by default and configured per host, never per study.

### Design principles borrowed from a reflex-trading substrate sketch (2026-09-30)

A reflex-trading substrate sketch solves a rhyming
problem: one writer that must never stall, many consumers at different speeds, contained failures, and decisions that
cannot wait for a round trip. Its claims map onto chairlift as follows.

1. **One artifact, extensible.** The manifest and the event stream share one record shape; every line carries `kind`
   and `schema`. A new capability is a new event kind; readers skip kinds they do not know, so old readers keep working.
2. **Drop-not-block (the infruptor).** The runner is the single writer and never waits on a reader. Each reader
   (watch, report timer, alert rules, collector) keeps its own byte offset. A dead or slow reader falls behind and is
   reported as stale; it can never slow or fail a run.
3. **Compute path and halo (the twins).** The runner and the stages are the compute path; watch, reports, alerts and
   ledger checks are the halo. The halo informs between runs, through the spec and the ledger, never inside one.
4. **Pre-armed authority (the reflex).** Budgets are frozen in the spec before launch: the trial budget, `MemoryMax`,
   `WatchdogSec`, gate thresholds. The runner and systemd enforce them locally. Failure responses are pre-composed as
   units (`OnFailure` renders the report and fires the alert) so no human is in the failure path.
5. **Keyed demux and explicit gaps.** One collector fans run events into per-study channels in the shared tree, each its
   own failure domain. A missed heartbeat becomes an explicit `gap` event, never silence.
6. **Status honesty.** Every run report carries a verification panel: invariants proven by tests in this build, and
   those assumed.

From an earlier daily automation (a dead timer went unnoticed; parameters moved without a log): a dead-man's switch for
scheduled runs (an expected run that did not happen is a critical alert), and parameters change only through the spec.


## Scope in v1 / out-of-scope

**In v1:**
- the event stream (layer 1) with failure capture and the contextvar progress/metric API
- `watch` and `status`
- the static report and index
- `submit` with the watchdog and `OnFailure`
- the first four alert rules with sinks (a) and (b)

**Out of scope:**
- the live-monitoring rules, which need milestone 3's production cadence
- the push sinks beyond a single tested webhook
- changes to lookout itself: those are in the shared devops repo and need Laurent's call
- a web server beyond the stdlib static unit

## Stakeholders & buy-in

- Laurent: owner; decides where the report tree lives and whether lookout links to it.

## Preconditions

- The runner and CLI exist (met: `run/dag.py`, `run/cli.py`).
- A shared, world-readable report location. The candidate is a `chairlift/` tree beside lookout's on NFS. **Pending.**
- A user systemd manager with linger on the target boxes (already true where lookout runs).

## Implementation sequence

1. **Event stream.**
   - `run/events.py`: sink, schema, `progress` / `metric` / `gate` / `warning`.
   - Runner integration, including failed stages.
   - Tests: event order, a failure run, hits recorded, events absent from keys.
   - Walkthrough `wt_events.py`.
2. **`status` and `watch`** over the stream. Test `status` exit codes on synthetic event files.
3. **Report and index** (stdlib HTML, atomic writes). Golden test on the page's structure (syrupy on a normalized DOM
   outline, not bytes).
4. **`submit`, the watchdog and `OnFailure`.** Unit templates in `systemd/` and an install script. Test the watchdog
   with a stage that sleeps past a short `WatchdogSec` under `systemd-run --user`, marked as an integration test that
   runs where a user manager exists.
5. **Alert rules and sinks (a) and (b).** Test each rule on synthetic histories.
6. **Link into lookout** (optional; a shared-repo change).

## Verification criteria

- **Replay:** a run's `events.jsonl` replays into the same report page after the run, regardless of how the run was
  launched (terminal, tmux, systemd).
- **Hang detection:** a deliberately hung stage under `chairlift submit` with `WatchdogSec=10s` is killed within 20 s,
  its `run_finished` says `watchdog`, and the alert unit fires.
- **Failure capture:** a failed stage's event carries the stage name, elapsed time and error, and the report shows it
  with critical severity.
- **Report/state consistency:** `chairlift report` on a toy run is covered by a structural golden test, and its stage
  table agrees with `RunReport` and the manifest.
- **No cost to reproducibility:** removing `events.jsonl` changes no cache key and no output.

## Open questions

- **Where the report tree lives:** beside lookout's on NFS with its own web unit, or inside lookout's `reports/`
  with an index link (a small lookout change).
- **Whether metrics should also be written as TSV in lookout's schema style,** so one renderer could chart both
  machine and model health.
- **Default `WatchdogSec`:** per study (a walk-forward fold can legitimately take an hour), or derived from the stage's
  history.
- **Push channel:** email, a Slack webhook, or both.

## Related plans / docs / PRs

- `plans/m0-skeleton.md` (the runner this instruments)
- `plans/speculative/chairlift-automated-strategy-search.md` (searches are long runs; per-trial metrics go through the
  same event stream)
- lookout: `~/devops/lookout/` (README, `report.py` alert model, `systemd/` units)

## Closeout / as-built

*Fill at completion.*
