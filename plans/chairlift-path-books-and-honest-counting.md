# chairlift on path-marked books: native overlapping-entry books, kept models, honest counts, outcome-free seals

**Status:** actionable
**Prepared:** 2026-10-14
**Owner:** Laurent Lanteigne
**Buy-in:** Laurent, 2026-10-14 ("1 and 2 and then let's see how well the ridge with ML rank sizing does on the dev?"; mogul log M-522)

---

## Problem / motivation

On 2026-10-13 chairlift was pointed at mogul, a delta-hedged VIX option book with a new entry every trading day, each
entry held about a month and marked daily along its path, with a stop. The client
(`mogul_trade_private/chairlift_client/`) ran 19 experiments and searches, 176 ledgered trials. The machinery did its
core job: no leakage path, stages rebuilt to the third decimal by an independent script, searches that judged
themselves (PBO 0.65 on the trade knobs, 0.25 on the ridge penalty). The review
(`chairlift_client/REVIEW.md`, part B) also found six places where chairlift, as it stands, fits this shape of study
badly. A seventh came from the SPXW book's engine review the same night.

None of the seven changed mogul's conclusions. Each one cost a workaround, a side script or an erratum, and each will
recur in every study of overlapping, path-marked trades: strangle_client, mogul, the SPXW book, any option-selling
client. That is the class of study the desk's research is drifting toward, so these are worth fixing in the core once,
not in every client.

The seven, with what each cost tonight:

1. **No native book for overlapping, path-marked entries.** chairlift's books (quantile, signal, event pool) assume a
   position per row with a return over a fixed horizon. mogul's entries live a variable number of days, end at a stop
   or at expiry, and are marked daily. The client built its own sizing books as stages, as strangle_client had. The
   study's declared book is a placeholder, so `chairlift report` on this study reports nothing meaningful.
2. **The fit stage drops the fitted model.** `fit_walk_forward` returns each fold's description (ridge's
   standardised coefficients, LightGBM's gain importances), and the fit stage keeps only the predictions. The review
   needed `review/export_review.py`, which calls the same functions on the same panels to recover them. It reproduced
   the stored books exactly, but it is a second code path for a first-class output.
3. **The ledger counts one headline per run, not every book.** The ledger showed 32 headlined trials and a deflated
   Sharpe probability of 0.979. The honest count was about 225 books (five models times the sizing rules times the
   framings, plus the search cells). A deflated statistic is only as good as its trial count.
4. **The deflated Sharpe answers the wrong question for a known premium.** It asks whether the best book's Sharpe
   beats the expected maximum of null trials. For a short-premium book the premium makes that trivially true (DSR 1.0
   even at 225 trials). The decision question was "does the model beat constant size", a paired difference on the
   same days. The client measured it (paired block-bootstrap CIs); chairlift's judgement did not use it. This is the
   SPY lesson from chairlift's own gallery: judge the active series.
5. **Books built from live positions drop flat days.** A gated book is flat on many days; dropping them overstated
   every gated Sharpe by 0.1–0.5 (mogul erratum M-486). The client now marks every book on the trading calendar; a
   first version of that fix itself used a calendar that ended early and dropped September 2026 (M-513). The core
   should own the calendar and refuse a calendar that misses a P&L date.
6. **No power check for this shape.** Entries overlap for about a month, so a fold's training set held 8 to 46
   independent months for 29 inputs. In-sample IC was 0.6–0.8; out-of-sample swung from −0.71 to +0.64. chairlift
   0.2.0 has a power stage (MDE, Fama-MacBeth IC), but it does not know that overlapping entries share outcomes.
7. **The holdout gate seals rows by an exit-date column, which selects on outcome.** The mogul client declared
   `Holdout(start=2024-01-01, column="exit_date")`. A stop moves a trade's exit earlier, so of the trades entered
   before the seal into later expiries, the stopped ones (exit before the seal) stay in dev and the held ones are
   sealed. On a synthetic twin crossing the seal, 13 stopped trades landed in dev and 7 held ones in the holdout. In
   mogul no straddling trade stopped, so no bias happened (M-516); a book with a December crash would have been
   biased, and the bias is always against the short-premium seller on the dev side.

## Proposed approach

Seven changes, smallest first. Each is independent; together they make "a book of path-marked, overlapping trades" a
first-class study shape, judged on its gain over a baseline, with honest counts.

### A. Outcome-free seals (item 7)

The holdout gate gains an assignment mode that uses only what is known at entry:
- `Holdout(start, entry="t0", horizon="expiry")`: a row is dev when its horizon (expiry) precedes the seal, holdout when
  its entry follows it, and **neither** when it straddles. The straddling rows are dropped from both windows and
  counted in the manifest.
- The existing `column=` mode stays for fixed-horizon studies (slalom: a straddle held a fixed number of days, where
  exit date = entry + horizon, so the column carries no outcome).
- The gate refuses `column="exit_date"` when the study declares a stop or any early exit (a spec flag on the
  trade source), with an error that names this plan.

*Rejected:* sealing by entry date alone. A trade entered just before the seal would then read holdout-period marks
while counted as dev, a leak.

### B. A kept model (item 2)

The fit stage stores each fold's description beside its predictions, as a second output of the same stage (same key,
same cache), e.g. `fit_ridge.models`: fold, input, coefficient or importance, in-sample IC, training rows. `chairlift
show` prints it; the report renders a coefficient-stability table (sign agreement across folds) by default.

*Rejected:* a separate "explain" stage. It would refit, which is the second code path this removes.

### C. A path book (item 1)

A new book type for overlapping path-marked trades, `PathBook`:
- input: a trades table with one row per (trade, date) carrying the trade's cumulative P&L, its entry, its exit
  reason, and an optional size column
- sizing rules as specs: constant, rank (2 × the prediction's percentile among earlier predictions), tercile, inverse
  risk, and the direct mean-variance policy the client wrote
- output: the daily book on the trading calendar (item E), the per-trade table, and the paired comparison against
  declared baselines (item D)

The mogul client's `sizes`, `daily_pnl`, `on_calendar` and `_book_stage` are the reference implementation; the
strangle client's books are the second. Both would shrink to a few lines each.

### D. Judge the gain, not the level (items 3 and 4)

- **Every book counts.** The ledger records one trial per *book evaluated*, not per run: each model × sizing rule ×
  framing that produced a judged book. The headline stays one per run; the count used for deflation is the book
  count.
- **A study may declare a baseline.** `judge = Gain(baseline="constant")`: the judgement then deflates the *paired
  Sharpe difference* against the baseline on the same days (a paired block bootstrap for the CI; the deflation over
  the book count). The plain deflated Sharpe stays available and is reported, but a study with a declared baseline is
  judged on the gain.
- **Searches inherit it.** Walk-forward selection and PBO are computed on the gain series when a baseline is declared.

*Open:* whether the deflation of a paired difference should use the trials' variance of *gains* (the natural
analogue) or of Sharpes. The gains' variance is the honest one; it needs every book's gain series stored, which item
B's storage makes cheap.

### E. The calendar is the core's (item 5)

The study declares its trading calendar as a source (the chain's quote dates, or an exchange calendar). Every book is
marked on it from its first to its last P&L date, zero when flat, and the book stage raises when a P&L date is missing
from the calendar (mogul M-513's lesson). A gated book's flat days are counted by construction.

### F. Power for overlapping outcomes (item 6)

The power stage learns the overlap: the effective number of independent outcomes is the number of entries divided by
the mean holding period in entry steps (or a block-bootstrap estimate of it). The power report then states, per fold,
inputs against independent outcomes, and warns when the ratio passes a declared threshold (the client's 29 inputs on 8
independent months would have been flagged before any fit).

## Decisions at promotion (2026-10-14)

Laurent's go named the plan, not its four open questions. The plan's own proposals are taken as the answers and
flagged at the first checkpoint, where any of them can still be reversed before the branch merges:

1. **chairlift never prices options.** It books, sizes, judges and counts. The path book consumes a trades table built
   by the client's own engine (mogul's, the SPXW book's).
2. **Order:** A, then B with the DAG feature it needs, then E with C, then D. F waits for the path book's holding
   periods.
3. **The public copy** gets these as one reviewed batch, after Laurent reviews the private branch (the standing rule).
4. **Straddling rows join neither window.** They are dropped from both and counted. Joining the holdout whole would
   carry dev-period marks into the holdout book.

How each item is built, and what was rejected:

- **A, the seal.** `Holdout` gains two fields, added at defaults that keep every existing spec's hash:
  - `horizon`, the column holding the date each row's outcome is final at the latest (an option's expiry). With it, a
    sealed row is one whose horizon is on or after the seal and a holdout row is one entered on or after it; a row
    entered before the seal into a later horizon straddles and is dropped from both.
  - `early_exit`, a declaration that rows can end before their horizon (a stop). With it and no horizon, `guard`
    refuses: sealing on any other column would select on the outcome.

  The straddling rows are counted in the run's events (a metric and a warning), not in the manifest: the manifest
  records keys and outputs, not row counts.
- **B, kept models, and side outputs.** A stage may now emit named side outputs (`Stage.side`, returned as
  `Outputs(value, **side)`). Each is stored beside the main output under the same stage key, recorded in the
  manifest, addressed downstream as `<stage>.<name>`, and shown by `chairlift show`. The fit stage keeps `models`:
  each fold's description (coefficients or importances), its inputs, in-sample IC and row counts.
  - *Rejected:* a separate models stage, which refits (twice the compute, a second code path).
  - *Rejected:* a dict as the fit stage's output, which breaks every consumer of a fit, in the core and in clients.
- **E with C, the calendar and the path book.** A new `book/path.py`:
  - `PathBook`, the spec: the sizing rule (constant, rank, tercile, inverse, direct), the column names, and the
    benchmark.
  - `PathBookDecl`, the study piece: the trades source, the calendar source, the frame that carries the baseline
    weights and the outcome, and the baselines.

  Each evaluated model gets an `eval_<name>` stage. It holds the book on the declared calendar from the first test
  entry to the last P&L date (zero when flat), each baseline on the same days, the paired difference against each,
  the active series against the benchmark, and how the sizing behaves. The per-trade table (weight, prediction,
  final P&L) is its side output. The book raises when a P&L date is missing from the calendar.
- **D, honest counting and the gain.**
  - **Every judged book is a ledger entry** (`kind: "book"`), one per distinct book key, labelled with what it is
    judged on (its family). Run-level trials stay as they were, so no ledger written before this changes meaning.
    `chairlift ledger` deflates each family over its books when any exist.
  - **The gain is judged on the active series** (book minus benchmark, day by day). The DSR formula needs a return
    series, and chairlift already judges timing books this way (`active_*`). The paired Sharpe difference stays
    reported, with its CI.
  - *Rejected:* deflating the paired Sharpe difference directly. The formula's √(T−1) scaling assumes a Sharpe of one
    series, and the difference's standard error depends on the two books' correlation.
- **Acceptance.** The mogul client ported onto A, B, C, D and E must reproduce its current stored books (built after
  the stale-day drop, mogul M-519) to the third decimal. Re-running every step-10 experiment and search on the ported
  client then gives the honest book count mechanically.

## Found on the way, outside this plan (2026-10-14)

The agents that rewrote the pipeline and operations pages read the code against the docs and reported these. Each is
verified from the code by them, not yet by a test here. They are follow-ups, not part of this branch:
- **The code digest skips code under site-packages,** so a study run against a non-editable chairlift install keys its
  stages without chairlift's own code. Edits also leave the digest unchanged for helpers behind `functools.cache` or
  `lru_cache`, for module-level `functools.partial` objects, and for `functools.wraps` decorators from another file.
- **`submit` gaps:**
  - It always wraps `run`, so submitting a search experiment runs the study once instead of the search.
  - It copies `CHAIRLIFT_SECRET_*` variables into the unit with `--setenv`, so they show in `--dry-run` output and
    in the unit's environment.
  - It does not forward `--profile`, and a bad `--watchdog` value gives a traceback instead of a usage error.
- **Alerts and reports:**
  - `alerts check` reads only each study's newest run, so a later `show` hides a failed run.
  - The `page` sink does nothing.
  - The index lists report paths as text, not links.
- **Config:** nothing reads `compute.polars_threads` or `compute.memory_default`.
- **Stale docs:** guide pages 1, 2, 5 and 7, `how-to-read.md`, `docs/README.md` and the guide README describe older
  behaviour (keys, secrets, sweep charging, reports, the test count, the version).

## Scope in v1 / out-of-scope

**In v1:**
- A (outcome-free seals) and B (kept models): small, independent, and A protects every future study
- E (the calendar) as part of C, since the path book is the first consumer

**Out of scope for v1:**
- C (the path book) beyond the mogul and strangle clients' needs
- D's search integration (walk-forward selection and PBO on gains): after D's ledger change has run on one study
- F (power for overlap): after C, since it needs the path book's holding periods

## Stakeholders & buy-in

- Laurent: which items, in which order; whether the public copy gets them (private first, reviewed, as agreed).

## Preconditions

- chairlift 0.2.0 on private (met, 2026-10-13).
- The mogul client and REVIEW.md as the reference (met; `mogul_trade_private/chairlift_client/`).
- A decision on how far chairlift should own trade simulation (pending; see open questions).

## Implementation sequence

1. A: the outcome-free seal mode, its refusal of exit-date sealing under early exits, a twin test that crosses the
   seal (a test written for another option-selling study is the template).
2. B: store fold models in the fit stage; `show` and the report table; the mogul client's export script deleted.
3. E + C: the calendar source and the path book, with the mogul client ported onto it as the acceptance test (its
   stored books must reproduce: 2.036 / 2.359 / 0.973).
4. D: the book count in the ledger, then the declared baseline and the paired-gain judgement; re-judge the mogul
   study and record both judgements.
5. F: overlap-aware power, checked against the mogul client's folds.

## Verification criteria

- A twin crossing the seal yields the same dev set whatever the stop level (no outcome dependence), and the straddling
  rows are counted in the manifest.
- The mogul client, ported onto C and E, reproduces its stored books to the third decimal.
- The fit stage's stored coefficients equal `export_review.py`'s, fold by fold.
- The ledger's trial count for the mogul study equals the number of judged books (about 225), and the judgement of
  the gated call-wing ridge reads its gain over constant: +0.64 [+0.15, +1.18] before deflation.
- The power report flags the 29-input, 8-month folds before any fit runs.

## Open questions for Laurent

1. How far should chairlift own trade simulation? The path book consumes a trades table built by the client's own
   engine (mogul's, the SPXW book's). Proposed: chairlift never prices options; it books, sizes, judges and counts.
2. Order: A and B first (small, protective), or C first (it removes the most client code)?
3. Should the public copy get these as they land, or as one reviewed batch?
4. Is "neither window" for straddling trades acceptable, or should they join the holdout whole (they would carry dev
   dates into the holdout book)?
