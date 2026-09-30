# Chairlift Automated Strategy Search

**Status:** speculative
**Prepared:** 2026-09-29
**Owner:** Laurent Lanteigne
**Buy-in:** <fill when promoting to actionable>

---

## Problem / motivation

Laurent wants chairlift to go beyond "run the study I specified". The target mode is to declare the study's objects (universe, target, feature families, book) plus a search-space config, launch one command, and have the pipeline search the rest: target transforms, feature catalog and funnel settings, model family and hyper-parameters, refit schedule, and possibly portfolio parameters. The output is a complete candidate strategy, ready for a human to freeze and take to the holdout.

The same requirement sets the acceptance bar. The search mode, like the rest of chairlift, must run on two very different studies:
- **slalom:** cross-sectional; ranks ~60–790 single names per day on a hedged-straddle P&L target. It failed its holdout.
- **VXX:** a time series; sizes one position through time.

A search engine that only works for one shape isn't modular, and forcing both through the same code is how we find out.

Why now: the design already has every ingredient the search needs, but none of the glue:
- fold loop, manifest cache, feature funnel, look ledger, holdout seal
- nested hyper-parameter search, currently "off by default"

If the interfaces are designed without the search in mind, they will need to change later. M0 is exactly where that is cheap.

**The central risk is the reason this needs a plan rather than a script.** Automated search is the most efficient way to overfit a backtest ever built. slalom quantified the cost of *hand* search:
- 11 amendments and 2 disclosed looks explained about ⅓ of the dev→holdout gap.
- The untuned ridge kept 37% of its dev Sharpe on the liquid book; the tuned candidate kept 21%.
- The single most damaging choice was a portfolio-level eligibility rule (amendment 8, volume-first), chosen on dev.

An automated search runs hundreds or thousands of trials. If it reports the best trial's dev Sharpe as the result, it produces slalom's failure mode at scale and faster.

So the deliverable is not "a search". It is a search whose output carries the full cost of having searched, and which reports:
- the out-of-sample performance of the **procedure** rather than of the chosen configuration
- a deflated Sharpe ratio and a probability of backtest overfitting
- what the same search finds when there is **nothing to find**

The cost of not doing this carefully is building a machine that manufactures the next slalom. The cost of not doing it at all is that every study keeps paying for manual tuning, and pays for it invisibly.

## Proposed approach

### 1. The search space is data, declared per study, in layers

A `SearchSpace` is a frozen spec object like every other chairlift spec, so its canonical JSON hash identifies the search in the ledger. Dimensions are typed (categorical, integer, float, log-float, conditional on another dimension), and grouped into layers. Each layer is either searchable or frozen by default.

| layer | examples | default |
|---|---|---|
| target transforms | grouping for demeaning (date, date × sector, none), scale (within-date sd, MAD, none), rank vs raw, winsor level, lag twin as training target | **searchable** |
| target definition | position spec (structure, hold, exit rule), accounting | **frozen**: changing what you predict is a new hypothesis and needs a new registration |
| feature catalog | which families are on, each family's grid (lookbacks, levels), which unary transforms | searchable |
| funnel | screen null percentile, top-K per family, cluster threshold, representation (member / mean / PC1), greedy max-k | searchable |
| model | family (ridge, ridge-on-representatives, GBM, ranking GBM, tail classifier), hyper-parameters within each family (conditional dimensions) | searchable |
| ensemble | members, weighting (z-mean, stacker on nested OOF) | searchable |
| schedule | refit cadence (12 / 6 / 3 / 1 months), window (12–48 months or expanding), weight half-life | searchable |
| book | quantile cut, eligibility thresholds, sizing rule, participation cap | **frozen by default**, opt-in behind an explicit flag that is logged |

Two defaults do the heavy lifting: the target definition and the book are not searched unless the researcher says so.
- **Target definition:** searching over what to predict turns a hypothesis test into a fishing trip. The target is the hypothesis.
- **Book:** slalom's most damaging dev choice was a book rule, capacity-driven eligibility, and book parameters interact with regime in ways the fold structure cannot see. When the book layer is opted in, its dimensions get their own sub-budget and are reported separately.

Every search space has a **trial 0**: the study's declared default configuration, i.e. the untuned baseline. It always runs, it is always reported beside the winner, and it is always carried to the holdout alongside the candidate. That is slalom's lesson made structural.

### 2. The objective is declared, in the target's units, after costs

The objective is a scalar computed by the existing evaluator on validation predictions, so search and final evaluation cannot disagree about what "good" means. Default:
- **objective:** daily book Sharpe (the overlapping-book statistic from amendment 10) after costs at a declared fraction of the quoted spread per trade (default 0.5, i.e. market orders)
- **constraints:** feasibility conditions, not penalties:
  - minimum positions per day
  - maximum turnover
  - maximum drawdown in target units
  - minimum number of validation half-years with data
- **stability term** (optional, off by default): penalize configurations whose validation Sharpe varies a lot across inner folds; reported regardless

Costs go into the objective on purpose. slalom's edge existed at mid and not at the touch: break-even at 30% of the quoted spread per trade against 50% for a market order. A search optimizing mid-price Sharpe would have selected for the cost-sensitive part of the edge.

For VXX the objective is the same statistic on a one-position book, with its own cost model (futures or ETF spread plus roll). No code difference.

Multi-objective search (a Pareto front over Sharpe, turnover and capacity) is deferred to v2. It is the right tool for the book layer, which is also deferred.

### 3. Nested walk-forward: we evaluate the procedure, not the configuration

This is the load-bearing decision.

- The **outer loop** is the dev walk-forward the study already reports (e.g. slalom's 7 test years 2017–2023 on rolling 4-year windows with a 21-day embargo on exit dates).
- For each outer fold, the search runs **only inside that fold's training window**, on inner walk-forward folds carved from it. For example: a 4-year window gives 3 inner test years on expanding inner windows, with the same embargo rule.
- The search picks a configuration using inner validation only. That configuration is refit on the full outer training window and predicts the outer test year.
- The outer test years are therefore never seen by the search that produced their predictions.

The concatenated outer predictions measure **what you get if you run this search procedure and trade its answer**, which is the thing we actually want to know. This is also exactly the nested fix proposed for scale_research's selection/scoring overlap, generalized.

What "the candidate" means afterwards is a separate step:
- The final configuration for the holdout comes from running the search once on the whole dev period (inner folds across all of dev).
- Its dev statistics are **not** the headline. The headline is the procedure's outer-OOS result, with the deflation of §5.
- The holdout is sealed throughout; the loader refuses it to the search exactly as it does to everything else.

A per-fold stability read comes for free: seven outer folds each choose a configuration.
- If they choose similar ones (same model family, similar feature set, schedule within one step), the procedure is stable, and the full-dev configuration is a believable summary of it.
- If every fold picks something different, the "strategy" is the search itself. That is a finding, not a failure to hide.

**Worked cost example (slalom):** 7 outer folds × 3 inner folds × 200 trials = 4,200 fits per search, plus 200 × 3 for the final full-dev run.
- Nearly all feature work is shared through the manifest cache, since families don't depend on the trial.
- Target transforms and funnel runs are per trial but cheap.
- The expensive part is GBM fits (5 seeds each, ~0.5M rows per window).
- Mitigations:
  - successive halving (score every trial on the first inner fold, drop the bottom half, continue)
  - model-family-conditional budgets (ridge trials are nearly free)
  - a process pool with `spawn` (the polars fork deadlock noted in CLAUDE.md)
- Target: one full slalom search in under ~8 hours on this box. That is to be measured, not assumed (open question).

### 4. Search algorithm: staged, simple first, with a random baseline

- **The funnel stays deterministic given its parameters.** A trial sets funnel parameters, and the funnel's four stages run as specified. The search does not choose individual features; it chooses how features are chosen. This keeps the number of effective degrees of freedom countable.
- **Samplers:** random search as the always-available baseline, Optuna TPE as the default, grid for small categorical spaces. Random search is a real competitor on low-signal problems, and running it beside TPE shows whether the clever sampler is finding structure or chasing noise.
- **Pruning:** successive halving / ASHA across inner folds, with the first inner fold as the rung.
- **Seeds:** every trial has a seed derived from (search hash, trial number), so a search is reproducible bit for bit. The report hash is part of the numbers card.

The Optuna dependency is a decision point (open question). An in-house random + successive-halving runner is ~200 lines, and would avoid a dependency in the core. Optuna's TPE and fANOVA importance are hard to match, though. Current lean: Optuna in an optional `chairlift[search]` extra, with random search in core.

### 5. Selection accounting: every trial is a look, and the report says what that costs

The ledger records the search as one registered event with:
- the trial count N
- the full distribution of trial objective values
- the search-space hash and the budget consumed

Four adjustments are computed and reported:
1. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014). It corrects the winner's Sharpe for N trials, the variance of trial Sharpes, and the skew and kurtosis of the returns. It is reported on the procedure's outer-OOS Sharpe *and*, for contrast, on the naive best-trial dev Sharpe.
2. **Probability of Backtest Overfitting via combinatorially symmetric cross-validation** (Bailey, Borwein, López de Prado & Zhu 2017). It uses the trials × time-block matrix of returns: how often is the in-sample best configuration below the median out of sample? PBO > 0.5 means the search is picking noise.
3. **Search null.** Run the identical search, same space, budget and seeds, on a target with no signal. Two versions:
   - a within-date permutation for cross-sectional studies, so the cross-section's structure is kept but the link is broken
   - a block-rolled target for time series, keeping autocorrelation (the circular-shift idea from scale_research)

   The null gives the distribution of "best procedure outer-OOS Sharpe when there is nothing to find". The real result is reported as a percentile of that null. **This is the single most honest number the search produces.** If a 200-trial search finds Sharpe 1.1 on noise, a real 1.5 is weak evidence, whatever its bootstrap CI says. With k null repetitions the cost is k × the real search, so the default is k = 5 with an option for more. Even k = 1 is far better than none.
4. **Reality check against the incumbent / trial 0.** Hansen's SPA test (or White's Reality Check) of the winner against trial 0 and, when present, the incumbent: is the improvement larger than the best of N random configurations would give?

These become registered gates for a search-produced candidate, **in addition** to the existing dev gates. Proposed defaults, to be frozen in the study's spec before the search runs:
- DSR > 0.95
- PBO < 0.3
- real result above the 95th percentile of the search null
- SPA p < 0.05 against trial 0

A candidate that fails any of them is reported, not promoted.

### 6. The candidate report

One builder, the same notebook kit as today (how-to-read → code → conclusion per section). Sections:
1. **Verdict line:** procedure outer-OOS Sharpe [CI], DSR, PBO, search-null percentile, SPA p against trial 0, pass/fail on each registered gate.
2. **The procedure's outer-OOS record:** daily book, by year, by half-year, both legs (cross-sectional) or long/short periods (time series). Beside it, trial 0 run through the same outer folds.
3. **Stability of choices across outer folds:** a table of the chosen configuration per fold; agreement per dimension.
4. **What mattered:** fANOVA / permutation importance of each search dimension on the trial objective, per layer.
5. **Budget:** trials run and pruned, wall time, cache hits, the trial-objective distribution (histogram), and the null distribution overlaid.
6. **The final configuration:** spec hash, the dimensions that moved from their defaults and by how much, and the economic reading of each move. The report prompts for one sentence per moved dimension. A move without a reason is flagged, not blocked.
7. **Diagnostics inherited from the standard run:** decay per input and per model, residual reads, the untuned baseline and the broad universe beside the candidate.

The human step stays human. Freezing the candidate and writing the OPENED ledger entry are separate commands. The search never opens a holdout.

### 7. How the two clients exercise it

A sketch of both search spaces as data (spec format pending the dataclass decision). The point is that the files differ and the engine doesn't.

```python
slalom_space = SearchSpace(
    target_transforms=Choice("demean_by", ["t0", "t0×sector"]) & Choice("scale", ["sd", "mad"]),
    funnel=Int("top_k_per_family", 2, 6) & Float("cluster_dist", 0.2, 0.45) & Choice("repr", ["member", "mean"]),
    model=Choice(
        "family",
        {
            "ridge": LogFloat("alpha", 1e-3, 1e2),
            "gbm": Int("num_leaves", 15, 127) & LogFloat("lr", 0.01, 0.2) & Int("min_data", 200, 5000),
            "lambdarank": Int("num_leaves", 15, 63) & LogFloat("lr", 0.01, 0.2),
        },
    ),
    ensemble=Choice("members", [("ridge",), ("gbm",), ("ridge", "gbm")]),
    schedule=Choice("refit_months", [12, 3]) & Choice("window_months", [24, 36, 48]),
    book=Frozen(),  # amendment-8 lesson
)
vxx_space = SearchSpace(
    target_transforms=Choice("scale", ["ewm_vol_21", "ewm_vol_63"]),
    funnel=Int("max_k", 3, 12) & Float("cluster_dist", 0.2, 0.45),
    model=Choice(
        "family",
        {"ridge": LogFloat("alpha", 1e-3, 1e2), "gbm": Int("num_leaves", 7, 31) & LogFloat("lr", 0.01, 0.2)},
    ),
    schedule=Choice("refit_months", [12, 6, 1])
    & Choice("window_months", [36, 60, None])
    & Choice("half_life_months", [None, 12, 24]),
    book=Frozen(),
)
```

The slalom run is a **known-answer test**, and the most valuable validation available anywhere. The holdout is spent, so the search's candidate cannot be promoted. But we know the truth: dev said 2.16, and the holdout said 0.45. A trustworthy search protocol, run on slalom's dev only, should do three things:
- report a procedure outer-OOS Sharpe materially below 2.16
- give a DSR and search-null percentile that express doubt
- ideally produce a PBO that flags the liquid-book concentration

If the automated protocol would have told us, on dev, that slalom was weaker than it looked, the protocol works. If it reports even more confidence than the hand-tuned path did, it is dangerous, and we find that out on a study where it cannot cost money.

Reading the spent holdout here is a read about the *tool*, not the strategy. It is recorded in slalom's `HOLDOUT_LEDGER.md` as a tool-validation read with no strategy decision attached.

VXX is the modularity test:
- one entity
- empty grouping, so every within-date operation must degrade to an identity or a time-only operation
- a return target with a vol scale
- a sizing book
- far fewer rows, so the inner folds are short and the search null matters even more

The acceptance bar is zero edits to the search engine. Any edit is written up as a finding about the abstraction.

### Rejected alternatives

- **Search on dev, report the best dev trial, trust the holdout to catch it.** This is what we did by hand with slalom, at smaller scale. It spends the holdout as the only guard, and a spent holdout cannot be reused. Rejected: the holdout must confirm, not filter.
- **One flat joint search over everything, including individual feature inclusion.** Degrees of freedom explode, and DSR's effective-N becomes unknowable. Rejected in favour of searching the funnel's *parameters*.
- **A full AutoML framework** (auto-sklearn, FLAML, AutoGluon). These assume i.i.d. rows, random K-fold cross-validation and a generic loss. Retrofitting embargoed walk-forward, grouping-aware transforms, a book-level objective after costs and a ledger would cost more than building the thin layer. Some of their samplers can be borrowed.
- **Penalized objective only** (e.g. Sharpe minus λ·complexity). A useful tie-breaker, but it does not replace the null or the procedure-level evaluation: a complexity penalty can't see how many configurations were tried.
- **Global configuration only, no per-fold choice.** Cheaper, but it loses the stability read and makes the outer-OOS number optimistic again (the configuration would have seen all folds). Rejected for the headline. The global run is still what produces the candidate.

## Scope in v1 / out-of-scope

**In v1:**
- `SearchSpace`, `Dimension`, `Trial` and `Objective` specs in `core/`, with hashing and ledger integration (a search is one registered event with N and the trial distribution).
- Nested schedule: inner walk-forward folds carved from an outer training window, with the embargo rule and the seal inherited.
- The trial runner on the manifest cache: process pool with spawn, deterministic seeds, resume-safe through the manifest (no silent skips).
- Samplers: random (core), TPE (optional extra), grid. Successive-halving pruning across inner folds.
- Trial 0 = the study's defaults, always run, always reported.
- Selection accounting: DSR, PBO via CSCV, search null (within-date permutation / block roll, k repetitions), SPA against trial 0.
- The candidate report builder, and the registered search gates in the study spec.
- Both clients: slalom's known-answer run on dev (plus the tool-validation holdout read) and a VXX search on its dev period.

**Out of scope:**
- Book-layer search (flag exists, disabled in v1; needs its own sub-budget and Pareto front).
- Target-definition search.
- Multi-objective / Pareto search.
- Neural architecture search.
- Distributed compute beyond a local process pool.
- Automatic promotion or holdout opening (never automated, not deferred).
- Live automatic re-search / redeploy (belongs with M3 production cadence, as a scheduled re-search with its own ledger policy).

## Stakeholders & buy-in

- Laurent: owner and sole approver, for the gate thresholds, the default frozen layers, and the compute budget.
- If chairlift is later shared with the team: whoever reviews the design page should also sign off on the search gates, since they define what a search-produced result is allowed to claim.

## Preconditions

- chairlift M0 skeleton (spec schema, manifest store, runner, property-test harness): **pending**.
- chairlift M1 (slalom through chairlift, dev card to 1e-12): **pending**. The known-answer test needs it.
- VXX client plug-ins (M2): **pending**.
- The six open design calls, especially orchestration (in-house runner) and spec format (dataclasses), since `SearchSpace` is a spec: **pending**.
- A compute measurement: one slalom GBM fit on a 4-year window at 5 seeds, wall time, to size the default budget. **Not measured.**
- Dependency decision: Optuna as an optional extra. **Pending.**

## Implementation sequence

1. **Specs first.** `SearchSpace` / `Dimension` (typed, conditional) / `Trial` / `Objective` / `SearchGate` in `core/`, with canonical-JSON hashing and ledger event schema. Nothing runs yet. Test: two equal spaces hash equal, and any change changes the hash.
2. **Nested schedule.** `Schedule.inner(outer_fold)` returns inner folds from the outer training window, with the embargo asserted. Test: the fold invariants hold at both levels, and no inner row's exit date reaches the outer test start.
3. **Trial runner.** It applies a trial's parameters to the study spec (a pure function: study × trial → study′), runs the fold loop through the manifest, and returns validation predictions and the objective. Caching makes feature work shared. Test: trial 0 reproduces the standard run's predictions bit for bit.
4. **Samplers and pruning.** Random, grid, TPE extra, successive halving across inner folds. Test: same search hash and seed give the same trial sequence and the same report hash.
5. **Procedure evaluation.** The outer loop runs the search per outer fold and concatenates outer predictions; then the full-dev search produces the candidate. Test: the search's access log shows no outer-test and no holdout rows.
6. **Selection accounting.** DSR, PBO-CSCV, search null, SPA. Test on synthetic data:
   - on pure noise, the procedure CI covers zero, PBO ≈ 0.5 or above, and the real result sits inside the null
   - with a planted signal, all four read as real
7. **Candidate report.** The builder, gates in the spec, stability table, importance, budget and null overlay.
8. **slalom known-answer run.** Dev-only search; record the four adjustments and the procedure outer-OOS. Then the tool-validation holdout read, disclosed in slalom's ledger. Write up whether the protocol would have flagged the failure.
9. **VXX run.** Search on its dev period through the same engine. Record every engine edit as a finding.
10. **Docs:** a how-to-read for the candidate report, and a page on what the four adjustments mean and how to disagree with them.

## Verification criteria

- **Leak-free by construction:** the seal test and outer-test access log show zero rows from the outer test years or the holdout touched by any search step. Enforced as a failing test, not a review item.
- **Reproducible:** the same search spec, seed and data hash give an identical trial sequence, candidate spec hash and report hash.
- **Trial 0 identity:** trial 0's validation and outer predictions equal the non-search standard run exactly.
- **Calibrated on synthetic data:**
  - on a pure-noise target with the slalom-sized space and budget, the procedure's outer-OOS Sharpe 99% CI contains 0, PBO ≥ 0.4, DSR < 0.95, and the search-null percentile is < 95 in at least 90% of 20 noise seeds
  - with a planted linear signal of known IC, the search recovers it: DSR > 0.95, null percentile > 99
- **slalom known answer:** the procedure outer-OOS Sharpe on dev is reported with its CI. The acceptance criterion is that the report is *less* confident than the naive dev 2.16 (a DSR or null percentile that would have failed at least one registered search gate, or a PBO ≥ 0.3). If the protocol would have passed slalom with confidence, that is a failed verification and a design problem, not a result to report around.
- **VXX modularity:** the search engine runs VXX with zero edits to `search/`, `schedule/` or `evaluate/`; any edit is listed in the closeout as an abstraction finding.
- **Budget:** a full slalom search (200 trials, 7 outer × 3 inner, 5 null repetitions deferred) completes within the budget agreed after the compute measurement, with its wall time and cache-hit rate in the report.

## Open questions

- **Objective:** Sharpe after costs at 0.5 of the spread per trade, or at a fill-fraction the study declares (slalom's execution assumption was passive fills)? Should the constraint set be study-declared only?
- **Per-fold vs global choice:** is the per-fold configuration table enough of a stability read, or should the procedure use a consensus configuration (mode across folds) as an alternative candidate?
- **Null construction for cross-sections:** a within-date permutation keeps each day's cross-section but breaks name persistence. Should it be permuted across names consistently for the whole period (keeps persistence, breaks the link)? Probably the latter; test both on synthetic data.
- **Default budget and k for the null** once the compute measurement exists.
- **Optuna or in-house for TPE.**
- **Whether book search is ever allowed**, or only as a separate, explicitly registered "capacity study" with its own holdout.
- **Gate thresholds** (DSR 0.95, PBO 0.3, null p95, SPA 0.05): sensible defaults, or should studies with fewer rows (VXX) use different ones?
- **How to present DSR when returns are overlapping** (daily book of 21-day positions): effective sample size for the skew/kurtosis terms.

## Related plans / docs / PRs

- slalom holdout verdict: NB11 `11_holdout_verdict`, FINDINGS 2026-09-24 (selection share of the gap, the amendment-8 lesson).
- Literature:
  - Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
  - Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest Overfitting*
  - Hansen (2005), *A Test for Superior Predictive Ability*
  - White (2000), *A Reality Check for Data Snooping*
  - Bergstra & Bengio (2012), random search for hyper-parameter optimization
  - Li et al. (2018), Hyperband / ASHA

## Closeout / as-built

*Fill when status becomes `complete`.*
