"""Models: a spec says what to fit; `build(spec)` returns an estimator with fit / predict / describe.

    RidgeSpec      closed-form ridge on standardised inputs, penalty alpha · n (alpha does not scale with sample size)
    LightGBMSpec   gradient boosting, several seeds averaged (needs the `ml` extra)
    RuleSpec       a regime rule searched on the training rows: be in the position only when one input is above (or
                   below) one of its training quantiles; the rule is chosen by the position's own Sharpe, not by a
                   regression fit, and it must beat "always in" by the margin a search over that many noise rules
                   would find by luck
    CompositeSpec  no fitted weights: the equal-weight sum of the inputs, each standardised on the training rows and
                   signed by a prior (or by its training correlation). For a few inputs and a weak signal, a prior
                   beats a regression that fits the noise in a short training window

An estimator is fitted on training rows only by the walk-forward fit; nothing here knows about folds or dates.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import numpy as np

from chairlift.core.spec import added, spec


class Estimator(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, names: Sequence[str] = ()) -> Estimator: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def describe(self) -> dict[str, Any]: ...


@spec(name="Ridge", version=1)
class RidgeSpec:
    """`prior`: (input, weight) pairs, a direction in standardised-input space. The penalty then shrinks the weights
    toward that direction (scaled by its own in-fold fit) instead of toward zero; inputs it does not name have prior
    weight 0."""

    alpha: float = 1e-3
    intercept: bool = False  # cross-section: irrelevant to a ranking; time series: the training mean is the carry
    prior: tuple[tuple[str, float], ...] = added(())


@spec(name="LightGBM", version=1)
class LightGBMSpec:
    params: Mapping[str, Any]  # LGBMRegressor keyword arguments, n_estimators included
    seeds: tuple[int, ...] = (0,)
    threads: int = 16
    deterministic: bool = True  # bit-identical refits on any thread count (LightGBM's deterministic + col-wise)


@spec(name="Rule", version=1)
class RuleSpec:
    """The target `y` handed to fit() must be the return the position earns (e.g. the next period's return)."""

    position: float = -1.0  # the position when the rule is on (−1: short-or-flat; +1: long-or-flat)
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    min_on: float = 0.25  # a rule must be on for at least this share of training rows
    periods: int = 252
    # a rule replaces "always in" only if its training Sharpe beats always-in by this margin. "deflated": the expected
    # maximum Sharpe of that many noise rules over the training window (conservative: candidate rules are correlated)
    margin: float | Literal["deflated"] = added("deflated")


@spec(name="Composite", version=1)
class CompositeSpec:
    """`signs`: (input, sign) pairs, the prior. Inputs without a prior get the sign of their training correlation with
    the target when `learn_signs`, else are left out. `min_abs_corr`: a learned sign needs at least this |corr|."""

    signs: tuple[tuple[str, float], ...] = ()
    learn_signs: bool = False
    min_abs_corr: float = 0.0


ModelSpec = RidgeSpec | LightGBMSpec | RuleSpec | CompositeSpec


class Ridge:
    def __init__(self, alpha: float, intercept: bool = False, prior: Sequence[tuple[str, float]] = ()) -> None:
        self.alpha = alpha
        self.intercept = intercept
        self.prior = dict(prior)

    def fit(self, X: np.ndarray, y: np.ndarray, names: Sequence[str] = ()) -> Ridge:
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.ym = float(y.mean())
        yc = y - self.ym
        lam = self.alpha * len(y)
        w0 = np.zeros(Z.shape[1])
        if self.prior:
            if len(names) != Z.shape[1]:
                raise ValueError("a ridge prior needs the input names")
            d = np.array([self.prior.get(n, 0.0) for n in names], float)
            zd = Z @ d
            # the prior direction's own scale, fitted in fold: shrink toward c·d, not toward an arbitrary magnitude
            c = float(zd @ yc / (zd @ zd)) if float(zd @ zd) > 0 else 0.0
            w0 = c * d
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ yc + lam * w0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        z = ((X - self.mu) / self.sd) @ self.w
        return z + self.ym if self.intercept else z

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "ridge",
            "alpha": self.alpha,
            "intercept": self.ym if self.intercept else None,
            "coef_standardised": [float(v) for v in self.w],
        }


class LightGBM:
    def __init__(self, s: LightGBMSpec) -> None:
        self.spec = s

    def fit(self, X: np.ndarray, y: np.ndarray, names: Sequence[str] = ()) -> LightGBM:
        lgb: Any = importlib.import_module("lightgbm")  # the `ml` extra; untyped
        p = {k: v for k, v in self.spec.params.items() if k != "n_estimators"}
        p.setdefault("objective", "regression")  # a study's own objective (huber, l1, …) is kept
        p.update(verbose=-1, num_threads=self.spec.threads)
        if self.spec.deterministic:
            p.update(deterministic=True, force_col_wise=True)
        n = int(self.spec.params.get("n_estimators", 100))
        self.models: list[Any] = [
            lgb.LGBMRegressor(n_estimators=n, random_state=s, **p).fit(X, y) for s in self.spec.seeds
        ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import warnings

        with warnings.catch_warnings():
            # LightGBM 4.6 names numpy columns at fit and then warns at predict that numpy has no names
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            return np.mean(np.stack([np.asarray(m.predict(X), dtype=float) for m in self.models]), axis=0)

    def describe(self) -> dict[str, Any]:
        gain = np.mean([m.booster_.feature_importance("gain") for m in self.models], axis=0)
        return {"kind": "lightgbm", "seeds": list(self.spec.seeds), "gain_importance_mean": [float(v) for v in gain]}


class Rule:
    def __init__(self, s: RuleSpec) -> None:
        self.spec = s

    @staticmethod
    def _sharpe(pnl: np.ndarray, periods: int) -> float:
        sd = pnl.std(ddof=1)
        return float(pnl.mean() / sd * np.sqrt(periods)) if sd > 0 else -np.inf

    def fit(self, X: np.ndarray, y: np.ndarray, names: Sequence[str] = ()) -> Rule:
        s = self.spec
        r = np.nan_to_num(y)
        always = (self._sharpe(s.position * r, s.periods), -1, 0.0, 0)
        best = (-np.inf, -1, 0.0, 0)
        self.candidates = 1
        for j in range(X.shape[1]):
            x = X[:, j]
            for q in s.quantiles:
                t = float(np.nanquantile(x, q))
                for side in (1, -1):
                    on = (x > t) if side == 1 else (x < t)
                    if on.mean() < s.min_on:
                        continue
                    self.candidates += 1
                    score = self._sharpe(np.where(on, s.position * r, 0.0), s.periods)
                    if score > best[0]:
                        best = (score, j, t, side)
        if s.margin == "deflated":
            from chairlift.evaluate.deflated import expected_max_sharpe

            # per-period Sharpe estimates over len(r) rows have variance ≈ 1/len(r)
            self.margin = expected_max_sharpe(self.candidates, 1 / max(len(r), 2)) * np.sqrt(s.periods)
        else:
            self.margin = float(s.margin)
        self.always_sharpe = always[0]
        self.best_rule_sharpe = best[0]
        chosen = best if best[1] >= 0 and best[0] - always[0] > self.margin else always
        self.train_sharpe, self.j, self.t, self.side = chosen
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.j < 0:
            return np.full(len(X), self.spec.position)
        x = X[:, self.j]
        on = (x > self.t) if self.side == 1 else (x < self.t)
        return np.where(on, self.spec.position, 0.0)

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "rule",
            "input": self.j,  # index into the fold's inputs; -1 = always on
            "threshold": self.t,
            "on_when": "above" if self.side == 1 else "below",
            "train_sharpe": self.train_sharpe,
            "always_on_train_sharpe": self.always_sharpe,
            "best_rule_train_sharpe": self.best_rule_sharpe,
            "margin": self.margin,
            "candidates": self.candidates,
        }


class Composite:
    def __init__(self, s: CompositeSpec) -> None:
        self.spec = s

    def fit(self, X: np.ndarray, y: np.ndarray, names: Sequence[str] = ()) -> Composite:
        if len(names) != X.shape[1]:
            raise ValueError("a composite needs the input names")
        prior = dict(self.spec.signs)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        w = np.zeros(X.shape[1])
        self.source: list[str] = []
        for j, n in enumerate(names):
            if n in prior:
                w[j], src = float(np.sign(prior[n])), "prior"
            elif self.spec.learn_signs:
                c = float(np.corrcoef(Z[:, j], y)[0, 1]) if Z[:, j].std() > 0 and y.std() > 0 else 0.0
                w[j], src = (float(np.sign(c)), "learned") if abs(c) >= self.spec.min_abs_corr else (0.0, "dropped")
            else:
                src = "no prior"
            self.source.append(src)
        # no prior and no learned sign strong enough: no evidence, so no tilt (a constant prediction; a study that can
        # live with one says so with FitSpec(allow_constant=True))
        self.w, self.names = (w / np.abs(w).sum() if np.any(w) else w), list(names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "composite",
            "weights": {n: float(v) for n, v in zip(self.names, self.w, strict=True)},
            "sign_source": dict(zip(self.names, self.source, strict=True)),
        }


def build(s: ModelSpec) -> Estimator:
    if isinstance(s, RuleSpec):
        return Rule(s)
    if isinstance(s, RidgeSpec):
        return Ridge(s.alpha, s.intercept, s.prior)
    if isinstance(s, CompositeSpec):
        return Composite(s)
    return LightGBM(s)
