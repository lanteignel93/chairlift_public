"""Models: a spec says what to fit; `build(spec)` returns an estimator with fit / predict / describe.

    RidgeSpec      closed-form ridge on standardised inputs, penalty alpha · n (alpha does not scale with sample size)
    LightGBMSpec   gradient boosting, several seeds averaged (needs the `ml` extra)
    RuleSpec       a regime rule searched on the training rows: be in the position only when one input is above (or
                   below) one of its training quantiles; the rule is chosen by the position's own Sharpe, not by a
                   regression fit, and "always in" is one of the candidates

An estimator is fitted on training rows only by the walk-forward fit; nothing here knows about folds or dates.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from chairlift.core.spec import spec


class Estimator(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Estimator: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def describe(self) -> dict[str, Any]: ...


@spec(name="Ridge", version=1)
class RidgeSpec:
    alpha: float = 1e-3
    intercept: bool = False  # cross-section: irrelevant to a ranking; time series: the training mean is the carry


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


ModelSpec = RidgeSpec | LightGBMSpec | RuleSpec


class Ridge:
    def __init__(self, alpha: float, intercept: bool = False) -> None:
        self.alpha = alpha
        self.intercept = intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> Ridge:
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        self.ym = float(y.mean())
        A = Z.T @ Z + self.alpha * len(y) * np.eye(Z.shape[1])
        self.w = np.linalg.solve(A, Z.T @ (y - self.ym))
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> LightGBM:
        lgb: Any = importlib.import_module("lightgbm")  # the `ml` extra; untyped
        p = {k: v for k, v in self.spec.params.items() if k != "n_estimators"}
        p.update(objective="regression", verbose=-1, num_threads=self.spec.threads)
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> Rule:
        s = self.spec
        r = np.nan_to_num(y)
        best = (self._sharpe(s.position * r, s.periods), -1, 0.0, 0)  # always on
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
        self.train_sharpe, self.j, self.t, self.side = best
        self.always_sharpe = self._sharpe(s.position * r, s.periods)
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
            "candidates": self.candidates,
        }


def build(s: ModelSpec) -> Estimator:
    if isinstance(s, RuleSpec):
        return Rule(s)
    if isinstance(s, RidgeSpec):
        return Ridge(s.alpha, s.intercept)
    return LightGBM(s)
