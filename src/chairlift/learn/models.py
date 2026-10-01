"""Models: a spec says what to fit; `build(spec)` returns an estimator with fit / predict / describe.

    RidgeSpec      closed-form ridge on standardised inputs, penalty alpha · n (alpha does not scale with sample size)
    LightGBMSpec   gradient boosting, several seeds averaged (needs the `ml` extra)

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


ModelSpec = RidgeSpec | LightGBMSpec


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
        return np.mean(np.stack([np.asarray(m.predict(X), dtype=float) for m in self.models]), axis=0)

    def describe(self) -> dict[str, Any]:
        gain = np.mean([m.booster_.feature_importance("gain") for m in self.models], axis=0)
        return {"kind": "lightgbm", "seeds": list(self.spec.seeds), "gain_importance_mean": [float(v) for v in gain]}


def build(s: ModelSpec) -> Estimator:
    if isinstance(s, RidgeSpec):
        return Ridge(s.alpha, s.intercept)
    return LightGBM(s)
