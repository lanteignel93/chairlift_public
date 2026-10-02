"""In-fold input transforms: fitted on a fold's training rows, applied unchanged to its test rows.

    Winsorize      clip each input at its training q and 1 − q quantiles
    Interactions   add the pairwise products of the `top` inputs with the largest |training correlation| to the
                   target, each centred at its training mean (for ranks: a "both high or both low" bet)
    PCA            replace (or extend) the inputs with the first k principal components of the standardised training
                   inputs

A transform sees the training rows only, so its quantiles, correlations and loadings are as unknown to the test year
as the model's weights. Transforms run after selection and the null fill, in the order given.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from chairlift.core.spec import spec


@spec(name="Winsorize", version=1)
class Winsorize:
    q: float = 0.01


@spec(name="Interactions", version=1)
class Interactions:
    top: int = 4


@spec(name="PCA", version=1)
class PCA:
    k: int = 5
    keep_inputs: bool = False


Transform = Winsorize | Interactions | PCA


class Fitted:
    """One fitted transform: `apply` maps a matrix with the training columns to the transformed matrix."""

    def __init__(self, s: Transform, X: np.ndarray, y: np.ndarray, names: list[str]) -> None:
        self.s = s
        self.names_in = names
        match s:
            case Winsorize(q=q):
                if not 0 <= q < 0.5:
                    raise ValueError(f"Winsorize q must be in [0, 0.5), got {q}")
                self.lo, self.hi = np.quantile(X, q, axis=0), np.quantile(X, 1 - q, axis=0)
                self.names_out = names
            case Interactions(top=top):
                ic = np.nan_to_num(np.array([_corr(X[:, j], y) for j in range(X.shape[1])]))
                order = sorted(range(len(names)), key=lambda j: (-abs(float(ic[j])), j))
                self.chosen = sorted(order[: min(top, len(names))])
                self.mu = X.mean(0)
                self.pairs = [(a, b) for i, a in enumerate(self.chosen) for b in self.chosen[i + 1 :]]
                self.names_out = names + [f"{names[a]}×{names[b]}" for a, b in self.pairs]
            case PCA(k=k, keep_inputs=keep):
                self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
                _, _, vt = np.linalg.svd((X - self.mu) / self.sd, full_matrices=False)
                k = min(k, vt.shape[0])
                # a component's sign is arbitrary; fixing it (largest loading positive) makes refits comparable
                signs = np.sign(vt[np.arange(k), np.abs(vt[:k]).argmax(1)])
                self.v = (vt[:k] * signs[:, None]).T
                self.names_out = (names if keep else []) + [f"pc{i + 1}" for i in range(k)]

    def apply(self, X: np.ndarray) -> np.ndarray:
        match self.s:
            case Winsorize():
                return np.clip(X, self.lo, self.hi)
            case Interactions():
                C = X - self.mu
                extra = [C[:, a] * C[:, b] for a, b in self.pairs]
                return np.column_stack([X, *extra]) if extra else X
            case PCA(keep_inputs=keep):
                Z = ((X - self.mu) / self.sd) @ self.v
                return np.column_stack([X, Z]) if keep else Z

    def describe(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": type(self.s).__name__}
        if isinstance(self.s, Interactions):
            d["pairs"] = self.names_out[len(self.names_in) :]
        return d


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = a.std(), b.std()
    if not (sa > 0 and sb > 0):
        return 0.0
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def fit_apply(
    specs: Sequence[Transform], X_tr: np.ndarray, y: np.ndarray, X_te: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    """Fit each transform on the (already transformed) training matrix, in order; apply it to both matrices."""
    cols: list[str] = list(names)
    info: list[dict[str, Any]] = []
    for s in specs:
        f = Fitted(s, X_tr, y, cols)
        X_tr, X_te, cols = f.apply(X_tr), f.apply(X_te), f.names_out
        info.append(f.describe())
    return X_tr, X_te, cols, info
