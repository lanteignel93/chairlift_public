import datetime as dt

import numpy as np
import polars as pl
import pytest

from chairlift.core.spec import SpecError, added, spec, spec_hash
from chairlift.learn.fit import FitSpec, fit_walk_forward
from chairlift.learn.models import RidgeSpec
from chairlift.learn.transforms import PCA, Interactions, Winsorize, fit_apply
from chairlift.schedule.walkforward import WalkForward, make_folds


def test_winsorize_clips_at_training_quantiles_only():
    X_tr = np.arange(101, dtype=float)[:, None]
    X_te = np.array([[-50.0], [50.0], [500.0]])
    _, te, cols, _ = fit_apply([Winsorize(q=0.1)], X_tr, X_tr[:, 0], X_te, ["a"])
    assert cols == ["a"]
    assert te[:, 0].tolist() == [10.0, 50.0, 90.0]


def test_interactions_pick_the_top_inputs_by_training_correlation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    y = X[:, 1] + 0.5 * X[:, 3] + 0.01 * rng.normal(size=500)
    _, te, cols, info = fit_apply([Interactions(top=2)], X, y, X[:10], ["a", "b", "c", "d"])
    assert cols == ["a", "b", "c", "d", "b×d"]
    assert info[0]["pairs"] == ["b×d"]
    mu = X.mean(0)
    np.testing.assert_allclose(te[:, 4], (X[:10, 1] - mu[1]) * (X[:10, 3] - mu[3]))


def test_pca_is_fitted_on_training_rows_and_its_sign_is_fixed():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(400, 1))
    X = np.column_stack([f[:, 0], f[:, 0] + 0.1 * rng.normal(size=400), rng.normal(size=400)])
    tr, _, cols, _ = fit_apply([PCA(k=1)], X, f[:, 0], -X[:5], ["a", "b", "c"])
    assert cols == ["pc1"]
    assert np.corrcoef(tr[:, 0], f[:, 0])[0, 1] > 0.99  # the largest loading is positive: pc1 moves with the factor
    tr2, _, _, _ = fit_apply([PCA(k=1)], -X, f[:, 0], X[:5], ["a", "b", "c"])
    np.testing.assert_allclose(np.abs(tr2), np.abs(tr), atol=1e-9)


def test_an_added_field_at_its_default_keeps_the_old_hash():
    @spec(name="Thing", version=1)
    class Old:
        a: int = 1

    @spec(name="Thing", version=1)
    class New:
        a: int = 1
        b: tuple[int, ...] = added(())

    assert spec_hash(Old()) == spec_hash(New())
    assert spec_hash(New(b=(1,))) != spec_hash(Old())
    with pytest.raises(SpecError):
        added([])


def test_a_fit_with_transforms_runs_in_fold():
    rng = np.random.default_rng(2)
    dates = [dt.date(2010, 1, 1) + dt.timedelta(days=30 * i) for i in range(72)]
    rows = []
    for d in dates:
        x = rng.uniform(size=(30, 3))
        y = x[:, 0] * x[:, 1] + 0.1 * rng.normal(size=30)
        rows += [
            {
                "root": f"n{i}",
                "t0": d,
                "exit_date": d + dt.timedelta(days=28),
                "a": x[i, 0],
                "b": x[i, 1],
                "c": x[i, 2],
                "y_z": y[i],
                "y_rank": y[i],
            }
            for i in range(30)
        ]
    panel = pl.DataFrame(rows)
    w = WalkForward(start=dates[0], first_test_year=2013, mode="expanding", embargo=1)
    folds = make_folds(panel, w)
    s = FitSpec(model=RidgeSpec(alpha=1e-4), transforms=(Winsorize(q=0.01), Interactions(top=2)))
    _, infos = fit_walk_forward(panel, folds, w, ["a", "b", "c"], s)
    assert all(i["model_inputs"] == ["a", "b", "c", "a×b"] for i in infos)
    assert FitSpec(model=RidgeSpec()) == FitSpec(model=RidgeSpec(), transforms=())
