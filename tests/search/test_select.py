"""search/select.py: the search judged without hindsight, and its overfitting probability."""

from __future__ import annotations

import datetime as dt

import numpy as np

from chairlift.search.select import pbo, walk_forward_selection


def _series(n_cand: int, edge: dict[int, float], days: int = 2000, seed: int = 0) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    dates = [(dt.date(2015, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(days)]
    return {
        f"c{j}": dict(zip(dates, (edge.get(j, 0.0) + rng.standard_normal(days)).tolist(), strict=True))
        for j in range(n_cand)
    }


def test_a_real_edge_is_selected_and_survives():
    s = _series(10, {3: 0.15})
    wf = walk_forward_selection(s)
    assert sum(p["picked"] == "c3" for p in wf["picks"]) >= len(wf["picks"]) - 1
    assert wf["selected_oos_sharpe"] > 1.0
    assert pbo(s)["pbo"] < 0.1  # the in-sample winner keeps winning


def test_pure_noise_is_overfit_and_selection_earns_nothing():
    s = _series(20, {})
    wf = walk_forward_selection(s)
    assert abs(wf["selected_oos_sharpe"]) < 1.0 < wf["best_in_hindsight_sharpe"] + 1.0
    assert pbo(s)["pbo"] > 0.3  # the in-sample winner is a coin flip out of sample


def test_a_complexity_cost_tips_selection_toward_the_simpler_candidate():
    s = _series(2, {0: 0.05, 1: 0.15}, days=3000)
    assert all(p["picked"] == "c1" for p in walk_forward_selection(s)["picks"][2:])  # c1 is better: chosen unpriced
    priced = walk_forward_selection(s, costs={"c1": 10.0})  # a prior larger than any Sharpe gap: the simpler one wins
    assert all(p["picked"] == "c0" for p in priced["picks"])
