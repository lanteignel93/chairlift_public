"""evaluate/attribution.py: alpha after known factors, with HAC errors."""

from __future__ import annotations

import numpy as np
import pytest

from chairlift.evaluate.attribution import factor_attribution


def _dates(n: int) -> list[str]:
    return [f"{2000 + i // 12}-{i % 12 + 1:02d}-28" for i in range(n)]


def test_a_pure_factor_copy_has_no_alpha_and_full_beta():
    rng = np.random.default_rng(0)
    d = _dates(240)
    f = rng.normal(0.01, 0.04, 240)
    out = factor_attribution(dict(zip(d, 0.5 * f, strict=True)), {"mom": dict(zip(d, f, strict=True))})
    assert (
        abs(out["alpha_per_period"]) < 1e-12
        and out["betas"]["mom"]["beta"] == pytest.approx(0.5)
        and out["r2"] == pytest.approx(1.0)
    )


def test_alpha_beyond_the_factor_is_found_and_significant():
    rng = np.random.default_rng(1)
    d = _dates(240)
    f = rng.normal(0.0, 0.04, 240)
    y = 0.01 + 0.3 * f + rng.normal(0, 0.02, 240)
    out = factor_attribution(dict(zip(d, y, strict=True)), {"f": dict(zip(d, f, strict=True))}, periods=12, lags=3)
    assert out["alpha_annual"] == pytest.approx(0.12, abs=0.04) and out["alpha_t"] > 4
    assert out["betas"]["f"]["beta"] == pytest.approx(0.3, abs=0.1)


def test_too_few_observations():
    assert factor_attribution({"a": 1.0}, {"f": {"a": 1.0}})["reason"]
