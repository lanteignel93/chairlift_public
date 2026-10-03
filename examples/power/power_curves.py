"""The pipeline's power function on twins built like the real panels: which nulls are informative, and which designs
could have found a large-cap anomaly. Writes power_curves.json beside this file.

    uv run python examples/power/power_curves.py
"""

from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from chairlift.verify.power import detection_rates
from chairlift.verify.twins import CrossSectionTwin

HERE = Path(__file__).parent
ICS = (0.01, 0.02, 0.03, 0.05)
SEEDS = range(60)
# style_vol 0.012 gives a feature's date-to-date IC sd ≈ 0.15 at 480 names, as measured on the S&P 500 panel;
# Student-t(5) idiosyncratic returns
REAL = {"style_vol": 0.012, "idio_df": 5.0, "noise_features": 1, "sector_tilt": 0.0}
DESIGNS = {
    "S&P 500, 5 years monthly (the equity test window)": {"names": 480, "dates": 60},
    "S&P 500, 8 years monthly (the whole equity dev window)": {"names": 480, "dates": 96},
    "2,500 liquid names, 5 years monthly": {"names": 2500, "dates": 60},
    "2,500 liquid names, 5 years weekly (same IC per week)": {"names": 2500, "dates": 260},
}


def run(job: tuple[str, float]) -> dict[str, object]:
    name, ic = job
    spec = CrossSectionTwin(signals=(ic,), **DESIGNS[name], **REAL)
    return {"design": name} | detection_rates(spec, SEEDS)


if __name__ == "__main__":
    jobs = [(d, ic) for d in DESIGNS for ic in ICS]
    with ProcessPoolExecutor(max_workers=16, mp_context=multiprocessing.get_context("spawn")) as pool:
        rows = list(pool.map(run, jobs))
    (HERE / "power_curves.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(
            f"{r['design'][:44]:44s} IC {r['ic']:.2f}  sd(IC) {r['sd_ic']:.3f}  power: IC test {r['power_ic_test']:.2f}"
            f"  decile test {r['power_decile_test']:.2f}  false+ {r['false_positive_rate']:.2f}"
        )
