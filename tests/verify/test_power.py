"""verify/power.py: detection rates on twins built like real panels."""

import dataclasses

from chairlift.verify.power import detection_rates, power_curve
from chairlift.verify.twins import CrossSectionTwin


def test_a_strong_signal_is_found_and_a_noise_feature_is_not():
    spec = CrossSectionTwin(names=300, dates=48, signals=(0.08,), noise_features=1)
    r = detection_rates(spec, range(8))
    assert r["power_ic_test"] == 1.0 and r["false_positive_rate"] <= 0.125


def test_factor_structure_costs_power():
    base = CrossSectionTwin(names=300, dates=48, noise_features=1, sector_tilt=0.0)
    clean = power_curve([0.03], base, range(12))[0]
    noisy = power_curve([0.03], dataclasses.replace(base, style_vol=0.012), range(12))[0]
    assert noisy["sd_ic"] > 2 * clean["sd_ic"]
    assert noisy["power_ic_test"] < clean["power_ic_test"]
