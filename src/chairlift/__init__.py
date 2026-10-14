"""chairlift: a machine-learning research pipeline in which the research protocol is code.

The names a study is written with are importable from the top level (`from chairlift import Study, Model, ...`);
they load on first use, so `import chairlift` and the CLI stay fast. `docs/guide/10-api.md` lists them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

_EXPORTS: dict[str, str] = {
    # declaring a study
    "Study": "chairlift.study.build",
    "Source": "chairlift.study.build",
    "Model": "chairlift.study.build",
    "FeatureSet": "chairlift.study.build",
    "Ensemble": "chairlift.study.build",
    "CrossSectionBook": "chairlift.study.build",
    "TimeSeriesBook": "chairlift.study.build",
    "PathBook": "chairlift.study.build",
    "Stage": "chairlift.run.dag",
    "Outputs": "chairlift.run.dag",
    "Pipeline": "chairlift.run.dag",
    "spec": "chairlift.core.spec",
    "added": "chairlift.core.spec",
    "spec_hash": "chairlift.core.spec",
    # data
    "DataRef": "chairlift.data.refs",
    "fingerprint_of": "chairlift.data.refs",
    "content_hash": "chairlift.data.refs",
    # folds, fits, transforms, models
    "WalkForward": "chairlift.schedule.walkforward",
    "make_folds": "chairlift.schedule.walkforward",
    "FitSpec": "chairlift.learn.fit",
    "fit_walk_forward": "chairlift.learn.fit",
    "zscore_ensemble": "chairlift.learn.fit",
    "RidgeSpec": "chairlift.learn.models",
    "LightGBMSpec": "chairlift.learn.models",
    "RuleSpec": "chairlift.learn.models",
    "CompositeSpec": "chairlift.learn.models",
    "ClusterSelection": "chairlift.learn.selection",
    "Winsorize": "chairlift.learn.transforms",
    "Interactions": "chairlift.learn.transforms",
    "PCA": "chairlift.learn.transforms",
    # books and evaluation
    "QuantileBook": "chairlift.book.quantile",
    "SignalBook": "chairlift.book.timeseries",
    "PathSizing": "chairlift.book.path",
    "DailyStatsSpec": "chairlift.evaluate.daily",
    "daily_stats": "chairlift.evaluate.daily",
    "paired_sharpe_diff": "chairlift.evaluate.daily",
    "deflated_sharpe": "chairlift.evaluate.deflated",
    "factor_attribution": "chairlift.evaluate.attribution",
    "fama_macbeth_ic": "chairlift.evaluate.power",
    "power_report": "chairlift.evaluate.power",
    "mde_ic": "chairlift.evaluate.power",
    "mde_sharpe": "chairlift.evaluate.power",
    "PowerSpec": "chairlift.study.build",
    "residualize": "chairlift.features.cross_section",
    # the ledger and the holdout
    "Holdout": "chairlift.ledger.holdout",
    "guard": "chairlift.ledger.holdout",
    "with_holdout": "chairlift.ledger.holdout",
    "Ledger": "chairlift.ledger.trials",
    "Headline": "chairlift.ledger.trials",
    # search
    "SearchSpace": "chairlift.search.space",
    "pbo": "chairlift.search.select",
    "walk_forward_selection": "chairlift.search.select",
}

__all__ = [  # literal, so type checkers see it; tests/docs checks it equals _EXPORTS
    "PCA",
    "ClusterSelection",
    "CompositeSpec",
    "CrossSectionBook",
    "DailyStatsSpec",
    "DataRef",
    "Ensemble",
    "FeatureSet",
    "FitSpec",
    "Headline",
    "Holdout",
    "Interactions",
    "Ledger",
    "LightGBMSpec",
    "Model",
    "Outputs",
    "PathBook",
    "PathSizing",
    "Pipeline",
    "PowerSpec",
    "QuantileBook",
    "RidgeSpec",
    "RuleSpec",
    "SearchSpace",
    "SignalBook",
    "Source",
    "Stage",
    "Study",
    "TimeSeriesBook",
    "WalkForward",
    "Winsorize",
    "__version__",
    "added",
    "content_hash",
    "daily_stats",
    "deflated_sharpe",
    "factor_attribution",
    "fama_macbeth_ic",
    "fingerprint_of",
    "fit_walk_forward",
    "guard",
    "make_folds",
    "mde_ic",
    "mde_sharpe",
    "paired_sharpe_diff",
    "pbo",
    "power_report",
    "residualize",
    "spec",
    "spec_hash",
    "walk_forward_selection",
    "with_holdout",
    "zscore_ensemble",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'chairlift' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


if TYPE_CHECKING:
    from chairlift.book.path import PathSizing
    from chairlift.book.quantile import QuantileBook
    from chairlift.book.timeseries import SignalBook
    from chairlift.core.spec import added, spec, spec_hash
    from chairlift.data.refs import DataRef, content_hash, fingerprint_of
    from chairlift.evaluate.attribution import factor_attribution
    from chairlift.evaluate.daily import DailyStatsSpec, daily_stats, paired_sharpe_diff
    from chairlift.evaluate.deflated import deflated_sharpe
    from chairlift.evaluate.power import fama_macbeth_ic, mde_ic, mde_sharpe, power_report
    from chairlift.features.cross_section import residualize
    from chairlift.learn.fit import FitSpec, fit_walk_forward, zscore_ensemble
    from chairlift.learn.models import CompositeSpec, LightGBMSpec, RidgeSpec, RuleSpec
    from chairlift.learn.selection import ClusterSelection
    from chairlift.learn.transforms import PCA, Interactions, Winsorize
    from chairlift.ledger.holdout import Holdout, guard, with_holdout
    from chairlift.ledger.trials import Headline, Ledger
    from chairlift.run.dag import Outputs, Pipeline, Stage
    from chairlift.schedule.walkforward import WalkForward, make_folds
    from chairlift.search.select import pbo, walk_forward_selection
    from chairlift.search.space import SearchSpace
    from chairlift.study.build import (
        CrossSectionBook,
        Ensemble,
        FeatureSet,
        Model,
        PathBook,
        PowerSpec,
        Source,
        Study,
        TimeSeriesBook,
    )
