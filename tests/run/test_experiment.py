"""run/experiment.py: experiment files and parameter validation against the factory's signature."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chairlift.run.experiment import ExperimentError, load_experiment, validate_params
from chairlift.verify import toy


def write(tmp_path: Path, text: str, name: str = "exp.toml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


# ---- loading -----------------------------------------------------------------------------------------------------


def test_a_full_file_loads_with_defaults_from_the_file_name(tmp_path: Path):
    exp = load_experiment(
        write(
            tmp_path,
            'study = "chairlift.verify.toy:pipeline"\nreason = "why"\n'
            '[params]\nic = 0.2\n[run]\ntargets = ["evaluate"]\n[sweep]\nwindow = [1, 3]\nseed = [7, 8]\n',
            "window-sweep.toml",
        )
    )
    assert exp.name == "window-sweep" and exp.reason == "why" and exp.study_name is None
    assert exp.params == {"ic": 0.2} and exp.targets == ("evaluate",)
    assert exp.sweep == {"window": (1, 3), "seed": (7, 8)}


def test_cells_are_the_cross_product_with_the_last_key_fastest(tmp_path: Path):
    exp = load_experiment(
        write(tmp_path, 'study = "m:f"\n[params]\nic = 0.2\n[sweep]\nwindow = [1, 3]\nseed = [7, 8]\n')
    )
    assert exp.cells() == [
        {"ic": 0.2, "window": 1, "seed": 7},
        {"ic": 0.2, "window": 1, "seed": 8},
        {"ic": 0.2, "window": 3, "seed": 7},
        {"ic": 0.2, "window": 3, "seed": 8},
    ]


def test_no_sweep_is_one_cell(tmp_path: Path):
    assert load_experiment(write(tmp_path, 'study = "m:f"\n[params]\nx = 1\n')).cells() == [{"x": 1}]


def test_a_relative_file_study_resolves_against_the_experiment_file(tmp_path: Path):
    (tmp_path / "exps").mkdir()
    exp = load_experiment(write(tmp_path / "exps", 'study = "../study.py:pipeline"\n'))
    assert exp.study == f"{(tmp_path / 'study.py').resolve()}:pipeline"


def test_the_record_keeps_the_text_and_its_hash(tmp_path: Path):
    text = 'study = "m:f"\n'
    exp = load_experiment(write(tmp_path, text))
    rec = exp.record()
    assert rec["text"] == text and len(rec["sha256"]) == 64 and rec["name"] == "exp"


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ('study = "m:f"\nwindow = 3\n', "unknown key"),
        ("[params]\nx = 1\n", "`study` must be"),
        ('study = "nocolon"\n', "`study` must be"),
        ('study = "m:f"\nparams = 3\n', r"\[params\] must be a table"),
        ('study = "m:f"\n[run]\ntarget = ["a"]\n', r"unknown key\(s\) in \[run\]"),
        ('study = "m:f"\n[run]\ntargets = "a"\n', "targets must be a list"),
        ('study = "m:f"\n[sweep]\nw = 3\n', "non-empty list"),
        ('study = "m:f"\n[sweep]\nw = []\n', "non-empty list"),
        ('study = "m:f"\n[params]\nw = 1\n[sweep]\nw = [2]\n', "both in"),
        ('study = "m:f"\nname = 3\n', "`name` must be a string"),
        ('study = "m:f\n', "exp.toml"),
    ],
)
def test_malformed_files_are_errors_that_name_the_key(tmp_path: Path, text: str, match: str):
    with pytest.raises(ExperimentError, match=match):
        load_experiment(write(tmp_path, text))


# ---- parameter validation ----------------------------------------------------------------------------------------


def test_params_are_converted_to_the_factory_annotations():
    out = validate_params(toy.pipeline, {"window": 3, "ic": 1})
    assert out == {"window": 3, "ic": 1.0} and isinstance(out["ic"], float)  # an int for a float becomes a float


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"lookback": 3}, r"unknown parameter\(s\) lookback; the factory takes window, ic, seed, pace"),
        ({"window": "3"}, r"window='3'.*Expected `int`, got `str`"),
        ({"window": True}, "Expected `int`, got `bool`"),
        ({"window": 1.5}, "Expected `int`, got `float`"),
    ],
)
def test_bad_params_are_named_before_anything_runs(params: dict[str, Any], match: str):
    with pytest.raises(ExperimentError, match=match):
        validate_params(toy.pipeline, params)


def test_missing_required_keyword_and_var_keyword_factories():
    def needs(root: Path, *, alpha: float) -> None: ...

    def anything(root: Path, **kw: Any) -> None: ...

    with pytest.raises(ExperimentError, match="missing required parameter"):
        validate_params(needs, {})
    assert validate_params(anything, {"z": [1, 2]}) == {"z": [1, 2]}


def test_containers_convert_to_their_annotated_shape():
    def f(root: Path, *, cuts: tuple[float, ...] = (), names: list[str] | None = None) -> None: ...

    assert validate_params(f, {"cuts": [1, 2.5], "names": ["a"]}) == {"cuts": (1.0, 2.5), "names": ["a"]}


@pytest.mark.parametrize("path", sorted((Path(__file__).parents[2] / "examples" / "experiments").glob("*.toml")))
def test_shipped_example_experiments_load_and_validate(path: Path):
    exp = load_experiment(path)
    assert exp.study == "chairlift.verify.toy:pipeline"
    for cell in exp.cells():
        validate_params(toy.pipeline, cell)
