"""Every debug walkthrough runs in its asserted mode as part of the suite, so a walkthrough can never go stale."""

import runpy
from pathlib import Path

import pytest

WALKTHROUGHS = sorted((Path(__file__).parents[1] / "debug_walkthroughs").glob("wt_*.py"))


@pytest.mark.parametrize("path", WALKTHROUGHS, ids=lambda p: p.stem)
def test_walkthrough_asserted_run(path, capsys):
    runpy.run_path(str(path), run_name="__main__")
    assert "walkthrough passed" in capsys.readouterr().out
