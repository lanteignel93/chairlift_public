"""Code identity recorded on every manifest entry: the chairlift version plus the git commit it was built from.

The commit is provenance, not part of a stage's cache key. A stage's key carries its declared version instead: an
author bumps the version when the stage computes something different, and the golden reproduction test catches the
changes nobody declared. Keying on the commit would invalidate every cache on every commit, including doc edits.
"""

from __future__ import annotations

import subprocess
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@cache
def code_identity() -> str:
    try:
        pkg = version("chairlift")
    except PackageNotFoundError:
        pkg = "unknown"
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return f"chairlift {pkg} (no git)"
    return f"chairlift {pkg} @ {commit}{'+dirty' if dirty else ''}"
