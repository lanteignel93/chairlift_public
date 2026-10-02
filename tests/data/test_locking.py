"""data/locking.py: appends from many processes stay whole lines."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROG = """
import json, sys
from pathlib import Path
from chairlift.data.locking import append_line
for i in range(200):
    append_line(Path(sys.argv[1]), json.dumps({"w": sys.argv[2], "i": i, "pad": "x" * 3000}))
"""


def test_parallel_appends_never_interleave(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    procs = [subprocess.Popen([sys.executable, "-c", PROG, str(path), str(k)]) for k in range(4)]
    assert all(p.wait(timeout=120) == 0 for p in procs)
    lines = path.read_text().splitlines()
    assert len(lines) == 800 and all(json.loads(line)["pad"] == "x" * 3000 for line in lines)
