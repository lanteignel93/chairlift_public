"""The study's ledger: every experiment tried, and every look at the holdout, append-only.

`<study dir>/ledger.jsonl`, one JSON object per line:

    trial          a run with a new signature: its run id, parameters, sweep cell, reason, and the headline numbers
                   (Sharpe and observation count) read from the stage output the study names in HEADLINE
    holdout_open   the one look: who opened it, when, and why

A trial is a signature, not a run: re-running an experiment is not a new trial; changing anything that moves the
signature is. The count of trials is what the deflated Sharpe ratio charges the best one for.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chairlift.data.locking import append_line


@dataclass(frozen=True)
class Headline:
    """Where a run's headline numbers live: a stage whose output is a JSON dict, and dotted paths into it."""

    stage: str
    sharpe: str  # annualised Sharpe, e.g. "ls.sharpe"
    n_obs: str  # observations behind it, e.g. "ls.n_days"
    periods: int = 252  # observations per year (de-annualises the Sharpe)
    daily: str | None = None  # dotted path to the daily series (a list of {date, ...}); a search needs it
    value: str = "pnl"  # the field of each daily row that is the P&L

    @staticmethod
    def parse(raw: Any) -> Headline | None:
        if raw is None:
            return None
        if isinstance(raw, Headline):
            return raw
        return Headline(**raw)


def dig(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = obj[part]
    return obj


class Ledger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line from a killed writer
        return out

    def _append(self, entry: dict[str, Any]) -> dict[str, Any]:
        entry = {"at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **entry}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        append_line(self.path, json.dumps(entry, sort_keys=True, default=str))
        return entry

    def trials(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for e in self.entries():
            if e.get("kind") == "trial" and e["signature"] not in seen:
                seen[e["signature"]] = e
        return list(seen.values())

    def holdout_opened(self) -> dict[str, Any] | None:
        return next((e for e in self.entries() if e.get("kind") == "holdout_open"), None)

    def record_trial(
        self,
        *,
        signature: str,
        run_id: str,
        params: dict[str, Any],
        reason: str,
        headline: dict[str, Any] | None,
        sweep: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Append a trial unless this signature is already one; returns the entry written, or None."""
        if any(t["signature"] == signature for t in self.trials()):
            return None
        opened = self.holdout_opened()
        return self._append(
            {
                "kind": "trial",
                "signature": signature,
                "run_id": run_id,
                "params": params,
                "reason": reason,
                "headline": headline,
                "sweep": sweep,
                "after_holdout_open": opened is not None,
            }
        )

    def open_holdout(self, *, reason: str, by: str) -> dict[str, Any]:
        """The one look. A second opening is refused: looking again is a new registration, not a second look."""
        if not reason.strip():
            raise ValueError("opening the holdout needs a reason")
        prior = self.holdout_opened()
        if prior is not None:
            raise PermissionError(
                f"the holdout was already opened at {prior['at']} by {prior['by']}: {prior['reason']!r}"
            )
        return self._append({"kind": "holdout_open", "reason": reason, "by": by, "trials_before": len(self.trials())})
