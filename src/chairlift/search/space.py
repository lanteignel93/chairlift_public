"""A search space over a study's experiment parameters, and the candidates drawn from it.

    [search]                       (in an experiment file)
    sampler = "random"             # "grid": every combination; "random": `budget` seeded draws
    budget  = 20
    seed    = 1
    [search.params]
    alpha  = { low = 1e-4, high = 1.0, log = true }
    window = [3, 5, 10]            # a list: choose among these
    q      = { low = 0.05, high = 0.2 }
    depth  = { low = 2, high = 6, int = true }

Every candidate is an ordinary run with its own signature, charged to the study's ledger; a draw repeated is the
same trial, not a new one.
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast


class SearchError(ValueError):
    pass


@dataclass(frozen=True)
class Range:
    low: float
    high: float
    log: bool = False
    integer: bool = False

    def draw(self, rng: random.Random) -> float | int:
        if self.log:
            if self.low <= 0:
                raise SearchError("a log range needs low > 0")
            v = math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        else:
            v = rng.uniform(self.low, self.high)
        return round(v) if self.integer else float(f"{v:.6g}")  # 6 significant digits: readable, stable keys


@dataclass(frozen=True)
class SearchSpace:
    params: dict[str, tuple[Any, ...] | Range] = field(default_factory=dict[str, "tuple[Any, ...] | Range"])
    sampler: str = "random"
    budget: int = 20
    seed: int = 0

    @staticmethod
    def parse(raw: Mapping[str, Any]) -> SearchSpace:
        unknown = set(raw) - {"params", "sampler", "budget", "seed", "objective"}
        if unknown:
            raise SearchError(f"[search]: unknown key(s) {sorted(unknown)}")
        params: dict[str, tuple[Any, ...] | Range] = {}
        raw_params: dict[str, Any] = dict(raw.get("params", {}))
        for k, v in raw_params.items():
            if isinstance(v, list):
                values = cast("list[Any]", v)
                if not values:
                    raise SearchError(f"[search.params] {k}: an empty list")
                params[k] = tuple(values)
            elif isinstance(v, dict):
                v = cast("dict[str, Any]", v)
                extra = set(v) - {"low", "high", "log", "int"}
                if extra or "low" not in v or "high" not in v:
                    raise SearchError(f"[search.params] {k}: a range is {{low, high, log?, int?}}")
                params[k] = Range(
                    float(v["low"]), float(v["high"]), bool(v.get("log", False)), bool(v.get("int", False))
                )
            else:
                raise SearchError(f"[search.params] {k}: a list of values or a {{low, high}} range")
        sampler = str(raw.get("sampler", "random"))
        if sampler not in ("random", "grid"):
            raise SearchError("[search] sampler is 'random' or 'grid'")
        if sampler == "grid" and any(isinstance(v, Range) for v in params.values()):
            raise SearchError("[search] a grid needs lists, not ranges")
        return SearchSpace(params, sampler, int(raw.get("budget", 20)), int(raw.get("seed", 0)))

    def candidates(self) -> list[dict[str, Any]]:
        """The parameter sets to run, in order, without duplicates."""
        names = list(self.params)
        if self.sampler == "grid":
            lists = [cast("tuple[Any, ...]", self.params[n]) for n in names]  # parse() refuses ranges in a grid
            out = [dict(zip(names, c, strict=True)) for c in itertools.product(*lists)]
            return out[: self.budget] if self.budget else out
        rng = random.Random(self.seed)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        tries = 0
        while len(out) < self.budget and tries < self.budget * 50:
            tries += 1
            c = {n: (v.draw(rng) if isinstance(v, Range) else rng.choice(v)) for n, v in self.params.items()}
            key = repr(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out
