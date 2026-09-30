"""A small DAG runner over the artifact store and the manifest.

Each stage is a pure function of its named inputs and its spec. The runner walks the graph in dependency order,
computes each stage's key from (name, version, spec hash, input references), reuses the stored output when the
manifest holds that key and the store holds the object, and otherwise runs the stage and records why.

A stage with no inputs reads the outside world, so it must say how to fingerprint what it reads. Without a
fingerprint it is volatile and runs every time: the runner never assumes external data is unchanged.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Literal

from chairlift.core.identity import code_identity
from chairlift.core.spec import spec_hash
from chairlift.data.manifest import Manifest
from chairlift.data.store import ArtifactStore

Status = Literal["hit", "ran"]
PlanStatus = Literal["hit", "run", "upstream"]  # upstream: an input will re-run; its bytes may or may not change


@dataclass(frozen=True)
class Stage:
    name: str
    fn: Callable[..., Any]
    inputs: tuple[str, ...] = ()
    spec: Any = None
    version: int = 1
    fingerprint: Callable[[], str] | None = None  # required for a source stage to be cacheable

    def key(self, input_refs: Mapping[str, str]) -> str:
        fp = self.fingerprint() if self.fingerprint is not None else ""
        parts = [self.name, str(self.version), spec_hash(self.spec), fp]
        parts += [f"{k}={input_refs[k]}" for k in sorted(input_refs)]
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


@dataclass
class StageResult:
    stage: str
    status: Status
    key: str
    output: str
    reason: str
    seconds: float = 0.0


@dataclass
class PlanItem:
    stage: str
    status: PlanStatus
    key: str | None
    reason: str


@dataclass
class RunReport:
    results: dict[str, StageResult] = field(default_factory=dict)

    def status(self) -> dict[str, Status]:
        return {name: r.status for name, r in self.results.items()}

    def ran(self) -> list[str]:
        return [name for name, r in self.results.items() if r.status == "ran"]


class Pipeline:
    def __init__(self, stages: Iterable[Stage], root: Path | str) -> None:
        self.stages = {s.name: s for s in stages}
        for s in self.stages.values():
            missing = [i for i in s.inputs if i not in self.stages]
            if missing:
                raise KeyError(f"stage {s.name!r} depends on unknown stage(s) {missing}")
        try:
            self.order = list(TopologicalSorter({n: s.inputs for n, s in self.stages.items()}).static_order())
        except CycleError as exc:
            raise ValueError(f"pipeline has a cycle: {exc.args[1]}") from exc
        root = Path(root)
        self.store = ArtifactStore(root / "store")
        self.manifest = Manifest(root / "manifest.jsonl")

    def _needed(self, targets: Iterable[str] | None) -> list[str]:
        if targets is None:
            return self.order
        need: set[str] = set()
        stack = list(targets)
        while stack:
            n = stack.pop()
            if n not in self.stages:
                raise KeyError(f"unknown stage {n!r}")
            if n not in need:
                need.add(n)
                stack.extend(self.stages[n].inputs)
        return [n for n in self.order if n in need]

    def _decide(self, stage: Stage, key: str) -> tuple[bool, str, str | None]:
        """(reuse?, why, stored output) — the one rule both run() and plan() apply."""
        rec = self.manifest.lookup(key)
        if not stage.inputs and stage.fingerprint is None:
            return False, "source stage without a fingerprint: always runs", None
        if rec is None:
            return False, "no record for this key (new, or an input / spec / version changed)", None
        if not self.store.has(rec.output):
            return False, "record exists but its output is missing from the store", None
        return True, "inputs, spec and version unchanged", rec.output

    def plan(self, targets: Iterable[str] | None = None) -> list[PlanItem]:
        """What run() would do, without running anything.

        A stage downstream of one that will re-run cannot be decided in advance: if the re-run produces identical
        bytes (content addressing), the downstream key is unchanged and it will be a hit. It is reported as `upstream`.
        """
        items: list[PlanItem] = []
        refs: dict[str, str] = {}
        for name in self._needed(targets):
            stage = self.stages[name]
            pending = [i for i in stage.inputs if i not in refs]
            if pending:
                items.append(PlanItem(name, "upstream", None, f"waits on re-running input(s): {', '.join(pending)}"))
                continue
            key = stage.key({i: refs[i] for i in stage.inputs})
            reuse, why, out = self._decide(stage, key)
            if reuse and out is not None:
                refs[name] = out
            items.append(PlanItem(name, "hit" if reuse else "run", key, why))
        return items

    def run(self, targets: Iterable[str] | None = None, reason: str = "") -> RunReport:
        report = RunReport()
        refs: dict[str, str] = {}
        for name in self._needed(targets):
            stage = self.stages[name]
            input_refs = {i: refs[i] for i in stage.inputs}
            key = stage.key(input_refs)
            reuse, why, out = self._decide(stage, key)
            if reuse and out is not None:
                refs[name] = out
                report.results[name] = StageResult(name, "hit", key, out, why)
                continue
            t0 = time.perf_counter()
            kwargs = {i: self.store.get(refs[i]) for i in stage.inputs}
            result = stage.fn(**kwargs) if stage.spec is None else stage.fn(stage.spec, **kwargs)
            ref = self.store.put(result)
            seconds = time.perf_counter() - t0
            self.manifest.append(
                stage=name,
                key=key,
                version=stage.version,
                spec_hash=spec_hash(stage.spec),
                inputs=input_refs,
                output=ref,
                code=code_identity(),
                reason=reason or why,
            )
            refs[name] = ref
            report.results[name] = StageResult(name, "ran", key, ref, why, seconds)
        return report

    def load(self, name: str, report: RunReport) -> Any:
        return self.store.get(report.results[name].output)
