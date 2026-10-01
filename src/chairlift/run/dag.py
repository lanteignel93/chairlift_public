"""A small DAG runner over the artifact store and the manifest.

Each stage is a pure function of its named inputs and its spec. The runner walks the graph in dependency order,
computes each stage's key from (name, version, spec hash, input references), reuses the stored output when the
manifest holds that key and the store holds the object, and otherwise runs the stage and records why.

A stage with no inputs reads the outside world, so it must say how to fingerprint what it reads. Without a
fingerprint it is volatile and runs every time: the runner never assumes external data is unchanged.

Two hashes identify work:
- a stage's *key* includes the references of its inputs' actual outputs: it decides reuse, after upstream has run;
- a stage's *plan key* includes its inputs' plan keys instead: a Merkle hash of the pipeline's definition (names,
  versions, specs, data fingerprints), known before anything runs. The run *signature* hashes every needed stage's
  plan key, so the same study, parameters, code versions and data give the same signature on any machine.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import resource
import socket
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import Any, Literal

import polars as _pl

from chairlift.core.identity import code_identity
from chairlift.core.spec import canonical, spec_hash
from chairlift.data.manifest import Manifest
from chairlift.data.refs import StatCache, bind
from chairlift.data.store import ArtifactStore
from chairlift.run.events import EventLog, Listener, bound

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

    def key(self, input_refs: Mapping[str, str], fingerprint: str = "") -> str:
        parts = [self.name, str(self.version), spec_hash(self.spec), fingerprint]
        parts += [f"{k}={input_refs[k]}" for k in sorted(input_refs)]
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def plan_key(self, input_plan_keys: Mapping[str, str], fingerprint: str = "") -> str:
        parts = ["plan", self.name, str(self.version), spec_hash(self.spec), fingerprint]
        parts += [f"{k}={input_plan_keys[k]}" for k in sorted(input_plan_keys)]
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


@dataclass(frozen=True)
class Signature:
    signature: str
    plan_keys: dict[str, str]
    fingerprints: dict[str, str]


@dataclass
class RunReport:
    results: dict[str, StageResult] = field(default_factory=dict)
    run_id: str = ""
    signature: str = ""

    def status(self) -> dict[str, Status]:
        return {name: r.status for name, r in self.results.items()}

    def ran(self) -> list[str]:
        return [name for name, r in self.results.items() if r.status == "ran"]


class Pipeline:
    def __init__(
        self,
        stages: Iterable[Stage],
        root: Path | str,
        store: Path | str | None = None,
        *,
        data: Mapping[str, Path] | None = None,
        cache: Path | str | None = None,
    ) -> None:
        """`root` holds this study's run records. `store` is the content store + manifest; given a shared store,
        every study on the machine reuses every other study's stages. Without one the root is self-contained.

        `data` maps the study's data aliases to directories on this machine (the site config's `[data]`); `cache`
        holds the fingerprint stat cache. Both are site settings: they never enter a key or the signature."""
        self.stages = {s.name: s for s in stages}
        for s in self.stages.values():
            missing = [i for i in s.inputs if i not in self.stages]
            if missing:
                raise KeyError(f"stage {s.name!r} depends on unknown stage(s) {missing}")
        try:
            self.order = list(TopologicalSorter({n: s.inputs for n, s in self.stages.items()}).static_order())
        except CycleError as exc:
            raise ValueError(f"pipeline has a cycle: {exc.args[1]}") from exc
        self.root = Path(root)
        self.data: dict[str, Path] = {k: Path(v) for k, v in (data or {}).items()}
        self.cache = StatCache(Path(cache)) if cache is not None else None
        if store is None:
            self.store = ArtifactStore(self.root / "store")
            self.manifest = Manifest(self.root / "manifest.jsonl")
        else:
            self.store = ArtifactStore(Path(store))
            self.manifest = Manifest(Path(store) / "manifest.jsonl", shared=True)

    def rebase(
        self,
        root: Path | str,
        store: Path | str | None = None,
        *,
        data: Mapping[str, Path] | None = None,
        cache: Path | str | None = None,
    ) -> Pipeline:
        """The same stages over another run root, store and data roots (how the CLI places a study on this machine).
        Data roots and cache not given are kept."""
        return Pipeline(
            self.stages.values(),
            root,
            store,
            data=self.data if data is None else data,
            cache=cache if cache is not None else (self.cache.directory if self.cache else None),
        )

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

    def _fingerprints(self, names: Iterable[str]) -> dict[str, str]:
        """Each source's fingerprint, computed once per call (a fingerprint may hash files)."""
        out: dict[str, str] = {}
        with bind(self.data, self.cache):
            for n in names:
                fp = self.stages[n].fingerprint
                if fp is not None:
                    out[n] = fp()
        return out

    def signature(self, targets: Iterable[str] | None = None) -> Signature:
        """The inputs-only identity of a run of `targets`: same definition and data → same signature anywhere."""
        needed = self._needed(targets)
        fps = self._fingerprints(needed)
        plan_keys: dict[str, str] = {}
        for name in needed:
            stage = self.stages[name]
            plan_keys[name] = stage.plan_key({i: plan_keys[i] for i in stage.inputs}, fps.get(name, ""))
        digest = hashlib.sha256("\x1f".join(f"{n}={plan_keys[n]}" for n in sorted(plan_keys)).encode()).hexdigest()
        return Signature(digest, plan_keys, fps)

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
        needed = self._needed(targets)
        fps = self._fingerprints(needed)
        for name in needed:
            stage = self.stages[name]
            pending = [i for i in stage.inputs if i not in refs]
            if pending:
                items.append(PlanItem(name, "upstream", None, f"waits on re-running input(s): {', '.join(pending)}"))
                continue
            key = stage.key({i: refs[i] for i in stage.inputs}, fps.get(name, ""))
            reuse, why, out = self._decide(stage, key)
            if reuse and out is not None:
                refs[name] = out
            items.append(PlanItem(name, "hit" if reuse else "run", key, why))
        return items

    def run(
        self,
        targets: Iterable[str] | None = None,
        reason: str = "",
        meta: Mapping[str, Any] | None = None,
        listeners: Iterable[Listener] = (),
    ) -> RunReport:
        """Run the needed stages; write a run record and an event stream under <root>/runs/.

        `meta` is caller context recorded verbatim: the study reference, the parameters, the resolved configuration.
        `listeners` receive every event as it is written (the live view); an exception in one never fails the run.
        """
        targets = list(targets) if targets is not None else None
        sig = self.signature(targets)
        started = dt.datetime.now(dt.UTC)
        report = RunReport(run_id=self._new_run_id(started, sig.signature), signature=sig.signature)
        record = self._record(report, sig, started, targets, reason, meta)
        self._write_record(record)
        log = EventLog(self.root / "runs" / f"{report.run_id}.events.jsonl", report.run_id, list(listeners))
        study = (meta or {}).get("name") or (meta or {}).get("study")
        log.emit("run_started", study=study, signature=sig.signature, stages=list(sig.plan_keys), reason=reason)
        refs: dict[str, str] = {}
        t_run = time.perf_counter()
        try:
            self._run_stages(sig, refs, report, reason, log)
        except BaseException as exc:
            err = f"{type(exc).__name__}: {exc}"
            record |= {"status": "failed", "error": err}
            log.emit("run_finished", status="failed", error=err, seconds=round(time.perf_counter() - t_run, 4))
            raise
        else:
            record |= {"status": "ok"}
            log.emit(
                "run_finished",
                status="ok",
                seconds=round(time.perf_counter() - t_run, 4),
                ran=len(report.ran()),
                reused=len(report.results) - len(report.ran()),
            )
        finally:
            record |= {
                "finished_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "results": {
                    n: {"status": r.status, "key": r.key, "output": r.output, "seconds": round(r.seconds, 4)}
                    for n, r in report.results.items()
                },
            }
            self._write_record(record)
        return report

    def _new_run_id(self, started: dt.datetime, signature: str) -> str:
        """`<UTC time to the microsecond>Z-<signature[:12]>`; unique within this root even for back-to-back runs."""
        base = f"{started:%Y%m%dT%H%M%S}.{started.microsecond:06d}Z-{signature[:12]}"
        run_id, n = base, 1
        while (self.root / "runs" / f"{run_id}.json").exists():
            n += 1
            run_id = f"{base}-{n}"
        return run_id

    def _run_stages(self, sig: Signature, refs: dict[str, str], report: RunReport, reason: str, log: EventLog) -> None:
        for name in sig.plan_keys:
            stage = self.stages[name]
            input_refs = {i: refs[i] for i in stage.inputs}
            key = stage.key(input_refs, sig.fingerprints.get(name, ""))
            reuse, why, out = self._decide(stage, key)
            log.emit("stage_started", stage=name, key=key, will="reuse" if reuse else "run", reason=why)
            if reuse and out is not None:
                refs[name] = out
                report.results[name] = StageResult(name, "hit", key, out, why)
                log.emit("stage_finished", stage=name, status="hit", key=key, output=out, seconds=0.0)
                continue
            t0, c0 = time.perf_counter(), time.process_time()
            try:
                kwargs = {i: self.store.get(refs[i]) for i in stage.inputs}
                with bound(log, name), bind(self.data, self.cache):
                    result = stage.fn(**kwargs) if stage.spec is None else stage.fn(stage.spec, **kwargs)
                ref = self.store.put(result)
            except BaseException as exc:
                log.emit(
                    "stage_finished",
                    stage=name,
                    status="failed",
                    key=key,
                    seconds=round(time.perf_counter() - t0, 4),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            seconds = time.perf_counter() - t0
            shape = result.shape if isinstance(result, _Frame) else (None, None)
            log.emit(
                "stage_finished",
                stage=name,
                status="ran",
                key=key,
                output=ref,
                seconds=round(seconds, 4),
                cpu_seconds=round(time.process_time() - c0, 4),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                rows=shape[0],
                cols=shape[1],
            )
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

    def _record(
        self,
        report: RunReport,
        sig: Signature,
        started: dt.datetime,
        targets: list[str] | None,
        reason: str,
        meta: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        stages: dict[str, Any] = {}
        for name, plan_key in sig.plan_keys.items():
            st = self.stages[name]
            stages[name] = {
                "version": st.version,
                "spec": canonical(st.spec),
                "spec_hash": spec_hash(st.spec),
                "fingerprint": sig.fingerprints.get(name),
                "inputs": list(st.inputs),
                "plan_key": plan_key,
            }
        return {
            "kind": "run",
            "schema": 1,
            "run_id": report.run_id,
            "signature": sig.signature,
            "started_at": started.isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "targets": targets,
            "reason": reason,
            "meta": dict(meta or {}),
            "data": {k: str(v) for k, v in sorted(self.data.items())},  # where the data was here; not hashed
            "stages": stages,
            "code": code_identity(),
            "environment": _environment(),
            "status": "running",
        }

    def _write_record(self, record: dict[str, Any]) -> None:
        path = self.root / "runs" / f"{record['run_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=1, sort_keys=True, default=str))
        os.replace(tmp, path)

    def load(self, name: str, report: RunReport) -> Any:
        return self.store.get(report.results[name].output)


def _environment() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str] = {}
    for name in ("chairlift", "polars", "numpy", "click"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "absent"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


_Frame = _pl.DataFrame
