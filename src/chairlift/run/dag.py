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

import contextlib
import datetime as dt
import functools
import hashlib
import inspect
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
from typing import Any, Literal, cast

import polars as _pl

from chairlift.core.identity import code_identity
from chairlift.core.spec import SpecError, canonical, canonical_json, is_spec, spec_hash
from chairlift.data.manifest import Manifest
from chairlift.data.refs import StatCache, bind
from chairlift.data.store import ArtifactStore
from chairlift.ledger.holdout import bind_ledger
from chairlift.ledger.trials import Headline, Ledger, dig
from chairlift.run.compute import bind_compute
from chairlift.run.events import EventLog, Listener, bound

Status = Literal["hit", "ran"]
PlanStatus = Literal["hit", "run", "upstream"]  # upstream: an input will re-run; its bytes may or may not change


def captured(fn: Callable[..., Any], ignore: Iterable[str] = ()) -> dict[str, Any]:
    """The values a stage function carries from outside its arguments: closure cells and functools.partial arguments.

    They change what the stage computes as surely as its spec does, so they are part of its identity. Functions,
    classes and modules are code (covered by `version`, like the stage function itself) and are skipped.
    """
    skip = set(ignore)
    out: dict[str, Any] = {}
    seen: set[int] = set()

    def walk(f: Any, prefix: str) -> None:
        if id(f) in seen:
            return
        seen.add(id(f))
        if isinstance(f, functools.partial):
            part = cast("functools.partial[Any]", f)
            for i, a in enumerate(part.args):
                put(f"{prefix}arg{i}", a)
            for k, v in part.keywords.items():
                put(prefix + k, v)
            walk(part.func, prefix)
            return
        code = getattr(f, "__code__", None)
        cells = getattr(f, "__closure__", None)
        if code is None or not cells:
            return
        for name, cell in zip(code.co_freevars, cells, strict=True):
            try:
                put(prefix + name, cell.cell_contents)
            except ValueError:  # an empty cell: a name assigned after the function was defined
                continue

    def put(name: str, value: Any) -> None:
        if name in skip or name.split(".")[-1] in skip:
            return
        if (callable(value) and not is_spec(value)) or inspect.ismodule(value) or isinstance(value, type):
            walk(value, name + ".")
            return
        out[name] = value

    walk(fn, "")
    return dict(sorted(out.items()))


def _defined_in_own_module(f: Any, code: Any) -> bool:
    """True when the function's code comes from the file of the module whose namespace it lives in (its globals:
    works for imported modules, file-path studies and runpy alike)."""
    g: dict[str, Any] = getattr(f, "__globals__", None) or {}
    if g.get("__name__") != getattr(f, "__module__", None):
        return False  # a wrapper renamed into another module (functools.wraps, dataclass __repr__)
    path = g.get("__file__")
    return isinstance(path, str) and os.path.realpath(path) == os.path.realpath(code.co_filename)


def _library_roots() -> tuple[str, ...]:
    import sysconfig

    roots = {sysconfig.get_paths()[k] for k in ("stdlib", "platstdlib", "purelib", "platlib")}
    return tuple(os.path.realpath(r) + os.sep for r in roots if r)


_LIBS = _library_roots()


def _user_file(path: str | None) -> bool:
    """Code a study author or chairlift owns: on disk, and not in the standard library or installed site-packages."""
    if not path or path.startswith("<"):
        return False
    real = os.path.realpath(path)
    return not real.startswith(_LIBS)


def code_digest(fn: Callable[..., Any]) -> str:
    """sha256 over the source of a stage function and of every function or class it reaches by name, transitively,
    within the stage's own module and within chairlift. Editing study code or chairlift's research code re-runs
    exactly the stages that reach it; third-party libraries are covered by the environment hash and `version`."""
    seen: dict[str, set[str]] = {}
    done: set[int] = set()

    def code_of(f: Any) -> Any:
        while isinstance(f, functools.partial):
            f = cast("functools.partial[Any]", f).func
        return getattr(f, "__code__", None), f

    def names(code: Any) -> set[str]:
        out = set(code.co_names)
        for c in code.co_consts:
            if inspect.iscode(c):
                out |= names(c)
        return out

    def ours(obj: Any, module: str | None) -> bool:
        if isinstance(obj, type):
            try:
                return _user_file(inspect.getsourcefile(obj))
            except TypeError:
                return False
        code = getattr(obj, "__code__", None)
        return code is not None and _user_file(code.co_filename)

    def visit_class(c: type, module: str | None) -> None:
        if id(c) in done:
            return
        done.add(id(c))
        try:
            label = os.path.basename(inspect.getsourcefile(c) or c.__module__)
        except TypeError:
            label = c.__module__
        key = f"{label}:{c.__qualname__}"
        try:
            seen.setdefault(key, set()).add(inspect.getsource(c))
        except (OSError, TypeError):
            seen.setdefault(key, set()).add(key)
        for v in vars(c).values():
            if inspect.isfunction(v):
                visit(v, module)

    def visit(f: Any, module: str | None, root: bool = False) -> None:
        code, f = code_of(f)
        if code is None:
            return
        if id(code) in done or (not root and not _defined_in_own_module(f, code)):
            # visited; or not code of the module it claims (dataclass-generated __init__ / __repr__ and stdlib
            # wrappers renamed to Class.method share one code object across classes: hashing it would make the
            # digest depend on which class the walk met first). The class source already covers what they derive from
            return
        done.add(id(code))
        # labelled by file name, not module name: the loader names a study module differently (CLI, runpy, import);
        # and not by line number, so moving a function in its file does not re-key what reaches it
        label = os.path.basename(code.co_filename)
        key = f"{label}:{getattr(f, '__qualname__', '')}"
        try:
            seen.setdefault(key, set()).add(inspect.getsource(f))
        except (OSError, TypeError):
            seen.setdefault(key, set()).add(code.co_code.hex())  # no source on disk: the bytecode
        g = getattr(f, "__globals__", {})
        for name in names(code):
            target = g.get(name)
            if inspect.ismodule(target) and (target.__name__.startswith("chairlift") or target.__name__ == module):
                continue  # attribute access through a module object: not followed (names are imported directly here)
            if inspect.isfunction(target) and ours(target, module):
                visit(target, module)
            elif isinstance(target, type) and ours(target, module):
                visit_class(target, module)
            elif name in g and not callable(target) and not inspect.ismodule(target):
                # a module-level value the code reads (START = date(2011, 6, 1), a parameter table): its value is
                # part of what the stage computes, though no function's source shows it
                with contextlib.suppress(SpecError):  # a frame, a logger, a lock: not a value to key a result on
                    seen.setdefault(f"{label}:global:{name}", set()).add(canonical_json(target))
        for cell in getattr(f, "__closure__", None) or ():
            try:
                c = cell.cell_contents
            except ValueError:
                continue
            if inspect.isfunction(c) and ours(c, module):
                visit(c, module)

    _, root = code_of(fn)
    visit(fn, getattr(root, "__module__", None), root=True)
    # every source under a key, sorted: the digest cannot depend on the order the walk met them
    body = "\x1f".join(f"{k}\x1e{chr(0x1D).join(sorted(v))}" for k, v in sorted(seen.items()))
    return hashlib.sha256(body.encode()).hexdigest()


@dataclass(frozen=True)
class Stage:
    """One node: `fn(spec?, *inputs)` with its identity (spec hash, version, captured values, code digest, input keys,
    and for a source its data fingerprint). A change to any of them re-keys the stage and everything downstream."""

    name: str
    fn: Callable[..., Any]
    inputs: tuple[str, ...] = ()
    spec: Any = None
    version: int = 1
    fingerprint: Callable[[], str] | None = None  # required for a source stage to be cacheable
    ignore: tuple[str, ...] = ()  # captured names deliberately left out of the identity (they cannot change a result)
    captured_hash: str = field(init=False, default="")
    code_hash: str = field(init=False, default="")

    def __post_init__(self) -> None:
        values = captured(self.fn, self.ignore)
        try:
            digest = spec_hash(values) if values else ""
        except SpecError as exc:
            names = ", ".join(f"{k} ({type(v).__name__})" for k, v in values.items())
            raise ValueError(
                f"stage {self.name!r} captures values that cannot be part of its identity: {names}. Put them in the"
                f" stage's spec, pass them as an input, or list them in ignore= if they cannot change a result ({exc})"
            ) from exc
        object.__setattr__(self, "captured_hash", digest)
        object.__setattr__(self, "code_hash", code_digest(self.fn))

    def key(self, input_refs: Mapping[str, str], fingerprint: str = "") -> str:
        parts = [self.name, str(self.version), spec_hash(self.spec), fingerprint, f"code={self.code_hash}"]
        parts += [f"captured={self.captured_hash}"] if self.captured_hash else []
        parts += [f"{k}={input_refs[k]}" for k in sorted(input_refs)]
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def plan_key(self, input_plan_keys: Mapping[str, str], fingerprint: str = "") -> str:
        parts = ["plan", self.name, str(self.version), spec_hash(self.spec), fingerprint, f"code={self.code_hash}"]
        parts += [f"captured={self.captured_hash}"] if self.captured_hash else []
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
    """A DAG of stages over a content-addressed store: `plan` says what would rebuild, `run` builds what is missing
    and records the run (signature, manifest, ledger trial, events), `load` reads a stage's output."""

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
        self.ledger_path = self.root / "ledger.jsonl"  # the study's trials and holdout opening
        self.workers = 1  # worker processes stages may use (a site setting, never in a key)
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
        p = Pipeline(
            self.stages.values(),
            root,
            store,
            data=self.data if data is None else data,
            cache=cache if cache is not None else (self.cache.directory if self.cache else None),
        )
        p.workers = self.workers
        return p

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
        with bind(self.data, self.cache), bind_ledger(self.ledger_path):
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
        headline: Headline | Mapping[str, Any] | None = None,
        charge: bool = True,
    ) -> RunReport:
        """Run the needed stages; write a run record and an event stream under <root>/runs/.

        `charge=False` keeps the run out of the study's ledger: inspecting one stage (`chairlift show`) is not an
        experiment.

        `meta` is caller context recorded verbatim: the study reference, the parameters, the resolved configuration.
        `listeners` receive every event as it is written (the live view); an exception in one never fails the run.
        """
        targets = list(targets) if targets is not None else None
        sig = self.signature(targets)
        started = dt.datetime.now(dt.UTC)
        report = RunReport(run_id=self._new_run_id(started, sig.signature), signature=sig.signature)
        record = self._record(report, sig, started, targets, reason, meta)
        self._write_record(record)
        last_event = [time.monotonic()]

        def _progress(_e: dict[str, Any]) -> None:
            last_event[0] = time.monotonic()

        log = EventLog(self.root / "runs" / f"{report.run_id}.events.jsonl", report.run_id, [*listeners, _progress])
        from chairlift.run.systemd import Heartbeat, watchdog_interval

        wd = watchdog_interval()
        heartbeat = Heartbeat(wd, lambda: last_event[0]).start() if wd else None  # under systemd with WatchdogSec
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
            if charge:
                self._record_trial(report, targets, reason, meta, Headline.parse(headline))
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
            if heartbeat is not None:
                heartbeat.stop()
        return report

    def _record_trial(
        self,
        report: RunReport,
        targets: list[str] | None,
        reason: str,
        meta: Mapping[str, Any] | None,
        headline: Headline | None,
    ) -> None:
        """Charge the run to the study's ledger: a new experiment is a new trial (a re-run is not). With a headline,
        the experiment is the headline stage's key; without one, the run signature."""
        numbers: dict[str, Any] | None = None
        trial_key: str | None = None
        if headline is not None and headline.stage in report.results:
            trial_key = report.results[headline.stage].key
            out = self.store.get(report.results[headline.stage].output)
            try:
                numbers = {
                    "sharpe": float(dig(out, headline.sharpe)),
                    "n_obs": int(dig(out, headline.n_obs)),
                    "periods": headline.periods,
                }
            except (KeyError, TypeError, ValueError):
                numbers = None
        m = dict(meta or {})
        Ledger(self.ledger_path).record_trial(
            signature=report.signature,
            run_id=report.run_id,
            params={"targets": targets, **m.get("params", {})},
            reason=reason,
            headline=numbers,
            sweep=m.get("sweep"),
            trial_key=trial_key,
        )

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
                with (
                    bound(log, name),
                    bind(self.data, self.cache),
                    bind_ledger(self.ledger_path),
                    bind_compute(self.workers),
                ):
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
                "captured": canonical(captured(st.fn, st.ignore)),
                "code": st.code_hash,
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
            "environment": environment(),
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


@functools.cache
def environment() -> dict[str, Any]:
    """What decides whether a re-run can give identical bytes; `hash` summarizes it and `rerun` compares it.

    The hash covers the interpreter, the machine architecture, every installed distribution and version, and the code
    identity (commit + dirty flag). The nearest uv.lock is recorded beside it for reference, not hashed: the installed
    set is what actually ran.
    """
    from importlib.metadata import distributions

    dists = sorted({f"{d.metadata['Name'].lower()}=={d.version}" for d in distributions() if d.metadata["Name"]})
    core = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "code": code_identity(),
        "distributions": dists,
    }
    digest = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    picked = ("chairlift", "polars", "numpy", "click", "msgspec", "blake3")
    packages = {n: next((d.split("==")[1] for d in dists if d.split("==")[0] == n), "absent") for n in picked}
    return {
        "hash": digest,
        "python": core["python"],
        "platform": platform.platform(),
        "packages": packages,
        "distributions": dists,
        "uv_lock": _uv_lock(),
    }


def _uv_lock() -> str | None:
    for d in (Path.cwd(), *Path.cwd().parents):
        f = d / "uv.lock"
        if f.is_file():
            return hashlib.sha256(f.read_bytes()).hexdigest()
    return None


_Frame = _pl.DataFrame
