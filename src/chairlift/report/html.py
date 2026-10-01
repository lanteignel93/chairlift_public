"""Static run reports: one self-contained HTML page per run, and an index of the latest run of every study.

    chairlift report RUN            → <study dir>/reports/<run_id>.html
    chairlift report --index        → <paths.reports>/index.html

Everything comes from files a run already wrote: its record, its event stream, the study's ledger. A report is a
view; it never changes a result, and nothing reads it back.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from chairlift.run.events import read_events
from chairlift.run.live import replay

CSS = """
:root { --ink:#1f2430; --muted:#6b7385; --line:#e3e6ee; --ok:#2e9e6a; --bad:#c94f4f; --run:#d39a2c; --bg:#fbfbfd; }
@media (prefers-color-scheme: dark) { :root { --ink:#e6e8ef; --muted:#9aa2b5; --line:#2c3140; --bg:#161922; } }
body { font: 14px/1.45 system-ui, sans-serif; color: var(--ink); background: var(--bg); margin: 0 auto;
       max-width: 1100px; padding: 16px; }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 24px 0 8px; }
.meta { color: var(--muted); font-size: 12px; word-break: break-all; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: 12px; }
.ok { color: var(--ok); } .failed, .died { color: var(--bad); }
.running { color: var(--run); } .hit { color: var(--muted); }
.wrap { overflow-x: auto; }
"""


HEAD = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
)


def _page(title: str, body: str) -> str:
    return f"{HEAD}<title>{_e(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"


def _e(v: Any) -> str:
    return html.escape(str(v))


def _table(head: list[str], rows: list[list[Any]], classes: list[str] | None = None) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    body: list[str] = []
    for i, r in enumerate(rows):
        cls = f' class="{classes[i]}"' if classes and classes[i] else ""
        body.append(f"<tr{cls}>" + "".join(f"<td>{_e(c)}</td>" for c in r) + "</tr>")
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _num(v: Any, fmt: str = "{:+.3f}") -> str:
    return fmt.format(v) if isinstance(v, int | float) else ("" if v is None else str(v))


def run_report(rec_path: Path, ledger_path: Path | None = None, headline: dict[str, Any] | None = None) -> str:
    """The HTML for one run. `headline`: numbers to show in the summary (read from a stage output by the caller)."""
    rec = json.loads(rec_path.read_text())
    evs, _ = read_events(rec_path.parent / rec_path.name.replace(".json", ".events.jsonl"))
    state = replay(evs)
    study = rec.get("meta", {}).get("name", "")
    status = rec.get("status", "?")
    stages: list[list[Any]] = []
    classes: list[str] = []
    for name, st in state.stages.items():
        shape = f"{st.rows}×{st.cols}" if st.rows is not None else ""
        stages.append(
            [name, st.status, _num(st.seconds, "{:.2f}"), shape, _num(st.peak_rss_mb, "{:.0f}"), st.error or st.reason]
        )
        classes.append(st.status)
    metrics = [[s, n, _num(v, "{:+.4f}"), " ".join(f"{k}={x}" for k, x in d.items())] for s, n, v, d in state.metrics]
    parts = [
        f"<h1>{_e(study)} · <span class='{_e(status)}'>{_e(status)}</span></h1>",
        f"<div class='meta'>run {_e(rec['run_id'])} · signature {_e(rec['signature'])}"
        f" · started {_e(rec.get('started_at'))} · finished {_e(rec.get('finished_at', '—'))}"
        f" · host {_e(rec.get('host'))}</div>",
        f"<div class='meta'>reason: {_e(rec.get('reason') or '—')} · code {_e(rec.get('code'))} · environment"
        f" {_e(str(rec.get('environment', {}).get('hash', ''))[:12])}</div>",
    ]
    if rec.get("error"):
        parts.append(f"<h2>Error</h2><pre>{_e(rec['error'])}</pre>")
    if headline:
        parts.append("<h2>Headline</h2>" + _table(list(headline), [[_num(v) for v in headline.values()]]))
    parts.append(
        "<h2>Stages</h2>" + _table(["stage", "status", "seconds", "shape", "peak RSS MB", "detail"], stages, classes)
    )
    if metrics:
        parts.append("<h2>Metrics</h2>" + _table(["stage", "metric", "value", "where"], metrics))
    if ledger_path is not None and ledger_path.exists():
        from chairlift.ledger.trials import Ledger

        led = Ledger(ledger_path)
        opened = led.holdout_opened()
        gate = f"OPEN since {opened['at']} ({opened['reason']})" if opened else "sealed"
        parts.append(f"<h2>Ledger</h2><p>{len(led.trials())} trial(s) · holdout {_e(gate)}</p>")
    params = rec.get("meta", {}).get("params", {})
    if params:
        parts.append("<h2>Parameters</h2>" + _table(["name", "value"], [[k, json.dumps(v)] for k, v in params.items()]))
    title = f"{study} {rec['run_id']}"
    return _page(title, "".join(parts))


def index_page(studies_dir: Path) -> str:
    """The latest run of every study, newest first, linking to its report when one exists."""
    rows: list[list[Any]] = []
    classes: list[str] = []
    for d in sorted(p for p in studies_dir.iterdir() if p.is_dir()) if studies_dir.exists() else []:
        recs = sorted((d / "runs").glob("*Z-*.json"), key=lambda p: p.stem)
        recs = [p for p in recs if not p.name.endswith(".events.jsonl")]
        if not recs:
            continue
        rec = json.loads(recs[-1].read_text())
        report = d / "reports" / f"{rec['run_id']}.html"
        rows.append(
            [
                d.name,
                rec["run_id"],
                rec.get("status"),
                rec["signature"][:12],
                rec.get("reason", ""),
                str(report) if report.exists() else "",
            ]
        )
        classes.append(str(rec.get("status")))
    order = sorted(range(len(rows)), key=lambda i: rows[i][1], reverse=True)
    body = _table(
        ["study", "latest run", "status", "signature", "reason", "report"],
        [rows[i] for i in order],
        [classes[i] for i in order],
    )
    return _page("chairlift runs", f"<h1>chairlift · latest runs</h1>{body}")
