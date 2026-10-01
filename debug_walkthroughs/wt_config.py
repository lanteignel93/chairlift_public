"""Walkthrough: two machines, one experiment, one signature.

    uv run python debug_walkthroughs/wt_config.py          # asserted run
    uv run python debug_walkthroughs/wt_config.py --pdb    # step through

Two simulated machines. Each has a user config (its own home and its own `[data] vendor` path) and the same data
bytes stored at a different place. machine B also has a profile chosen through the host map, and an environment
variable that moves its store. The walkthrough resolves both configs, prints the provenance of each key, builds the
same study on each with the configured data roots, and checks:
- the signatures are equal
- the run roots differ
- a changed byte moves the signature on both

Under --pdb, useful breakpoints:
  chairlift/core/config.py   load_config   step over the layers loop; `prov` shows which layer set each key
  chairlift/core/config.py   _expand       path expansion; `~`, `${VAR}`, and relative-to-the-file
  chairlift/data/refs.py     content_hash  `lines` holds the (relpath, hash) pairs the directory hash is made of;
                                           identical on both machines although the paths differ
  chairlift/run/dag.py       signature     `fps` (data fingerprints) and `plan_keys` (one per stage)
What to expect:
- machine A: paths.home from the user file; compute.workers from the default
- machine B: compute.workers = 4 from profile `big` (its host map); paths.store from env (CHAIRLIFT_PATHS__STORE)
- one signature for both machines; a different one after a single byte changes
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import polars as pl

from chairlift.core.config import Resolved, load_config
from chairlift.data.refs import DataRef, fingerprint_of
from chairlift.run.dag import Pipeline, Stage

PRICES = DataRef(alias="vendor", relpath="prices")


def study(root: Path, *, window: int = 2) -> Pipeline:
    def prices() -> pl.DataFrame:
        return pl.read_csv(PRICES.path() / "*.csv")

    def smooth(prices: pl.DataFrame) -> pl.DataFrame:
        return prices.with_columns(pl.col("px").rolling_mean(window, min_samples=1).alias("smooth"))

    return Pipeline(
        [Stage("prices", prices, fingerprint=fingerprint_of(PRICES)), Stage("smooth", smooth, ("prices",))], root
    )


def machine(base: Path, name: str, extra: str = "") -> tuple[Path, Path]:
    """A user config file and a data directory for one machine."""
    data = base / name / "mnt" / "vendor"
    (data / "prices").mkdir(parents=True)
    (data / "prices" / "a.csv").write_text("day,px\n1,100.0\n2,101.5\n3,99.0\n")
    user = base / name / "config.toml"
    user.write_text(f'[paths]\nhome = "{base / name / "home"}"\n[data]\nvendor = "{data}"\n{extra}')
    return user, data


def resolve(user: Path, base: Path, host: str, env: dict[str, str]) -> Resolved:
    return load_config(cwd=base, env=env, user_file=user, system_file=base / "none.toml", hostname=host)


def show(label: str, r: Resolved) -> None:
    print(f"--- {label}: profile {r.profile or '-'}")
    for key, value, layer in r.explain():
        print(f"  {key:22} = {value!s:60} ({layer.split(' /')[0]})")


def main() -> None:
    base = Path(tempfile.mkdtemp(prefix="wt_config_"))
    user_a, _ = machine(base, "a")
    user_b, data_b = machine(base, "b", '[profile.big]\ncompute.workers = 4\n[profile.hosts]\n"box-b" = "big"\n')
    ra = resolve(user_a, base, "box-a", {})
    rb = resolve(user_b, base, "box-b", {"CHAIRLIFT_PATHS__STORE": str(base / "b" / "fast" / "store")})
    show("machine A", ra)
    show("machine B", rb)
    assert ra.provenance["paths.home"].startswith("user") and "compute.workers" not in ra.provenance
    assert rb.profile == "big" and rb.config.compute.workers == 4
    assert rb.provenance["paths.store"] == "env"

    pipes = []
    for r in (ra, rb):
        c = r.config
        pipes.append(
            study(base / "unused").rebase(
                c.paths.study_dir("wt"), c.paths.store_dir(), data=c.data, cache=c.paths.cache_dir()
            )
        )
    sig_a, sig_b = (p.signature().signature for p in pipes)
    print("signature A", sig_a[:16], "· root", pipes[0].root)
    print("signature B", sig_b[:16], "· root", pipes[1].root)
    assert sig_a == sig_b and pipes[0].root != pipes[1].root  # same experiment, different places

    reports = [p.run(reason="walkthrough") for p in pipes]
    assert reports[0].results["smooth"].output == reports[1].results["smooth"].output  # same bytes out

    with (data_b / "prices" / "a.csv").open("a") as fh:
        fh.write("4,98.0\n")  # machine B's data moves on
    sig_b2 = pipes[1].signature().signature
    print("after one appended row on B:", sig_b2[:16])
    assert sig_b2 != sig_a
    shutil.rmtree(base)
    print("walkthrough passed")


if __name__ == "__main__":
    if "--pdb" in sys.argv:
        import pdb

        pdb.set_trace()  # `s` into main(), then `b chairlift/core/config.py:<load_config line>`
    main()
