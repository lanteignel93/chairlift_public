"""The repository holds no machine paths, host names, firm, vendor-account or user detail: it is meant to be public.

The blocked words are listed by the first 16 hex digits of their sha256, so this file does not publish what it guards
against. Every word of every tracked text file (lowercased, split on non-word characters) is hashed and checked.
"""

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]
BLOCKED = {
    "a1aa8ed605d6edb3",
    "1bcc42eeeb811b05",
    "311046542a31edfb",
    "8d31757a7496af23",
    "e66681074b7bf707",
    "3043266aa5087540",
    "e38c31cf45fafe69",
    "bdde3af2631bf9fd",
    "22032506f6268d7d",
}
WORD = re.compile(r"\w+")


def _h(word: str) -> str:
    return hashlib.sha256(word.encode()).hexdigest()[:16]


def scan(text: str, blocked: set[str]) -> list[tuple[int, str]]:
    return [
        (i, w) for i, line in enumerate(text.splitlines(), 1) for w in WORD.findall(line.lower()) if _h(w) in blocked
    ]


def test_no_tracked_file_names_a_site_path_or_firm():
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    hits = []
    for f in files:
        if f.endswith((".lock", ".png")):
            continue
        try:
            text = (ROOT / f).read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        hits += [f"{f}:{i}: {w}" for i, w in scan(text, BLOCKED)]
    assert not hits, "site detail in tracked files:\n  " + "\n  ".join(hits)


def test_the_check_sees_a_blocked_word_as_a_whole_word_any_case():
    fake = {_h("acme")}
    assert scan("data under /mnt/ACME/x\nacmeish is fine", fake) == [(1, "acme")]
