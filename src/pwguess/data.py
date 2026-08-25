"""Loading RockYou and the held-out eval set, and the train/val split.

The split needs no file on disk: a password's bucket is a deterministic
hash of its own text, so "is this password train or val" can be recomputed
anywhere without ever risking two copies of a split list drifting apart.
"""

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from .tokenizer import CHAR_TO_ID, MAX_LEN

VOCAB_CHARS = set(CHAR_TO_ID.keys())
_BUDGET = MAX_LEN - 2  # room left for a password after <bos>/<eos>

N_BUCKETS = 1000
TRAIN_BUCKETS = 980  # buckets [0, 980) -> train  (98%)
# buckets [980, 1000) -> val                       (2%)


def iter_rockyou(path: str | Path) -> Iterator[str]:
    """Yields RockYou passwords as text, one per line.

    The file is ~99.998% valid UTF-8 (measured); the remainder is legacy
    Latin-1 (accented characters like 'contraseña'). Both are real
    passwords and neither is dropped — decode falls back rather than
    raising, and out-of-vocabulary characters become <unk> at encode time.
    """
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.rstrip(b"\n").rstrip(b"\r")
            if not raw:
                continue
            try:
                pw = raw.decode("utf-8")
            except UnicodeDecodeError:
                pw = raw.decode("latin-1")
            yield pw


def password_hash(password: str) -> int:
    """A stable 64-bit fingerprint used for both split assignment and
    overlap detection — the same primitive serves two callers rather than
    each inventing its own notion of "identity" for a password."""
    digest = hashlib.blake2b(password.encode("utf-8", "surrogateescape"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def bucket_of(password: str) -> int:
    return password_hash(password) % N_BUCKETS


def split_of(password: str) -> Literal["train", "val"]:
    return "train" if bucket_of(password) < TRAIN_BUCKETS else "val"


def iter_rockyou_split(path: str | Path, split: Literal["train", "val"]) -> Iterator[str]:
    want_train = split == "train"
    for pw in iter_rockyou(path):
        if (bucket_of(pw) < TRAIN_BUCKETS) == want_train:
            yield pw


def _has_oov_char(pw: str) -> bool:
    return not set(pw) <= VOCAB_CHARS


def corpus_stats(path: str | Path) -> tuple[dict, np.ndarray]:
    """One pass over RockYou. Returns (stats, sorted_hashes).

    sorted_hashes is every password's hash, sorted, for O(log n) overlap
    lookups elsewhere — a 14.3M-entry uint64 array is ~115 MB, versus well
    over 1 GB for the equivalent Python set of strings.
    """
    hashes: list[int] = []
    total = 0
    with_oov = 0
    truncated = 0

    for pw in iter_rockyou(path):
        total += 1
        if _has_oov_char(pw):
            with_oov += 1
        if len(pw) > _BUDGET:
            truncated += 1
        hashes.append(password_hash(pw))

    sorted_hashes = np.array(hashes, dtype=np.uint64)
    sorted_hashes.sort()

    n_train = int(np.count_nonzero(sorted_hashes % N_BUCKETS < TRAIN_BUCKETS))

    stats = {
        "total": total,
        "with_oov_chars": with_oov,
        "with_oov_frac": with_oov / total,
        "truncated": truncated,
        "truncated_frac": truncated / total,
        "n_train": n_train,
        "n_val": total - n_train,
    }
    return stats, sorted_hashes


def load_eval_passwords(sqlite_path: str | Path) -> list[str]:
    """The held-out cross-corpus set. `strength` is dropped on load — see
    docs/00-audit-v1.md for why that column carries no signal."""
    con = sqlite3.connect(sqlite_path)
    try:
        rows = con.execute("SELECT password FROM Users").fetchall()
    finally:
        con.close()
    return [r[0] for r in rows]


def overlap_with_eval(sorted_rockyou_hashes: np.ndarray, eval_passwords: list[str]) -> dict:
    """How many eval passwords appear verbatim in RockYou.

    This number is reported, never used to filter the eval set: a real
    attacker also has RockYou, so overlap is part of the true attack
    surface, not leakage to be scrubbed away. See docs/00-audit-v1.md §7.
    """
    if len(sorted_rockyou_hashes) == 0 or not eval_passwords:
        return {"n_eval": len(eval_passwords), "n_overlap": 0, "overlap_frac": 0.0}

    eval_hashes = np.array([password_hash(p) for p in eval_passwords], dtype=np.uint64)
    idx = np.clip(np.searchsorted(sorted_rockyou_hashes, eval_hashes), 0, len(sorted_rockyou_hashes) - 1)
    found = sorted_rockyou_hashes[idx] == eval_hashes
    n = int(found.sum())
    return {"n_eval": len(eval_passwords), "n_overlap": n, "overlap_frac": n / len(eval_passwords)}
