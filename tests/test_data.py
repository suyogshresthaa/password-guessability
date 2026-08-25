import sqlite3
from pathlib import Path

import numpy as np
import pytest

from pwguess.data import (
    N_BUCKETS, TRAIN_BUCKETS, bucket_of, load_eval_passwords,
    overlap_with_eval, password_hash, split_of,
)


def test_bucket_is_deterministic():
    assert bucket_of("hunter2") == bucket_of("hunter2")


def test_same_password_always_same_split():
    # The whole point of hash-based splitting: no split file to desync.
    for _ in range(5):
        assert split_of("correcthorsebatterystaple") == split_of("correcthorsebatterystaple")


def test_different_passwords_can_land_in_different_buckets():
    buckets = {bucket_of(f"pw{i}") for i in range(200)}
    assert len(buckets) > 1


def test_bucket_range():
    for pw in ["a", "hunter2", "P@ssw0rd123!", ""]:
        assert 0 <= bucket_of(pw) < N_BUCKETS


def test_split_proportions_roughly_98_2():
    # Not exact for any finite sample, but should be in the right ballpark
    # over a few thousand distinct strings.
    n = 5000
    n_train = sum(1 for i in range(n) if split_of(f"password{i}") == "train")
    frac = n_train / n
    assert 0.93 < frac < 1.0  # TRAIN_BUCKETS/N_BUCKETS == 0.98, generous tolerance


def test_password_hash_is_a_64bit_uint():
    h = password_hash("hunter2")
    assert 0 <= h < 2**64


def test_load_eval_passwords_drops_strength_column(tmp_path):
    db = tmp_path / "toy.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE Users (password TEXT, strength INTEGER)")
    con.executemany("INSERT INTO Users VALUES (?, ?)", [("abc123", 1), ("hunter2", 0)])
    con.commit()
    con.close()

    pw = load_eval_passwords(db)
    assert sorted(pw) == ["abc123", "hunter2"]


def test_overlap_with_eval_finds_exact_matches():
    rockyou = ["password", "123456", "iloveyou", "qwerty"]
    hashes = np.array(sorted(password_hash(p) for p in rockyou), dtype=np.uint64)

    eval_set = ["password", "not_in_rockyou_xyz", "qwerty"]
    result = overlap_with_eval(hashes, eval_set)

    assert result["n_eval"] == 3
    assert result["n_overlap"] == 2
    assert result["overlap_frac"] == pytest.approx(2 / 3)


def test_overlap_with_empty_rockyou_hashes_is_zero():
    result = overlap_with_eval(np.array([], dtype=np.uint64), ["a", "b"])
    assert result["n_overlap"] == 0
    assert result["overlap_frac"] == 0.0
