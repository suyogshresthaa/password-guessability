"""The vocabulary. One definition, imported everywhere a password is turned
into numbers — training, scoring, and the app all call this module rather
than each carrying their own copy.

(v1's bug was exactly two copies of the same feature code drifting apart
between the notebook and app.py. This file exists so that can't happen again.)
"""

import hashlib

import numpy as np

PAD, BOS, EOS, UNK = 0, 1, 2, 3
_SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Printable ASCII 0x20 (space) .. 0x7E (~) -> ids 4..98. 95 characters.
_PRINTABLE = [chr(c) for c in range(0x20, 0x7F)]

ID_TO_CHAR = _SPECIAL + _PRINTABLE
CHAR_TO_ID = {ch: i for i, ch in enumerate(ID_TO_CHAR) if i >= 4}

VOCAB_SIZE = len(ID_TO_CHAR)  # 99
assert VOCAB_SIZE == 99

MAX_LEN = 32  # includes <bos> and <eos>; covers 99.99%+ of real passwords

# Frozen fingerprint of the vocab. A model checkpoint is only meaningful
# paired with the tokenizer it was trained under — asserting this hash at
# load time makes a silent mismatch (e.g. a reordered vocab) impossible.
VOCAB_HASH = hashlib.sha256("".join(ID_TO_CHAR).encode()).hexdigest()[:12]


def encode(password: str, max_len: int = MAX_LEN) -> tuple[np.ndarray, bool]:
    """password -> fixed-length uint8 array: <bos> chars... <eos> <pad>...

    Returns (ids, saturated) where saturated=True means the password was
    longer than fits and was truncated — callers must not treat that
    silently as a normal-length password.
    """
    budget = max_len - 2  # room for <bos> and <eos>
    saturated = len(password) > budget
    body = password[:budget]

    ids = np.full(max_len, PAD, dtype=np.uint8)
    ids[0] = BOS
    for i, ch in enumerate(body):
        ids[1 + i] = CHAR_TO_ID.get(ch, UNK)
    if not saturated:
        ids[1 + len(body)] = EOS
    # else: no <eos> written — the sequence fills the whole budget and is
    # scored as truncated by the caller (see Scorer / model docs).

    return ids, saturated


def decode(ids: np.ndarray) -> str:
    """Inverse of encode: strips <bos>/<eos>/<pad>, renders <unk> as '�'."""
    chars = []
    for i in ids:
        i = int(i)
        if i in (PAD, BOS):
            continue
        if i == EOS:
            break
        chars.append("�" if i == UNK else ID_TO_CHAR[i])
    return "".join(chars)
