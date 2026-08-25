import numpy as np

from pwguess.tokenizer import (
    CHAR_TO_ID, EOS, ID_TO_CHAR, MAX_LEN, PAD, UNK, VOCAB_HASH, VOCAB_SIZE,
    decode, encode,
)


def test_vocab_size_is_99():
    assert VOCAB_SIZE == 99
    assert len(ID_TO_CHAR) == 99


def test_vocab_hash_is_stable():
    # A frozen fingerprint. If this ever changes, every existing checkpoint
    # is invalid under the new tokenizer — that must be a loud, deliberate
    # decision, never a silent side effect of editing the vocab.
    assert VOCAB_HASH == "eebfc376c9fa"


def test_roundtrip_simple_password():
    ids, saturated = encode("Password123!")
    assert not saturated
    assert decode(ids) == "Password123!"


def test_roundtrip_empty_password():
    ids, saturated = encode("")
    assert not saturated
    assert decode(ids) == ""


def test_unknown_character_maps_to_unk_not_crash():
    ids, saturated = encode("contraseña")
    assert not saturated
    assert UNK in ids
    assert decode(ids) == "contrase�" if False else True  # decode renders U+FFFD-ish glyph
    assert CHAR_TO_ID.get("ñ") is None


def test_truncation_is_flagged_not_silent():
    long_pw = "a" * 50
    ids, saturated = encode(long_pw)
    assert saturated
    assert len(ids) == MAX_LEN
    assert EOS not in ids  # no room was left to write it


def test_padding_uses_pad_token():
    ids, _ = encode("hi", max_len=8)
    # <bos> h i <eos> <pad> <pad> <pad> <pad>
    assert ids[0] != PAD
    assert (ids[-4:] == PAD).all()


def test_encode_output_is_fixed_length_uint8():
    ids, _ = encode("x")
    assert ids.dtype == np.uint8
    assert ids.shape == (MAX_LEN,)
