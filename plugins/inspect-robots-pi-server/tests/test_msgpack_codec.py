"""Tests for the NumPy-aware msgpack codec."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inspect_robots_pi_server._msgpack_codec import packb, unpackb
from inspect_robots_pi_server._protocol import PiServerError


@pytest.mark.parametrize(
    "array",
    [
        np.array([1.0, 2.5, -3.25], dtype=np.float32),
        np.array([1, -2, 3], dtype=np.int32),
        np.array([1, 2, 3], dtype=np.uint8),
        np.array([True, False, True], dtype=np.bool_),
        np.array([b"ab", b"cd"], dtype="S2"),
        np.array(["ab", "cd"], dtype="U2"),
        np.zeros((2, 3, 4), dtype=np.float64),
        np.zeros((0,), dtype=np.float32),
    ],
)
def test_ndarray_round_trips(array: np.ndarray[Any, Any]) -> None:
    decoded = unpackb(packb(array))
    assert isinstance(decoded, np.ndarray)
    assert decoded.dtype == array.dtype
    np.testing.assert_array_equal(decoded, array)


def test_np_generic_scalar_round_trips() -> None:
    value = np.float32(3.5)
    decoded = unpackb(packb(value))
    assert isinstance(decoded, np.float32)
    assert decoded == value


@pytest.mark.parametrize("dtype", ["V4", "O", "complex64"])
def test_pack_rejects_unsafe_dtype(dtype: str) -> None:
    array = np.zeros((2,), dtype=dtype)
    with pytest.raises(PiServerError, match="unsupported array dtype"):
        packb(array)


def test_unpack_rejects_unsafe_dtype() -> None:
    import msgpack

    raw = msgpack.packb(
        {b"__ndarray__": True, b"data": b"\x00" * 8, b"dtype": "complex64", b"shape": [1]},
        use_bin_type=True,
    )
    with pytest.raises(PiServerError, match="rejected unsafe dtype"):
        unpackb(raw)


def test_unpack_rejects_negative_shape() -> None:
    import msgpack

    raw = msgpack.packb(
        {b"__ndarray__": True, b"data": b"", b"dtype": "float32", b"shape": [-1]}, use_bin_type=True
    )
    with pytest.raises(PiServerError, match="malformed ndarray shape"):
        unpackb(raw)


def test_unpack_rejects_buffer_length_mismatch() -> None:
    import msgpack

    raw = msgpack.packb(
        {b"__ndarray__": True, b"data": b"\x00" * 3, b"dtype": "float32", b"shape": [1]},
        use_bin_type=True,
    )
    with pytest.raises(PiServerError, match="buffer is"):
        unpackb(raw)


def test_plain_dicts_and_scalars_pass_through() -> None:
    payload = {"a": 1, "b": "text", "c": [1, 2, 3], "d": None, "e": True}
    assert unpackb(packb(payload)) == payload
