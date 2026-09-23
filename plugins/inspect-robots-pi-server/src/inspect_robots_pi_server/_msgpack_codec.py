"""NumPy-aware msgpack codec, a direct port of ``pi_inference_client.msgpack_numpy``.

Reimplemented rather than depending on the ``msgpack-numpy`` PyPI package:
its wire format is not confirmed identical to ``pi_inference_client``'s own
in-package codec, and this protocol is security-sensitive (arbitrary
dtype/shape arriving from the wire), so this file is the single source of
truth for the exact bytes exchanged, kept intentionally small and diffable
against the original rather than delegated to an unrelated dependency.

Wire format: an ``ndarray`` becomes ``{b"__ndarray__": True, b"data":
tobytes(), b"dtype": dtype.str, b"shape": shape}``; an ``np.generic`` scalar
becomes ``{b"__npgeneric__": True, b"data": item(), b"dtype": dtype.str}``.
Unpacking only accepts dtype kinds ``{f, i, u, b, S, U}`` (float, signed/
unsigned int, bool, byte string, unicode string) — a deliberate security
control against arbitrary object/void/complex reconstruction from the wire;
do not loosen it.
"""

from __future__ import annotations

import math
from typing import Any

import msgpack
import numpy as np

from inspect_robots_pi_server._protocol import PiServerError

_SAFE_DTYPE_KINDS = frozenset({"f", "i", "u", "b", "S", "U"})


def _pack_hook(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind not in _SAFE_DTYPE_KINDS:
            raise PiServerError(
                "invalid_frame", f"unsupported array dtype for encoding: {obj.dtype}"
            )
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": list(obj.shape),
        }
    if isinstance(obj, np.generic):
        if obj.dtype.kind not in _SAFE_DTYPE_KINDS:
            raise PiServerError(
                "invalid_frame", f"unsupported scalar dtype for encoding: {obj.dtype}"
            )
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    raise TypeError(f"object of type {type(obj).__name__} is not msgpack-serializable")


def _unpack_hook(obj: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in obj:
        dtype = _safe_dtype(obj[b"dtype"])
        shape_raw = obj[b"shape"]
        if not isinstance(shape_raw, (list, tuple)) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape_raw
        ):
            raise PiServerError("invalid_frame", f"malformed ndarray shape: {shape_raw!r}")
        shape = tuple(int(dim) for dim in shape_raw)
        data = obj[b"data"]
        if not isinstance(data, (bytes, bytearray)):
            raise PiServerError("invalid_frame", "ndarray 'data' field must be bytes")
        expected = dtype.itemsize * (math.prod(shape) if shape else 1)
        if expected != len(data):
            raise PiServerError(
                "invalid_frame",
                f"ndarray buffer is {len(data)} bytes, expected {expected} for shape {shape}",
            )
        return np.frombuffer(data, dtype=dtype).reshape(shape).copy()
    if b"__npgeneric__" in obj:
        dtype = _safe_dtype(obj[b"dtype"])
        return dtype.type(obj[b"data"])
    return obj


def _safe_dtype(dtype_str: Any) -> np.dtype[Any]:
    if not isinstance(dtype_str, str):
        raise PiServerError(
            "invalid_frame", f"dtype field must be a string, got {type(dtype_str).__name__}"
        )
    try:
        dtype = np.dtype(dtype_str)
    except TypeError as exc:
        raise PiServerError("invalid_frame", f"unparseable dtype {dtype_str!r}: {exc}") from exc
    if dtype.kind not in _SAFE_DTYPE_KINDS:
        raise PiServerError("invalid_frame", f"rejected unsafe dtype on decode: {dtype}")
    return dtype


def packb(obj: object) -> bytes:
    """Encode one Python/NumPy object tree to msgpack bytes."""
    try:
        return msgpack.packb(obj, default=_pack_hook, use_bin_type=True)  # type: ignore[no-any-return]
    except PiServerError:
        raise
    except Exception as exc:
        raise PiServerError("invalid_frame", f"could not encode msgpack payload: {exc}") from exc


def unpackb(data: bytes) -> object:
    """Decode msgpack bytes back to a Python/NumPy object tree."""
    try:
        return msgpack.unpackb(data, raw=False, object_hook=_unpack_hook, strict_map_key=False)
    except PiServerError:
        raise
    except Exception as exc:
        raise PiServerError("invalid_frame", f"could not decode msgpack payload: {exc}") from exc
