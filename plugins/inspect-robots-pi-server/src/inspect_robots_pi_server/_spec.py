"""Parse and query the ``PolicySpec`` a PI-protocol server returns from ``load``.

The ``load`` RPC's response carries ``result.spec`` as a JSON **string**
(not a nested object) — this module parses and validates it once, at connect
time, into a typed :class:`PolicySpec`, and exposes the two small pieces of
logic every caller needs: resolving one declared state/camera sub-key to its
actual wire key (:func:`payload_key`), and resolving the image resolution a
given camera must be resized to before encoding (:func:`resolution_for`).

Framework-agnostic on purpose (no ``inspect_robots`` imports): callers decide
what an unmappable key means for their own error taxonomy.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from inspect_robots_pi_server._protocol import PiServerError

_VALID_RESIZE_MODES = frozenset({"stretch", "pad"})
_VALID_INTERPOLATIONS = frozenset({"bilinear", "lanczos"})


@dataclass(frozen=True)
class ImagePreprocessSpec:
    """How the server wants images resized before encoding, per ``load``'s response."""

    target_resolution: tuple[int, int] | None
    image_resolutions: Mapping[str, tuple[int, int]]
    resize_mode: str
    interpolation: str


@dataclass(frozen=True)
class PolicySpec:
    """A parsed and validated ``load`` response."""

    input_spec: Mapping[str, tuple[list[int], str]]
    output_spec: Mapping[str, Any]
    action_keys: tuple[str, ...]
    action_horizon: int | None
    action_dim: int | None
    image_preprocess: ImagePreprocessSpec
    camera_names: frozenset[str]


def is_camera_key(key: str, shape: list[int]) -> bool:
    """A key is a camera iff it isn't a ``*_mask`` sibling and looks like an ``(H, W, 3)`` image."""
    return not key.endswith("_mask") and len(shape) >= 3 and shape[-1] == 3


def parse_policy_spec(raw_json: str) -> PolicySpec:
    """Parse and validate the ``load`` response's ``result.spec`` JSON string."""
    try:
        raw = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PiServerError("ProtocolError", f"load() spec is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise PiServerError("ProtocolError", "load() spec JSON must be an object")

    input_spec_raw = raw.get("input_spec") or {}
    if not isinstance(input_spec_raw, dict):
        raise PiServerError("ProtocolError", "load() spec.input_spec must be an object")
    input_spec: dict[str, tuple[list[int], str]] = {}
    for key, entry in input_spec_raw.items():
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
            or not isinstance(entry[0], list)
            or not isinstance(entry[1], str)
        ):
            raise PiServerError(
                "ProtocolError", f"load() spec.input_spec[{key!r}] must be [shape, dtype]"
            )
        input_spec[key] = (list(entry[0]), entry[1])

    output_spec = raw.get("output_spec") or {}
    if not isinstance(output_spec, dict):
        raise PiServerError("ProtocolError", "load() spec.output_spec must be an object")

    action_keys_raw = raw.get("action_keys")
    if not isinstance(action_keys_raw, list) or not action_keys_raw:
        raise PiServerError("ProtocolError", "load() spec.action_keys must be a non-empty list")
    action_keys = tuple(str(key) for key in action_keys_raw)

    action_horizon = raw.get("action_horizon")
    action_dim = raw.get("action_dim")
    if (action_horizon is None) != (action_dim is None):
        raise PiServerError(
            "ProtocolError", "load() spec.action_horizon and .action_dim must be set together"
        )

    image_preprocess = _parse_image_preprocess(raw.get("image_preprocess"))

    camera_names = frozenset(
        key for key, (shape, _dtype) in input_spec.items() if is_camera_key(key, shape)
    )

    return PolicySpec(
        input_spec=input_spec,
        output_spec=output_spec,
        action_keys=action_keys,
        action_horizon=action_horizon,
        action_dim=action_dim,
        image_preprocess=image_preprocess,
        camera_names=camera_names,
    )


def _parse_image_preprocess(raw: Any) -> ImagePreprocessSpec:
    if not isinstance(raw, dict):
        raise PiServerError("ProtocolError", "load() spec.image_preprocess must be an object")

    resize_mode = raw.get("resize_mode")
    if resize_mode not in _VALID_RESIZE_MODES:
        raise PiServerError(
            "ProtocolError",
            f"load() spec.image_preprocess.resize_mode must be one of "
            f"{sorted(_VALID_RESIZE_MODES)}, got {resize_mode!r}",
        )
    interpolation = raw.get("interpolation")
    if interpolation not in _VALID_INTERPOLATIONS:
        raise PiServerError(
            "ProtocolError",
            f"load() spec.image_preprocess.interpolation must be one of "
            f"{sorted(_VALID_INTERPOLATIONS)}, got {interpolation!r}",
        )

    target_resolution = _parse_resolution(raw.get("target_resolution"), "target_resolution")
    image_resolutions_raw = raw.get("image_resolutions") or {}
    if not isinstance(image_resolutions_raw, dict):
        raise PiServerError(
            "ProtocolError", "load() spec.image_preprocess.image_resolutions must be an object"
        )
    image_resolutions: dict[str, tuple[int, int]] = {}
    for key, value in image_resolutions_raw.items():
        resolution = _parse_resolution(value, f"image_resolutions[{key!r}]")
        if resolution is None:
            raise PiServerError(
                "ProtocolError", f"load() spec.image_preprocess.image_resolutions[{key!r}] is null"
            )
        image_resolutions[key] = resolution

    if target_resolution is None and not image_resolutions:
        raise PiServerError(
            "ProtocolError",
            "load() spec.image_preprocess must set target_resolution or image_resolutions",
        )

    return ImagePreprocessSpec(
        target_resolution=target_resolution,
        image_resolutions=image_resolutions,
        resize_mode=resize_mode,
        interpolation=interpolation,
    )


def _parse_resolution(value: Any, label: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in value)
    ):
        raise PiServerError(
            "ProtocolError", f"load() spec.image_preprocess.{label} must be [height, width]"
        )
    return int(value[0]), int(value[1])


def resolution_for(spec: ImagePreprocessSpec, camera_wire_key: str) -> tuple[int, int]:
    """The ``(height, width)`` to resize ``camera_wire_key`` to before encoding."""
    if camera_wire_key in spec.image_resolutions:
        return spec.image_resolutions[camera_wire_key]
    if spec.target_resolution is not None:
        return spec.target_resolution
    raise PiServerError(
        "ProtocolError",
        f"no image_preprocess resolution resolves for camera {camera_wire_key!r}",
    )


def payload_key(bare_key: str, input_spec: Mapping[str, Any]) -> str | None:
    """Resolve one declared sub-key to its wire key, or ``None`` if unmappable.

    Mirrors ``pi_inference_client``'s own ``_state_payload_key``/
    ``_image_payload_key`` exactly: the bare form wins if it is itself a key
    in ``input_spec``; else the ``observation/<key>`` form wins if *that* is
    in ``input_spec``; else, when ``input_spec`` is non-empty, neither form
    is recognized by this deployment and the caller should treat that as a
    configuration error (the real client instead silently drops the value
    here — this module intentionally does not do that, returning ``None``
    for the caller to fail loudly on instead).
    """
    if not input_spec:
        return bare_key
    if bare_key in input_spec:
        return bare_key
    prefixed = f"observation/{bare_key}"
    if prefixed in input_spec:
        return prefixed
    return None
