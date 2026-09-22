"""Dotted-path field access for config-driven observation/action selectors.

Rosboard already delivers nested ROS submessages as same-named nested dicts
(``payload["pose"]["pose"]["position"]["x"]``, confirmed against rosboard's own
``ros2dict()``), so a selector name like ``"pose.pose.position.x"`` is a direct
key path, not a schema the adapter has to know in advance. This is what lets
:mod:`inspect_robots_rosboard._config`-driven observations/actions cover
arbitrary message types without a per-type parser: the config file's
``selector.names`` list *is* the schema.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def get_path(payload: Mapping[str, Any], dotted_name: str) -> float:
    """Read one numeric scalar out of ``payload`` at a dot-separated key path."""
    value: Any = payload
    for part in dotted_name.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"field path {dotted_name!r} not found in message payload")
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"field path {dotted_name!r} did not resolve to a number, got {value!r}")
    return float(value)


def select_state(payload: Mapping[str, Any], names: Sequence[str]) -> npt.NDArray[np.float64]:
    """Build a ``(len(names),)`` vector by reading each dotted path from ``payload`` in order."""
    return np.asarray([get_path(payload, name) for name in names], dtype=np.float64)


def build_from_selector(names: Sequence[str], values: Sequence[float]) -> dict[str, Any]:
    """Build a nested message-fields dict, writing each value at its dotted path.

    The inverse of :func:`select_state`: ``names[i]`` is where ``values[i]``
    is written. Missing intermediate dicts are created as needed; a name
    reused with conflicting nesting (e.g. ``"a"`` and ``"a.b"`` together) is
    rejected rather than silently overwriting a scalar with a dict or a dict
    with a scalar.
    """
    if len(names) != len(values):
        raise ValueError(f"selector has {len(names)} names but received {len(values)} values")
    root: dict[str, Any] = {}
    for name, value in zip(names, values, strict=True):
        parts = name.split(".")
        node = root
        for part in parts[:-1]:
            existing = node.get(part)
            if existing is None:
                existing = {}
                node[part] = existing
            elif not isinstance(existing, dict):
                raise ValueError(f"selector name {name!r} conflicts with an earlier scalar entry")
            node = existing
        leaf = parts[-1]
        if isinstance(node.get(leaf), dict):
            raise ValueError(f"selector name {name!r} conflicts with an earlier nested entry")
        node[leaf] = float(value)
    return root
