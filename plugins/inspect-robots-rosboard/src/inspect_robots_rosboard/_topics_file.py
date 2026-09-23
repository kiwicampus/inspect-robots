"""Parse the ROS2 ``rosboard_client`` node's own ``topics_to_subscribe.yaml``.

That file (schema: ``url``, ``topics: [...]``, ``topics_to_stream: [...]``) is
unrelated to :mod:`inspect_robots_rosboard._config`'s richer robot-config
schema: it carries no message type, field-selector, image-resize, or clamp
metadata, since the ROS2 client discovers types dynamically from rosboard's
own subscribe acknowledgement. This module lets the fixed-schema embodiment
mode (``-E topics_file=...``) treat that file as the single source of truth
for two things only: the rosboard ``url``, and which of the fixed schema's
role topics (``odometry_topic``/``imu_topic``/``camera_topic``/
``command_topic``) are actually enabled, by checking each one's presence in
``topics``/``topics_to_stream``. Field semantics, clamps, and camera
dimensions still come from the fixed-schema ``-E`` args, unchanged.

Commented-out YAML list entries (``# /foo/bar,`` inside the flow sequence)
are simply absent from the parsed list, which is exactly how the ROS2 node's
own operators already toggle topics in this file today.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from inspect_robots_rosboard._config import ConfigError


@dataclass(frozen=True)
class TopicsFile:
    """A parsed ``topics_to_subscribe.yaml``: the rosboard URL plus two topic sets."""

    url: str
    topics: frozenset[str]
    topics_to_stream: frozenset[str]


def load_topics_file(path: str) -> TopicsFile:
    """Read and validate a ``topics_to_subscribe.yaml`` file at ``path``."""
    try:
        raw_text = Path(path).read_text()
    except OSError as exc:
        raise ConfigError(f"could not read topics file at {path!r}: {exc}") from exc
    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"topics file at {path!r} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"topics file at {path!r} must be a YAML mapping at the top level")

    url = raw.get("url")
    if not isinstance(url, str) or not url:
        raise ConfigError(f"{path!r}: missing or empty required field 'url'")

    return TopicsFile(
        url=url,
        topics=_topic_set(raw.get("topics"), path, "topics"),
        topics_to_stream=_topic_set(raw.get("topics_to_stream"), path, "topics_to_stream"),
    )


def _topic_set(value: Any, path: str, field: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ConfigError(f"{path!r}: {field!r} must be a list of non-empty strings")
    return frozenset(value)
