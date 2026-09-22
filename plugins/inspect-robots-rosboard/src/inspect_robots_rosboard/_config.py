"""Parse the optional YAML robot-config file, the plugin's one non-scalar setting.

``docs/guide/adapters.md`` requires scalar-only ``-E key=value`` constructor
kwargs so the setup wizard and CLI stay uniform across embodiments. This file
is what keeps that promise while still letting ``RosboardEmbodiment`` describe
an arbitrary set of observation/action topics: the constructor accepts exactly
one new scalar, ``-E config=path/to/robot.yaml``, and everything structural
lives inside that file, parsed here.

Known, intentional gaps against a general "robot config" schema (see the
plugin README's Configuration-file section for the full explanation of each):
``qos`` blocks are parsed and validated but never applied, since rosboard's
subscribe frame has no reliability/history/depth/throttle knobs at all.
``align.strategy``/``align.stamp`` are parsed but only ``tol_ms`` is honored,
as a per-key staleness bound against wall-clock receive time, not true
cross-topic closest-timestamp buffering: ``RosboardClient`` caches one latest
sample per topic, not a time-indexed history to search. ``actions[].publish.
strategy`` (fifo/nearest pacing) is parsed but unused: the embodiment already
executes one action per closed-loop ``step()`` call, so there is no queued
chunk to re-pace here (that pacing question belongs to whatever plays a
policy's ``ActionChunk``, i.e. ``inspect_robots.controller``, not this
adapter). ``robot_type``, ``max_duration_s``, ``metadata``, and ``recording``
are accepted for forward/tooling compatibility with configs shared across
other systems, but this adapter does not read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """A malformed, inconsistent, or unreadable rosboard robot-config YAML file."""


@dataclass(frozen=True)
class ImageSpec:
    """A camera observation's post-decode target size, ``(height, width)``."""

    height: int
    width: int


@dataclass(frozen=True)
class ObservationSpec:
    """One ``observations[]`` entry: a topic decoded into one observation key.

    Exactly one of ``image``/``selector_names`` is set: ``image`` entries
    populate ``Observation.images[key]``; ``selector_names`` entries populate
    ``Observation.state[key]`` with one scalar per dotted-path name, read from
    the topic's payload in the given order (this is also how the built-in
    xyzw-to-wxyz quaternion reorder is expressed: list ``orientation.w``
    before ``orientation.x``/``y``/``z``, since rosboard already delivers
    orientation as a named dict, not a flat array).
    """

    key: str
    topic: str
    message_type: str
    image: ImageSpec | None
    selector_names: tuple[str, ...] | None
    tol_ms: float


@dataclass(frozen=True)
class ActionSpec:
    """One ``actions[]`` entry: an action slice published to one command topic.

    ``selector_names`` gives the dotted field path each action dimension is
    written to inside the outgoing message dict, in order (e.g.
    ``twist.linear.x``); the embodiment's flat action vector is the
    concatenation of every ``ActionSpec``'s dimensions, in file order.
    """

    key: str
    publish_topic: str
    publish_type: str
    selector_names: tuple[str, ...]
    clamp_low: float
    clamp_high: float
    safety_behavior: str


@dataclass(frozen=True)
class RobotConfig:
    """A fully parsed and validated robot-config YAML file."""

    name: str
    rate_hz: float
    observations: tuple[ObservationSpec, ...]
    actions: tuple[ActionSpec, ...]


_SUPPORTED_SAFETY_BEHAVIORS = frozenset({"zeros"})


def load_robot_config(path: str) -> RobotConfig:
    """Read and validate a robot-config YAML file at ``path``."""
    try:
        raw_text = Path(path).read_text()
    except OSError as exc:
        raise ConfigError(f"could not read rosboard robot config at {path!r}: {exc}") from exc
    try:
        raw = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"rosboard robot config at {path!r} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(
            f"rosboard robot config at {path!r} must be a YAML mapping at the top level"
        )

    name = _require_str(raw, "name", path)
    rate_hz = float(_require(raw, "rate_hz", path))
    if rate_hz <= 0:
        raise ConfigError(f"{path!r}: rate_hz must be positive, got {rate_hz!r}")

    observations_raw = _require(raw, "observations", path)
    if not isinstance(observations_raw, list) or not observations_raw:
        raise ConfigError(f"{path!r}: observations must be a non-empty list")
    observations = tuple(_parse_observation(entry, path) for entry in observations_raw)
    _require_unique_keys(observations, path, "observations")

    actions_raw = _require(raw, "actions", path)
    if not isinstance(actions_raw, list) or not actions_raw:
        raise ConfigError(f"{path!r}: actions must be a non-empty list")
    actions = tuple(_parse_action(entry, path) for entry in actions_raw)
    _require_unique_keys(actions, path, "actions")

    return RobotConfig(name=name, rate_hz=rate_hz, observations=observations, actions=actions)


def _parse_observation(entry: Any, path: str) -> ObservationSpec:
    if not isinstance(entry, dict):
        raise ConfigError(f"{path!r}: each observations[] entry must be a mapping")
    key = _require_str(entry, "key", path)
    topic = _require_str(entry, "topic", path)
    message_type = _require_str(entry, "type", path)

    image_raw = entry.get("image")
    selector_raw = entry.get("selector")
    if (image_raw is None) == (selector_raw is None):
        raise ConfigError(
            f"{path!r}: observation {key!r} must set exactly one of 'image' or 'selector'"
        )

    image = None
    selector_names = None
    if image_raw is not None:
        if not isinstance(image_raw, dict) or "resize" not in image_raw:
            raise ConfigError(
                f"{path!r}: observation {key!r} needs 'image.resize: [height, width]'; "
                "this adapter declares camera shapes up front and never introspects them"
            )
        resize = image_raw["resize"]
        if (
            not isinstance(resize, (list, tuple))
            or len(resize) != 2
            or not all(isinstance(v, int) and v > 0 for v in resize)
        ):
            raise ConfigError(
                f"{path!r}: observation {key!r}: image.resize must be [height, width] "
                "positive integers"
            )
        image = ImageSpec(height=int(resize[0]), width=int(resize[1]))
    else:
        names = _selector_names(selector_raw, path, key)
        selector_names = names

    align = entry.get("align") or {}
    tol_ms = float(align.get("tol_ms", 0.0)) if isinstance(align, dict) else 0.0
    if tol_ms < 0:
        raise ConfigError(f"{path!r}: observation {key!r}: align.tol_ms must be >= 0")

    return ObservationSpec(
        key=key,
        topic=topic,
        message_type=message_type,
        image=image,
        selector_names=selector_names,
        tol_ms=tol_ms,
    )


def _parse_action(entry: Any, path: str) -> ActionSpec:
    if not isinstance(entry, dict):
        raise ConfigError(f"{path!r}: each actions[] entry must be a mapping")
    key = _require_str(entry, "key", path)
    publish = _require(entry, "publish", path)
    if not isinstance(publish, dict):
        raise ConfigError(f"{path!r}: action {key!r}: 'publish' must be a mapping")
    publish_topic = _require_str(publish, "topic", path)
    publish_type = _require_str(publish, "type", path)

    selector = _require(entry, "selector", path)
    names = _selector_names(selector, path, key)

    from_tensor = entry.get("from_tensor") or {}
    clamp = from_tensor.get("clamp") if isinstance(from_tensor, dict) else None
    if (
        not isinstance(clamp, (list, tuple))
        or len(clamp) != 2
        or not all(isinstance(v, (int, float)) for v in clamp)
    ):
        raise ConfigError(
            f"{path!r}: action {key!r} needs 'from_tensor.clamp: [low, high]'; this adapter "
            "hard-clamps every published action independent of any framework guardrail"
        )
    clamp_low, clamp_high = float(clamp[0]), float(clamp[1])
    if not clamp_low < clamp_high:
        raise ConfigError(
            f"{path!r}: action {key!r}: from_tensor.clamp low must be < high, "
            f"got [{clamp_low!r}, {clamp_high!r}]"
        )

    safety_behavior = str(entry.get("safety_behavior", "zeros"))
    if safety_behavior not in _SUPPORTED_SAFETY_BEHAVIORS:
        raise ConfigError(
            f"{path!r}: action {key!r}: safety_behavior {safety_behavior!r} is not supported, "
            f"only {sorted(_SUPPORTED_SAFETY_BEHAVIORS)} (publishes on close())"
        )

    return ActionSpec(
        key=key,
        publish_topic=publish_topic,
        publish_type=publish_type,
        selector_names=names,
        clamp_low=clamp_low,
        clamp_high=clamp_high,
        safety_behavior=safety_behavior,
    )


def _selector_names(selector: Any, path: str, key: str) -> tuple[str, ...]:
    if not isinstance(selector, dict) or "names" not in selector:
        raise ConfigError(f"{path!r}: {key!r} needs 'selector.names: [field.path, ...]'")
    names = selector["names"]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n for n in names):
        raise ConfigError(f"{path!r}: {key!r}: selector.names must be a non-empty list of strings")
    return tuple(names)


def _require(mapping: dict[str, Any], field_name: str, path: str) -> Any:
    if field_name not in mapping:
        raise ConfigError(f"{path!r}: missing required field {field_name!r}")
    return mapping[field_name]


def _require_str(mapping: dict[str, Any], field_name: str, path: str) -> str:
    value = _require(mapping, field_name, path)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{path!r}: field {field_name!r} must be a non-empty string")
    return value


def _require_unique_keys(entries: tuple[Any, ...], path: str, section: str) -> None:
    keys = [entry.key for entry in entries]
    if len(keys) != len(set(keys)):
        raise ConfigError(f"{path!r}: {section}[].key values must be unique, got {keys!r}")
