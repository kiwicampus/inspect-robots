"""Build and parse the rosboard websocket wire protocol.

Every frame is a JSON **array** ``[identifier, payload]`` (rosbridge, by
contrast, uses ``{"op": ...}`` objects; this is the single biggest structural
difference from ``inspect_robots_ros._protocol``). Identifiers used here:
``"s"`` subscribe, ``"u"`` unsubscribe, ``"m"`` message (both directions).
``"t"`` (a server->client topic announcement, sent once shortly after
connect) is received but ignored: this adapter already knows its topics from
configuration, so there is nothing to discover.

Unlike rosbridge, rosboard has no ``status`` operation: there is no
asynchronous error report for a bad publish. A malformed or type-mismatched
command can silently no-op server-side; only a malformed frame or a dropped
socket surfaces here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

JsonObject = dict[str, Any]
Frame = list[Any]


class RosboardError(Exception):
    """A rosboard protocol failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class TopicMessage:
    """One ``"m"`` frame in either direction: topic name plus its full payload."""

    topic_name: str
    payload: JsonObject


def subscribe(topic: str) -> Frame:
    """Build a subscribe frame: ``["s", {"topicName": topic}]``."""
    return ["s", {"topicName": topic}]


def unsubscribe(topic: str) -> Frame:
    """Build an unsubscribe frame: ``["u", {"topicName": topic}]``."""
    return ["u", {"topicName": topic}]


def publish(topic: str, topic_type: str, fields: Mapping[str, Any]) -> Frame:
    """Build a command frame carrying ``_topic_name``/``_topic_type`` plus fields."""
    payload: JsonObject = {"_topic_name": topic, "_topic_type": topic_type, **dict(fields)}
    return ["m", payload]


def encode_frame(frame: Frame) -> str:
    """Encode one ``[identifier, payload]`` frame as compact JSON text."""
    try:
        return json.dumps(frame, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RosboardError("invalid_frame", f"could not encode JSON frame: {exc}") from exc


def decode_frame(raw: str | bytes) -> Frame:
    """Decode and validate one ``[identifier, payload]`` websocket JSON array."""
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RosboardError("invalid_frame", f"could not decode JSON frame: {exc}") from exc
    if not isinstance(decoded, list) or len(decoded) != 2:
        raise RosboardError("invalid_frame", "rosboard frame must be a 2-element JSON array")
    if not isinstance(decoded[0], str):
        raise RosboardError("invalid_frame", "rosboard frame identifier must be a string")
    return decoded


def parse_incoming(frame: Frame) -> TopicMessage | None:
    """Parse ``"m"`` message frames; ``"t"`` and anything else are ignored (returns ``None``)."""
    identifier, payload = frame[0], frame[1]
    if identifier != "m":
        return None
    if not isinstance(payload, Mapping):
        raise RosboardError("invalid_frame", "'m' frame payload must be a JSON object")
    topic_name = payload.get("_topic_name")
    if not isinstance(topic_name, str):
        raise RosboardError("invalid_frame", "'m' frame payload missing string '_topic_name'")
    return TopicMessage(topic_name=topic_name, payload=dict(payload))
