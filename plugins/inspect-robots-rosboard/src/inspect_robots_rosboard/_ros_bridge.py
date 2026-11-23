"""Republish the rosboard connection's traffic into local ROS 2, and record it to a rosbag.

Optional: rclpy and ``rclpy_message_converter`` are imported only when a bridge is built,
so the plugin still runs on a machine without ROS.
"""

from __future__ import annotations

import atexit
import base64
import io
import os
import signal
import subprocess
import threading
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

_IMAGE_TYPES = frozenset({"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"})
_GRID_TYPE = "nav_msgs/msg/OccupancyGrid"
_COMPRESSED_TYPE = "sensor_msgs/msg/CompressedImage"
# Never bagged: ROS 2 bookkeeping, and RViz's own interaction topics.
DEFAULT_BAG_EXCLUDE = r"^/(rosout|parameter_events|clicked_point|goal_pose|initialpose)$"


def local_topic(topic: str, topic_type: str) -> tuple[str, str]:
    """The local name and type a rosboard topic is republished as.

    Images go out as the JPEG rosboard already sent, on ``<topic>/compressed``.
    """
    if topic_type in _IMAGE_TYPES:
        return f"{topic}/compressed", _COMPRESSED_TYPE
    return topic, topic_type


def payload_to_fields(topic_type: str, payload: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Turn a rosboard payload into ``(local type, message dict)`` for the converter.

    Byte arrays stay base64, the converter's encoding for ``uint8[]``. Raises
    ``ValueError`` when an image or grid payload has no ``_data_jpeg``.
    """
    fields = {k: v for k, v in payload.items() if not k.startswith("_")}
    if topic_type in _IMAGE_TYPES:
        jpeg = payload.get("_data_jpeg")
        if not isinstance(jpeg, str):
            raise ValueError("image payload has no _data_jpeg")
        return _COMPRESSED_TYPE, {
            "header": fields.get("header", {}),
            "format": "jpeg",
            "data": jpeg,
        }
    if topic_type == _GRID_TYPE:
        return _GRID_TYPE, _grid_fields(fields, payload.get("_data_jpeg"))
    return topic_type, fields


def _grid_fields(fields: dict[str, Any], png: Any) -> dict[str, Any]:
    """Undo rosboard's PNG subsampling of an OccupancyGrid, as rosboard_client does."""
    import numpy as np
    from PIL import Image as PILImage

    if not isinstance(png, str):
        raise ValueError("occupancy grid payload has no _data_jpeg")
    cells = np.asarray(PILImage.open(io.BytesIO(base64.b64decode(png))))
    if cells.ndim == 3:
        cells = cells[..., 0]
    height, width = cells.shape
    info = dict(fields.get("info", {}))
    original_width = float(info.get("width") or width)
    info["resolution"] = float(info.get("resolution", 0.0)) * original_width / width
    info["width"], info["height"] = width, height
    return {**fields, "info": info, "data": cells.astype(np.int8).flatten("C").tolist()}


class RosBridge:
    """Publish every received sample, and every command sent, on a local ROS 2 node.

    The rosboard receive thread only stores the latest sample per topic; a worker thread
    converts and publishes, so a slow conversion never delays observations.
    """

    def __init__(
        self,
        *,
        command_prefix: str = "/eval",
        node_name: str = "inspect_robots_rosboard_bridge",
        notice: Callable[[str], None] = print,
    ) -> None:
        try:
            import rclpy
            from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
            from rclpy_message_converter.message_converter import (
                convert_dictionary_to_ros_message,
            )
        except ImportError as exc:
            raise RuntimeError(
                "ros_bridge needs rclpy and rclpy_message_converter: source ROS 2 before "
                "the eval, or disable the bridge (ros_bridge.enabled: 0)"
            ) from exc
        self._convert = convert_dictionary_to_ros_message
        self._qos_default = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        self._qos_static = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._rclpy = rclpy
        self._context = rclpy.Context()
        rclpy.init(context=self._context)
        self._node = rclpy.create_node(node_name, context=self._context)
        self.command_prefix = command_prefix.rstrip("/")
        self._notice = notice
        self._publishers: dict[tuple[str, str], Any] = {}
        self._warned: set[str] = set()
        self._pending: dict[str, tuple[str, str, Mapping[str, Any]]] = {}
        self._commands: list[tuple[str, str, Mapping[str, Any]]] = []
        self._wake = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run, name="inspect-robots-rosboard-ros-bridge", daemon=True
        )
        self._worker.start()

    def on_sample(self, topic: str, payload: dict[str, Any], stamp: float) -> None:
        """Client listener: keep the newest sample of ``topic`` for the worker."""
        del stamp
        topic_type = payload.get("_topic_type")
        if not isinstance(topic_type, str):
            return
        with self._wake:
            self._pending[topic] = (topic, topic_type, payload)
            self._wake.notify()

    def on_command(self, topic: str, topic_type: str, fields: Mapping[str, Any]) -> None:
        """Client publish listener: mirror a command sent to the robot under the prefix."""
        with self._wake:
            self._commands.append((self.command_prefix + topic, topic_type, dict(fields)))
            self._wake.notify()

    def close(self) -> None:
        """Flush, stop the worker, and shut the node down; idempotent."""
        with self._wake:
            if self._closed:
                return
            self._closed = True
            self._wake.notify()
        self._worker.join(timeout=2.0)
        with suppress(Exception):
            self._node.destroy_node()
        with suppress(Exception):
            self._rclpy.shutdown(context=self._context)

    def _run(self) -> None:
        while True:
            with self._wake:
                while not self._pending and not self._commands and not self._closed:
                    self._wake.wait()
                samples = list(self._pending.values())
                commands = self._commands
                self._pending, self._commands = {}, []
                closed = self._closed
            for topic, topic_type, payload in samples:
                self._publish_sample(topic, topic_type, payload)
            for topic, topic_type, fields in commands:
                self._publish(topic, topic_type, dict(fields))
            if closed:
                return

    def _publish_sample(self, topic: str, topic_type: str, payload: Mapping[str, Any]) -> None:
        try:
            out_type, fields = payload_to_fields(topic_type, payload)
        except ValueError as exc:
            self._warn_once(topic, f"[ros bridge] {topic}: {exc}")
            return
        self._publish(local_topic(topic, topic_type)[0], out_type, fields)

    def _publish(self, topic: str, topic_type: str, fields: dict[str, Any]) -> None:
        try:
            message = self._convert(topic_type, fields, strict_mode=False)
            key = (topic, topic_type)
            publisher = self._publishers.get(key)
            if publisher is None:
                qos = self._qos_static if topic == "/tf_static" else self._qos_default
                publisher = self._node.create_publisher(type(message), topic, qos)
                self._publishers[key] = publisher
            publisher.publish(message)
        except Exception as exc:
            text = f"[ros bridge] could not republish {topic} ({topic_type}): {exc}"
            self._warn_once(topic, text)

    def _warn_once(self, topic: str, text: str) -> None:
        if topic not in self._warned:
            self._warned.add(topic)
            self._notice(text)


class RosbagRecorder:
    """Run ``ros2 bag record`` on every local topic for the length of one trial."""

    def __init__(
        self,
        *,
        exclude: str = DEFAULT_BAG_EXCLUDE,
        popen: Callable[..., Any] = subprocess.Popen,
        stop_timeout_s: float = 10.0,
    ) -> None:
        self.exclude = exclude
        self._popen = popen
        self._stop_timeout_s = stop_timeout_s
        self._process: Any = None
        self._path: Path | None = None
        self._log: Any = None

    def start(self, path: Path) -> None:
        """Start recording into ``path`` (a bag directory that must not exist yet)."""
        self.stop()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._log = (path.parent / f"{path.name}.record.log").open("w")
        command = ["ros2", "bag", "record", "-a", "-x", self.exclude, "-s", "mcap"]
        command += ["--storage-preset-profile", "zstd_fast", "-o", str(path)]
        # No terminal stdin: the recorder's keyboard controls would eat the operator's keys.
        self._process = self._popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._path = path
        # Its own session misses Ctrl+C, so close the bag when the interpreter exits.
        atexit.register(self.stop)

    def stop(self) -> Path | None:
        """Stop with SIGINT so the bag is closed cleanly; returns its path, if recording."""
        process, path, log = self._process, self._path, self._log
        self._process = self._path = self._log = None
        if process is None:
            return None
        atexit.unregister(self.stop)
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=self._stop_timeout_s)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if log is not None:
            log.close()
        return path
