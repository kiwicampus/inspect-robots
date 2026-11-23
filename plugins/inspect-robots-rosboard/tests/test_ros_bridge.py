"""Tests for the local ROS 2 bridge and the per trial rosbag, with rclpy faked."""

from __future__ import annotations

import base64
import io
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _stub_server import StubRosboardServer
from PIL import Image as PILImage
from test_config import _MINIMAL_CONFIG, _write_config
from test_rosboard_embodiment import _SCENE

from inspect_robots.rollout import TrialRecord
from inspect_robots_rosboard import RosboardEmbodiment
from inspect_robots_rosboard._client import RosboardClient
from inspect_robots_rosboard._config import ConfigError, RosBridgeSpec, load_robot_config
from inspect_robots_rosboard._ros_bridge import (
    DEFAULT_BAG_EXCLUDE,
    RosbagRecorder,
    RosBridge,
    local_topic,
    payload_to_fields,
)

_JPEG = base64.b64encode(b"\xff\xd8fake-jpeg\xff\xd9").decode()


class _FakeMessage:
    def __init__(self, topic_type: str, fields: dict[str, Any]) -> None:
        self.topic_type = topic_type
        self.fields = fields


class _FakePublisher:
    def __init__(self, topic: str, qos: Any) -> None:
        self.topic = topic
        self.qos = qos
        self.messages: list[_FakeMessage] = []

    def publish(self, message: _FakeMessage) -> None:
        self.messages.append(message)


class _FakeNode:
    def __init__(self) -> None:
        self.publishers: dict[str, _FakePublisher] = {}
        self.destroyed = False

    def create_publisher(self, msg_type: Any, topic: str, qos: Any) -> _FakePublisher:
        assert msg_type is _FakeMessage
        publisher = _FakePublisher(topic, qos)
        self.publishers[topic] = publisher
        return publisher

    def destroy_node(self) -> None:
        self.destroyed = True


class _FakeRos:
    """The parts of rclpy, rclpy.qos and rclpy_message_converter the bridge touches."""

    def __init__(self) -> None:
        self.node = _FakeNode()
        self.shutdowns = 0

    def modules(self) -> dict[str, types.ModuleType]:
        rclpy = types.ModuleType("rclpy")
        rclpy.Context = object  # type: ignore[attr-defined]
        rclpy.init = lambda context: None  # type: ignore[attr-defined]
        rclpy.create_node = lambda name, context: self.node  # type: ignore[attr-defined]

        def shutdown(context: Any) -> None:
            self.shutdowns += 1

        rclpy.shutdown = shutdown  # type: ignore[attr-defined]
        qos = types.ModuleType("rclpy.qos")
        qos.QoSProfile = lambda **kw: kw  # type: ignore[attr-defined]
        qos.QoSDurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL="transient_local")  # type: ignore[attr-defined]
        qos.QoSReliabilityPolicy = types.SimpleNamespace(RELIABLE="reliable")  # type: ignore[attr-defined]
        converter = types.ModuleType("rclpy_message_converter.message_converter")

        def convert(topic_type: str, fields: dict[str, Any], strict_mode: bool) -> _FakeMessage:
            assert strict_mode is False
            if fields.get("explode"):
                raise ValueError("bad field")
            return _FakeMessage(topic_type, fields)

        converter.convert_dictionary_to_ros_message = convert  # type: ignore[attr-defined]
        return {
            "rclpy": rclpy,
            "rclpy.qos": qos,
            "rclpy_message_converter": types.ModuleType("rclpy_message_converter"),
            "rclpy_message_converter.message_converter": converter,
        }


@pytest.fixture()
def fake_ros(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeRos]:
    ros = _FakeRos()
    for name, module in ros.modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    yield ros


def _wait(predicate: Callable[[], bool], timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def _png(cells: np.ndarray[Any, Any]) -> str:
    buffer = io.BytesIO()
    PILImage.fromarray(cells.astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


# --------------------------------------------------------------------------
# conversion helpers: no ROS needed
# --------------------------------------------------------------------------


def test_images_become_compressed_on_their_own_topic() -> None:
    for kind in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"):
        assert local_topic("/cam", kind) == ("/cam/compressed", "sensor_msgs/msg/CompressedImage")
        payload = {"_topic_type": kind, "_data_jpeg": _JPEG, "header": {"frame_id": "c"}}
        out_type, fields = payload_to_fields(kind, payload)
        assert out_type == "sensor_msgs/msg/CompressedImage"
        assert fields == {"header": {"frame_id": "c"}, "format": "jpeg", "data": _JPEG}
    assert local_topic("/odom", "nav_msgs/msg/Odometry") == ("/odom", "nav_msgs/msg/Odometry")


def test_other_types_drop_rosboard_fields() -> None:
    payload = {"_topic_name": "/odom", "_time": 1.0, "pose": {"x": 1}}
    assert payload_to_fields("nav_msgs/msg/Odometry", payload) == (
        "nav_msgs/msg/Odometry",
        {"pose": {"x": 1}},
    )


@pytest.mark.parametrize("kind", ["sensor_msgs/msg/Image", "nav_msgs/msg/OccupancyGrid"])
def test_binary_payload_without_data_is_rejected(kind: str) -> None:
    with pytest.raises(ValueError, match="_data_jpeg"):
        payload_to_fields(kind, {"_topic_type": kind})


@pytest.mark.parametrize("channels", [1, 3])
def test_occupancy_grid_undoes_rosboard_subsampling(channels: int) -> None:
    cells = np.array([[0, 100], [255, 50]], dtype=np.uint8)
    image = cells if channels == 1 else np.repeat(cells[..., None], 3, axis=2)
    payload = {
        "_data_jpeg": _png(image),
        "header": {"frame_id": "odom"},
        "info": {"width": 8, "height": 8, "resolution": 0.05},
    }
    out_type, fields = payload_to_fields("nav_msgs/msg/OccupancyGrid", payload)
    assert out_type == "nav_msgs/msg/OccupancyGrid"
    assert fields["info"] == {"width": 2, "height": 2, "resolution": pytest.approx(0.2)}
    assert fields["data"] == [0, 100, -1, 50]
    assert fields["header"] == {"frame_id": "odom"}


# --------------------------------------------------------------------------
# RosBridge
# --------------------------------------------------------------------------


def test_bridge_needs_ros(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rclpy", None)
    with pytest.raises(RuntimeError, match="source ROS 2"):
        RosBridge()


def test_bridge_republishes_samples_and_mirrors_commands(fake_ros: _FakeRos) -> None:
    notices: list[str] = []
    bridge = RosBridge(command_prefix="/eval/", notice=notices.append)
    image = {"_topic_type": "sensor_msgs/msg/Image", "_data_jpeg": _JPEG}
    bridge.on_sample("/cam", image, 0.0)
    bridge.on_sample("/tf_static", {"_topic_type": "tf2_msgs/msg/TFMessage", "transforms": []}, 0.0)
    bridge.on_sample("/untyped", {"x": 1}, 0.0)  # no _topic_type: skipped
    bridge.on_command("/cmd", "geometry_msgs/msg/TwistStamped", {"twist": {"linear": {"x": 0.2}}})
    publishers = fake_ros.node.publishers
    _wait(lambda: {"/cam/compressed", "/tf_static", "/eval/cmd"} <= set(publishers))
    assert publishers["/cam/compressed"].messages[0].fields["data"] == _JPEG
    assert publishers["/tf_static"].qos["durability"] == "transient_local"
    assert "durability" not in publishers["/cam/compressed"].qos
    command = publishers["/eval/cmd"].messages[0]
    assert command.topic_type == "geometry_msgs/msg/TwistStamped"
    assert command.fields == {"twist": {"linear": {"x": 0.2}}}
    assert "/untyped" not in publishers
    bridge.on_sample("/cam", image, 1.0)
    _wait(lambda: len(publishers["/cam/compressed"].messages) == 2)  # publisher reused
    assert notices == []
    bridge.close()
    bridge.close()
    assert fake_ros.node.destroyed and fake_ros.shutdowns == 1


def test_bridge_warns_once_per_broken_topic(fake_ros: _FakeRos) -> None:
    notices: list[str] = []
    bridge = RosBridge(notice=notices.append)
    for _ in range(2):
        bridge.on_sample("/cam", {"_topic_type": "sensor_msgs/msg/Image"}, 0.0)
        bridge.on_sample("/odom", {"_topic_type": "nav_msgs/msg/Odometry", "explode": 1}, 0.0)
        time.sleep(0.05)
    bridge.close()
    assert len(notices) == 2
    assert any("/cam" in n and "_data_jpeg" in n for n in notices)
    assert any("could not republish /odom" in n for n in notices)
    assert fake_ros.node.publishers == {}


def test_client_publish_listener_runs_after_each_send(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    sent: list[tuple[str, str, dict[str, Any]]] = []

    def broken(topic: str, topic_type: str, fields: Any) -> None:
        raise RuntimeError("listener bug")

    client.add_publish_listener(broken)
    client.add_publish_listener(lambda t, k, f: sent.append((t, k, dict(f))))
    client.connect()
    client.publish("/cmd", "geometry_msgs/msg/Twist", {"linear": {"x": 1.0}})
    assert sent == [("/cmd", "geometry_msgs/msg/Twist", {"linear": {"x": 1.0}})]
    client.close()


# --------------------------------------------------------------------------
# RosbagRecorder
# --------------------------------------------------------------------------


class _FakeProcess:
    def __init__(self, hang: bool = False) -> None:
        self.pid = 4242
        self.hang = hang
        self.waits = 0

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        if self.hang and timeout is not None:
            raise subprocess.TimeoutExpired("ros2", timeout)
        return 0


def _recorder(
    monkeypatch: pytest.MonkeyPatch, process: _FakeProcess
) -> tuple[RosbagRecorder, list[list[str]], list[int]]:
    commands: list[list[str]] = []
    signals: list[int] = []

    def popen(command: list[str], **kwargs: Any) -> _FakeProcess:
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] is subprocess.DEVNULL
        commands.append(command)
        return process

    def killpg(pid: int, sig: int) -> None:
        assert pid == process.pid
        signals.append(sig)
        if len(signals) > 2:
            raise ProcessLookupError

    monkeypatch.setattr("inspect_robots_rosboard._ros_bridge.os.killpg", killpg)
    return RosbagRecorder(popen=popen, stop_timeout_s=0.1), commands, signals


def test_bag_records_every_local_topic_and_stops_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import signal

    recorder, commands, signals = _recorder(monkeypatch, _FakeProcess())
    assert recorder.stop() is None
    path = tmp_path / "bags" / "run1" / "s0-e0"
    recorder.start(path)
    assert commands[0][:5] == ["ros2", "bag", "record", "-a", "-x"]
    assert commands[0][5] == DEFAULT_BAG_EXCLUDE
    assert commands[0][-2:] == ["-o", str(path)]
    assert "mcap" in commands[0] and "zstd_fast" in commands[0]
    assert (tmp_path / "bags" / "run1" / "s0-e0.record.log").exists()
    assert recorder.stop() == path
    assert signals == [signal.SIGINT]


def test_bag_is_closed_at_exit_until_stopped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registered: list[Any] = []
    monkeypatch.setattr("atexit.register", registered.append)
    monkeypatch.setattr("atexit.unregister", registered.remove)
    recorder, _commands, signals = _recorder(monkeypatch, _FakeProcess())
    recorder.start(tmp_path / "a")
    assert registered == [recorder.stop]
    registered[0]()  # what the interpreter runs on exit, e.g. after Ctrl+C
    assert registered == [] and len(signals) == 1


def test_bag_that_ignores_sigint_is_killed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import signal

    process = _FakeProcess(hang=True)
    recorder, _commands, signals = _recorder(monkeypatch, process)
    recorder.start(tmp_path / "a")
    recorder.start(tmp_path / "b")  # a second start closes the first bag
    assert signals == [signal.SIGINT, signal.SIGKILL]
    assert recorder.stop() == tmp_path / "b"  # third signal: the process is already gone
    assert process.waits == 4


# --------------------------------------------------------------------------
# config and embodiment wiring
# --------------------------------------------------------------------------

_BRIDGE_CONFIG = (
    _MINIMAL_CONFIG
    + """
ros_bridge:
  enabled: ${TEST_BRIDGE:-1}
  viz: ${TEST_VIZ:-0}
  viz_topics: [/tf, /tf_static, /tf]
  bag: "yes"
"""
)


def test_config_parses_ros_bridge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_robot_config(_write_config(tmp_path)).ros_bridge == RosBridgeSpec()
    monkeypatch.setenv("TEST_VIZ", "true")
    spec = load_robot_config(_write_config(tmp_path, _BRIDGE_CONFIG)).ros_bridge
    assert spec == RosBridgeSpec(enabled=True, viz=True, viz_topics=("/tf", "/tf_static"), bag=True)
    monkeypatch.setenv("TEST_BRIDGE", "")  # off turns viz and bag off too
    spec = load_robot_config(_write_config(tmp_path, _BRIDGE_CONFIG)).ros_bridge
    assert (spec.enabled, spec.viz, spec.bag) == (False, False, False)


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ("ros_bridge: [1]", "must be a mapping"),
        ("ros_bridge: {viz_topics: /tf}", "viz_topics"),
        ("ros_bridge: {command_prefix: eval}", "command_prefix"),
        ("ros_bridge: {bag_exclude: 3}", "bag_exclude"),
        ("ros_bridge: {viz: 2}", "boolean"),
        ("ros_bridge: {enabled: maybe}", "boolean"),
    ],
)
def test_config_rejects_bad_ros_bridge(tmp_path: Path, block: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_robot_config(_write_config(tmp_path, _MINIMAL_CONFIG + block + "\n"))


def test_embodiment_bridges_subscribes_viz_and_bags_each_trial(
    fake_ros: _FakeRos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_VIZ", "1")
    embodiment = RosboardEmbodiment(config=_write_config(tmp_path, _BRIDGE_CONFIG))
    assert embodiment._bridge is not None and embodiment._bag is not None
    assert embodiment._viz_topics == ("/tf", "/tf_static")
    client = embodiment._client
    assert embodiment._bridge.on_sample in client._listeners
    assert embodiment._bridge.on_command in client._publish_listeners

    subscribed: list[str] = []
    monkeypatch.setattr(client, "subscribe", subscribed.append)
    embodiment._subscribe_recorded(("/odometry/local",))
    assert subscribed == ["/tf", "/tf_static"]

    recorder, commands, _signals = _recorder(monkeypatch, _FakeProcess())
    embodiment._bag = recorder
    embodiment.on_trial_start("s/0", 0, str(tmp_path), "run1")
    assert commands[0][-1] == str(tmp_path / "bags" / "run1" / "s_0-e0")
    record = TrialRecord(scene_id="s/0", epoch=0, seed=0)
    embodiment.on_trial_end(record, str(tmp_path), "run1")
    assert record.metadata["rosbag"] == "bags/run1/s_0-e0"
    assert "telemetry" not in record.metadata

    embodiment._bag.start(tmp_path / "late")
    embodiment.close()  # stops the bag and the bridge
    assert fake_ros.shutdowns == 1
    assert embodiment._bag.stop() is None


def test_viz_topics_must_be_in_the_topics_file(fake_ros: _FakeRos, tmp_path: Path) -> None:
    del fake_ros
    topics = tmp_path / "topics.yaml"
    topics.write_text(
        "url: ws://robot:80\n"
        "topics: [/camera/color/image_raw, /odometry/local, /tf]\n"
        "topics_to_stream: []\n"
    )
    config = _BRIDGE_CONFIG.replace("${TEST_VIZ:-0}", "1")
    with pytest.raises(ValueError, match=r"ros_bridge.viz_topics \['/tf_static'\]"):
        RosboardEmbodiment(config=_write_config(tmp_path, config), topics_file=str(topics))


def test_bridge_off_leaves_the_embodiment_untouched(tmp_path: Path) -> None:
    config = _BRIDGE_CONFIG.replace("${TEST_BRIDGE:-1}", "0")
    embodiment = RosboardEmbodiment(config=_write_config(tmp_path, config))
    assert embodiment._bridge is None and embodiment._bag is None
    assert embodiment._viz_topics == ()
    record = TrialRecord(scene_id=_SCENE.id, epoch=0, seed=0)
    embodiment.on_trial_start(_SCENE.id, 0, str(tmp_path), "run1")
    embodiment.on_trial_end(record, str(tmp_path), "run1")
    assert record.metadata == {}


def test_bridge_worker_does_not_block_the_receive_thread(fake_ros: _FakeRos) -> None:
    gate = threading.Event()
    original = fake_ros.node.create_publisher

    def slow(msg_type: Any, topic: str, qos: Any) -> _FakePublisher:
        gate.wait(2.0)
        return original(msg_type, topic, qos)

    fake_ros.node.create_publisher = slow  # type: ignore[method-assign]
    bridge = RosBridge()
    start = time.monotonic()
    for seq in range(50):
        bridge.on_sample("/odom", {"_topic_type": "nav_msgs/msg/Odometry", "seq": seq}, 0.0)
    assert time.monotonic() - start < 0.5
    gate.set()
    _wait(lambda: "/odom" in fake_ros.node.publishers)
    _wait(lambda: fake_ros.node.publishers["/odom"].messages[-1].fields["seq"] == 49)
    assert len(fake_ros.node.publishers["/odom"].messages) <= 2  # only the newest is kept
    bridge.close()
