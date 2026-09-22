"""Tests for the rosboard embodiment adapter and its wire protocol."""

from __future__ import annotations

import base64
import threading
import time
from collections.abc import Callable, Mapping
from io import BytesIO
from typing import Any, cast

import numpy as np
import pytest
from _stub_server import StubRosboardServer
from PIL import Image

from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from inspect_robots_rosboard import RosboardEmbodiment, rosboard_embodiment
from inspect_robots_rosboard._client import RosboardClient, TopicSample, _rosboard_socket_url
from inspect_robots_rosboard._msgs import (
    build_twist,
    parse_compressed_image,
    parse_imu,
    parse_odometry,
)
from inspect_robots_rosboard._protocol import (
    RosboardError,
    TopicMessage,
    decode_frame,
    encode_frame,
    parse_incoming,
    publish,
    subscribe,
    unsubscribe,
)

_SCENE = Scene(id="s0", instruction="drive forward")


# --------------------------------------------------------------------------
# Layer 1: protocol golden tests
# --------------------------------------------------------------------------


def test_rosboard_socket_url_normalization() -> None:
    assert _rosboard_socket_url("ws://host:80") == "ws://host:80/rosboard/v1"
    assert _rosboard_socket_url("wss://host:443") == "wss://host:443/rosboard/v1"
    assert _rosboard_socket_url("192.168.0.1:8888") == "ws://192.168.0.1:8888/rosboard/v1"
    assert _rosboard_socket_url("host:443") == "wss://host:443/rosboard/v1"
    assert _rosboard_socket_url("no-port-here") == "ws://no-port-here/rosboard/v1"


def test_protocol_outbound_frames_match_rosboard_shapes() -> None:
    assert subscribe("/odometry/local") == ["s", {"topicName": "/odometry/local"}]
    assert unsubscribe("/odometry/local") == ["u", {"topicName": "/odometry/local"}]
    assert publish("/cmd", "geometry_msgs/msg/Twist", {"linear": {"x": 1.0}}) == [
        "m",
        {"_topic_name": "/cmd", "_topic_type": "geometry_msgs/msg/Twist", "linear": {"x": 1.0}},
    ]


def test_encode_frame_rejects_non_json_serializable_payload() -> None:
    with pytest.raises(RosboardError, match="invalid_frame"):
        encode_frame(["m", {"nan": float("nan")}])


def test_encode_decode_frame_round_trip() -> None:
    frame = ["m", {"_topic_name": "/odometry/local", "_topic_type": "nav_msgs/msg/Odometry"}]
    assert decode_frame(encode_frame(frame)) == frame


def test_decode_frame_rejects_malformed_input() -> None:
    with pytest.raises(RosboardError, match="invalid_frame"):
        decode_frame("not json")
    with pytest.raises(RosboardError, match="invalid_frame"):
        decode_frame("{}")
    with pytest.raises(RosboardError, match="invalid_frame"):
        decode_frame("[1, 2, 3]")
    with pytest.raises(RosboardError, match="invalid_frame"):
        decode_frame("[1, {}]")


def test_parse_incoming_returns_topic_message_for_m_frames() -> None:
    parsed = parse_incoming(["m", {"_topic_name": "/odometry/local", "value": 1}])
    assert parsed == TopicMessage(
        topic_name="/odometry/local", payload={"_topic_name": "/odometry/local", "value": 1}
    )


def test_parse_incoming_ignores_topics_announcement_and_unknown_identifiers() -> None:
    assert parse_incoming(["t", {"/odometry/local": "nav_msgs/msg/Odometry"}]) is None
    assert parse_incoming(["q", {}]) is None


def test_parse_incoming_rejects_malformed_m_payload() -> None:
    with pytest.raises(RosboardError, match="invalid_frame"):
        parse_incoming(["m", "not a dict"])
    with pytest.raises(RosboardError, match="invalid_frame"):
        parse_incoming(["m", {"no_topic_name": True}])


# --------------------------------------------------------------------------
# Layer 2: client tests against the real stub server
# --------------------------------------------------------------------------


def test_client_subscribe_receive_latest_and_sequence(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()
    client.subscribe("/odometry/local")
    stub_server.wait_for(lambda frames: any(f[0] == "s" for f in frames))
    stub_server.publish("/odometry/local", "nav_msgs/msg/Odometry", {"seq": 1})

    sample = client.wait_for_sample("/odometry/local", timeout_s=2.0)
    assert sample.payload["seq"] == 1
    assert client.sequence("/odometry/local") == 1
    assert client.latest("/odometry/local") is sample
    client.close()


def test_client_wait_for_sample_times_out(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()
    with pytest.raises(TimeoutError):
        client.wait_for_sample("/never/published", timeout_s=0.1)
    client.close()


def test_client_ignores_topics_announcement(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()
    client.subscribe("/odometry/local")
    stub_server.wait_for(lambda frames: any(f[0] == "s" for f in frames))
    stub_server.send_topics_announcement({"/odometry/local": "nav_msgs/msg/Odometry"})
    time.sleep(0.05)
    assert client.sequence("/odometry/local") == 0
    client.close()


def test_client_malformed_frame_latches_and_surfaces(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()
    stub_server.send_raw("not json")
    time.sleep(0.1)
    with pytest.raises(RosboardError):
        client.latest("/anything")
    client.close()


def test_client_connection_death_latches(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()
    stub_server.drop_connections()
    time.sleep(0.1)
    with pytest.raises(ConnectionError):
        client.latest("/anything")
    client.close()


def test_client_close_is_idempotent(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()
    client.close()
    client.close()


def test_client_properties_and_unsubscribe(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    assert client.connected is False
    assert client.latched_error is None
    client.connect()
    assert client.connected is True
    assert client.receiver_alive is True
    client.subscribe("/odometry/local")
    client.unsubscribe("/odometry/local")
    stub_server.wait_for(lambda frames: any(f[0] == "u" for f in frames))
    client.close()
    assert client.connected is False


def test_client_connect_is_idempotent(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()
    client.connect()  # second call is a no-op, does not reconnect
    client.close()


def test_client_connect_after_close_raises(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()
    client.close()
    with pytest.raises(RuntimeError, match="closed"):
        client.connect()


def test_client_connect_failure_raises_connection_error() -> None:
    client = RosboardClient("ws://127.0.0.1:1", connect_timeout_s=0.5)
    with pytest.raises(ConnectionError):
        client.connect()


def test_client_send_failure_latches_and_raises(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()

    class _BrokenSocket:
        def send(self, _frame: str) -> None:
            raise OSError("broken pipe")

    client._ws = cast(Any, _BrokenSocket())
    with pytest.raises(ConnectionError):
        client.publish("/cmd", "geometry_msgs/msg/Twist", {})
    assert isinstance(client.latched_error, ConnectionError)


def test_client_threaded_send_while_receiving(stub_server: StubRosboardServer) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    client.connect()
    client.subscribe("/odometry/local")
    stub_server.wait_for(lambda frames: any(f[0] == "s" for f in frames))

    stop = threading.Event()

    def publisher() -> None:
        seq = 0
        while not stop.is_set():
            seq += 1
            stub_server.publish("/odometry/local", "nav_msgs/msg/Odometry", {"seq": seq})
            time.sleep(0.005)

    thread = threading.Thread(target=publisher, daemon=True)
    thread.start()
    try:
        for _ in range(20):
            client.publish("/cmd", "geometry_msgs/msg/Twist", build_twist(0.1, 0.0))
        client.wait_for_sample("/odometry/local", timeout_s=2.0)
    finally:
        stop.set()
        thread.join(timeout=2)
    client.close()


# --------------------------------------------------------------------------
# Layer 3: message conversion unit tests
# --------------------------------------------------------------------------


def _image_payload(
    image: np.ndarray[Any, np.dtype[np.uint8]], image_format: str = "JPEG"
) -> dict[str, Any]:
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"format": image_format.lower(), "_data_jpeg": encoded}


def test_parse_compressed_image_round_trips() -> None:
    # PNG (lossless) on a tiny image: JPEG's block-based DCT distorts a 2x3
    # image with a single sharp pixel far beyond a reasonable tolerance.
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = [10, 20, 30]
    parsed = parse_compressed_image(_image_payload(image, image_format="PNG"))
    assert parsed.shape == (2, 3, 3)
    assert parsed.dtype == np.uint8
    np.testing.assert_array_equal(parsed, image)


@pytest.mark.parametrize(
    "payload",
    [
        {"format": "jpeg"},
        {"format": "jpeg", "_data_jpeg": 123},
        {"format": "jpeg", "_data_jpeg": "not base64!!"},
        {"format": "jpeg", "_data_jpeg": base64.b64encode(b"not an image").decode()},
    ],
)
def test_parse_compressed_image_rejects_bad_payloads(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        parse_compressed_image(payload)


def _odometry_payload() -> dict[str, Any]:
    return {
        "pose": {
            "pose": {
                "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            "covariance": [0.0] * 36,
        },
        "twist": {
            "twist": {
                "linear": {"x": 0.5, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.1},
            },
            "covariance": [0.0] * 36,
        },
    }


def test_parse_odometry_reorders_quaternion_to_wxyz() -> None:
    pose, twist = parse_odometry(_odometry_payload())
    np.testing.assert_allclose(pose, [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(twist, [0.5, 0.0, 0.0, 0.0, 0.0, 0.1])


def test_parse_odometry_rejects_missing_fields() -> None:
    with pytest.raises(ValueError):
        parse_odometry({"pose": {}})


def _imu_payload() -> dict[str, Any]:
    return {
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "angular_velocity": {"x": 0.01, "y": 0.02, "z": 0.03},
        "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 9.8},
    }


def test_parse_imu_reorders_quaternion_to_wxyz() -> None:
    parsed = parse_imu(_imu_payload())
    np.testing.assert_allclose(parsed, [1.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.03, 0.0, 0.0, 9.8])


def test_parse_imu_rejects_missing_fields() -> None:
    with pytest.raises(ValueError):
        parse_imu({"orientation": {}})


def test_build_twist_shape() -> None:
    assert build_twist(0.3, -0.2) == {
        "linear": {"x": 0.3, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": -0.2},
    }


# --------------------------------------------------------------------------
# Layer 4: embodiment tests (fake client for fast unit coverage)
# --------------------------------------------------------------------------


def _embodiment(**kwargs: Any) -> RosboardEmbodiment:
    kwargs.setdefault("camera_height", 2)
    kwargs.setdefault("camera_width", 3)
    kwargs.setdefault("obs_timeout_s", 1.0)
    return RosboardEmbodiment(**kwargs)


class _FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.sleep_calls: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        self.sleep_calls.append(duration)
        self.now += duration


class _FakeClient:
    def __init__(self, clock: _FakeClock) -> None:
        self.clock = clock
        self.samples: dict[str, TopicSample] = {}
        self.published: list[tuple[str, str, dict[str, Any]]] = []
        # Fired on every wait_for_sample() call (both reset()'s per-topic
        # freshness wait and step()'s post-publish odometry wait); the default
        # installed by _ready_embodiment simulates a live sensor stream by
        # re-publishing whatever is already cached, bumping its sequence.
        self.on_wait: Callable[[str, int], None] | None = None
        self.closed = False

    def put(self, topic: str, payload: Mapping[str, Any]) -> None:
        previous = self.samples.get(topic)
        seq = 1 if previous is None else previous.seq + 1
        self.samples[topic] = TopicSample(payload=dict(payload), stamp=self.clock(), seq=seq)

    def connect(self) -> None:
        pass

    def subscribe(self, topic: str) -> None:
        pass

    def unsubscribe(self, topic: str) -> None:
        pass

    def publish(self, topic: str, topic_type: str, fields: Mapping[str, Any]) -> None:
        self.published.append((topic, topic_type, dict(fields)))

    def latest(self, topic: str) -> TopicSample | None:
        return self.samples.get(topic)

    def sequence(self, topic: str) -> int:
        sample = self.samples.get(topic)
        return 0 if sample is None else sample.seq

    def wait_for_sample(self, topic: str, *, after_seq: int = 0, timeout_s: float) -> TopicSample:
        if self.on_wait is not None:
            self.on_wait(topic, after_seq)
        sample = self.samples.get(topic)
        if sample is not None and sample.seq > after_seq:
            return sample
        self.clock.sleep(timeout_s)
        raise TimeoutError(topic)

    def close(self) -> None:
        self.closed = True


def _ready_embodiment(
    *, clock: _FakeClock | None = None, **kwargs: Any
) -> tuple[RosboardEmbodiment, _FakeClient, _FakeClock]:
    fake_clock = clock or _FakeClock()
    kwargs.setdefault("clock", fake_clock)
    kwargs.setdefault("sleep", fake_clock.sleep)
    embodiment = _embodiment(**kwargs)
    client = _FakeClient(fake_clock)
    client.put(embodiment.odometry_topic, _odometry_payload())
    client.put(embodiment.imu_topic, _imu_payload())
    image = np.zeros((embodiment.camera_height, embodiment.camera_width, 3), dtype=np.uint8)
    client.put(embodiment.camera_topic, _image_payload(image))

    def fresh(topic: str, _after_seq: int) -> None:
        if topic in client.samples:
            client.put(topic, client.samples[topic].payload)

    client.on_wait = fresh
    embodiment._client = cast(Any, client)
    embodiment._initialized = True
    embodiment._validated_camera = True
    embodiment._instruction = _SCENE.instruction
    return embodiment, client, fake_clock


def test_info_shape_and_capabilities() -> None:
    embodiment = _embodiment()
    info = embodiment.info
    assert info.action_space.shape == (2,)
    np.testing.assert_allclose(info.action_space.low, [-0.5, -0.5])
    np.testing.assert_allclose(info.action_space.high, [0.5, 0.5])
    assert info.action_space.semantics is not None
    assert info.action_space.semantics.control_mode == "base_velocity"
    assert info.action_space.semantics.dim_labels == ("linear_x", "angular_z")
    assert info.observation_space.state is not None
    state_keys = {f.key for f in info.observation_space.state.fields}
    assert state_keys == {"odom_pose", "odom_twist", "imu"}
    assert info.capabilities == frozenset({"self_paced"})
    assert info.is_simulated is False


def test_construction_touches_no_network() -> None:
    # An unroutable URL must not be dialed at construction time.
    _embodiment(url="ws://198.51.100.1:1")


@pytest.mark.parametrize("kwargs", [{"camera_height": None}, {"camera_width": None}])
def test_constructor_requires_camera_resolution(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="camera_height and camera_width"):
        RosboardEmbodiment(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"camera_height": 1.5, "camera_width": 3},
        {"camera_height": 2, "camera_width": 0},
        {"camera_height": -1, "camera_width": 3},
    ],
)
def test_constructor_validates_camera_resolution_type_and_sign(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="camera_height and camera_width"):
        RosboardEmbodiment(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"linear_x_limit": 0.0},
        {"linear_x_limit": -1.0},
        {"angular_z_limit": float("inf")},
        {"control_hz": 0.0},
        {"obs_timeout_s": -1.0},
        {"staleness_s": -1.0},
    ],
)
def test_constructor_validates_numeric_bounds(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _embodiment(**kwargs)


def test_conformance() -> None:
    assert_embodiment_conformant(_embodiment().info)


def test_reset_returns_fresh_observation() -> None:
    embodiment, _client, _clock = _ready_embodiment()
    observation = embodiment.reset(_SCENE)
    assert observation.instruction == "drive forward"
    assert "front" in observation.images
    assert set(observation.state) == {"odom_pose", "odom_twist", "imu"}


def test_reset_missing_topic_times_out() -> None:
    embodiment, client, _clock = _ready_embodiment()
    del client.samples[embodiment.imu_topic]
    with pytest.raises(TimeoutError, match="imu"):
        embodiment.reset(_SCENE)


def test_reset_camera_resolution_mismatch_raises() -> None:
    embodiment, client, _clock = _ready_embodiment()
    embodiment._validated_camera = False
    embodiment._initialized = False
    wrong_image = np.zeros((99, 99, 3), dtype=np.uint8)
    client.put(embodiment.camera_topic, _image_payload(wrong_image))
    with pytest.raises(ValueError, match="declared resolution"):
        embodiment.reset(_SCENE)


def test_reset_warns_once_from_second_reset(capsys: pytest.CaptureFixture[str]) -> None:
    embodiment, _client, _clock = _ready_embodiment()
    embodiment.reset(_SCENE)
    capsys.readouterr()
    embodiment.reset(_SCENE)
    err = capsys.readouterr().err
    assert "no reset mechanism" in err
    embodiment.reset(_SCENE)
    err = capsys.readouterr().err
    assert err == ""


def test_step_clamps_linear_and_angular_to_configured_limits() -> None:
    embodiment, client, _clock = _ready_embodiment(linear_x_limit=0.5, angular_z_limit=0.5)
    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.array([10.0, 10.0])))
    topic, topic_type, fields = client.published[-1]
    assert topic == embodiment.command_topic
    assert topic_type == "geometry_msgs/msg/Twist"
    assert fields["linear"]["x"] == 0.5
    assert fields["angular"]["z"] == 0.5

    embodiment.step(Action(data=np.array([-10.0, -10.0])))
    _topic, _topic_type, fields = client.published[-1]
    assert fields["linear"]["x"] == -0.5
    assert fields["angular"]["z"] == -0.5


def test_step_clamp_is_not_the_frameworks_approver() -> None:
    # No Approver/guardrail chain is constructed anywhere in this test: the
    # clamp observed above comes from the embodiment's own step(), not from
    # ClampApprover/DeltaLimitApprover.
    embodiment, client, _clock = _ready_embodiment(linear_x_limit=0.2, angular_z_limit=0.2)
    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.array([100.0, -100.0])))
    _topic, _topic_type, fields = client.published[-1]
    assert fields["linear"]["x"] == 0.2
    assert fields["angular"]["z"] == -0.2


def test_step_within_bounds_is_passed_through_unclamped() -> None:
    embodiment, client, _clock = _ready_embodiment()
    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.array([0.1, -0.2])))
    _topic, _topic_type, fields = client.published[-1]
    assert fields["linear"]["x"] == pytest.approx(0.1)
    assert fields["angular"]["z"] == pytest.approx(-0.2)


def test_step_rejects_wrong_action_shape() -> None:
    embodiment, _client, _clock = _ready_embodiment()
    embodiment.reset(_SCENE)
    with pytest.raises(ValueError, match="shape"):
        embodiment.step(Action(data=np.array([0.1, 0.1, 0.1])))


def test_step_self_paces_between_publishes() -> None:
    clock = _FakeClock()
    embodiment, _client, _clock = _ready_embodiment(clock=clock, control_hz=10.0)
    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.array([0.1, 0.0])))
    before = clock.now
    embodiment.step(Action(data=np.array([0.1, 0.0])))
    assert clock.now - before >= 1.0 / 10.0 - 1e-9


def test_step_fresh_obs_timeout_names_control_hz() -> None:
    embodiment, client, _clock = _ready_embodiment(control_hz=5.0)
    embodiment.reset(_SCENE)
    client.on_wait = None  # stop simulating a live odometry stream
    with pytest.raises(TimeoutError, match="control_hz=5"):
        embodiment.step(Action(data=np.array([0.1, 0.0])))


def test_step_staleness_raises() -> None:
    embodiment, _client, clock = _ready_embodiment(staleness_s=1.0)
    embodiment.reset(_SCENE)
    clock.now += 10.0
    with pytest.raises(TimeoutError, match="stale"):
        embodiment._assemble_observation()


def test_close_publishes_zero_twist_and_is_idempotent() -> None:
    embodiment, client, _clock = _ready_embodiment()
    embodiment.reset(_SCENE)
    embodiment.close()
    topic, topic_type, fields = client.published[-1]
    assert topic == embodiment.command_topic
    assert topic_type == "geometry_msgs/msg/Twist"
    assert fields["linear"]["x"] == 0.0
    assert fields["angular"]["z"] == 0.0
    assert client.closed is True
    embodiment.close()  # idempotent


def test_close_before_reset_does_not_publish() -> None:
    embodiment = _embodiment()
    embodiment.close()  # never initialized; must not touch the client's publish path


def test_factory_constructs_embodiment() -> None:
    assert isinstance(rosboard_embodiment(camera_height=2, camera_width=3), RosboardEmbodiment)


def test_required_sample_raises_without_a_cached_message() -> None:
    embodiment, _client, _clock = _ready_embodiment()
    with pytest.raises(RuntimeError, match="no cached message"):
        embodiment._required_sample("/never/subscribed")


def test_context_manager_closes_on_exit() -> None:
    embodiment, client, _clock = _ready_embodiment()
    with embodiment:
        embodiment.reset(_SCENE)
    assert client.closed is True


def test_ensure_initialized_wraps_connect_failure() -> None:
    embodiment = _embodiment(url="ws://127.0.0.1:1", connect_timeout_s=0.2, obs_timeout_s=0.2)
    with pytest.raises(ConnectionError, match="rosboard node is already running"):
        embodiment.reset(_SCENE)


# --------------------------------------------------------------------------
# End-to-end: real stub server, exercising the real _ensure_initialized() path
# --------------------------------------------------------------------------


def test_end_to_end_reset_and_step_against_real_stub_server(
    stub_server: StubRosboardServer,
) -> None:
    embodiment = RosboardEmbodiment(
        url=stub_server.url,
        camera_height=2,
        camera_width=3,
        control_hz=20.0,
        obs_timeout_s=2.0,
    )

    stop = threading.Event()

    def stream() -> None:
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        while not stop.is_set():
            stub_server.publish(
                embodiment.odometry_topic, "nav_msgs/msg/Odometry", _odometry_payload()
            )
            stub_server.publish(embodiment.imu_topic, "sensor_msgs/msg/Imu", _imu_payload())
            stub_server.publish(
                embodiment.camera_topic, "sensor_msgs/msg/Image", _image_payload(image, "PNG")
            )
            time.sleep(0.01)

    thread = threading.Thread(target=stream, daemon=True)
    thread.start()
    try:
        observation = embodiment.reset(_SCENE)
        assert observation.instruction == "drive forward"
        result = embodiment.step(Action(data=np.array([0.1, 0.0])))
        assert result.terminated is False
        assert set(result.observation.state) == {"odom_pose", "odom_twist", "imu"}

        # The stub server actually recorded the clamped command over the wire.
        stub_server.wait_for(lambda frames: any(f[0] == "s" for f in frames))
    finally:
        stop.set()
        thread.join(timeout=2)
        embodiment.close()
