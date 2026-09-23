"""Tests for the YAML robot-config path: parsing, selectors, and the embodiment mode it drives."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from _stub_server import StubRosboardServer
from PIL import Image
from test_rosboard_embodiment import _FakeClient, _FakeClock

from inspect_robots.conformance import assert_embodiment_conformant
from inspect_robots.scene import Scene
from inspect_robots.types import Action
from inspect_robots_rosboard import RosboardEmbodiment
from inspect_robots_rosboard._config import ConfigError, load_robot_config
from inspect_robots_rosboard._selectors import build_from_selector, get_path, select_state

_SCENE = Scene(id="s0", instruction="drive to the waypoint")

_MINIMAL_CONFIG = """
name: rover_diff_drive
rate_hz: 10.0
observations:
  - key: observation.image.main
    topic: /camera/color/image_raw
    type: sensor_msgs/msg/Image
    image:
      resize: [2, 3]
  - key: observation.state
    topic: /odometry/local
    type: nav_msgs/msg/Odometry
    selector:
      names: [twist.twist.linear.x, twist.twist.angular.z]
    align: {tol_ms: 150}
actions:
  - key: action
    publish:
      topic: /motion_control/speed_controller/output_cmd
      type: geometry_msgs/msg/TwistStamped
    selector:
      names: [twist.linear.x, twist.angular.z]
    from_tensor:
      clamp: [-2.0, 2.0]
    safety_behavior: zeros
"""


def _write_config(tmp_path: Path, text: str = _MINIMAL_CONFIG) -> str:
    path = tmp_path / "robot.yaml"
    path.write_text(text)
    return str(path)


# --------------------------------------------------------------------------
# _selectors.py: dotted-path get/set, no I/O
# --------------------------------------------------------------------------


def test_get_path_reads_nested_scalar() -> None:
    payload = {"twist": {"twist": {"linear": {"x": 1.5}}}}
    assert get_path(payload, "twist.twist.linear.x") == 1.5


def test_get_path_rejects_missing_or_non_numeric() -> None:
    with pytest.raises(ValueError, match="not found"):
        get_path({"a": 1}, "b")
    with pytest.raises(ValueError, match="not found"):
        get_path({"a": {"b": 1}}, "a.c")
    with pytest.raises(ValueError, match="did not resolve to a number"):
        get_path({"a": "text"}, "a")
    with pytest.raises(ValueError, match="did not resolve to a number"):
        get_path({"a": True}, "a")


def test_select_state_builds_vector_in_order() -> None:
    payload = {"longitude": -74.1, "latitude": 4.6}
    vector = select_state(payload, ["latitude", "longitude"])
    np.testing.assert_allclose(vector, [4.6, -74.1])


def test_build_from_selector_round_trips_through_select_state() -> None:
    names = ["twist.linear.x", "twist.angular.z"]
    fields = build_from_selector(names, [1.0, -0.5])
    assert fields == {"twist": {"linear": {"x": 1.0}, "angular": {"z": -0.5}}}
    assert select_state(fields, names).tolist() == [1.0, -0.5]


def test_build_from_selector_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="2 names but received 1"):
        build_from_selector(["a", "b"], [1.0])


def test_build_from_selector_rejects_conflicting_nesting() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        build_from_selector(["a", "a.b"], [1.0, 2.0])
    with pytest.raises(ValueError, match="conflicts"):
        build_from_selector(["a.b", "a"], [1.0, 2.0])


# --------------------------------------------------------------------------
# _config.py: YAML parsing and validation, no I/O beyond reading the file
# --------------------------------------------------------------------------


def test_load_robot_config_happy_path(tmp_path: Path) -> None:
    config = load_robot_config(_write_config(tmp_path))
    assert config.name == "rover_diff_drive"
    assert config.rate_hz == 10.0
    assert [o.key for o in config.observations] == ["observation.image.main", "observation.state"]
    image_spec = config.observations[0]
    assert image_spec.image is not None
    assert (image_spec.image.height, image_spec.image.width) == (2, 3)
    state_spec = config.observations[1]
    assert state_spec.selector_names == ("twist.twist.linear.x", "twist.twist.angular.z")
    assert state_spec.tol_ms == 150.0
    assert len(config.actions) == 1
    action = config.actions[0]
    assert action.publish_topic == "/motion_control/speed_controller/output_cmd"
    assert action.selector_names == ("twist.linear.x", "twist.angular.z")
    assert (action.clamp_low, action.clamp_high) == (-2.0, 2.0)
    assert action.safety_behavior == "zeros"


def test_load_robot_config_rejects_missing_file() -> None:
    with pytest.raises(ConfigError, match="could not read"):
        load_robot_config("/nonexistent/robot.yaml")


def test_load_robot_config_rejects_invalid_yaml(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_robot_config(_write_config(tmp_path, "name: [unterminated"))


def test_load_robot_config_rejects_non_mapping_top_level(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_robot_config(_write_config(tmp_path, "- just\n- a\n- list\n"))


@pytest.mark.parametrize("field", ["name", "rate_hz", "observations", "actions"])
def test_load_robot_config_requires_top_level_fields(tmp_path: Path, field: str) -> None:
    # Renaming the top-level key (not any nested key of the same name) makes
    # it disappear from the parsed mapping without disturbing the rest.
    text = _MINIMAL_CONFIG.replace(f"\n{field}:", f"\n_{field}:", 1)
    with pytest.raises(ConfigError, match=f"missing required field {field!r}"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_non_positive_rate(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace("rate_hz: 10.0", "rate_hz: 0")
    with pytest.raises(ConfigError, match="rate_hz must be positive"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_observation_with_both_image_and_selector(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace(
        "    image:\n      resize: [2, 3]\n",
        "    image:\n      resize: [2, 3]\n    selector:\n      names: [a]\n",
    )
    with pytest.raises(ConfigError, match="exactly one of 'image' or 'selector'"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_observation_with_neither(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace("    image:\n      resize: [2, 3]\n", "")
    with pytest.raises(ConfigError, match="exactly one of 'image' or 'selector'"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_missing_image_resize(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace("    image:\n      resize: [2, 3]\n", "    image: {}\n")
    with pytest.raises(ConfigError, match=r"image\.resize"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_missing_action_clamp(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace("    from_tensor:\n      clamp: [-2.0, 2.0]\n", "")
    with pytest.raises(ConfigError, match=r"from_tensor\.clamp"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_inverted_clamp(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace("clamp: [-2.0, 2.0]", "clamp: [2.0, -2.0]")
    with pytest.raises(ConfigError, match="low must be < high"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_unsupported_safety_behavior(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace("safety_behavior: zeros", "safety_behavior: hold")
    with pytest.raises(ConfigError, match="safety_behavior 'hold' is not supported"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_action_zero_fields_defaults_empty(tmp_path: Path) -> None:
    config = load_robot_config(_write_config(tmp_path))
    assert config.actions[0].zero_fields == ()


def test_load_robot_config_parses_action_zero_fields(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace(
        "    from_tensor:", "    zero_fields: [twist.linear.y, twist.linear.z]\n    from_tensor:"
    )
    config = load_robot_config(_write_config(tmp_path, text))
    assert config.actions[0].zero_fields == ("twist.linear.y", "twist.linear.z")


def test_load_robot_config_rejects_non_list_zero_fields(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace(
        "    from_tensor:", "    zero_fields: not-a-list\n    from_tensor:"
    )
    with pytest.raises(ConfigError, match="zero_fields must be a list of strings"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_non_string_zero_fields_entries(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace("    from_tensor:", "    zero_fields: [1]\n    from_tensor:")
    with pytest.raises(ConfigError, match="zero_fields must be a list of strings"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_zero_fields_overlapping_selector_names(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace(
        "    from_tensor:", "    zero_fields: [twist.linear.x]\n    from_tensor:"
    )
    with pytest.raises(ConfigError, match=r"zero_fields overlaps selector\.names"):
        load_robot_config(_write_config(tmp_path, text))


def test_load_robot_config_rejects_duplicate_observation_keys(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace("observation.state", "observation.image.main", 1)
    with pytest.raises(ConfigError, match="key values must be unique"):
        load_robot_config(_write_config(tmp_path, text))


# --------------------------------------------------------------------------
# Embodiment, config-driven mode (fake client for fast unit coverage)
# --------------------------------------------------------------------------


def _image_payload(image: Any, image_format: str = "JPEG") -> dict[str, Any]:
    buffer = BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"format": image_format.lower(), "_data_jpeg": encoded}


def _ready_config_embodiment(
    tmp_path: Path,
) -> tuple[RosboardEmbodiment, _FakeClient, _FakeClock]:
    fake_clock = _FakeClock()
    embodiment = RosboardEmbodiment(
        config=_write_config(tmp_path), clock=fake_clock, sleep=fake_clock.sleep
    )
    client = _FakeClient(fake_clock)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    client.put("/camera/color/image_raw", _image_payload(image))
    client.put(
        "/odometry/local",
        {"twist": {"twist": {"linear": {"x": 1.0}, "angular": {"z": 0.5}}}},
    )

    def fresh(topic: str, _after_seq: int) -> None:
        if topic in client.samples:
            client.put(topic, client.samples[topic].payload)

    client.on_wait = fresh
    embodiment._client = cast(Any, client)
    embodiment._initialized = True
    embodiment._instruction = _SCENE.instruction
    return embodiment, client, fake_clock


def test_config_info_matches_yaml_shape(tmp_path: Path) -> None:
    embodiment = RosboardEmbodiment(config=_write_config(tmp_path))
    info = embodiment.info
    assert info.action_space.shape == (2,)
    np.testing.assert_allclose(info.action_space.low, [-2.0, -2.0])
    np.testing.assert_allclose(info.action_space.high, [2.0, 2.0])
    assert info.action_space.semantics is not None
    assert info.action_space.semantics.dim_labels == ("twist.linear.x", "twist.angular.z")
    assert [c.name for c in info.observation_space.cameras] == ["observation.image.main"]
    assert info.observation_space.state is not None
    assert [f.key for f in info.observation_space.state.fields] == ["observation.state"]
    assert info.control_hz == 10.0
    assert info.capabilities == frozenset({"self_paced"})
    assert_embodiment_conformant(info)


def test_config_construction_touches_no_network(tmp_path: Path) -> None:
    RosboardEmbodiment(config=_write_config(tmp_path), url="ws://198.51.100.1:1")


def test_config_reset_assembles_image_and_state(tmp_path: Path) -> None:
    embodiment, _client, _clock = _ready_config_embodiment(tmp_path)
    obs = embodiment.reset(_SCENE)
    assert set(obs.images) == {"observation.image.main"}
    assert obs.images["observation.image.main"].shape == (2, 3, 3)
    assert set(obs.state) == {"observation.state"}
    np.testing.assert_allclose(obs.state["observation.state"], [1.0, 0.5])
    assert obs.instruction == _SCENE.instruction


def test_config_step_clamps_and_publishes_each_action_spec(tmp_path: Path) -> None:
    embodiment, client, _clock = _ready_config_embodiment(tmp_path)
    embodiment.reset(_SCENE)

    result = embodiment.step(Action(data=np.array([10.0, -10.0])))

    assert len(client.published) == 1
    topic, topic_type, fields = client.published[0]
    assert topic == "/motion_control/speed_controller/output_cmd"
    assert topic_type == "geometry_msgs/msg/TwistStamped"
    assert fields == {"twist": {"linear": {"x": 2.0}, "angular": {"z": -2.0}}}
    assert result.terminated is False
    assert result.truncated is False


def test_config_step_publishes_zero_fields_alongside_commanded_values(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace(
        "    from_tensor:",
        "    zero_fields: [twist.linear.y, twist.linear.z, twist.angular.x, twist.angular.y]\n"
        "    from_tensor:",
    )
    fake_clock = _FakeClock()
    embodiment = RosboardEmbodiment(
        config=_write_config(tmp_path, text), clock=fake_clock, sleep=fake_clock.sleep
    )
    client = _FakeClient(fake_clock)
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    client.put("/camera/color/image_raw", _image_payload(image))
    client.put(
        "/odometry/local",
        {"twist": {"twist": {"linear": {"x": 1.0}, "angular": {"z": 0.5}}}},
    )

    def fresh(topic: str, _after_seq: int) -> None:
        if topic in client.samples:
            client.put(topic, client.samples[topic].payload)

    client.on_wait = fresh
    embodiment._client = cast(Any, client)
    embodiment._initialized = True
    embodiment._instruction = _SCENE.instruction

    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.array([10.0, -10.0])))

    topic, _topic_type, fields = client.published[-1]
    assert topic == "/motion_control/speed_controller/output_cmd"
    assert fields == {
        "twist": {
            "linear": {"x": 2.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": -2.0},
        }
    }


def test_config_step_rejects_wrong_action_shape(tmp_path: Path) -> None:
    embodiment, _client, _clock = _ready_config_embodiment(tmp_path)
    embodiment.reset(_SCENE)
    with pytest.raises(ValueError, match="expected"):
        embodiment.step(Action(data=np.array([1.0, 2.0, 3.0])))


def test_config_close_publishes_zeros_for_safety_behavior(tmp_path: Path) -> None:
    embodiment, client, _clock = _ready_config_embodiment(tmp_path)
    embodiment.reset(_SCENE)
    embodiment.close()

    assert client.closed is True
    command_topic = "/motion_control/speed_controller/output_cmd"
    zero_publishes = [p for p in client.published if p[0] == command_topic]
    assert zero_publishes[-1][2] == {"twist": {"linear": {"x": 0.0}, "angular": {"z": 0.0}}}


def test_config_close_is_a_noop_before_reset(tmp_path: Path) -> None:
    embodiment = RosboardEmbodiment(config=_write_config(tmp_path))
    embodiment.close()  # must not raise even though nothing was ever initialized


# --------------------------------------------------------------------------
# Embodiment, config-driven mode: end-to-end against the real stub server
# --------------------------------------------------------------------------


def test_config_end_to_end_against_stub_server(
    tmp_path: Path, stub_server: StubRosboardServer
) -> None:
    embodiment = RosboardEmbodiment(
        config=_write_config(tmp_path), url=stub_server.url, obs_timeout_s=2.0
    )

    def publish_samples() -> None:
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        stub_server.publish(
            "/camera/color/image_raw", "sensor_msgs/msg/Image", _image_payload(image)
        )
        stub_server.publish(
            "/odometry/local",
            "nav_msgs/msg/Odometry",
            {"twist": {"twist": {"linear": {"x": 1.0}, "angular": {"z": 0.5}}}},
        )

    import threading
    import time

    def publisher_loop(stop: threading.Event) -> None:
        while not stop.is_set():
            publish_samples()
            time.sleep(0.02)

    stop = threading.Event()
    thread = threading.Thread(target=publisher_loop, args=(stop,), daemon=True)
    thread.start()
    try:
        obs = embodiment.reset(_SCENE)
        assert obs.images["observation.image.main"].shape == (2, 3, 3)
        result = embodiment.step(Action(data=np.array([0.1, 0.0])))
        assert result.terminated is False
    finally:
        stop.set()
        thread.join(timeout=2)
        embodiment.close()

    stub_server.wait_for(
        lambda frames: any(
            f[0] == "m" and f[1].get("_topic_name") == "/motion_control/speed_controller/output_cmd"
            for f in frames
        )
    )
