"""Tests for the topics_to_subscribe.yaml-driven mode: parsing and the embodiment it drives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from test_config import _image_payload
from test_rosboard_embodiment import _embodiment, _FakeClient, _FakeClock, _ready_embodiment

from inspect_robots.scene import Scene
from inspect_robots.types import Action
from inspect_robots_rosboard import RosboardEmbodiment
from inspect_robots_rosboard._config import ConfigError
from inspect_robots_rosboard._topics_file import TopicsFile, load_topics_file

_SCENE = Scene(id="s0", instruction="drive forward")

_REAL_TOPICS_FILE = """
url: ws://robot.example.com:80
topics:
  [
    /odometry/local,
    /imu/data_abs_heading,
    /camera/color/image_raw,
    # /video_mapping/right/image_raw,
  ]
topics_to_stream:
  [
   #/motion_control/speed_controller/reference_cmd,
   #/predicted_waypoints
  ]
"""


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "topics_to_subscribe.yaml"
    path.write_text(text)
    return str(path)


# --------------------------------------------------------------------------
# load_topics_file
# --------------------------------------------------------------------------


def test_load_topics_file_parses_commented_flow_sequences(tmp_path: Path) -> None:
    path = _write(tmp_path, _REAL_TOPICS_FILE)
    parsed = load_topics_file(path)
    assert parsed == TopicsFile(
        url="ws://robot.example.com:80",
        topics=frozenset({"/odometry/local", "/imu/data_abs_heading", "/camera/color/image_raw"}),
        topics_to_stream=frozenset(),
    )


def test_load_topics_file_uncommented_stream_topic_is_included(tmp_path: Path) -> None:
    text = _REAL_TOPICS_FILE.replace(
        "#/motion_control/speed_controller/reference_cmd,",
        "/motion_control/speed_controller/reference_cmd,",
    )
    parsed = load_topics_file(_write(tmp_path, text))
    assert "/motion_control/speed_controller/reference_cmd" in parsed.topics_to_stream


def test_load_topics_file_missing_file_raises() -> None:
    with pytest.raises(ConfigError, match="could not read"):
        load_topics_file("/nonexistent/topics_to_subscribe.yaml")


def test_load_topics_file_invalid_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_topics_file(_write(tmp_path, "url: [unclosed"))


def test_load_topics_file_requires_mapping_at_top_level(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_topics_file(_write(tmp_path, "- just\n- a\n- list\n"))


def test_load_topics_file_requires_url(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="url"):
        load_topics_file(_write(tmp_path, "topics: [/a]\n"))


def test_load_topics_file_rejects_empty_url(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="url"):
        load_topics_file(_write(tmp_path, 'url: ""\n'))


def test_load_topics_file_omitted_topic_lists_default_empty(tmp_path: Path) -> None:
    parsed = load_topics_file(_write(tmp_path, "url: ws://x:80\n"))
    assert parsed.topics == frozenset()
    assert parsed.topics_to_stream == frozenset()


def test_load_topics_file_rejects_non_list_topics(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="topics"):
        load_topics_file(_write(tmp_path, "url: ws://x:80\ntopics: not-a-list\n"))


def test_load_topics_file_rejects_non_string_topic_entries(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="topics"):
        load_topics_file(_write(tmp_path, "url: ws://x:80\ntopics: [1, 2]\n"))


# --------------------------------------------------------------------------
# RosboardEmbodiment(topics_file=...)
# --------------------------------------------------------------------------


def test_topics_file_sources_the_url(tmp_path: Path) -> None:
    embodiment = _embodiment(topics_file=_write(tmp_path, _REAL_TOPICS_FILE))
    assert embodiment.url == "ws://robot.example.com:80"


def test_topics_file_and_explicit_url_are_mutually_exclusive(tmp_path: Path) -> None:
    path = _write(tmp_path, _REAL_TOPICS_FILE)
    with pytest.raises(ValueError, match="mutually exclusive"):
        RosboardEmbodiment(topics_file=path, url="ws://other:80", camera_height=2, camera_width=3)


def test_topics_file_missing_required_role_topic_raises(tmp_path: Path) -> None:
    text = _REAL_TOPICS_FILE.replace("/odometry/local,", "")
    path = _write(tmp_path, text)
    with pytest.raises(ValueError, match=r"odometry_topic=.*/odometry/local"):
        _embodiment(topics_file=path)


def test_topics_file_command_topic_absent_disables_publishing(tmp_path: Path) -> None:
    embodiment = _embodiment(topics_file=_write(tmp_path, _REAL_TOPICS_FILE))
    assert embodiment._publish_enabled is False
    assert "DRY RUN" in (embodiment.info.docs or "")


def test_topics_file_command_topic_present_enables_publishing(tmp_path: Path) -> None:
    text = _REAL_TOPICS_FILE.replace(
        "#/motion_control/speed_controller/reference_cmd,",
        "/motion_control/speed_controller/reference_cmd,",
    )
    embodiment = _embodiment(topics_file=_write(tmp_path, text))
    assert embodiment._publish_enabled is True
    assert "DRY RUN" not in (embodiment.info.docs or "")


def test_step_dry_run_does_not_publish_but_still_advances(tmp_path: Path) -> None:
    clock = _FakeClock()
    embodiment, client, _clock = _ready_embodiment(
        clock=clock, topics_file=_write(tmp_path, _REAL_TOPICS_FILE)
    )
    embodiment.reset(_SCENE)
    published_before = len(client.published)
    result = embodiment.step(Action(data=np.array([0.3, -0.3])))
    assert len(client.published) == published_before  # nothing sent
    assert result.observation is not None


def test_step_dry_run_logs_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    embodiment, _client, _clock = _ready_embodiment(topics_file=_write(tmp_path, _REAL_TOPICS_FILE))
    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.array([0.1, 0.0])))
    captured = capsys.readouterr()
    assert "dry-run" in captured.err
    assert embodiment.command_topic in captured.err


def test_close_dry_run_does_not_publish_zero_twist(tmp_path: Path) -> None:
    embodiment, client, _clock = _ready_embodiment(topics_file=_write(tmp_path, _REAL_TOPICS_FILE))
    embodiment.reset(_SCENE)
    published_before = len(client.published)
    embodiment.close()
    assert len(client.published) == published_before
    assert client.closed is True


# --------------------------------------------------------------------------
# RosboardEmbodiment(config=..., topics_file=...) -- the two modes combined
# --------------------------------------------------------------------------

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
actions:
  - key: action
    publish:
      topic: /motion_control/speed_controller/reference_cmd
      type: geometry_msgs/msg/Twist
    selector:
      names: [linear.x, angular.z]
    from_tensor:
      clamp: [-2.0, 2.0]
    safety_behavior: zeros
"""


def _write_config(tmp_path: Path, text: str = _MINIMAL_CONFIG) -> str:
    path = tmp_path / "robot.yaml"
    path.write_text(text)
    return str(path)


def _ready_config_embodiment_with_topics_file(
    tmp_path: Path, topics_text: str
) -> tuple[RosboardEmbodiment, _FakeClient, _FakeClock]:
    fake_clock = _FakeClock()
    embodiment = RosboardEmbodiment(
        config=_write_config(tmp_path),
        topics_file=_write(tmp_path, topics_text),
        clock=fake_clock,
        sleep=fake_clock.sleep,
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


def test_config_mode_sources_url_from_topics_file(tmp_path: Path) -> None:
    embodiment = RosboardEmbodiment(
        config=_write_config(tmp_path), topics_file=_write(tmp_path, _REAL_TOPICS_FILE)
    )
    assert embodiment.url == "ws://robot.example.com:80"


def test_config_mode_missing_observation_topic_in_topics_file_raises(tmp_path: Path) -> None:
    text = _REAL_TOPICS_FILE.replace("/odometry/local,", "")
    with pytest.raises(ValueError, match=r"observation\.state.*odometry/local"):
        RosboardEmbodiment(config=_write_config(tmp_path), topics_file=_write(tmp_path, text))


def test_config_mode_action_topic_absent_from_stream_disables_publishing(tmp_path: Path) -> None:
    embodiment = RosboardEmbodiment(
        config=_write_config(tmp_path), topics_file=_write(tmp_path, _REAL_TOPICS_FILE)
    )
    assert embodiment._action_publish_enabled == (False,)
    assert "DRY RUN" in (embodiment.info.docs or "")


def test_config_mode_action_topic_present_enables_publishing(tmp_path: Path) -> None:
    text = _REAL_TOPICS_FILE.replace(
        "#/motion_control/speed_controller/reference_cmd,",
        "/motion_control/speed_controller/reference_cmd,",
    )
    embodiment = RosboardEmbodiment(
        config=_write_config(tmp_path), topics_file=_write(tmp_path, text)
    )
    assert embodiment._action_publish_enabled == (True,)
    assert "DRY RUN" not in (embodiment.info.docs or "")


def test_config_mode_without_topics_file_still_publishes(tmp_path: Path) -> None:
    embodiment = RosboardEmbodiment(config=_write_config(tmp_path))
    assert embodiment._action_publish_enabled == (True,)


def test_step_config_dry_run_does_not_publish_but_still_advances(tmp_path: Path) -> None:
    embodiment, client, _clock = _ready_config_embodiment_with_topics_file(
        tmp_path, _REAL_TOPICS_FILE
    )
    embodiment.reset(_SCENE)
    published_before = len(client.published)
    result = embodiment.step(Action(data=np.array([0.3, -0.3])))
    assert len(client.published) == published_before
    assert result.observation is not None


def test_step_config_dry_run_logs_to_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    embodiment, _client, _clock = _ready_config_embodiment_with_topics_file(
        tmp_path, _REAL_TOPICS_FILE
    )
    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.array([0.1, 0.0])))
    captured = capsys.readouterr()
    assert "dry-run" in captured.err
    assert "/motion_control/speed_controller/reference_cmd" in captured.err


def test_close_config_dry_run_does_not_publish_zero_twist(tmp_path: Path) -> None:
    embodiment, client, _clock = _ready_config_embodiment_with_topics_file(
        tmp_path, _REAL_TOPICS_FILE
    )
    embodiment.reset(_SCENE)
    published_before = len(client.published)
    embodiment.close()
    assert len(client.published) == published_before
    assert client.closed is True
