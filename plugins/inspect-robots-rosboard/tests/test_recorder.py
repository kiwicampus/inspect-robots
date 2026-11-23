"""Tests for the read only telemetry recorder: client listener, file rows, embodiment wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _stub_server import StubRosboardServer
from test_config import _MINIMAL_CONFIG, _write_config
from test_rosboard_embodiment import _SCENE, _odometry_payload, _ready_embodiment

from inspect_robots.rollout import TrialRecord
from inspect_robots.types import Action
from inspect_robots_rosboard import RosboardEmbodiment
from inspect_robots_rosboard._client import RosboardClient
from inspect_robots_rosboard._config import ConfigError, load_robot_config
from inspect_robots_rosboard._recorder import TopicRecorder


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_client_listener_sees_every_message_and_cannot_break_it(
    stub_server: StubRosboardServer,
) -> None:
    client = RosboardClient(stub_server.url, connect_timeout_s=2.0)
    seen: list[tuple[str, int]] = []

    def broken(topic: str, payload: dict[str, Any], stamp: float) -> None:
        raise RuntimeError("listener bug")

    client.add_listener(broken)
    client.add_listener(lambda topic, payload, stamp: seen.append((topic, payload["seq"])))
    client.connect()
    client.subscribe("/odometry/local")
    stub_server.wait_for(lambda frames: any(f[0] == "s" for f in frames))
    for seq in (1, 2, 3):
        stub_server.publish("/odometry/local", "nav_msgs/msg/Odometry", {"seq": seq})
    client.wait_for_sample("/odometry/local", after_seq=2, timeout_s=2.0)
    assert seen == [("/odometry/local", 1), ("/odometry/local", 2), ("/odometry/local", 3)]
    assert client.latched_error is None
    client.close()


def test_recorder_rows_share_the_trial_clock_and_drop_covariance(tmp_path: Path) -> None:
    now = [100.0]
    recorder = TopicRecorder(["/odometry/local"], lambda: now[0])
    path = tmp_path / "t" / "trial.jsonl"
    recorder.start(path, {"trial": "s0-e0"})
    payload = {
        "_topic_name": "/odometry/local",
        "header": {"stamp": {"sec": 5, "nanosec": 500_000_000}},
        "pose": {"pose": {"position": {"x": 1.0}}, "covariance": [0.0] * 36},
    }
    recorder.on_sample("/odometry/local", payload, 100.25)
    recorder.on_sample("/other", payload, 100.3)
    recorder.command(0, 100.5, [0.3, -0.1], True)
    assert recorder.stop() == path
    recorder.on_sample("/odometry/local", payload, 101.0)  # after stop: ignored
    recorder.command(1, 101.0, [0.3, 0.0], True)

    header, sample, command = _rows(path)
    assert header == {"kind": "header", "topics": ["/odometry/local"], "trial": "s0-e0"}
    assert sample == {
        "topic": "/odometry/local",
        "stamp": 0.25,
        "header_stamp": 5.5,
        "msg": {
            "header": {"stamp": {"sec": 5, "nanosec": 500_000_000}},
            "pose": {"pose": {"position": {"x": 1.0}}},
        },
    }
    assert command == {
        "kind": "command",
        "t": 0,
        "stamp": 0.5,
        "action": [0.3, -0.1],
        "published": True,
    }


def test_recorder_degrades_when_the_file_cannot_be_written(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")
    recorder = TopicRecorder(["/fix"], lambda: 0.0)
    recorder.start(blocker / "sub" / "trial.jsonl", {})
    recorder.on_sample("/fix", {"latitude": 1.0}, 0.0)
    assert recorder.stop() is None


def test_recorder_stops_writing_after_a_write_error(tmp_path: Path) -> None:
    recorder = TopicRecorder(["/fix"], lambda: 0.0)
    recorder.start(tmp_path / "trial.jsonl", {})
    assert recorder._file is not None
    recorder._file.close()  # the next write raises ValueError
    recorder.on_sample("/fix", {"latitude": 1.0}, 0.0)
    assert recorder._file is None


def test_embodiment_records_samples_commands_and_the_stop(tmp_path: Path) -> None:
    embodiment, client, clock = _ready_embodiment(record_topics="/odometry/local, /fix")
    assert embodiment._recorder is not None
    assert embodiment._recorder.topics == ("/odometry/local", "/fix")
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)

    embodiment.on_trial_start("s0", 0, str(tmp_path), "run1")
    embodiment.reset(_SCENE)
    embodiment._recorder.on_sample("/odometry/local", _odometry_payload(), clock())
    embodiment.step(Action(data=np.asarray([0.2, -0.1])))
    embodiment.step(Action(data=np.asarray([9.0, 0.0])))  # clamped before it is recorded
    embodiment.observe_parked()
    embodiment.on_trial_end(record, str(tmp_path), "run1")

    assert record.metadata["telemetry"] == "telemetry/run1/s0-e0.jsonl"
    rows = _rows(tmp_path / "telemetry" / "run1" / "s0-e0.jsonl")
    assert rows[0]["kind"] == "header" and rows[0]["trial"] == "s0-e0"
    assert rows[1]["topic"] == "/odometry/local"
    commands = [row for row in rows if row.get("kind") == "command"]
    assert [(row["t"], row["action"], row["published"]) for row in commands] == [
        (0, [0.2, -0.1], True),
        (1, [embodiment.linear_x_limit, 0.0], True),
        (None, [], True),
    ]
    assert len(client.published) == 3  # the recorder itself never publishes


def test_trial_ended_mid_command_is_stopped_and_logged(tmp_path: Path) -> None:
    embodiment, client, _clock = _ready_embodiment(record_topics="/odometry/local")
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)
    embodiment.on_trial_start("s0", 0, str(tmp_path), "run1")
    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.asarray([0.2, 0.1])))
    embodiment.on_trial_end(record, str(tmp_path), "run1")  # Ctrl+C: no observe_parked
    rows = _rows(tmp_path / "telemetry" / "run1" / "s0-e0.jsonl")
    assert [row["t"] for row in rows if row.get("kind") == "command"] == [0, None]
    assert len(client.published) == 2

    embodiment.on_trial_start("s0", 1, str(tmp_path), "run1")
    embodiment.observe_parked()  # already stopped: the trial end sends nothing more
    embodiment.on_trial_end(TrialRecord(scene_id="s0", epoch=1, seed=0), str(tmp_path), "run1")
    assert len(client.published) == 3


def test_embodiment_without_record_topics_adds_no_metadata(tmp_path: Path) -> None:
    embodiment, _client, _clock = _ready_embodiment()
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)
    embodiment.on_trial_start("s0", 0, str(tmp_path), "run1")
    embodiment.on_trial_end(record, str(tmp_path), "run1")
    assert embodiment._recorder is None
    assert "telemetry" not in record.metadata
    assert not (tmp_path / "telemetry").exists()


def test_telemetry_outside_log_dir_keeps_the_absolute_path(tmp_path: Path) -> None:
    embodiment, _client, _clock = _ready_embodiment(record_topics=["/fix"])
    record = TrialRecord(scene_id="s/0", epoch=1, seed=0)
    embodiment.on_trial_start("s/0", 1, str(tmp_path / "a"), "run1")
    embodiment.on_trial_end(record, str(tmp_path / "b"), "run1")
    assert record.metadata["telemetry"] == str(
        (tmp_path / "a" / "telemetry" / "run1" / "s_0-e1.jsonl").resolve()
    )


def test_record_topics_must_be_in_the_topics_file(tmp_path: Path) -> None:
    topics = tmp_path / "topics.yaml"
    topics.write_text(
        "url: ws://localhost:1\n"
        "topics: [/odometry/local, /imu/data_abs_heading, /camera/color/image_raw]\n"
        "topics_to_stream: []\n"
    )
    with pytest.raises(ValueError, match=r"record_topics \['/fix'\]"):
        RosboardEmbodiment(
            topics_file=str(topics), camera_height=2, camera_width=3, record_topics="/fix"
        )


def test_config_record_topics_parse_and_cli_override(tmp_path: Path) -> None:
    text = _MINIMAL_CONFIG.replace(
        "rate_hz: 10.0", "rate_hz: 10.0\nrecord_topics: [/odometry/local, /fix, /fix]"
    )
    path = _write_config(tmp_path, text)
    assert load_robot_config(path).record_topics == ("/odometry/local", "/fix")
    from_yaml = RosboardEmbodiment(config=path)
    assert from_yaml._recorder is not None
    assert from_yaml._recorder.topics == ("/odometry/local", "/fix")
    overridden = RosboardEmbodiment(config=path, record_topics="/imu")
    assert overridden._recorder is not None and overridden._recorder.topics == ("/imu",)


@pytest.mark.parametrize("value", ["/fix", "[1, 2]", "['']"])
def test_config_rejects_bad_record_topics(tmp_path: Path, value: str) -> None:
    text = _MINIMAL_CONFIG.replace("rate_hz: 10.0", f"rate_hz: 10.0\nrecord_topics: {value}")
    with pytest.raises(ConfigError, match="record_topics must be a list"):
        load_robot_config(_write_config(tmp_path, text))


def test_config_mode_records_the_clamped_command(tmp_path: Path) -> None:
    from test_config import _ready_config_embodiment

    embodiment, _client, clock = _ready_config_embodiment(tmp_path)
    embodiment._recorder = TopicRecorder(["/odometry/local"], clock)
    embodiment.on_trial_start("s0", 0, str(tmp_path), "run1")
    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.asarray([5.0, -0.5])))
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)
    embodiment.on_trial_end(record, str(tmp_path), "run1")
    rows = [r for r in _rows(tmp_path / record.metadata["telemetry"]) if r.get("kind") != "obs"]
    assert rows[-2]["action"] == [2.0, -0.5] and rows[-2]["published"] is True
    assert rows[-1]["t"] is None  # the trial ended mid command, so it was stopped


def test_recorded_topics_are_subscribed_once() -> None:
    embodiment, client, clock = _ready_embodiment()
    subscribed: list[str] = []
    setattr(client, "subscribe", subscribed.append)  # noqa: B010
    embodiment._recorder = TopicRecorder(["/odometry/local", "/fix"], clock)
    embodiment._subscribe_recorded(embodiment._all_topics())
    assert subscribed == ["/fix"]


def test_recorder_writes_input_ages(tmp_path: Path) -> None:
    recorder = TopicRecorder([], lambda: 10.0)
    path = tmp_path / "trial.jsonl"
    recorder.start(path, {})
    ages = {"/fix": {"rx": 0.1, "sensor": 0.3, "seq": 4, "repeated": False}}
    recorder.observation(2, 10.5, ages, -7.0)
    recorder.stop()
    recorder.observation(3, 11.0, ages, -7.0)  # after stop: ignored
    assert _rows(path)[1:] == [
        {"kind": "obs", "t": 2, "stamp": 0.5, "clock_offset": -7.0, "ages": ages}
    ]


def test_clock_offset_keeps_the_smallest_link_delay(tmp_path: Path) -> None:
    from test_config import _ready_config_embodiment

    embodiment, _client, _clock = _ready_config_embodiment(tmp_path)
    stamped = {"header": {"stamp": {"sec": 100, "nanosec": 0}}}
    embodiment._track_clock_offset("/a", {}, 5.0)  # no header: ignored
    assert embodiment._clock_offset is None
    embodiment._track_clock_offset("/a", stamped, 5.5)
    embodiment._track_clock_offset("/b", stamped, 5.2)
    embodiment._track_clock_offset("/a", stamped, 6.0)
    assert embodiment._clock_offset == pytest.approx(-94.8)


def test_config_mode_records_each_steps_input_ages(tmp_path: Path) -> None:
    from test_config import _ready_config_embodiment

    embodiment, client, clock = _ready_config_embodiment(tmp_path)
    odometry = dict(client.samples["/odometry/local"].payload)
    odometry["header"] = {"stamp": {"sec": 1000, "nanosec": 0}}
    client.put("/odometry/local", odometry)
    embodiment._clock_offset = clock.now - 1000.2  # the fastest message took 0.2 s less
    embodiment._recorder = TopicRecorder([], clock)
    embodiment.on_trial_start("s0", 0, str(tmp_path), "run1")
    embodiment.reset(_SCENE)
    clock.now += 0.5  # the step only waits for odometry, so the camera frame stays the same
    embodiment.step(Action(data=np.asarray([0.1, 0.0])))
    record = TrialRecord(scene_id="s0", epoch=0, seed=0)
    embodiment.on_trial_end(record, str(tmp_path), "run1")

    obs = [r for r in _rows(tmp_path / record.metadata["telemetry"]) if r.get("kind") == "obs"]
    assert [r["t"] for r in obs] == [0, 1]
    first = obs[0]["ages"]
    assert first["/odometry/local"]["sensor"] == pytest.approx(0.2)
    assert first["/camera/color/image_raw"]["sensor"] is None  # no header in the fake frame
    assert not first["/odometry/local"]["repeated"]
    last = obs[1]["ages"]
    assert last["/odometry/local"]["repeated"] is False  # waited for a new sample
    assert last["/camera/color/image_raw"]["repeated"] is True  # same frame as before
    assert last["/camera/color/image_raw"]["rx"] >= 0.5
