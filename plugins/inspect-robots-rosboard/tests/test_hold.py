"""Tests for holding each step's command: resends, expiry zeros, config keys, embodiment wiring."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from _stub_server import StubRosboardServer
from test_config import _MINIMAL_CONFIG, _image_payload, _write_config
from test_rosboard_embodiment import _SCENE, _FakeClient, _FakeClock

from inspect_robots.rollout import TrialRecord
from inspect_robots.types import Action
from inspect_robots_rosboard import RosboardEmbodiment
from inspect_robots_rosboard._config import ConfigError, load_robot_config
from inspect_robots_rosboard._hold import CommandHold

_TOPIC = "/motion_control/speed_controller/output_cmd"
_TYPE = "geometry_msgs/msg/TwistStamped"
_CMD = (_TOPIC, _TYPE, {"twist": {"linear": {"x": 0.3}, "angular": {"z": 0.2}}})
_ZERO = (_TOPIC, _TYPE, {"twist": {"linear": {"x": 0.0}, "angular": {"z": 0.0}}})


def _hold(clock: _FakeClock, sent: list[Any], expired: list[float]) -> CommandHold:
    return CommandHold(
        lambda *frame: sent.append(frame),
        hold_s=0.35,
        resend_hz=10.0,
        clock=clock,
        on_expire=expired.append,
        background=False,
    )


def test_hold_resends_until_the_deadline_then_sends_zeros_once() -> None:
    clock = _FakeClock()
    sent: list[Any] = []
    expired: list[float] = []
    hold = _hold(clock, sent, expired)
    hold.arm([_CMD], [_ZERO], sent_at=0.0)
    assert hold.armed

    assert hold.tick() == pytest.approx(0.1)  # nothing due yet: the step itself sent it
    assert sent == []
    for now in (0.1, 0.2, 0.3):
        clock.now = now
        hold.tick()
    assert sent == [_CMD, _CMD, _CMD]

    clock.now = 0.35
    assert hold.tick() is None
    assert sent[-1] == _ZERO and len(sent) == 4
    assert expired == [0.35]
    assert not hold.armed
    assert hold.tick() is None and len(sent) == 4


def test_hold_returns_the_wait_until_the_next_due_time() -> None:
    clock = _FakeClock()
    sent: list[Any] = []
    expired: list[float] = []
    hold = _hold(clock, sent, expired)
    hold.arm([_CMD], [_ZERO], sent_at=0.0)
    clock.now = 0.32
    hold.tick()  # late resend: the next one is not crowded right after it
    assert hold.tick() == pytest.approx(0.03)  # the deadline comes first


def test_disarm_stops_resending_without_zeros() -> None:
    clock = _FakeClock()
    sent: list[Any] = []
    expired: list[float] = []
    hold = _hold(clock, sent, expired)
    hold.arm([_CMD], [_ZERO], sent_at=0.0)
    hold.disarm()
    clock.now = 1.0
    assert hold.tick() is None
    assert sent == [] and expired == []


def test_arming_nothing_or_after_close_holds_nothing() -> None:
    clock = _FakeClock()
    sent: list[Any] = []
    expired: list[float] = []
    hold = _hold(clock, sent, expired)
    hold.arm([], [_ZERO], sent_at=0.0)
    assert not hold.armed
    hold.close()
    hold.arm([_CMD], [_ZERO], sent_at=0.0)
    assert not hold.armed


def test_publish_errors_do_not_escape_the_hold() -> None:
    def broken(*_frame: Any) -> None:
        raise ConnectionError("socket lost")

    clock = _FakeClock()
    expired: list[float] = []
    hold = CommandHold(
        broken, hold_s=0.2, resend_hz=10.0, clock=clock, on_expire=expired.append, background=False
    )
    hold.arm([_CMD], [_ZERO], sent_at=0.0)
    clock.now = 0.1
    hold.tick()
    clock.now = 0.2
    hold.tick()
    assert expired == [0.2]


def test_background_thread_resends_and_expires_on_its_own() -> None:
    sent: list[Any] = []
    expired: list[float] = []
    hold = CommandHold(
        lambda *frame: sent.append(frame),
        hold_s=0.2,
        resend_hz=50.0,
        clock=time.monotonic,
        on_expire=expired.append,
    )
    hold.arm([_CMD], [_ZERO], sent_at=time.monotonic())
    deadline = time.monotonic() + 2.0
    while not expired and time.monotonic() < deadline:
        time.sleep(0.01)
    hold.close()
    assert expired, "the hold never expired"
    assert sent[-1] == _ZERO
    assert 5 <= sent.count(_CMD) <= 11  # about 0.2 s at 50 Hz


def test_config_hold_keys_default_off_and_parse(tmp_path: Path) -> None:
    config = load_robot_config(_write_config(tmp_path))
    assert config.command_hold_s == 0.0 and config.command_resend_hz == 10.0
    assert RosboardEmbodiment(config=_write_config(tmp_path))._hold is None

    text = _MINIMAL_CONFIG.replace(
        "rate_hz: 10.0", "rate_hz: 10.0\ncommand_hold_s: 0.7\ncommand_resend_hz: 20"
    )
    config = load_robot_config(_write_config(tmp_path, text))
    assert config.command_hold_s == 0.7 and config.command_resend_hz == 20.0


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("command_hold_s: -1", "command_hold_s must be >= 0"),
        ("command_hold_s: .nan", "command_hold_s must be >= 0"),
        ("command_resend_hz: 0", "command_resend_hz must be positive"),
    ],
)
def test_config_rejects_bad_hold_keys(tmp_path: Path, line: str, message: str) -> None:
    text = _MINIMAL_CONFIG.replace("rate_hz: 10.0", f"rate_hz: 10.0\n{line}")
    with pytest.raises(ConfigError, match=message):
        load_robot_config(_write_config(tmp_path, text))


def _held_embodiment(
    tmp_path: Path, topics_file: str | None = None
) -> tuple[RosboardEmbodiment, _FakeClient, _FakeClock]:
    text = _MINIMAL_CONFIG.replace(
        "rate_hz: 10.0", "rate_hz: 3.0\ncommand_hold_s: 0.35\nrecord_topics: [/odometry/local]"
    )
    clock = _FakeClock()
    embodiment = RosboardEmbodiment(
        config=_write_config(tmp_path, text),
        topics_file=topics_file,
        clock=clock,
        sleep=clock.sleep,
    )
    assert embodiment._hold is not None
    embodiment._hold._background = False  # the tests drive tick() on the fake clock
    client = _FakeClient(clock)
    client.put("/camera/color/image_raw", _image_payload(np.zeros((2, 3, 3), dtype=np.uint8)))
    odometry = {"twist": {"twist": {"linear": {"x": 1.0}, "angular": {"z": 0.5}}}}
    client.put("/odometry/local", odometry)
    client.on_wait = lambda topic, _seq: client.put(topic, client.samples[topic].payload)
    embodiment._client = cast(Any, client)
    embodiment._initialized = True
    return embodiment, client, clock


def test_embodiment_holds_each_command_and_logs_its_end(tmp_path: Path) -> None:
    embodiment, client, clock = _held_embodiment(tmp_path)
    hold = embodiment._hold
    assert hold is not None
    embodiment.on_trial_start("s0", 0, str(tmp_path), "run1")
    embodiment.reset(_SCENE)

    embodiment.step(Action(data=np.array([0.3, 0.2])))
    command = client.published[-1]
    clock.now += 0.1
    hold.tick()
    assert client.published[-1] == command and len(client.published) == 2

    clock.now += 0.25
    hold.tick()  # 0.35 s old: zeros go out
    assert client.published[-1][2] == {"twist": {"linear": {"x": 0.0}, "angular": {"z": 0.0}}}

    embodiment.step(Action(data=np.array([0.3, -0.2])))  # a new command arms it again
    assert hold.armed
    embodiment.observe_parked()  # the stop disarms it
    assert not hold.armed
    embodiment.on_trial_end(TrialRecord(scene_id="s0", epoch=0, seed=0), str(tmp_path), "run1")

    rows = [json.loads(line) for line in (tmp_path / "telemetry/run1/s0-e0.jsonl").open()]
    assert rows[0]["command_hold_s"] == 0.35 and rows[0]["command_resend_hz"] == 10.0
    assert [row["kind"] for row in rows if row.get("kind") != "header"] == [
        "command",
        "hold_end",
        "command",
        "command",
    ]
    embodiment.close()
    assert hold._closed


def test_dry_run_holds_nothing(tmp_path: Path) -> None:
    topics = tmp_path / "topics.yaml"
    topics.write_text(
        "url: ws://127.0.0.1:1\n"
        "topics: [/camera/color/image_raw, /odometry/local]\n"
        "topics_to_stream: []\n"
    )
    embodiment, client, _clock = _held_embodiment(tmp_path, str(topics))
    embodiment.reset(_SCENE)
    embodiment.step(Action(data=np.array([0.3, 0.2])))
    assert embodiment._hold is not None and not embodiment._hold.armed
    assert client.published == []


def test_held_command_reaches_a_real_socket_then_zeros(
    tmp_path: Path, stub_server: StubRosboardServer
) -> None:
    text = _MINIMAL_CONFIG.replace(
        "rate_hz: 10.0", "rate_hz: 10.0\ncommand_hold_s: 0.3\ncommand_resend_hz: 20"
    )
    embodiment = RosboardEmbodiment(
        config=_write_config(tmp_path, text), url=stub_server.url, obs_timeout_s=2.0
    )
    stop = threading.Event()

    def publish_samples() -> None:
        while not stop.is_set():
            image = np.zeros((2, 3, 3), dtype=np.uint8)
            stub_server.publish("/camera/color/image_raw", _TYPE, _image_payload(image))
            odometry = {"twist": {"twist": {"linear": {"x": 1.0}, "angular": {"z": 0.5}}}}
            stub_server.publish("/odometry/local", "nav_msgs/msg/Odometry", odometry)
            time.sleep(0.02)

    thread = threading.Thread(target=publish_samples, daemon=True)
    thread.start()
    try:
        embodiment.reset(_SCENE)
        embodiment.step(Action(data=np.array([0.3, 0.2])))
        hold = embodiment._hold
        assert hold is not None
        deadline = time.monotonic() + 2.0
        while hold.armed and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        stop.set()
        thread.join(timeout=2)
        embodiment.close()

    sent = [f for f in stub_server.frames_of("m") if f["_topic_name"] == _TOPIC]
    commands = [f for f in sent if f["twist"]["angular"]["z"] == 0.2]
    assert 4 <= len(commands) <= 8  # the step's send plus resends at 20 Hz for 0.3 s
    assert sent[len(commands)]["twist"]["angular"]["z"] == 0.0  # zeros right after the hold


def test_config_values_expand_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = _MINIMAL_CONFIG.replace(
        "rate_hz: 10.0", "rate_hz: 10.0\ncommand_hold_s: ${HOLD_TIME:-0.7}"
    )
    path = _write_config(tmp_path, text)
    monkeypatch.delenv("HOLD_TIME", raising=False)
    assert load_robot_config(path).command_hold_s == 0.7
    monkeypatch.setenv("HOLD_TIME", "0.4")
    assert load_robot_config(path).command_hold_s == 0.4
    monkeypatch.setenv("HOLD_TIME", "0")
    assert RosboardEmbodiment(config=path)._hold is None


def test_config_rejects_an_unset_variable_without_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ROSBOARD_TEST_UNSET", raising=False)
    text = _MINIMAL_CONFIG.replace("rate_hz: 10.0", "rate_hz: ${ROSBOARD_TEST_UNSET}")
    with pytest.raises(ConfigError, match="ROSBOARD_TEST_UNSET is not set and has no default"):
        load_robot_config(_write_config(tmp_path, text))
