"""Tests for the PiServerPolicy adapter, against a real in-process stub server."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _stub_server import StubPiServer

from inspect_robots.compat import check_compatibility
from inspect_robots.embodiment import EmbodimentInfo
from inspect_robots.errors import ConfigError
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, ObservationSpace
from inspect_robots.types import Observation
from inspect_robots_pi_server import PiServerPolicy, pi_server_policy
from inspect_robots_pi_server._protocol import PiServerError

_SCENE = Scene(id="s0", instruction="drive forward")


def _policy(server: StubPiServer, **kwargs: object) -> PiServerPolicy:
    kwargs.setdefault("url", server.url)
    kwargs.setdefault("control_mode", "base_velocity")
    kwargs.setdefault("action_dim", server.action_dim)
    kwargs.setdefault("cameras", "front:cam_head")
    kwargs.setdefault("state_map", "odom:joint_position")
    kwargs.setdefault("camera_height", 8)
    kwargs.setdefault("camera_width", 8)
    kwargs.setdefault("request_timeout_s", 1.0)
    kwargs.setdefault("connect_open_timeout_s", 1.0)
    kwargs.setdefault("connect_max_retries", 1)
    return PiServerPolicy(**kwargs)  # type: ignore[arg-type]


def _observation() -> Observation:
    return Observation(
        images={"front": np.zeros((16, 16, 3), dtype=np.uint8)},
        state={"odom": np.zeros((4,), dtype=np.float32)},
        instruction="drive forward",
    )


# --------------------------------------------------------------------------
# Construction (network-free)
# --------------------------------------------------------------------------


def test_construct_and_info_touch_no_network() -> None:
    policy = PiServerPolicy(
        url="ws://198.51.100.1:1",
        control_mode="base_velocity",
        action_dim=4,
        cameras="front:cam_head",
        state_map="odom:joint_position",
        camera_height=8,
        camera_width=8,
    )
    assert policy.info.action_space.shape == (4,)
    assert policy.info.action_space.semantics is not None
    assert policy.info.action_space.semantics.control_mode == "base_velocity"
    assert [c.name for c in policy.info.observation_space.cameras] == ["front"]
    assert policy.info.observation_space.state_keys == frozenset({"odom"})


def test_action_dim_must_be_positive() -> None:
    with pytest.raises(ValueError, match="action_dim"):
        PiServerPolicy(url="ws://x", control_mode="base_velocity", action_dim=0)


def test_camera_height_and_width_required_together() -> None:
    with pytest.raises(ValueError, match="camera_height and camera_width"):
        PiServerPolicy(url="ws://x", control_mode="base_velocity", action_dim=2, camera_height=8)


def test_pi_server_policy_factory_function() -> None:
    policy = pi_server_policy(url="ws://x", control_mode="base_velocity", action_dim=2)
    assert isinstance(policy, PiServerPolicy)


# --------------------------------------------------------------------------
# reset() / act() against the real stub server
# --------------------------------------------------------------------------


def test_reset_and_act_returns_action_chunk(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    chunk = policy.act(_observation())

    assert len(chunk.actions) == stub_server.action_horizon
    for action in chunk.actions:
        assert action.data.shape == (stub_server.action_dim,)
        assert action.data.dtype == np.float64
    assert chunk.meta["server_processing_time_ms"] == 1.0
    policy.close()


def test_act_before_reset_raises(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server)
    with pytest.raises(RuntimeError, match="before reset"):
        policy.act(_observation())


def test_state_key_bare_form(stub_server: StubPiServer) -> None:
    # _policy()'s default state_map is "odom:joint_position": the IR key is
    # "odom", the sub-key checked against input_spec is "joint_position".
    stub_server.input_spec = {"cam_head": [[8, 8, 3], "uint8"], "joint_position": [[4], "float32"]}
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    policy.act(_observation())
    infer_req = next(r for r in stub_server.requests() if r[0] == "infer")[1]
    assert set(infer_req["inference_input"]["state"]) == {"joint_position"}
    policy.close()


def test_state_key_prefixed_form(stub_server: StubPiServer) -> None:
    stub_server.input_spec = {
        "cam_head": [[8, 8, 3], "uint8"],
        "observation/joint_position": [[4], "float32"],
    }
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    policy.act(_observation())
    infer_req = next(r for r in stub_server.requests() if r[0] == "infer")[1]
    assert set(infer_req["inference_input"]["state"]) == {"observation/joint_position"}
    policy.close()


def test_unrecognized_state_key_raises_config_error(stub_server: StubPiServer) -> None:
    stub_server.input_spec = {"cam_head": [[8, 8, 3], "uint8"]}
    policy = _policy(stub_server)
    with pytest.raises(ConfigError, match="state_map"):
        policy.reset(_SCENE)


def test_unrecognized_camera_raises_config_error(stub_server: StubPiServer) -> None:
    stub_server.input_spec = {"joint_position": [[4], "float32"]}
    policy = _policy(stub_server)
    with pytest.raises(ConfigError, match="cameras"):
        policy.reset(_SCENE)


def test_camera_mask_sent_when_declared(stub_server: StubPiServer) -> None:
    stub_server.input_spec = {
        "cam_head": [[8, 8, 3], "uint8"],
        "cam_head_mask": [[], "bool"],
        "joint_position": [[4], "float32"],
    }
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    policy.act(_observation())
    infer_req = next(r for r in stub_server.requests() if r[0] == "infer")[1]
    assert infer_req["inference_input"]["image"]["cam_head_mask"] is True
    policy.close()


def test_camera_mask_absent_when_not_declared(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    policy.act(_observation())
    infer_req = next(r for r in stub_server.requests() if r[0] == "infer")[1]
    assert "cam_head_mask" not in infer_req["inference_input"]["image"]
    policy.close()


def test_missing_image_in_observation_raises_config_error(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    obs = Observation(images={}, state={"odom": np.zeros((4,), dtype=np.float32)})
    with pytest.raises(ConfigError, match="no image"):
        policy.act(obs)
    policy.close()


def test_image_resized_to_advertised_resolution_before_encode(stub_server: StubPiServer) -> None:
    from io import BytesIO

    from PIL import Image

    stub_server.target_resolution = (12, 20)
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    obs = Observation(
        images={"front": np.zeros((64, 48, 3), dtype=np.uint8)},
        state={"odom": np.zeros((4,), dtype=np.float32)},
    )
    policy.act(obs)
    infer_req = next(r for r in stub_server.requests() if r[0] == "infer")[1]
    encoded = infer_req["inference_input"]["image"]["cam_head"]
    decoded = Image.open(BytesIO(encoded))
    assert decoded.size == (20, 12)  # PIL size is (width, height)
    policy.close()


def test_action_keys_local_override_wins(stub_server: StubPiServer) -> None:
    stub_server.action_keys = ["actions", "aux"]
    policy = _policy(stub_server, action_keys="actions")
    policy.reset(_SCENE)
    chunk = policy.act(_observation())
    assert len(chunk.actions) == stub_server.action_horizon
    for action in chunk.actions:
        assert action.data.shape == (stub_server.action_dim,)
    policy.close()


def test_action_dim_mismatch_raises(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server, action_dim=stub_server.action_dim + 1)
    policy.reset(_SCENE)
    with pytest.raises(PiServerError, match="assembled action last-dim"):
        policy.act(_observation())
    policy.close()


def test_missing_action_key_in_response_raises(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server, action_keys="actions,missing_key")
    policy.reset(_SCENE)
    with pytest.raises(PiServerError, match="missing action key"):
        policy.act(_observation())
    policy.close()


def test_action_horizon_dim_present_sends_null_prefix_defaults(stub_server: StubPiServer) -> None:
    assert stub_server.action_horizon is not None  # the default fixture already advertises it
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    policy.act(_observation())
    infer_req = next(r for r in stub_server.requests() if r[0] == "infer")[1]
    assert infer_req["inference_input"]["prefix_info"] == {
        "actions": None,
        "t": 0.0,
        "change_start": 0,
        "max_horizon": 0,
        "max_guidance_weight": 0.0,
        "renorm_min": 0.0,
    }
    np.testing.assert_array_equal(
        infer_req["inference_input"]["initial_noise"],
        np.zeros((stub_server.action_horizon, stub_server.action_dim), dtype=np.float32),
    )
    policy.close()


def test_action_horizon_dim_absent_sends_null_fields(stub_server: StubPiServer) -> None:
    stub_server.action_horizon = None
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    policy.act(_observation())
    infer_req = next(r for r in stub_server.requests() if r[0] == "infer")[1]
    assert infer_req["inference_input"]["prefix_info"] is None
    assert infer_req["inference_input"]["initial_noise"] is None
    policy.close()


def test_server_error_frame_surfaces_and_connection_stays_usable(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    obs = Observation(
        images={"front": np.zeros((16, 16, 3), dtype=np.uint8)},
        state={"odom": np.zeros((4,), dtype=np.float32)},
        instruction="__error__",
    )
    with pytest.raises(PiServerError, match="InferenceError: boom"):
        policy.act(obs)
    # The connection should still be usable for a normal request afterward.
    chunk = policy.act(_observation())
    assert len(chunk.actions) == stub_server.action_horizon
    policy.close()


def test_transport_text_frame_raises_connection_error(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server, connect_max_retries=1)
    policy.reset(_SCENE)
    obs = Observation(
        images={"front": np.zeros((16, 16, 3), dtype=np.uint8)},
        state={"odom": np.zeros((4,), dtype=np.float32)},
        instruction="__text__",
    )
    # A dead connection after the text frame triggers one reconnect+retry,
    # which also fails (server sends another text frame on the retry) —
    # the failure surfaces as a ConnectionError from the retried infer.
    with pytest.raises(ConnectionError):
        policy.act(obs)
    policy.close()


def test_hang_triggers_timeout(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server, request_timeout_s=0.2, connect_max_retries=1)
    policy.reset(_SCENE)
    obs = Observation(
        images={"front": np.zeros((16, 16, 3), dtype=np.uint8)},
        state={"odom": np.zeros((4,), dtype=np.float32)},
        instruction="__hang__",
    )
    with pytest.raises((TimeoutError, ConnectionError)):
        policy.act(obs)
    policy.close()


def test_close_is_idempotent(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server)
    policy.reset(_SCENE)
    policy.close()
    policy.close()


def test_close_before_reset_is_a_noop(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server)
    policy.close()


def test_api_key_sent_and_required(stub_server: StubPiServer) -> None:
    server = StubPiServer(require_api_key="secret")
    try:
        policy = _policy(server, env={"PI_SERVER_API_KEY": "secret"}, connect_open_timeout_s=2.0)
        policy.reset(_SCENE)
        policy.close()
    finally:
        server.stop()


def test_missing_api_key_fails_to_connect(stub_server: StubPiServer) -> None:
    server = StubPiServer(require_api_key="secret")
    try:
        policy = _policy(server, env={}, connect_open_timeout_s=1.0, connect_max_retries=1)
        with pytest.raises(ConnectionError, match="PI_SERVER_API_KEY"):
            policy.reset(_SCENE)
    finally:
        server.stop()


# --------------------------------------------------------------------------
# Compatibility with a real embodiment
# --------------------------------------------------------------------------


def test_compatible_with_matching_embodiment(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server)
    embodiment_info = EmbodimentInfo(
        name="fake",
        action_space=Box(
            shape=(stub_server.action_dim,),
            semantics=ActionSemantics(control_mode="base_velocity", rotation_repr="none"),
        ),
        observation_space=ObservationSpace(
            cameras=policy.info.observation_space.cameras, state_keys=frozenset({"odom"})
        ),
    )
    report = check_compatibility(policy, _FakeEmbodiment(embodiment_info))
    assert report.ok, report.errors


def test_incompatible_action_dim_fails_fast(stub_server: StubPiServer) -> None:
    policy = _policy(stub_server)
    embodiment_info = EmbodimentInfo(
        name="fake",
        action_space=Box(
            shape=(stub_server.action_dim + 1,),
            semantics=ActionSemantics(control_mode="base_velocity", rotation_repr="none"),
        ),
        observation_space=ObservationSpace(),
    )
    report = check_compatibility(policy, _FakeEmbodiment(embodiment_info))
    assert not report.ok
    assert any(issue.code == "action_dim" for issue in report.errors)


class _FakeEmbodiment:
    def __init__(self, info: EmbodimentInfo) -> None:
        self.info = info

    def reset(self, scene: Scene, *, seed: int | None = None) -> Any:
        raise NotImplementedError

    def step(self, action: Any) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        pass
