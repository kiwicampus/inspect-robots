"""Tests for the LeRobot policy adapter, against a real (untrained) synthetic checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inspect_robots.scene import Scene
from inspect_robots.types import Observation
from inspect_robots_lerobot import LeRobotPolicy, lerobot_policy
from inspect_robots_lerobot.policy import _as_mapping, _as_str_tuple, _resolve_device

_SCENE = Scene(id="s0", instruction="drive forward")


def _image(height: int = 64, width: int = 64) -> np.ndarray[Any, np.dtype[np.uint8]]:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _observation() -> Observation:
    return Observation(
        images={"front": _image()},
        state={"state": np.zeros((4,), dtype=np.float32)},
        instruction="drive forward",
    )


def _policy(checkpoint: Path, **kwargs: object) -> LeRobotPolicy:
    kwargs.setdefault("image_keys", "front:observation.images.front")
    kwargs.setdefault("state_keys", "state")
    kwargs.setdefault("device", "cpu")
    return LeRobotPolicy(checkpoint=str(checkpoint), **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Small pure-function unit tests
# --------------------------------------------------------------------------


def test_resolve_device_passes_through_explicit_value() -> None:
    assert _resolve_device("cpu") == "cpu"


def test_resolve_device_auto_resolves_to_cpu_or_cuda() -> None:
    import torch

    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert _resolve_device("auto") == expected


def test_as_mapping_parses_string_form() -> None:
    assert _as_mapping("a:b,c:d", "arg") == {"a": "b", "c": "d"}


def test_as_mapping_passes_through_dict() -> None:
    assert _as_mapping({"a": "b"}, "arg") == {"a": "b"}


@pytest.mark.parametrize("bad", ["a", "a:", ":b"])
def test_as_mapping_rejects_malformed_entries(bad: str) -> None:
    with pytest.raises(ValueError, match="not 'key:value'"):
        _as_mapping(bad, "arg")


def test_as_mapping_ignores_a_trailing_comma() -> None:
    assert _as_mapping("a:b,", "arg") == {"a": "b"}


def test_as_str_tuple_parses_string_and_sequence() -> None:
    assert _as_str_tuple("a, b ,c") == ("a", "b", "c")
    assert _as_str_tuple(["a", "b"]) == ("a", "b")


# --------------------------------------------------------------------------
# Construction against the real synthetic checkpoint
# --------------------------------------------------------------------------


def test_info_derived_from_checkpoint_feature_shapes(synthetic_checkpoint: Path) -> None:
    policy = _policy(synthetic_checkpoint)
    info = policy.info
    assert info.action_space.shape == (2,)
    assert info.action_space.semantics is not None
    assert info.action_space.semantics.control_mode == "base_velocity"
    assert [(c.name, c.height, c.width) for c in info.observation_space.cameras] == [
        ("front", 64, 64)
    ]
    assert info.observation_space.state_keys == frozenset({"state"})


def test_construction_accepts_mapping_form_of_image_and_state_keys(
    synthetic_checkpoint: Path,
) -> None:
    policy = _policy(
        synthetic_checkpoint,
        image_keys={"front": "observation.images.front"},
        state_keys=["state"],
    )
    assert policy.info.observation_space.state_keys == frozenset({"state"})


def test_lerobot_policy_factory_function(synthetic_checkpoint: Path) -> None:
    policy = lerobot_policy(
        checkpoint=str(synthetic_checkpoint),
        image_keys="front:observation.images.front",
        state_keys="state",
        device="cpu",
    )
    assert isinstance(policy, LeRobotPolicy)


def test_construction_requires_image_keys_when_checkpoint_has_image_features(
    synthetic_checkpoint: Path,
) -> None:
    with pytest.raises(ValueError, match="image_keys"):
        LeRobotPolicy(checkpoint=str(synthetic_checkpoint), state_keys="state", device="cpu")


def test_construction_rejects_image_keys_missing_a_checkpoint_feature(
    synthetic_checkpoint: Path,
) -> None:
    with pytest.raises(ValueError, match="missing checkpoint feature"):
        LeRobotPolicy(
            checkpoint=str(synthetic_checkpoint),
            image_keys="",
            state_keys="state",
            device="cpu",
        )


def test_construction_rejects_image_keys_naming_an_unknown_checkpoint_feature(
    synthetic_checkpoint: Path,
) -> None:
    with pytest.raises(ValueError, match="does not have"):
        LeRobotPolicy(
            checkpoint=str(synthetic_checkpoint),
            image_keys="front:observation.images.front,extra:observation.images.bogus",
            state_keys="state",
            device="cpu",
        )


def test_construction_requires_state_keys_when_checkpoint_has_a_state_feature(
    synthetic_checkpoint: Path,
) -> None:
    with pytest.raises(ValueError, match="state_keys"):
        LeRobotPolicy(
            checkpoint=str(synthetic_checkpoint),
            image_keys="front:observation.images.front",
            device="cpu",
        )


def test_construction_fails_clearly_for_a_nonexistent_checkpoint() -> None:
    with pytest.raises(RuntimeError, match="could not load LeRobot checkpoint"):
        LeRobotPolicy(checkpoint="/nonexistent/checkpoint/dir", device="cpu")


# --------------------------------------------------------------------------
# reset() / act() against the real synthetic checkpoint
# --------------------------------------------------------------------------


def test_act_returns_a_correctly_shaped_action_chunk(synthetic_checkpoint: Path) -> None:
    policy = _policy(synthetic_checkpoint)
    policy.reset(_SCENE)
    chunk = policy.act(_observation())

    assert len(chunk.actions) == 10  # n_action_steps from the fixture's ACTConfig
    for action in chunk.actions:
        assert action.data.shape == (2,)
        assert action.data.dtype == np.float64
        assert np.all(np.isfinite(action.data))
    assert chunk.inference_latency_s is not None
    assert chunk.inference_latency_s >= 0


def test_act_uses_observation_instruction_over_configured_task(synthetic_checkpoint: Path) -> None:
    # Not directly observable from the chunk (ACT ignores task), but this
    # exercises the instruction-fallback branch without raising.
    policy = _policy(synthetic_checkpoint, task="fallback task")
    policy.reset(_SCENE)
    chunk = policy.act(_observation())
    assert len(chunk.actions) == 10


def test_act_raises_on_missing_mapped_image(synthetic_checkpoint: Path) -> None:
    policy = _policy(synthetic_checkpoint)
    policy.reset(_SCENE)
    obs = Observation(images={}, state={"state": np.zeros((4,), dtype=np.float32)})
    with pytest.raises(KeyError, match="observation has no image 'front'"):
        policy.act(obs)


def test_act_raises_on_missing_mapped_state_key(synthetic_checkpoint: Path) -> None:
    policy = _policy(synthetic_checkpoint)
    policy.reset(_SCENE)
    obs = Observation(images={"front": _image()}, state={})
    with pytest.raises(KeyError, match="observation has no state 'state'"):
        policy.act(obs)


def test_act_raises_on_wrong_state_dimension(synthetic_checkpoint: Path) -> None:
    policy = _policy(synthetic_checkpoint)
    policy.reset(_SCENE)
    obs = Observation(images={"front": _image()}, state={"state": np.zeros((3,), dtype=np.float32)})
    with pytest.raises(ValueError, match="checkpoint expects \\(4,\\)"):
        policy.act(obs)


def test_reset_is_idempotent_and_does_not_raise(synthetic_checkpoint: Path) -> None:
    policy = _policy(synthetic_checkpoint)
    policy.reset(_SCENE)
    policy.reset(_SCENE)
    chunk = policy.act(_observation())
    assert len(chunk.actions) == 10


# --------------------------------------------------------------------------
# Compatibility with a real embodiment (inspect-robots-rosboard)
# --------------------------------------------------------------------------


def test_compatible_with_rosboard_embodiment_via_remap(synthetic_checkpoint: Path) -> None:
    from inspect_robots_rosboard import RosboardEmbodiment

    from inspect_robots.compat import check_compatibility

    policy = _policy(synthetic_checkpoint)
    embodiment = RosboardEmbodiment(camera_height=64, camera_width=64, camera_name="front")
    report = check_compatibility(policy, embodiment, remap={"state": "odom_twist"})
    assert report.ok, report.errors


def test_incompatible_with_rosboard_embodiment_without_matching_state_key(
    synthetic_checkpoint: Path,
) -> None:
    from inspect_robots_rosboard import RosboardEmbodiment

    from inspect_robots.compat import check_compatibility

    policy = _policy(synthetic_checkpoint)
    embodiment = RosboardEmbodiment(camera_height=64, camera_width=64, camera_name="front")
    report = check_compatibility(policy, embodiment)
    assert not report.ok
    assert any(issue.code == "missing_state" for issue in report.errors)
