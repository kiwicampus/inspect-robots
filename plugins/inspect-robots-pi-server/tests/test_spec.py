"""Tests for PolicySpec parsing and wire-key resolution."""

from __future__ import annotations

import json
from typing import Any

import pytest

from inspect_robots_pi_server._protocol import PiServerError
from inspect_robots_pi_server._spec import parse_policy_spec, payload_key, resolution_for

_BASE_SPEC: dict[str, Any] = {
    "input_spec": {
        "cam_head": [[8, 8, 3], "uint8"],
        "cam_head_mask": [[], "bool"],
        "joint_position": [[4], "float32"],
    },
    "output_spec": {"actions": [[2, 4], "float32"]},
    "action_keys": ["actions"],
    "action_horizon": 2,
    "action_dim": 4,
    "image_preprocess": {
        "target_resolution": [8, 8],
        "image_resolutions": {},
        "resize_mode": "pad",
        "interpolation": "bilinear",
    },
}


def test_parse_policy_spec_happy_path() -> None:
    spec = parse_policy_spec(json.dumps(_BASE_SPEC))
    assert spec.action_keys == ("actions",)
    assert spec.action_horizon == 2
    assert spec.action_dim == 4
    assert spec.camera_names == frozenset({"cam_head"})
    assert spec.image_preprocess.target_resolution == (8, 8)


def test_parse_policy_spec_rejects_invalid_json() -> None:
    with pytest.raises(PiServerError, match="not valid JSON"):
        parse_policy_spec("{not json")


def test_parse_policy_spec_rejects_missing_action_keys() -> None:
    bad = dict(_BASE_SPEC, action_keys=[])
    with pytest.raises(PiServerError, match="action_keys"):
        parse_policy_spec(json.dumps(bad))


def test_parse_policy_spec_rejects_horizon_without_dim() -> None:
    bad = dict(_BASE_SPEC, action_dim=None)
    with pytest.raises(PiServerError, match="must be set together"):
        parse_policy_spec(json.dumps(bad))


def test_parse_policy_spec_allows_both_horizon_and_dim_absent() -> None:
    bad = dict(_BASE_SPEC, action_horizon=None, action_dim=None)
    spec = parse_policy_spec(json.dumps(bad))
    assert spec.action_horizon is None
    assert spec.action_dim is None


def test_parse_policy_spec_rejects_bad_resize_mode() -> None:
    bad = dict(_BASE_SPEC)
    bad["image_preprocess"] = dict(bad["image_preprocess"], resize_mode="crop")
    with pytest.raises(PiServerError, match="resize_mode"):
        parse_policy_spec(json.dumps(bad))


def test_parse_policy_spec_rejects_missing_resolution() -> None:
    bad = dict(_BASE_SPEC)
    bad["image_preprocess"] = dict(
        bad["image_preprocess"], target_resolution=None, image_resolutions={}
    )
    with pytest.raises(PiServerError, match="target_resolution or image_resolutions"):
        parse_policy_spec(json.dumps(bad))


def test_payload_key_prefers_bare_form() -> None:
    assert payload_key("joint_position", {"joint_position": [[4], "float32"]}) == "joint_position"


def test_payload_key_falls_back_to_prefixed_form() -> None:
    assert (
        payload_key("joint_position", {"observation/joint_position": [[4], "float32"]})
        == "observation/joint_position"
    )


def test_payload_key_returns_none_when_unrecognized() -> None:
    assert payload_key("gripper", {"joint_position": [[4], "float32"]}) is None


def test_payload_key_identity_when_input_spec_empty() -> None:
    assert payload_key("anything", {}) == "anything"


def test_resolution_for_prefers_per_camera_override() -> None:
    spec = parse_policy_spec(
        json.dumps(
            dict(
                _BASE_SPEC,
                image_preprocess={
                    "target_resolution": [8, 8],
                    "image_resolutions": {"cam_head": [16, 24]},
                    "resize_mode": "pad",
                    "interpolation": "bilinear",
                },
            )
        )
    )
    assert resolution_for(spec.image_preprocess, "cam_head") == (16, 24)
    assert resolution_for(spec.image_preprocess, "cam_wrist") == (8, 8)
