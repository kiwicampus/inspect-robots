"""The ``Policy`` adapter that drives a PI-protocol policy server.

This is the "brain" half of an Inspect Robots eval, backed by a checkpoint
served over the PI protocol (the wire format ``pi_inference_client`` speaks,
e.g. a ``policy_server``/Modal deployment). It conforms to
:class:`inspect_robots.Policy` (the runtime-checkable protocol), so once
installed it can be paired with any compatible
:class:`inspect_robots.Embodiment`, for example
`inspect-robots-rosboard <../inspect-robots-rosboard/>`_.

Design mirrors `inspect-robots-xpolicylab <../inspect-robots-xpolicylab/>`_'s
laziness: constructing the adapter and reading ``.info`` never touch the
network (so ``inspect-robots list policies`` and fail-fast compatibility
checks work with no server running); the websocket connects, and calls
``load``, on the first ``reset()``/``act()``. A dead socket during ``act()``
is reconnected and the request replayed exactly once (replaying ``load()``
too, since a fresh connection could in principle land on a different-
versioned deployment) before raising.

**A protocol subtlety worth stating plainly**: the ``load`` response's
``action_horizon``/``action_dim`` fields are present on essentially every
real deployment (including this plugin's primary target, ``policy_server``,
which sets them unconditionally per checkpoint) — their presence enables
AsyncRTC-style realtime chunk streaming, but does **not** mean a client must
implement it. This adapter always sends the protocol's own well-defined
non-realtime defaults (a null ``actions`` prefix, zeroed ``initial_noise``)
whenever these fields are present, exactly as the real ``pi_inference_client``
does for a caller that isn't doing realtime prefixing. True overlapped
AsyncRTC (reusing a prediction as the next request's prefix) is a documented
non-goal — see the README's Configuration section.
"""

from __future__ import annotations

import atexit
import os
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

import numpy as np

from inspect_robots import (
    Action,
    ActionChunk,
    ActionSemantics,
    Box,
    CameraSpec,
    Observation,
    ObservationSpace,
    PolicyConfig,
    PolicyInfo,
    Scene,
)
from inspect_robots.errors import ConfigError
from inspect_robots.spaces import ControlMode, Frame, GripperKind, RotationRepr
from inspect_robots_pi_server._client import PolicyClient
from inspect_robots_pi_server._image import (
    jpeg_encode,
    resample_for,
    resize_stretch,
    resize_with_pad,
    validate_image,
)
from inspect_robots_pi_server._protocol import PiServerError
from inspect_robots_pi_server._spec import PolicySpec, payload_key, resolution_for


def _parse_str_mapping(value: str, arg: str) -> dict[str, str]:
    """Parse the compact ``"key:value,key:value"`` form used by CLI ``-P`` args."""
    out: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, val = item.partition(":")
        if not sep or not key.strip() or not val.strip():
            raise ValueError(
                f"{arg} entry {item!r} is not 'key:value'; expected e.g. 'front:cam_head'"
            )
        out[key.strip()] = val.strip()
    return out


def _as_mapping(value: Mapping[str, str] | str | None, arg: str) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, str):
        return _parse_str_mapping(value, arg)
    return dict(value)


def _as_str_tuple(value: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(value)


class PiServerPolicy:
    """An Inspect Robots ``Policy`` served by a PI-protocol policy server."""

    def __init__(
        self,
        *,
        url: str,
        control_mode: ControlMode,
        action_dim: int,
        rotation_repr: RotationRepr = "none",
        gripper: GripperKind = "none",
        frame: Frame = "base",
        action_keys: Sequence[str] | str | None = None,
        cameras: Mapping[str, str] | str | None = None,
        state_map: Mapping[str, str] | str | None = None,
        camera_height: int | None = None,
        camera_width: int | None = None,
        control_hz: float | None = None,
        prompt: str | None = None,
        name: str = "pi_server",
        api_key_env: str = "PI_SERVER_API_KEY",
        env: Mapping[str, str] | None = None,
        connect_open_timeout_s: float = 360.0,
        connect_max_retries: int = 3,
        request_timeout_s: float = 120.0,
        jpeg_quality: int = 85,
    ) -> None:
        if action_dim < 1:
            raise ValueError(f"action_dim must be >= 1, got {action_dim!r}")
        if jpeg_quality < 1 or jpeg_quality > 95:
            raise ValueError(f"jpeg_quality must be in [1, 95], got {jpeg_quality!r}")
        if (camera_height is None) != (camera_width is None):
            raise ValueError("camera_height and camera_width must be given together")

        environ = dict(os.environ) if env is None else env
        self._api_key = environ.get(api_key_env, "")
        self._api_key_env = api_key_env

        self._cameras = _as_mapping(cameras, "cameras")
        self._state_map = _as_mapping(state_map, "state_map")
        self._action_keys_override = _as_str_tuple(action_keys)
        self._prompt = prompt
        self._jpeg_quality = jpeg_quality

        semantics = ActionSemantics(
            control_mode=control_mode, rotation_repr=rotation_repr, gripper=gripper, frame=frame
        )
        camera_specs: tuple[CameraSpec, ...] = ()
        if camera_height is not None and camera_width is not None:
            camera_specs = tuple(
                CameraSpec(name=ir_key, height=camera_height, width=camera_width)
                for ir_key in self._cameras
            )
        self.info = PolicyInfo(
            name=name,
            action_space=Box(shape=(action_dim,), semantics=semantics),
            observation_space=ObservationSpace(
                cameras=camera_specs, state_keys=frozenset(self._state_map)
            ),
            control_hz=control_hz,
        )
        self.config = PolicyConfig()
        self.server_url = url

        self._client = PolicyClient(
            url,
            self._api_key,
            connect_open_timeout_s=connect_open_timeout_s,
            connect_max_retries=connect_max_retries,
            request_timeout_s=request_timeout_s,
        )
        self._resolved_for_spec: PolicySpec | None = None
        self._state_wire_keys: dict[str, str] = {}
        self._camera_wire_keys: dict[str, str] = {}
        self._camera_mask_needed: frozenset[str] = frozenset()
        self._prefix_info: dict[str, Any] | None = None
        self._initial_noise_shape: tuple[int, int] | None = None
        self._instruction: str | None = None
        self._connected_once = False
        self._closed = False
        # `eval()` closes embodiments it resolves, not policies; the atexit
        # hook is the safety net for registry-resolved CLI runs.
        atexit.register(self._atexit_close)

    # ------------------------------------------------------------------ #
    # Policy protocol
    # ------------------------------------------------------------------ #

    def reset(self, scene: Scene) -> None:
        """Connect lazily (if needed), reset the server's episode state."""
        if self._closed:
            raise RuntimeError("PiServerPolicy is closed")
        self._ensure_connected()
        self._client.reset()
        self._instruction = scene.instruction
        self._connected_once = True

    def act(self, observation: Observation) -> ActionChunk:
        """One inference round trip: build the request, infer, assemble the action chunk."""
        if self._closed:
            raise RuntimeError("PiServerPolicy is closed")
        if not self._connected_once:
            raise RuntimeError("act() called before reset(); call reset(scene) first")
        self._ensure_connected()

        start = time.perf_counter()
        try:
            request = self._build_infer_request(observation)
            result = self._client.infer(request)
        except ConnectionError:
            self._ensure_connected()
            request = self._build_infer_request(observation)
            result = self._client.infer(request)
        latency_s = time.perf_counter() - start

        actions = self._assemble_action_chunk(result)
        meta: dict[str, Any] = {
            key: result[key]
            for key in ("server_processing_time_ms", "server_total_time_ms")
            if key in result
        }
        return ActionChunk(
            actions=actions,
            control_hz=self.info.control_hz,
            inference_latency_s=latency_s,
            meta=meta,
        )

    def close(self) -> None:
        """Drop the websocket connection; idempotent."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self._atexit_close)
        self._client.close()

    def __enter__(self) -> PiServerPolicy:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _atexit_close(self) -> None:
        with suppress(Exception):
            self.close()

    def _ensure_connected(self) -> PolicySpec:
        try:
            spec = self._client.connect()
        except ConnectionError as exc:
            if not self._api_key:
                raise ConnectionError(
                    f"{exc}\nNo API key was found in ${self._api_key_env}. If this server "
                    f"requires one, set ${self._api_key_env} or pass -P api_key_env=NAME."
                ) from exc
            raise
        if spec is not self._resolved_for_spec:
            self._resolve_wire_keys(spec)
            self._resolved_for_spec = spec
        return spec

    def _resolve_wire_keys(self, spec: PolicySpec) -> None:
        state_wire: dict[str, str] = {}
        for ir_key, sub_key in self._state_map.items():
            wire_key = payload_key(sub_key, spec.input_spec)
            if wire_key is None:
                raise ConfigError(
                    f"pi_server policy: state_map key {ir_key!r} -> {sub_key!r} is not recognized "
                    f"by the connected server (tried {sub_key!r} and 'observation/{sub_key}'); "
                    f"server input_spec keys: {sorted(spec.input_spec)}"
                )
            state_wire[ir_key] = wire_key

        camera_wire: dict[str, str] = {}
        mask_needed: set[str] = set()
        for ir_key, sub_key in self._cameras.items():
            wire_key = payload_key(sub_key, spec.input_spec)
            if wire_key is None or wire_key not in spec.camera_names:
                raise ConfigError(
                    f"pi_server policy: cameras key {ir_key!r} -> {sub_key!r} is not recognized as "
                    f"a camera by the connected server (tried {sub_key!r} and "
                    f"'observation/{sub_key}'); server cameras: {sorted(spec.camera_names)}"
                )
            camera_wire[ir_key] = wire_key
            if f"{wire_key}_mask" in spec.input_spec:
                mask_needed.add(wire_key)

        self._state_wire_keys = state_wire
        self._camera_wire_keys = camera_wire
        self._camera_mask_needed = frozenset(mask_needed)

        if spec.action_horizon is not None and spec.action_dim is not None:
            self._initial_noise_shape = (spec.action_horizon, spec.action_dim)
            self._prefix_info = {
                "actions": None,
                "t": 0.0,
                "change_start": 0,
                "max_horizon": 0,
                "max_guidance_weight": 0.0,
                "renorm_min": 0.0,
            }
        else:
            self._initial_noise_shape = None
            self._prefix_info = None

    def _build_infer_request(self, observation: Observation) -> dict[str, Any]:
        assert self._resolved_for_spec is not None
        spec = self._resolved_for_spec

        state: dict[str, Any] = {}
        for ir_key, wire_key in self._state_wire_keys.items():
            if ir_key in observation.state:
                state[wire_key] = np.asarray(observation.state[ir_key], dtype=np.float32)

        image: dict[str, Any] = {}
        for ir_key, wire_key in self._camera_wire_keys.items():
            if ir_key not in observation.images:
                raise ConfigError(
                    f"pi_server policy: observation has no image {ir_key!r} (mapped to server "
                    f"camera {wire_key!r}); available images: {sorted(observation.images)}"
                )
            source = observation.images[ir_key]
            validate_image(source, f"observation.images[{ir_key!r}]")
            height, width = resolution_for(spec.image_preprocess, wire_key)
            resample = resample_for(spec.image_preprocess.interpolation)
            if spec.image_preprocess.resize_mode == "pad":
                resized = resize_with_pad(source, height, width, resample)
            else:
                resized = resize_stretch(source, height, width, resample)
            image[wire_key] = jpeg_encode(resized, quality=self._jpeg_quality)
            if wire_key in self._camera_mask_needed:
                image[f"{wire_key}_mask"] = True

        prompt = observation.instruction or self._instruction or self._prompt
        inputs: dict[str, str] = {}
        if prompt:
            inputs["prompt"] = prompt
            inputs["robot_task_string"] = prompt

        initial_noise = (
            np.zeros(self._initial_noise_shape, dtype=np.float32)
            if self._initial_noise_shape is not None
            else None
        )
        inference_input: dict[str, Any] = {
            "state": state,
            "image": image,
            "inputs": inputs,
            "prefix_info": self._prefix_info,
            "initial_noise": initial_noise,
            "extra": None,
            "return_context": False,
            "timestep_mask": True if "timestep_mask" in spec.input_spec else None,
        }
        return {"inference_input": inference_input, "encode_as_video": False}

    def _assemble_action_chunk(self, payload: dict[str, Any]) -> list[Action]:
        assert self._resolved_for_spec is not None
        result = payload.get("result")
        outputs = result.get("outputs") if isinstance(result, dict) else None
        if not isinstance(outputs, dict):
            raise PiServerError(
                "ProtocolError", "infer() result is missing an object 'outputs' field"
            )

        action_keys = self._action_keys_override or self._resolved_for_spec.action_keys
        arrays: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        for key in action_keys:
            value = outputs.get(key)
            if value is None:
                raise PiServerError(
                    "ProtocolError",
                    f"infer() outputs is missing action key {key!r}; got {sorted(outputs)}",
                )
            arr = np.asarray(value, dtype=np.float64)
            if arr.ndim != 2:
                raise PiServerError(
                    "ProtocolError",
                    f"infer() outputs[{key!r}] must be 2-D (horizon, dim), got {arr.shape}",
                )
            arrays.append(arr)

        if len({arr.shape[0] for arr in arrays}) != 1:
            raise PiServerError(
                "ProtocolError",
                f"infer() outputs for {action_keys} disagree on horizon length: "
                f"{[arr.shape for arr in arrays]}",
            )
        combined = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=-1)

        expected_dim = self.info.action_space.shape[0]
        if combined.shape[1] != expected_dim:
            raise PiServerError(
                "ProtocolError",
                f"assembled action last-dim {combined.shape[1]} != declared action_dim "
                f"{expected_dim}; check -P action_dim= against the checkpoint served",
            )
        return [Action(data=combined[step]) for step in range(combined.shape[0])]
