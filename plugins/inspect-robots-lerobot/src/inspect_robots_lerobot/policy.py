"""The ``Policy`` adapter that runs a local LeRobot checkpoint in-process.

This is the "brain" half of an Inspect Robots eval, backed directly by a
trained LeRobot checkpoint (ACT, Diffusion Policy, SmolVLA, or any other
architecture registered with ``lerobot.policies.factory.get_policy_class``) —
no inference server, no websocket, unlike
`inspect-robots-xpolicylab <../inspect-robots-xpolicylab/>`_. It conforms to
:class:`inspect_robots.Policy` (the runtime-checkable protocol), so once
installed it can be paired with any compatible
:class:`inspect_robots.Embodiment`, for example
`inspect-robots-rosboard <../inspect-robots-rosboard/>`_.

Checkpoint loading (``PreTrainedConfig.from_pretrained`` +
``<PolicyClass>.from_pretrained`` + ``make_pre_post_processors``) happens at
construction, not lazily on first ``act()``: this is a local disk read and a
weights-to-device move, not a network call, and no CLI command constructs a
policy just to list it, so there is no laziness benefit to defer it for.

Observation/action mapping:

- ``Observation.images[ir_key]`` -> the checkpoint's ``observation.images.<cam>``
  feature named by ``image_keys``.
- ``Observation.state[ir_key]`` for each key in ``state_keys``, concatenated
  in order, -> the checkpoint's single ``observation.state`` feature.
- ``policy.predict_action_chunk()`` (not ``select_action()``, which owns an
  internal per-architecture action queue/temporal ensembler that would
  double-buffer against Inspect Robots' own
  :class:`inspect_robots.controller.DefaultController` chunk buffering) ->
  one :class:`inspect_robots.Action` per predicted step, forming the returned
  :class:`inspect_robots.ActionChunk`.

Verified against the actually-installed lerobot 0.4.4 source (this API has
changed across lerobot versions); **not** verified against a real trained
checkpoint end to end, since none was available while writing this. Run
``inspect-robots doctor --policy lerobot -P checkpoint=...`` and a short
``scripted``-policy-free dry run before trusting this against real hardware.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.utils.constants import OBS_STATE

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
from inspect_robots.spaces import (
    ControlMode,
    Frame,
    GripperKind,
    RotationRepr,
)


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` when available, else ``"cpu"``; else pass through."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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
                f"{arg} entry {item!r} is not 'key:value'; expected e.g. "
                "'front:observation.images.laptop'"
            )
        out[key.strip()] = val.strip()
    return out


def _as_mapping(value: Mapping[str, str] | str, arg: str) -> dict[str, str]:
    if isinstance(value, str):
        return _parse_str_mapping(value, arg)
    return dict(value)


def _as_str_tuple(value: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(value)


def _load_policy(checkpoint: str, device: str) -> PreTrainedPolicy:
    try:
        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        policy_cls = get_policy_class(config.type)
        return policy_cls.from_pretrained(checkpoint, config=config)
    except Exception as exc:
        raise RuntimeError(f"could not load LeRobot checkpoint at {checkpoint!r}: {exc}") from exc


class LeRobotPolicy:
    """An Inspect Robots ``Policy`` backed by a locally loaded LeRobot checkpoint."""

    def __init__(
        self,
        *,
        checkpoint: str,
        image_keys: Mapping[str, str] | str | None = None,
        state_keys: Sequence[str] | str | None = None,
        device: str = "auto",
        task: str | None = None,
        robot_type: str | None = None,
        control_mode: ControlMode = "base_velocity",
        rotation_repr: RotationRepr = "none",
        gripper: GripperKind = "none",
        frame: Frame = "base",
        control_hz: float | None = None,
        name: str = "lerobot",
    ) -> None:
        resolved_device = _resolve_device(device)
        self._policy = _load_policy(checkpoint, resolved_device)
        self._device = resolved_device
        self._preprocessor, self._postprocessor = make_pre_post_processors(
            self._policy.config, pretrained_path=checkpoint
        )
        self._task = task
        self._robot_type = robot_type

        action_feature = self._policy.config.action_feature
        if action_feature is None:
            raise ValueError(f"checkpoint {checkpoint!r} declares no 'action' output feature")
        action_dim = action_feature.shape[0]

        image_features = self._policy.config.image_features
        if image_features and image_keys is None:
            raise ValueError(
                f"checkpoint {checkpoint!r} expects image feature(s) "
                f"{sorted(image_features)} but no image_keys mapping was given; pass "
                f"-P image_keys='<observation-key>:{next(iter(image_features))},...'"
            )
        self._image_keys = _as_mapping(image_keys, "image_keys") if image_keys is not None else {}
        unknown = set(self._image_keys.values()) - set(image_features)
        if unknown:
            raise ValueError(
                f"image_keys maps to checkpoint feature(s) {sorted(unknown)} the checkpoint "
                f"does not have; it expects {sorted(image_features)}"
            )
        missing = set(image_features) - set(self._image_keys.values())
        if missing:
            raise ValueError(
                f"image_keys is missing checkpoint feature(s) {sorted(missing)}; "
                f"it expects {sorted(image_features)}"
            )

        state_feature = self._policy.config.robot_state_feature
        if state_feature is not None and state_keys is None:
            raise ValueError(
                f"checkpoint {checkpoint!r} expects a {state_feature.shape[0]}-D "
                f"{OBS_STATE!r} but no state_keys was given; pass "
                "-P state_keys='ir_key1,ir_key2,...' (concatenated in that order)"
            )
        self._state_keys = _as_str_tuple(state_keys) if state_keys is not None else ()
        self._state_dim = state_feature.shape[0] if state_feature is not None else 0

        semantics = ActionSemantics(
            control_mode=control_mode, rotation_repr=rotation_repr, gripper=gripper, frame=frame
        )
        camera_specs = tuple(
            CameraSpec(
                name=ir_key,
                height=image_features[ckpt_key].shape[1],
                width=image_features[ckpt_key].shape[2],
            )
            for ir_key, ckpt_key in self._image_keys.items()
        )
        self.info = PolicyInfo(
            name=name,
            action_space=Box(shape=(action_dim,), semantics=semantics),
            observation_space=ObservationSpace(
                cameras=camera_specs, state_keys=frozenset(self._state_keys)
            ),
            control_hz=control_hz,
        )
        self.config = PolicyConfig()

    # ------------------------------------------------------------------ #
    # Policy protocol
    # ------------------------------------------------------------------ #

    def reset(self, scene: Scene) -> None:
        """Clear the checkpoint's own action queue/temporal-ensembler state for a fresh trial."""
        self._policy.reset()

    def act(self, observation: Observation) -> ActionChunk:
        """One local inference call: build the checkpoint's observation dict, predict a chunk."""
        obs_dict: dict[str, np.ndarray[Any, Any]] = {}
        for ir_key, ckpt_key in self._image_keys.items():
            image = observation.images.get(ir_key)
            if image is None:
                raise KeyError(
                    f"observation has no image {ir_key!r} (mapped to checkpoint feature "
                    f"{ckpt_key!r}); available images: {sorted(observation.images)}"
                )
            obs_dict[ckpt_key] = np.asarray(image)

        if self._state_keys:
            parts: list[np.ndarray[Any, Any]] = []
            for ir_key in self._state_keys:
                value = observation.state.get(ir_key)
                if value is None:
                    raise KeyError(
                        f"observation has no state {ir_key!r}; available state: "
                        f"{sorted(observation.state)}"
                    )
                parts.append(np.asarray(value, dtype=np.float32).ravel())
            state_vector = np.concatenate(parts)
            if state_vector.shape != (self._state_dim,):
                raise ValueError(
                    f"state_keys {self._state_keys} concatenate to shape "
                    f"{state_vector.shape}, checkpoint expects ({self._state_dim},)"
                )
            obs_dict[OBS_STATE] = state_vector

        task = observation.instruction or self._task
        prepared = prepare_observation_for_inference(
            obs_dict, torch.device(self._device), task, self._robot_type
        )
        processed = self._preprocessor(prepared)

        start = time.perf_counter()
        with torch.inference_mode():
            chunk = self._policy.predict_action_chunk(processed)
        latency_s = time.perf_counter() - start

        if chunk.ndim != 3 or chunk.shape[0] != 1:
            raise RuntimeError(
                f"predict_action_chunk() returned shape {tuple(chunk.shape)}, expected "
                "(1, n_action_steps, action_dim); this checkpoint's architecture may not "
                "be supported by this adapter yet"
            )

        expected_shape = self.info.action_space.shape
        actions: list[Action] = []
        with torch.inference_mode():
            for step in range(chunk.shape[1]):
                processed_step = self._postprocessor(chunk[:, step, :])
                vector = processed_step.squeeze(0).detach().to("cpu").numpy().astype(np.float64)
                if vector.shape != expected_shape:
                    raise ValueError(
                        f"checkpoint action has shape {vector.shape}, declared action_space "
                        f"expects {expected_shape}"
                    )
                actions.append(Action(data=vector))

        return ActionChunk(
            actions=actions, control_hz=self.info.control_hz, inference_latency_s=latency_s, meta={}
        )


def lerobot_policy(**kwargs: Any) -> LeRobotPolicy:
    """Construct the registry-facing LeRobot policy from CLI or programmatic arguments."""
    return LeRobotPolicy(**kwargs)
