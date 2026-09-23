"""Drive a mobile ground robot through a rosboard websocket connection.

The adapter is configuration-driven and needs no ROS installation on the
evaluation machine: rosboard performs all ROS message conversion server-side.
Construction builds only static spaces; the websocket connects lazily on the
first ``reset()``. Every ``step()`` hard-clips the outgoing action to the
configured limits before publishing, independent of any framework guardrail,
because this embodiment actuates a real vehicle.

Two ways to configure the observation/action layout, selected by whether
``-E config=...`` is passed:

- **Fixed schema** (default, no ``config``): exactly the odometry/IMU/single
  camera/Twist layout plan 0082 shipped with, via scalar ``-E`` args
  (``odometry_topic``, ``camera_topic``, ``linear_x_limit``, ...).
- **Config-driven** (``-E config=path/to/robot.yaml``): an arbitrary set of
  observation/action topics described in a YAML file, parsed by
  :mod:`inspect_robots_rosboard._config`. This is the one place the plugin's
  otherwise scalar-only ``-E key=value`` convention bends: the file path is
  still a single scalar arg, and everything structural lives inside it.
  See the plugin README's Configuration-file section for the schema and its
  documented gaps against rosboard's actual wire protocol (no QoS, no true
  cross-topic timestamp alignment, no action re-pacing).

These two paths are independent implementations sharing only the low-level
``RosboardClient`` and the generic staleness/sequence-wait helpers: the fixed
schema is untouched by the config path so its existing behavior and tests
stay exactly as they were before config support was added.

Independently of the above, either mode additionally accepts ``-E
topics_file=path/to/topics_to_subscribe.yaml``, the ROS2 ``rosboard_client``
node's own config file, parsed by :mod:`inspect_robots_rosboard._topics_file`.
This makes that file the source of truth for the rosboard ``url`` and for
whether each subscribed/published topic (the fixed schema's four roles, or
every ``observations[]``/``actions[]`` topic in a ``config`` file) is
actually active, without adding a ROS dependency: the file is read once, at
construction, as plain YAML. In particular, whether a command topic appears
in the file's ``topics_to_stream:`` list decides, once, whether ``step()``
publishes a real command on that topic or only logs what it would have sent
(a dry run) -- independently per ``actions[]`` entry in config-driven mode.
Toggling this requires editing the file and starting a new run, not mid-run:
it is read once, not re-checked on every ``step()``.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any

import numpy as np

from inspect_robots import (
    Action,
    ActionSemantics,
    Box,
    CameraSpec,
    EmbodimentBase,
    EmbodimentInfo,
    Observation,
    ObservationSpace,
    Scene,
    StateField,
    StateSpec,
    StepResult,
)
from inspect_robots.embodiment import SELF_PACED
from inspect_robots_rosboard._client import RosboardClient, TopicSample
from inspect_robots_rosboard._config import (
    ActionSpec,
    ObservationSpec,
    RobotConfig,
    load_robot_config,
)
from inspect_robots_rosboard._msgs import (
    build_twist,
    parse_compressed_image,
    parse_imu,
    parse_odometry,
)
from inspect_robots_rosboard._selectors import build_from_selector, select_state
from inspect_robots_rosboard._topics_file import TopicsFile, load_topics_file

_TWIST_TYPE = "geometry_msgs/msg/Twist"


def _dry_run_action_topics(actions: tuple[ActionSpec, ...], enabled: tuple[bool, ...]) -> list[str]:
    """The publish topics of every ``actions[]`` entry not armed by ``topics_file``."""
    return [a.publish_topic for a, e in zip(actions, enabled, strict=True) if not e]


class RosboardEmbodiment(EmbodimentBase):
    """Drive a mobile ground robot's twist controller through a rosboard websocket."""

    def __init__(
        self,
        *,
        config: str | None = None,
        topics_file: str | None = None,
        url: str = "ws://localhost:8888",
        odometry_topic: str = "/odometry/local",
        imu_topic: str = "/imu/data_abs_heading",
        camera_topic: str = "/camera/color/image_raw",
        camera_name: str = "front",
        camera_height: int | None = None,
        camera_width: int | None = None,
        command_topic: str = "/motion_control/speed_controller/reference_cmd",
        linear_x_limit: float = 0.5,
        angular_z_limit: float = 0.5,
        control_hz: float = 10.0,
        fresh_obs_timeout_s: float | None = None,
        obs_timeout_s: float = 5.0,
        staleness_s: float = 2.0,
        simulated: bool = False,
        name: str = "rosboard",
        connect_timeout_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        resolved_url = url
        topics_cfg: TopicsFile | None = None
        if topics_file is not None:
            if url != "ws://localhost:8888":
                raise ValueError(
                    "topics_file and url are mutually exclusive; the rosboard url is "
                    "read from topics_file's 'url:' field, so drop -E url= when using "
                    "-E topics_file="
                )
            topics_cfg = load_topics_file(topics_file)
            resolved_url = topics_cfg.url

        if config is not None:
            # Config-driven mode: the YAML file is the sole source of topic
            # wiring, action clamps, and control rate. Every fixed-schema
            # argument above (odometry_topic, camera_*, *_limit, control_hz,
            # fresh_obs_timeout_s, staleness_s) is ignored here; only the
            # generic connection/runtime args below still apply. ``topics_file``
            # composes with ``config``: it still sources ``url`` and still
            # gates which of the config file's observations[]/actions[]
            # topics are actually subscribed/published, same as fixed-schema
            # mode, just checked per config-file topic instead of per fixed
            # role.
            self._init_from_config(
                config,
                url=resolved_url,
                topics_file=topics_file,
                topics_cfg=topics_cfg,
                obs_timeout_s=obs_timeout_s,
                simulated=simulated,
                name=name,
                connect_timeout_s=connect_timeout_s,
                clock=clock,
                sleep=sleep,
            )
            return

        publish_enabled = True
        if topics_cfg is not None:
            for role_arg, topic in (
                ("odometry_topic", odometry_topic),
                ("imu_topic", imu_topic),
                ("camera_topic", camera_topic),
            ):
                if topic not in topics_cfg.topics:
                    raise ValueError(
                        f"topics_file {topics_file!r} does not list {role_arg}={topic!r} "
                        "under 'topics:'; uncomment or add it there to subscribe to it, "
                        f"or pass a different -E {role_arg}="
                    )
            publish_enabled = command_topic in topics_cfg.topics_to_stream

        if camera_height is None or camera_width is None:
            raise ValueError(
                "camera_height and camera_width are required; rosboard performs no "
                "resolution introspection, so declare the camera's real resolution "
                "explicitly, e.g. -E camera_height=480 -E camera_width=640"
            )
        if not isinstance(camera_height, int) or not isinstance(camera_width, int):
            raise ValueError("camera_height and camera_width must be integers")
        if camera_height < 1 or camera_width < 1:
            raise ValueError("camera_height and camera_width must be positive")
        if not math.isfinite(linear_x_limit) or linear_x_limit <= 0:
            raise ValueError(f"linear_x_limit must be positive and finite, got {linear_x_limit!r}")
        if not math.isfinite(angular_z_limit) or angular_z_limit <= 0:
            raise ValueError(
                f"angular_z_limit must be positive and finite, got {angular_z_limit!r}"
            )
        if not math.isfinite(control_hz) or control_hz <= 0:
            raise ValueError(f"control_hz must be positive and finite, got {control_hz!r}")
        resolved_fresh_timeout = (
            2.0 / control_hz if fresh_obs_timeout_s is None else float(fresh_obs_timeout_s)
        )
        for arg, timeout in (
            ("fresh_obs_timeout_s", resolved_fresh_timeout),
            ("obs_timeout_s", obs_timeout_s),
            ("connect_timeout_s", connect_timeout_s),
        ):
            if timeout <= 0 or not math.isfinite(timeout):
                raise ValueError(f"{arg} must be positive and finite, got {timeout!r}")
        if staleness_s < 0 or not math.isfinite(staleness_s):
            raise ValueError(f"staleness_s must be finite and >= 0, got {staleness_s!r}")

        self.info = EmbodimentInfo(
            name=name,
            action_space=Box(
                shape=(2,),
                low=np.asarray([-linear_x_limit, -angular_z_limit], dtype=np.float64),
                high=np.asarray([linear_x_limit, angular_z_limit], dtype=np.float64),
                semantics=ActionSemantics(
                    control_mode="base_velocity",
                    rotation_repr="none",
                    gripper="none",
                    frame="base",
                    dim_labels=("linear_x", "angular_z"),
                ),
            ),
            observation_space=ObservationSpace(
                cameras=(CameraSpec(camera_name, camera_height, camera_width),),
                state=StateSpec(
                    fields=(
                        StateField("odom_pose", (7,), unit="m+quat"),
                        StateField("odom_twist", (6,), unit="m/s+rad/s"),
                        StateField("imu", (10,), unit="quat+rad/s+m/s^2"),
                    )
                ),
            ),
            control_hz=control_hz,
            is_simulated=simulated,
            capabilities=frozenset({SELF_PACED}),
            supported_setups=frozenset(),
            supported_target_kinds=frozenset(),
            docs=(
                "Mobile ground robot over rosboard. Action is [linear_x, angular_z] "
                "base velocity in m/s / rad/s, base frame, clamped to "
                f"+/-{linear_x_limit:g} m/s and +/-{angular_z_limit:g} rad/s both by "
                "the declared action bounds and by an independent hard clamp inside "
                "step(). Observation state: 'odom_pose' = [x, y, z, qw, qx, qy, qz] "
                "(odometry pose, orientation reordered to wxyz); 'odom_twist' = "
                "[linear.x, linear.y, linear.z, angular.x, angular.y, angular.z] "
                "(odometry twist, native units); 'imu' = [qw, qx, qy, qz, wx, wy, wz, "
                "ax, ay, az] (orientation wxyz, angular velocity rad/s, linear "
                "acceleration m/s^2). No absolute-mode proprioceptive reference is "
                "declared because base_velocity is a rate command, not an absolute "
                f"target. Camera is a single forward-facing RGB stream ('{camera_name}')."
                + (
                    f" DRY RUN: topics_file={topics_file!r} does not list "
                    f"{command_topic!r} under topics_to_stream:, so step() logs the "
                    "would-be command instead of publishing it."
                    if topics_file is not None and not publish_enabled
                    else ""
                )
            ),
        )

        self.url = resolved_url
        self.odometry_topic = odometry_topic
        self.imu_topic = imu_topic
        self.camera_topic = camera_topic
        self.camera_name = camera_name
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.command_topic = command_topic
        self.linear_x_limit = linear_x_limit
        self.angular_z_limit = angular_z_limit
        self.control_hz = control_hz
        self.fresh_obs_timeout_s = resolved_fresh_timeout
        self.obs_timeout_s = obs_timeout_s
        self.staleness_s = staleness_s
        self._clock = clock
        self._sleep = sleep
        self._client = RosboardClient(
            resolved_url, connect_timeout_s=connect_timeout_s, clock=clock, sleep=sleep
        )
        self._publish_enabled = publish_enabled
        self._initialized = False
        self._instruction: str | None = None
        self._last_publish_time: float | None = None
        self._reset_count = 0
        self._warned_no_physical_reset = False
        self._validated_camera = False
        self._robot_config: RobotConfig | None = None

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        """Connect lazily, wait for a fresh sample per topic, and return the initial observation."""
        if self._robot_config is not None:
            return self._reset_config(scene, seed=seed)
        del seed
        self._instruction = scene.instruction
        self._ensure_initialized()

        if self._reset_count >= 1 and not self._warned_no_physical_reset:
            print(
                "warning: rosboard embodiment has no reset mechanism; between-trial "
                "reset does not change the physical world",
                file=sys.stderr,
            )
            self._warned_no_physical_reset = True

        sequences = {topic: self._client.sequence(topic) for topic in self._all_topics()}
        self._wait_for_sequences(sequences, self.obs_timeout_s, "obs_timeout_s")
        observation = self._assemble_observation()
        self._last_publish_time = None
        self._reset_count += 1
        return observation

    def step(self, action: Action) -> StepResult:
        """Hard-clamp, self-pace, publish the drive command, and assemble a fresh step."""
        if self._robot_config is not None:
            return self._step_config(action)
        data = np.asarray(action.data, dtype=np.float64)
        if data.shape != self.info.action_space.shape:
            raise ValueError(
                f"action has shape {data.shape}, expected {self.info.action_space.shape}"
            )

        # Hard safety clamp, independent of any framework approver (ClampApprover
        # stays on by default too, but must not be the only thing standing between
        # a policy bug and this real vehicle moving unexpectedly).
        linear_x = float(np.clip(data[0], -self.linear_x_limit, self.linear_x_limit))
        angular_z = float(np.clip(data[1], -self.angular_z_limit, self.angular_z_limit))

        now = self._clock()
        if self._last_publish_time is not None:
            remaining = self._last_publish_time + (1.0 / self.control_hz) - now
            if remaining > 0:
                self._sleep(remaining)

        seq_at_publish = self._client.sequence(self.odometry_topic)
        publish_time = self._clock()
        if self._publish_enabled:
            self._client.publish(self.command_topic, _TWIST_TYPE, build_twist(linear_x, angular_z))
        else:
            print(
                f"[rosboard dry-run] {self.command_topic!r} is not listed under "
                "topics_to_stream: in the configured topics_file; would publish "
                f"linear_x={linear_x:g} angular_z={angular_z:g} but sent nothing",
                file=sys.stderr,
            )
        self._last_publish_time = publish_time

        try:
            self._client.wait_for_sample(
                self.odometry_topic, after_seq=seq_at_publish, timeout_s=self.fresh_obs_timeout_s
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"no post-publish odometry within fresh_obs_timeout_s="
                f"{self.fresh_obs_timeout_s:g}s at control_hz={self.control_hz:g}; "
                "lower control_hz or raise fresh_obs_timeout_s"
            ) from exc

        return StepResult(
            observation=self._assemble_observation(),
            reward=None,
            terminated=False,
            truncated=False,
        )

    def close(self) -> None:
        """Publish a zero-velocity stop command, then release the websocket connection.

        Publishing zero before closing is a safety default beyond what a bare
        rosbridge-style adapter would do (``inspect_robots_ros.RosEmbodiment``
        does not do this): this embodiment drives a real moving vehicle, and an
        eval ending mid-command should not leave the last nonzero twist latched
        on the robot's controller.
        """
        if self._robot_config is not None:
            self._close_config()
            return
        if self._initialized and self._publish_enabled:
            with suppress(Exception):
                self._client.publish(self.command_topic, _TWIST_TYPE, build_twist(0.0, 0.0))
        self._client.close()

    def __enter__(self) -> RosboardEmbodiment:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            self._client.connect()
        except Exception as exc:
            raise ConnectionError(
                f"could not connect to rosboard at {self.url}. Confirm the robot's "
                "rosboard node is already running and reachable at this URL; no "
                "local ROS bringup is needed on the eval machine."
            ) from exc
        for topic in self._all_topics():
            self._client.subscribe(topic)
        self._wait_for_sequences(
            dict.fromkeys(self._all_topics(), 0), self.obs_timeout_s, "obs_timeout_s"
        )
        self._validate_camera_resolution()
        self._initialized = True

    def _all_topics(self) -> tuple[str, ...]:
        return (self.odometry_topic, self.imu_topic, self.camera_topic)

    def _wait_for_sequences(
        self, sequences: Mapping[str, int], timeout_s: float, timeout_name: str
    ) -> None:
        deadline = self._clock() + timeout_s
        for topic, sequence in sequences.items():
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError(f"missing topic {topic!r} within {timeout_name}={timeout_s:g}s")
            try:
                self._client.wait_for_sample(topic, after_seq=sequence, timeout_s=remaining)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"missing topic {topic!r} within {timeout_name}={timeout_s:g}s"
                ) from exc

    def _validate_camera_resolution(self) -> None:
        if self._validated_camera:
            return
        sample = self._required_sample(self.camera_topic)
        image = parse_compressed_image(sample.payload)
        actual = image.shape[:2]
        declared = (self.camera_height, self.camera_width)
        if actual != declared:
            raise ValueError(
                f"camera {self.camera_name!r} declared resolution "
                f"{self.camera_width}x{self.camera_height} but first frame is "
                f"{actual[1]}x{actual[0]}"
            )
        self._validated_camera = True

    def _assemble_observation(self) -> Observation:
        now = self._clock()
        odom_sample = self._required_sample(self.odometry_topic)
        self._check_staleness(self.odometry_topic, odom_sample, now)
        odom_pose, odom_twist = parse_odometry(odom_sample.payload)

        imu_sample = self._required_sample(self.imu_topic)
        self._check_staleness(self.imu_topic, imu_sample, now)
        imu = parse_imu(imu_sample.payload)

        camera_sample = self._required_sample(self.camera_topic)
        self._check_staleness(self.camera_topic, camera_sample, now)
        image = parse_compressed_image(camera_sample.payload)

        return Observation(
            images={self.camera_name: image},
            state={"odom_pose": odom_pose, "odom_twist": odom_twist, "imu": imu},
            instruction=self._instruction,
            image_times={self.camera_name: camera_sample.stamp},
            state_time=min(odom_sample.stamp, imu_sample.stamp),
        )

    def _required_sample(self, topic: str) -> TopicSample:
        sample = self._client.latest(topic)
        if sample is None:
            raise RuntimeError(f"no cached message for subscribed topic {topic!r}")
        return sample

    def _check_staleness(self, topic: str, sample: TopicSample, now: float) -> None:
        age = now - sample.stamp
        if age > self.staleness_s:
            raise TimeoutError(
                f"cached message on {topic!r} is stale by {age:g}s, exceeding "
                f"staleness_s={self.staleness_s:g}"
            )

    # -- Config-driven mode (``-E config=...``) -----------------------------
    #
    # Independent of the fixed-schema implementation above other than sharing
    # ``RosboardClient``/``_wait_for_sequences``/``_required_sample``, which are
    # already topic-string-generic. See the module docstring and the plugin
    # README's Configuration-file section for what this mode does and does not
    # honor from the YAML schema (no QoS, tol_ms as staleness not true
    # cross-topic alignment, no action re-pacing strategy).

    def _init_from_config(
        self,
        config_path: str,
        *,
        url: str,
        topics_file: str | None,
        topics_cfg: TopicsFile | None,
        obs_timeout_s: float,
        simulated: bool,
        name: str,
        connect_timeout_s: float,
        clock: Callable[[], float],
        sleep: Callable[[float], None],
    ) -> None:
        if obs_timeout_s <= 0 or not math.isfinite(obs_timeout_s):
            raise ValueError(f"obs_timeout_s must be positive and finite, got {obs_timeout_s!r}")
        if connect_timeout_s <= 0 or not math.isfinite(connect_timeout_s):
            raise ValueError(
                f"connect_timeout_s must be positive and finite, got {connect_timeout_s!r}"
            )

        robot_config = load_robot_config(config_path)

        if topics_cfg is not None:
            for spec in robot_config.observations:
                if spec.topic not in topics_cfg.topics:
                    raise ValueError(
                        f"topics_file {topics_file!r} does not list observation "
                        f"{spec.key!r}'s topic {spec.topic!r} under 'topics:'; uncomment "
                        "or add it there to subscribe to it"
                    )

        action_publish_enabled = tuple(
            topics_cfg is None or action_spec.publish_topic in topics_cfg.topics_to_stream
            for action_spec in robot_config.actions
        )

        cameras = tuple(
            CameraSpec(spec.key, spec.image.height, spec.image.width)
            for spec in robot_config.observations
            if spec.image is not None
        )
        state_fields = tuple(
            StateField(spec.key, (len(spec.selector_names or ()),), unit="config")
            for spec in robot_config.observations
            if spec.selector_names is not None
        )

        action_dims: list[str] = []
        low: list[float] = []
        high: list[float] = []
        for action_spec in robot_config.actions:
            action_dims.extend(action_spec.selector_names)
            low.extend([action_spec.clamp_low] * len(action_spec.selector_names))
            high.extend([action_spec.clamp_high] * len(action_spec.selector_names))

        self.info = EmbodimentInfo(
            name=name,
            action_space=Box(
                shape=(len(action_dims),),
                low=np.asarray(low, dtype=np.float64),
                high=np.asarray(high, dtype=np.float64),
                semantics=ActionSemantics(
                    control_mode="base_velocity",
                    rotation_repr="none",
                    gripper="none",
                    frame="base",
                    dim_labels=tuple(action_dims),
                ),
            ),
            observation_space=ObservationSpace(
                cameras=cameras, state=StateSpec(fields=state_fields)
            ),
            control_hz=robot_config.rate_hz,
            is_simulated=simulated,
            capabilities=frozenset({SELF_PACED}),
            supported_setups=frozenset(),
            supported_target_kinds=frozenset(),
            docs=(
                f"Mobile ground robot over rosboard, wired from robot config "
                f"{robot_config.name!r} ({config_path}). Observation images: "
                f"{[c.name for c in cameras]!r}. Observation state keys (dims): "
                f"{[(f.key, f.shape[0]) for f in state_fields]!r}. Action is the "
                "concatenation, in config file order, of each actions[] entry's "
                f"selector.names: {action_dims!r}, each independently hard-clamped "
                "to its own from_tensor.clamp inside step()."
                + (
                    f" DRY RUN: topics_file={topics_file!r} does not list "
                    f"{_dry_run_action_topics(robot_config.actions, action_publish_enabled)!r} "
                    "under topics_to_stream:, so step() logs the would-be command for "
                    "those actions instead of publishing it."
                    if topics_cfg is not None and not all(action_publish_enabled)
                    else ""
                )
            ),
        )

        self.url = url
        self.control_hz = robot_config.rate_hz
        self.fresh_obs_timeout_s = 2.0 / robot_config.rate_hz
        self.obs_timeout_s = obs_timeout_s
        self._clock = clock
        self._sleep = sleep
        self._client = RosboardClient(
            url, connect_timeout_s=connect_timeout_s, clock=clock, sleep=sleep
        )
        self._action_publish_enabled = action_publish_enabled
        self._robot_config = robot_config
        self._initialized = False
        self._instruction = None
        self._last_publish_time = None
        self._reset_count = 0
        self._warned_no_physical_reset = False

    def _reset_config(self, scene: Scene, *, seed: int | None = None) -> Observation:
        del seed
        assert self._robot_config is not None
        self._instruction = scene.instruction
        self._ensure_initialized_config()

        if self._reset_count >= 1 and not self._warned_no_physical_reset:
            print(
                "warning: rosboard embodiment has no reset mechanism; between-trial "
                "reset does not change the physical world",
                file=sys.stderr,
            )
            self._warned_no_physical_reset = True

        topics = self._all_topics_config()
        sequences = {topic: self._client.sequence(topic) for topic in topics}
        self._wait_for_sequences(sequences, self.obs_timeout_s, "obs_timeout_s")
        observation = self._assemble_observation_config()
        self._last_publish_time = None
        self._reset_count += 1
        return observation

    def _step_config(self, action: Action) -> StepResult:
        assert self._robot_config is not None
        data = np.asarray(action.data, dtype=np.float64)
        if data.shape != self.info.action_space.shape:
            raise ValueError(
                f"action has shape {data.shape}, expected {self.info.action_space.shape}"
            )

        now = self._clock()
        if self._last_publish_time is not None:
            remaining = self._last_publish_time + (1.0 / self.control_hz) - now
            if remaining > 0:
                self._sleep(remaining)

        reference_spec = self._freshness_reference_spec()
        seq_at_publish = self._client.sequence(reference_spec.topic)
        publish_time = self._clock()

        offset = 0
        for action_spec, enabled in zip(
            self._robot_config.actions, self._action_publish_enabled, strict=True
        ):
            n = len(action_spec.selector_names)
            clamped = np.clip(
                data[offset : offset + n], action_spec.clamp_low, action_spec.clamp_high
            )
            offset += n
            names = action_spec.selector_names + action_spec.zero_fields
            values = [*clamped.tolist(), *([0.0] * len(action_spec.zero_fields))]
            fields = build_from_selector(names, values)
            if enabled:
                self._client.publish(action_spec.publish_topic, action_spec.publish_type, fields)
            else:
                print(
                    f"[rosboard dry-run] {action_spec.publish_topic!r} is not listed "
                    "under topics_to_stream: in the configured topics_file; would "
                    f"publish {fields!r} but sent nothing",
                    file=sys.stderr,
                )
        self._last_publish_time = publish_time

        try:
            self._client.wait_for_sample(
                reference_spec.topic, after_seq=seq_at_publish, timeout_s=self.fresh_obs_timeout_s
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"no post-publish sample on {reference_spec.topic!r} within "
                f"fresh_obs_timeout_s={self.fresh_obs_timeout_s:g}s at control_hz="
                f"{self.control_hz:g}; lower control_hz or raise fresh_obs_timeout_s"
            ) from exc

        return StepResult(
            observation=self._assemble_observation_config(),
            reward=None,
            terminated=False,
            truncated=False,
        )

    def _close_config(self) -> None:
        assert self._robot_config is not None
        if self._initialized:
            for action_spec, enabled in zip(
                self._robot_config.actions, self._action_publish_enabled, strict=True
            ):
                if action_spec.safety_behavior != "zeros" or not enabled:
                    continue
                names = action_spec.selector_names + action_spec.zero_fields
                zeros = [0.0] * len(names)
                with suppress(Exception):
                    fields = build_from_selector(names, zeros)
                    self._client.publish(
                        action_spec.publish_topic, action_spec.publish_type, fields
                    )
        self._client.close()

    def _ensure_initialized_config(self) -> None:
        if self._initialized:
            return
        try:
            self._client.connect()
        except Exception as exc:
            raise ConnectionError(
                f"could not connect to rosboard at {self.url}. Confirm the robot's "
                "rosboard node is already running and reachable at this URL; no "
                "local ROS bringup is needed on the eval machine."
            ) from exc
        topics = self._all_topics_config()
        for topic in topics:
            self._client.subscribe(topic)
        self._wait_for_sequences(dict.fromkeys(topics, 0), self.obs_timeout_s, "obs_timeout_s")
        self._initialized = True

    def _all_topics_config(self) -> tuple[str, ...]:
        assert self._robot_config is not None
        return tuple(spec.topic for spec in self._robot_config.observations)

    def _freshness_reference_spec(self) -> ObservationSpec:
        """Pick the topic whose freshness proves a published action had an effect.

        The first state (non-image) observation, mirroring the fixed schema's
        choice of odometry over the camera: a fast, small message is a better
        step-time freshness gate than waiting on the largest payload. Falls
        back to the first observation of any kind if the config is all-image.
        """
        assert self._robot_config is not None
        for spec in self._robot_config.observations:
            if spec.selector_names is not None:
                return spec
        return self._robot_config.observations[0]

    def _assemble_observation_config(self) -> Observation:
        assert self._robot_config is not None
        now = self._clock()
        images: dict[str, Any] = {}
        state: dict[str, Any] = {}
        image_times: dict[str, float] = {}
        state_times: list[float] = []

        for spec in self._robot_config.observations:
            sample = self._required_sample(spec.topic)
            self._check_staleness_ms(spec, sample, now)
            if spec.image is not None:
                images[spec.key] = parse_compressed_image(
                    sample.payload, resize=(spec.image.height, spec.image.width)
                )
                image_times[spec.key] = sample.stamp
            else:
                assert spec.selector_names is not None
                state[spec.key] = select_state(sample.payload, spec.selector_names)
                state_times.append(sample.stamp)

        return Observation(
            images=images,
            state=state,
            instruction=self._instruction,
            image_times=image_times,
            state_time=min(state_times) if state_times else now,
        )

    def _check_staleness_ms(self, spec: ObservationSpec, sample: TopicSample, now: float) -> None:
        """Enforce ``align.tol_ms`` as a staleness bound; ``tol_ms <= 0`` means unbounded.

        Not true cross-topic closest-timestamp alignment (``RosboardClient``
        caches one latest sample per topic, not a searchable history), see the
        module docstring.
        """
        if spec.tol_ms <= 0:
            return
        age_ms = (now - sample.stamp) * 1000.0
        if age_ms > spec.tol_ms:
            raise TimeoutError(
                f"cached message on {spec.topic!r} (observation {spec.key!r}) is stale by "
                f"{age_ms:g}ms, exceeding align.tol_ms={spec.tol_ms:g}ms"
            )


def rosboard_embodiment(**kwargs: Any) -> RosboardEmbodiment:
    """Construct the registry-facing rosboard embodiment from CLI or programmatic arguments."""
    return RosboardEmbodiment(**kwargs)
