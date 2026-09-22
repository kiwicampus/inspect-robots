# inspect-robots-rosboard

## Safety

> This adapter drives a real mobile robot. Keep a working e-stop or kill switch
> within reach for the whole session. The `linear_x_limit`/`angular_z_limit`
> clamp is enforced independently inside `step()`, but a clamp is a bound, not
> a guarantee of safe motion at that bound.

- Clear the area around the robot and confirm line of sight before connecting.
- Start with much lower `linear_x_limit`/`angular_z_limit` than the defaults
  for a robot's first connection, and verify direction and sign conventions at
  low speed before trusting them.
- Supervise the first run with a human hand near the e-stop.
- rosboard has no `status: error` equivalent: a malformed or mistyped command
  can fail silently server-side. Watch the robot's own rosboard logs during
  first runs.
- `close()` publishes a zero-velocity stop command before disconnecting, but
  that is a best-effort courtesy, not a substitute for a real e-stop.

The package registers the `rosboard` Inspect Robots embodiment. It connects
directly to a running rosboard server (dheera/rosboard) over its own websocket
protocol, so the evaluation machine needs no ROS installation, no ROS message
packages, and no local ROS graph.

## Install

> [!NOTE]
> Not on PyPI yet. This plugin has no committed history in this repository,
> and its `publish-rosboard` CI job targets a `pypi-rosboard` GitHub
> Environment that has not been created. `pip install inspect-robots-rosboard`
> will fail until a maintainer creates that environment (with its PyPI
> trusted-publisher configuration) and a release runs.

Until then, install it from source as part of the `inspect_robots` uv
workspace:

```bash
cd inspect_robots
uv sync --all-packages --extra dev
```

Once released, `pip install inspect-robots-rosboard` will work on its own.
The embodiment then appears in `inspect-robots list embodiments`.

## Robot-side bringup

The robot's rosboard node must already be running and reachable at the
configured `url`. There is nothing to launch on the evaluation machine: no
ROS installation, no rosbridge, no local ROS graph. This plugin does not start
rosboard or install anything on the robot.

## Quickstart

```bash
inspect-robots run --task my-nav-task --policy scripted --embodiment rosboard \
    -E url=ws://robot.example.com:80 \
    -E camera_height=480 -E camera_width=640
```

Construction and `.info` are network-free. The websocket connects on the first
`reset()`, after compatibility and guardrail checks have inspected the
declared spaces.

## Configuration

Pass values as `-E key=value` arguments or as keyword arguments to
`RosboardEmbodiment`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `url` | `ws://localhost:8888` | rosboard websocket host. `/rosboard/v1` is appended automatically. |
| `odometry_topic` | `/odometry/local` | `nav_msgs/msg/Odometry` source; also the step-freshness reference. |
| `imu_topic` | `/imu/data_abs_heading` | `sensor_msgs/msg/Imu` source. |
| `camera_topic` | `/camera/color/image_raw` | Image source; rosboard compresses it to JPEG server-side regardless of the underlying ROS type. |
| `camera_name` | `front` | Observation key for the camera image. |
| `camera_height` | required | Declared frame height in pixels. Validated against the first real frame at connect time. |
| `camera_width` | required | Declared frame width in pixels. Validated against the first real frame at connect time. |
| `command_topic` | `/motion_control/speed_controller/reference_cmd` | `geometry_msgs/msg/Twist` drive command topic. |
| `linear_x_limit` | `0.5` | Max absolute forward/back speed (m/s). Both the declared action bound and an independent hard clamp inside `step()`. |
| `angular_z_limit` | `0.5` | Max absolute turn rate (rad/s). Both the declared action bound and an independent hard clamp inside `step()`. |
| `control_hz` | `10.0` | Command rate. The adapter sleep-gates each publish to this rate. |
| `fresh_obs_timeout_s` | `2/control_hz` | Maximum step-time wait for a sequence-newer odometry message. |
| `obs_timeout_s` | `5.0` | Reset-time wait for the first (and subsequent) messages on every configured topic. |
| `staleness_s` | `2.0` | Maximum cached sample age and cross-modal skew bound during observation assembly. |
| `simulated` | `False` | Set true only if pointed at a simulated rosboard source. |
| `name` | `rosboard` | Embodiment name recorded in logs, for example `rosboard:kiwibot-3`. |
| `connect_timeout_s` | `10.0` | Websocket connection timeout. |

There is no `-E` for the command topic's message type: it is always
`geometry_msgs/msg/Twist`, and only `linear.x`/`angular.z` are ever set (the
remaining four degrees of freedom are sent as zero).

## Configuration file (`-E config=...`)

The table above is the fixed schema plan 0082 shipped with: exactly one
odometry topic, one IMU topic, one camera, one Twist command. Passing
`-E config=path/to/robot.yaml` instead switches to a config-driven mode with
an arbitrary set of observation and action topics, each described in a YAML
file. This is the one place the plugin's otherwise scalar-only `-E key=value`
convention bends: the file path is still a single scalar argument, and
everything structural lives inside the file. `config` and the fixed-schema
arguments above (`odometry_topic`, `camera_topic`, `linear_x_limit`,
`control_hz`, ...) are mutually exclusive: passing `config` ignores all of
them, since the file becomes the sole source of topic wiring, action clamps,
and control rate.

```bash
inspect-robots run --task my-task --policy scripted --embodiment rosboard \
    -E config=rover_diff_drive.yaml
```

```yaml
name: rover_diff_drive
rate_hz: 10.0

observations:
  - key: observation.image.main
    topic: /camera/color/image_raw
    type: sensor_msgs/msg/Image
    image:
      resize: [360, 640]     # [height, width]; required for every image entry
    align: {tol_ms: 0}       # 0 (or omitted) means unbounded: use whatever is latest

  - key: observation.state
    topic: /odometry/local
    type: nav_msgs/msg/Odometry
    selector:
      names: [twist.twist.linear.x, twist.twist.angular.z]
    align: {tol_ms: 150}     # message older than 150ms at assembly time raises

actions:
  - key: action
    publish:
      topic: /motion_control/speed_controller/output_cmd
      type: geometry_msgs/msg/TwistStamped
    selector:
      names: [twist.linear.x, twist.angular.z]
    from_tensor:
      clamp: [-2.0, 2.0]     # hard-clamped inside step(), independent of any guardrail
    safety_behavior: zeros   # publish a zero vector to this topic on close()
```

- **`observations[]`**: each entry becomes one key in the policy's
  `Observation`. Set exactly one of:
  - `image: {resize: [height, width]}`, populating `Observation.images[key]`.
    `resize` is required (this adapter declares camera shapes up front and
    never introspects them, same as the fixed schema's `camera_height`/
    `camera_width`) and is applied client-side with Pillow after decode,
    since rosboard's own server-side resize is a fixed 800px-max-dimension
    downsample, not a configurable per-topic target.
  - `selector: {names: [field.path, ...]}`, populating
    `Observation.state[key]` with one scalar per dotted path, read from
    rosboard's JSON payload in the given order (rosboard delivers nested ROS
    submessages as same-named nested dicts, so `twist.twist.linear.x` is a
    literal key path, not a schema the adapter has to know about). Order
    matters: it is how a wxyz quaternion reorder is expressed, by listing
    `orientation.w` before `orientation.x`/`y`/`z`.
  - `align.tol_ms`: a **staleness bound**, not true cross-topic
    synchronization. The underlying client caches one latest sample per
    topic, not a searchable time-indexed history, so this is "reject a
    message older than `tol_ms` at assembly time," not "find the sample
    closest to some reference timestamp." `0` or omitted means unbounded
    (accept whatever is latest).
  - `qos`, `align.strategy`, `align.stamp`, `type` are parsed and validated
    but not applied: rosboard's subscribe frame carries only a topic name,
    with no reliability/history/depth/throttle knobs to set, and no
    "strategy" concept.
- **`actions[]`**: each entry publishes a slice of the flat action vector to
  its own topic. `RosboardEmbodiment`'s action space is the concatenation, in
  file order, of every entry's `selector.names`. Only `from_tensor.clamp`
  (required, `[low, high]`, applied to every dimension of that entry) and
  `safety_behavior: zeros` (the only supported value; publishes a zero vector
  to that topic on `close()`) are honored. `publish.qos` and `publish.strategy`
  (fifo/nearest re-pacing) are parsed but unused: this adapter already
  executes one action per closed-loop `step()` call, so there is no queued
  chunk to re-pace here; that question belongs to whatever plays a policy's
  predicted action chunk (`inspect_robots.controller`), not this adapter.
- **Images must be plain ROS images.** `type: sensor_msgs/msg/Image` or
  `sensor_msgs/msg/CompressedImage` are the only camera types this rosboard
  deployment can actually serve as a decodable frame: server-side, it
  JPEG-compresses exactly those two types to the `_data_jpeg` field this
  adapter decodes. A codec-compressed video type (e.g.
  `foxglove_msgs/msg/CompressedVideo`) is not supported: rosboard's own
  `ros2dict()` has no special case for it, so it would arrive as a raw,
  undecoded video-bitstream chunk, not a self-contained frame.
- **`name`, `robot_type`, `version`, `metadata`, `max_duration_s`,
  `recording`** (top level) are accepted, for compatibility with config files
  shared across other tooling, but this adapter reads only `name` (echoed
  into `EmbodimentInfo.docs`) and `rate_hz`; the rest are ignored.

## Observation and action contract

- Action is `[linear_x, angular_z]`, base frame, no rotation representation:
  declared with `control_mode="base_velocity"` (there is no absolute-target
  reference to align it with, since it is a rate command).
- `odom_pose` is `[x, y, z, qw, qx, qy, qz]`: odometry position and
  orientation, with the quaternion reordered from ROS's native xyzw to wxyz,
  the same convention [inspect-robots-ros](../inspect-robots-ros/) uses for
  `eef_pose`.
- `odom_twist` is `[linear.x, linear.y, linear.z, angular.x, angular.y, angular.z]`
  in native units (m/s, rad/s).
- `imu` is `[qw, qx, qy, qz, wx, wy, wz, ax, ay, az]`: orientation (wxyz),
  angular velocity (rad/s), linear acceleration (m/s^2).
- `state_time` is the older of the odometry and IMU receive times.
  `image_times[camera_name]` records the camera frame's receive time.
- Every observation is checked against `staleness_s`; the odometry, IMU, or
  camera message being older than that raises.

## Reset behavior

`reset()` is a no-op beyond connecting, subscribing, and waiting for a fresh
sample on every topic: there is no reset service, no operator confirmation,
and no way to change the robot's physical position between trials. From the
second reset onward, the adapter prints one stderr warning that between-trial
reset does not change the physical world.

## Troubleshooting

- Connection failures name the URL and note that no local ROS bringup is
  needed; confirm the robot's rosboard node is actually running and reachable.
- A camera resolution mismatch names both the declared and the actual first
  frame's resolution.
- Fresh-observation timeouts on `step()` usually mean `control_hz` is set
  higher than the robot's real odometry rate. Lower `control_hz` or raise
  `fresh_obs_timeout_s`.
- Staleness errors mean a configured topic stopped arriving; check the robot's
  rosboard session.
- rosboard has no `status: error` op. A command with the wrong shape or an
  unreachable topic can silently no-op server-side rather than raising here;
  check the robot's own rosboard/ROS logs if commands appear to have no
  effect.
- The `agent` LLM policy plugin does not yet recognize `base_velocity` and
  raises `ToolsetError` at bind time. Use `scripted`, `random`,
  `xpolicylab`, or a custom policy that emits a two-element action until
  `agent` adds mobile-base support.
