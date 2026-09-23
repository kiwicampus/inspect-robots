# inspect-robots-pi-server

The package registers the `pi_server` Inspect Robots policy. It connects
directly to a running PI-protocol policy server (the wire format
`pi_inference_client` speaks, e.g. a `policy_server`/Modal deployment) over
its own websocket protocol, so the evaluation machine needs no ROS
installation and no `pi_inference_client` checkout.

> [!NOTE]
> Verified against the real protocol contract and the real
> `pi_inference_client`/server source (checkpoint loading isn't this
> plugin's concern; that's `policy_server`'s job), including a full
> end-to-end run against an in-process stub server that speaks the exact
> wire framing. **Not** verified against a live Modal deployment, since none
> was available while writing this. Run a real `reset()`/`act()` round trip
> against your deployed server before trusting this in an eval.

## Install

```bash
cd inspect_robots
uv sync --all-packages --extra dev
```

## Quickstart

```bash
inspect-robots run --task my-nav-task --policy pi_server --embodiment rosboard \
    -P url=wss://pi-policy-my-model.modal.run \
    -P control_mode=base_velocity -P action_dim=2 \
    -P cameras=front:cam_head -P state_map=odom_twist:joint_position \
    -P camera_height=224 -P camera_width=224 \
    -E config=rover_diff_drive.yaml
```

Construction and `.info` are network-free: the websocket connects, and calls
`load`, on the first `reset()`/`act()`, so `inspect-robots list policies` and
fail-fast compatibility checks work with no server running.

## Configuration

Pass values as `-P key=value` arguments or as keyword arguments to
`PiServerPolicy`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `url` | required | The server's websocket URL, e.g. `wss://pi-policy-<model>.modal.run`. |
| `control_mode` | required | Declared `ActionSemantics.control_mode` (e.g. `base_velocity`, `joint_pos`), checked exactly against the paired embodiment's. |
| `action_dim` | required | The checkpoint's action vector width. This plugin cannot discover it network-free, so it must be declared; a mismatch against what the server actually returns raises at `act()` time, naming both numbers. |
| `rotation_repr` | `none` | Declared `ActionSemantics.rotation_repr`, checked exactly against the embodiment's. |
| `gripper` | `none` | Declared `ActionSemantics.gripper` (mismatch is a warning, not an error). |
| `frame` | `base` | Declared `ActionSemantics.frame` (mismatch is a warning, not an error). |
| `cameras` | none declared | `"ir_key:sub_key,..."` mapping `Observation.images` keys to the server's declared camera sub-keys. Every entry is resolved against the server's `input_spec` at first connect (bare or `observation/<sub_key>` form); an unrecognized entry raises `ConfigError` naming what was tried. |
| `state_map` | none declared | `"ir_key:sub_key,..."` mapping `Observation.state` keys to the server's declared state sub-keys, resolved the same way as `cameras`. A key present in `state_map` but absent from one particular observation is silently skipped for that step (matches the protocol's "all state fields optional" contract); an entry the server's `input_spec` never recognizes at all raises `ConfigError` at connect time. |
| `action_keys` | server's own `action_keys` | Override which of the server's declared output keys to concatenate into the final action vector, `"key1,key2"`. Rarely needed: the server's `load` response already names its own `action_keys`. |
| `camera_height` / `camera_width` | `None` | Advisory metadata only (declared once, shared across every camera in `cameras`): `inspect_robots.compat.check_compatibility` checks camera names, not resolution, so this never blocks a real mismatch. The resolution actually used to resize images is whatever the server's `load` response advertises (`image_preprocess`), not this. |
| `control_hz` | `None` | Declared policy rate; only used for a compatibility warning if it disagrees with the embodiment's (the rollout does not enforce it). |
| `prompt` | `None` | Fallback task/language string used when `Observation.instruction` is `None`, sent as both `inputs.prompt` and `inputs.robot_task_string`. |
| `name` | `pi_server` | Policy name recorded in logs. |
| `api_key_env` | `PI_SERVER_API_KEY` | The **name** of an environment variable holding the server's API key, never the literal key. `Authorization: Api-Key <value>` is sent on every connect, even if the variable is unset (some deployments sit behind a network-level gate with no per-request check); if the connection then fails, the error names `api_key_env` as a thing to check. |
| `connect_open_timeout_s` | `360.0` | Websocket handshake timeout, generous by default for a cold-starting serverless GPU container. |
| `connect_max_retries` | `3` | Connection attempts before giving up, with exponential backoff. Not retried for a malformed `load` response or a timeout. |
| `request_timeout_s` | `120.0` | Per-RPC (`load`/`infer`/`reset`) timeout. |
| `jpeg_quality` | `85` | JPEG quality used to encode every camera image before sending. |

## Observation and action contract

- `Observation.images[ir_key]` for every `cameras` entry is resized
  client-side to whatever resolution the connected server's `load` response
  advertises (`image_preprocess.target_resolution` or a per-camera override
  in `image_preprocess.image_resolutions`, `resize_mode` `stretch` or `pad`,
  `interpolation` `bilinear` or `lanczos`), then JPEG-encoded. A required
  camera missing from a given observation raises `ConfigError`.
- `Observation.state[ir_key]` for every `state_map` entry is sent as a
  `float32` array under its resolved wire key; a mapped key absent from one
  particular observation is silently skipped for that step, matching the
  protocol's own "all state fields optional" contract.
- `policy.predict_action_chunk()`-equivalent: every `act()` call is exactly
  one `infer` RPC, returning the server's full, unsliced predicted horizon
  (`outputs[action_key]`, shape `(horizon, action_dim)`). Multiple
  `action_keys` are concatenated on the last axis. Each horizon step becomes
  one `Action`; how many of them actually get played before the next `act()`
  call is `inspect_robots.controller.DefaultController`'s decision, not this
  adapter's.
- **The `action_horizon`/`action_dim` spec fields.** Most real deployments
  (including `policy_server`) declare these unconditionally per checkpoint —
  their presence enables AsyncRTC-style realtime chunk streaming, but this
  adapter never does that streaming. Instead, whenever they're present, it
  sends exactly the protocol's own well-defined non-realtime defaults (a
  null `actions` prefix, `max_horizon: 0`, zeroed `initial_noise`) on every
  request, unconditionally — the same thing the real `pi_inference_client`
  sends for a caller that isn't doing realtime prefixing. **True overlapped
  AsyncRTC (reusing a prediction as the next request's prefix) is a
  documented non-goal.**

## Non-goals

- **`telemetry`**: not sent. `policy_server`'s own benchmarking tooling talks
  to the server directly; this adapter doesn't duplicate it.
- **True overlapped AsyncRTC prefixing**: see above.
- **Per-camera resolution declaration for compatibility checking**:
  `camera_height`/`camera_width` are one shared value for every declared
  camera. If a checkpoint's `image_resolutions` genuinely differs per
  camera, images are still resized correctly (the real resolution always
  comes from the server's `load` response), but this plugin's advisory
  `CameraSpec` metadata will be wrong for some cameras. Harmless for
  compatibility checking (name-only), just imprecise metadata.
- **HTTP redirect-following** during the websocket handshake: not
  implemented. A Modal deployment serves a stable URL with no redirect
  chain, so this hasn't mattered in practice.

## Troubleshooting

- `ConfigError: ... state_map key ... is not recognized by the connected
  server`: the error names both wire-key forms tried (bare and
  `observation/<key>`) and the server's actual declared `input_spec` keys —
  fix the `-P state_map=`/`-P cameras=` mapping to match one of them.
- `ConfigError: ... observation has no image ...`: a camera declared in
  `-P cameras=` is missing from the embodiment's actual observation; check
  the embodiment side of the mapping.
- `PiServerError: ProtocolError: assembled action last-dim ... !=
  declared action_dim ...`: the server's real action width disagrees with
  `-P action_dim=`; fix the declared value to match the checkpoint.
- `ConnectionError: ... No API key was found in $PI_SERVER_API_KEY ...`: set
  the environment variable named by `api_key_env` (default
  `PI_SERVER_API_KEY`), or pass `-P api_key_env=NAME` if the server expects a
  differently-named one.
- `PiServerError: InferenceError: ...`: an in-band server-side failure; the
  connection stays usable afterward (this is not a dead-socket condition).
