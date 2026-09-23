# inspect-robots-lerobot

The package registers the `lerobot` Inspect Robots policy. It loads a trained
[LeRobot](https://github.com/huggingface/lerobot) checkpoint (ACT, Diffusion
Policy, SmolVLA, or any other architecture registered with
`lerobot.policies.factory.get_policy_class`) directly in-process and runs
inference locally: no inference server, no websocket, unlike
[inspect-robots-xpolicylab](../inspect-robots-xpolicylab/). The evaluation
machine needs `lerobot` (and therefore `torch`) installed, and enough compute
to run the checkpoint at the embodiment's `control_hz`.

> [!NOTE]
> Verified against the actually-installed `lerobot==0.4.4` API (checkpoint
> loading, `PolicyFeature` shapes, `predict_action_chunk`, and the separate
> processor-pipeline normalization step were all confirmed by reading that
> exact version's source). **Not** verified end to end against a real trained
> checkpoint: none was available while writing this. A synthetic
> untrained-weights ACT checkpoint was used to confirm the loading and
> inference call chain doesn't error, not that outputs are correct. Smoke-test
> against your own checkpoint with `inspect-robots doctor` and a short dry run
> before trusting this against real hardware.

## Install

```bash
cd inspect_robots
uv sync --all-packages --extra dev
```

## A note on checkpoint format: the processor migration

Newer `lerobot` versions moved observation/action normalization out of the
policy's own weights and into a separate `PolicyProcessorPipeline`, saved as
`policy_preprocessor.json`/`policy_postprocessor.json` alongside
`config.json`/`model.safetensors`. If your checkpoint predates this (only
`config.json` and `model.safetensors` in the directory), loading it here
fails with `ProcessorMigrationError`. Fix it once, in place:

```bash
python -m lerobot.processor.migrate_policy_normalization --pretrained-path /path/to/checkpoint
```

This writes a `<checkpoint>_migrated` directory; point `-P checkpoint=` at
that instead.

## Quickstart

```bash
inspect-robots run --task my-nav-task --policy lerobot --embodiment rosboard \
    -P checkpoint=/path/to/checkpoint \
    -P image_keys=observation.image.main:observation.images.front \
    -P state_keys=observation.state \
    -E config=rover_diff_drive.yaml
```

Construction loads the checkpoint immediately (not lazily): unlike a network
connection, there is no `inspect-robots list policies` code path that would
construct this adapter just to enumerate it, so eager loading costs nothing
extra and fails fast with a clear error instead of at the first `act()`.

## Configuration

Pass values as `-P key=value` arguments or as keyword arguments to
`LeRobotPolicy`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | required | Local directory (or Hugging Face Hub repo id) holding `config.json`, `model.safetensors`, and the processor files. |
| `image_keys` | required if the checkpoint has image features | `"ir_key:ckpt_key,..."` mapping `Observation.images` keys to the checkpoint's trained camera feature names (e.g. `observation.images.front`). Every checkpoint image feature must appear exactly once. |
| `state_keys` | required if the checkpoint has a state feature | `"ir_key1,ir_key2,..."`, concatenated in order into the checkpoint's single `observation.state` vector. |
| `device` | `auto` | `auto` resolves to `cuda` if available, else `cpu`. |
| `task` | `None` | Fallback task/language string when `Observation.instruction` is `None`. Only load-bearing for language-conditioned architectures (e.g. SmolVLA); ACT/Diffusion Policy ignore it. |
| `robot_type` | `None` | Passed through to the checkpoint's `prepare_observation_for_inference`; metadata only. |
| `control_mode` | `base_velocity` | Declared `ActionSemantics.control_mode`, checked exactly against the paired embodiment's. Set to `joint_pos` etc. for an arm checkpoint. |
| `rotation_repr` | `none` | Declared `ActionSemantics.rotation_repr`, checked exactly against the paired embodiment's. |
| `gripper` | `none` | Declared `ActionSemantics.gripper`. |
| `frame` | `base` | Declared `ActionSemantics.frame`. |
| `control_hz` | `None` | Declared policy rate; only used for a compatibility warning if it disagrees with the embodiment's (the rollout does not enforce it). |
| `name` | `lerobot` | Policy name recorded in logs. |

The action dimension and every image/state feature shape are read from the
checkpoint's own `config.json` (`PreTrainedConfig.action_feature` /
`.image_features` / `.robot_state_feature`), not configured here — pass an
`image_keys`/`state_keys` mapping that matches what the checkpoint actually
expects, and construction fails immediately, naming the checkpoint's real
feature names, if it doesn't.

## Observation and action contract

- Uses `policy.predict_action_chunk()`, not `select_action()`. `select_action()`
  owns its own per-architecture action queue/temporal ensembler, designed for
  a caller that re-infers every physical step; that would double-buffer
  against Inspect Robots' own `DefaultController` chunk buffering and
  replanning cadence. `predict_action_chunk()` returns the full raw
  `(1, n_action_steps, action_dim)` chunk once per `act()` call, and
  `DefaultController` (or `SmoothingController`, or a custom `Controller`)
  owns deciding how much of it to play before calling `act()` again.
- Each step of the returned chunk is passed through the checkpoint's own
  postprocessor individually (matching `select_action()`'s per-step
  normalization contract, since `PolicyProcessorPipeline` transforms one
  action's worth of tensor at a time, not a whole chunk) before being wrapped
  into an `Action`.
- `reset(scene)` calls `policy.reset()`, clearing whatever queue/ensembler
  state the checkpoint's own architecture keeps between trials.

## Troubleshooting

- `ProcessorMigrationError` at construction: see the migration section above.
- `checkpoint expects image feature(s) [...] but no image_keys mapping was
  given`: the error names the checkpoint's real trained feature names; build
  `-P image_keys=` from those, not guesses.
- `image_keys is missing checkpoint feature(s) [...]`/`maps to checkpoint
  feature(s) [...] the checkpoint does not have`: `image_keys` must cover
  every image feature the checkpoint declares, exactly, no more and no fewer.
- `state_keys [...] concatenate to shape (...), checkpoint expects (...)`:
  the named `Observation.state` keys, concatenated in the given order, must
  add up to exactly the checkpoint's trained state dimension.
- `predict_action_chunk() returned shape [...], expected (1, n_action_steps,
  action_dim)`: an architecture whose `predict_action_chunk()` doesn't follow
  the standard batch-first convention; not supported by this adapter as
  written.
- `control_mode`/`rotation_repr` compatibility errors at `eval()` time (not
  from this plugin): set `-P control_mode=`/`-P rotation_repr=` to match the
  paired embodiment's declared action semantics.
