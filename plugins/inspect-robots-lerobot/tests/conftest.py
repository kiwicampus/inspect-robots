from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def synthetic_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build and migrate a tiny, untrained-weights ACT checkpoint via lerobot's own API.

    Real lerobot 0.4.4 code, real files on disk, real ``from_pretrained`` /
    ``predict_action_chunk`` calls in the tests that use this: this is not a
    mock of lerobot, only of "the weights are meaningful," which nothing in
    this adapter cares about. ``pretrained_backbone_weights=None`` avoids a
    network fetch of pretrained ResNet18 weights.

    The normalization-pipeline migration (``config.json``/``model.safetensors``
    -> also ``policy_preprocessor.json``/``policy_postprocessor.json``) has no
    importable function in lerobot 0.4.4, only a ``python -m ...`` CLI entry
    point (``migrate_policy_normalization.main()``, argparse-only) — invoked
    as a subprocess rather than monkeypatching ``sys.argv``.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    input_features = {
        "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 64, 64)),
        "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
    }
    output_features = {"action": PolicyFeature(FeatureType.ACTION, (2,))}
    config = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        n_obs_steps=1,
        chunk_size=10,
        n_action_steps=10,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    policy = ACTPolicy(config)
    policy.eval()

    raw_dir = tmp_path_factory.mktemp("act_checkpoint_raw")
    policy.save_pretrained(raw_dir)

    migrated_dir = tmp_path_factory.mktemp("act_checkpoint") / "migrated"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "lerobot.processor.migrate_policy_normalization",
            "--pretrained-path",
            str(raw_dir),
            "--output-dir",
            str(migrated_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return migrated_dir
