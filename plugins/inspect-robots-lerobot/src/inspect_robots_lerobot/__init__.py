"""Run a locally loaded LeRobot checkpoint as an Inspect Robots ``Policy``.

The ``lerobot`` policy is discovered through the ``inspect_robots.policies``
entry-point group. Loading the checkpoint (disk read plus moving weights onto
the target device) happens at construction time, not lazily: unlike a network
connection, there is no CLI command (``list policies``) that constructs this
adapter just to enumerate it, so there is nothing eager loading would break.
"""

from __future__ import annotations

from inspect_robots_lerobot.policy import LeRobotPolicy, lerobot_policy

__all__ = ["LeRobotPolicy", "lerobot_policy"]

__version__ = "0.1.0"
