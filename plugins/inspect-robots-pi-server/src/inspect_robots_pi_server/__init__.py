"""Drive a checkpoint served over the PI protocol as an Inspect Robots policy.

The ``pi_server`` policy is discovered through the ``inspect_robots.policies``
entry-point group. Construction and ``.info`` are network-free; the websocket
connects on the first ``reset()``/``act()``.
"""

from __future__ import annotations

from typing import Any

from inspect_robots_pi_server.policy import PiServerPolicy

__all__ = ["PiServerPolicy", "pi_server_policy"]

__version__ = "0.1.0"


def pi_server_policy(**kwargs: Any) -> PiServerPolicy:
    """Construct the registry-facing PI-protocol policy from CLI or programmatic arguments."""
    return PiServerPolicy(**kwargs)
