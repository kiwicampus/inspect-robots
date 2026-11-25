"""Run Inspect Robots evaluations against a real mobile robot's rosboard server.

The ``rosboard`` embodiment is discovered through the
``inspect_robots.embodiments`` entry-point group. Construction and ``.info``
remain network-free; the websocket connects on the first reset.
"""

from __future__ import annotations

from inspect_robots_rosboard.embodiment import RosboardEmbodiment, rosboard_embodiment

__all__ = ["RosboardEmbodiment", "rosboard_embodiment"]

__version__ = "0.1.0"
