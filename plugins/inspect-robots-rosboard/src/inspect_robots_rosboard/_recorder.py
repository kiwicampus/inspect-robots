"""Per trial JSONL of every sample on chosen topics plus each command, on one clock.

Read only: it listens to messages the client already receives and never publishes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Any


def _trim(value: Any) -> Any:
    """Drop covariance arrays and rosboard's private ``_`` keys to keep rows small."""
    if isinstance(value, dict):
        return {
            k: _trim(v) for k, v in value.items() if not k.startswith("_") and "covariance" not in k
        }
    if isinstance(value, list):
        return [_trim(v) for v in value]
    return value


def header_stamp(payload: dict[str, Any]) -> float | None:
    """The message's own ``header.stamp`` in seconds, when it has one."""
    stamp = (payload.get("header") or {}).get("stamp") or {}
    if "sec" not in stamp:
        return None
    return float(stamp["sec"]) + float(stamp.get("nanosec", 0)) * 1e-9


class TopicRecorder:
    """Write ``topics`` samples and published commands of the current trial to a JSONL file.

    Rows carry ``stamp``, seconds on the client clock since the trial started, so commands
    and samples line up; ``header_stamp`` is the robot's own time when the message has one.
    """

    def __init__(self, topics: Sequence[str], clock: Callable[[], float]) -> None:
        self.topics = tuple(topics)
        self._clock = clock
        self._lock = threading.Lock()
        self._file: IO[str] | None = None
        self._path: Path | None = None
        self._t0 = 0.0

    def start(self, path: Path, header: dict[str, Any]) -> None:
        """Open ``path`` for a new trial and write its header row; best effort."""
        self.stop()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("w")
        except OSError:
            return
        with self._lock:
            self._file, self._path, self._t0 = handle, path, self._clock()
        self._write({"kind": "header", "topics": list(self.topics), **header})

    def stop(self) -> Path | None:
        """Close the current file and return its path, if a trial was being recorded."""
        with self._lock:
            handle, path = self._file, self._path
            self._file = self._path = None
        if handle is not None:
            handle.close()
        return path

    def on_sample(self, topic: str, payload: dict[str, Any], stamp: float) -> None:
        """Client listener: record a sample of a chosen topic."""
        if topic not in self.topics or self._file is None:
            return
        self._write(
            {
                "topic": topic,
                "stamp": round(stamp - self._t0, 4),
                "header_stamp": header_stamp(payload),
                "msg": _trim(payload),
            }
        )

    def command(
        self, t: int | None, stamp: float, values: Sequence[float], published: bool
    ) -> None:
        """Record one command: step ``t`` (None for a stop), and whether it was sent."""
        if self._file is None:
            return
        self._write(
            {
                "kind": "command",
                "t": t,
                "stamp": round(stamp - self._t0, 4),
                "action": [float(v) for v in values],
                "published": published,
            }
        )

    def observation(
        self, t: int, stamp: float, ages: dict[str, dict[str, Any]], clock_offset: float | None
    ) -> None:
        """Record how old each input of step ``t`` was when the observation was assembled.

        ``clock_offset`` is the smallest receive time minus header stamp seen so far, so
        ``sensor`` ages count from the robot's stamp, net of the fastest link delay.
        """
        if self._file is None:
            return
        self._write(
            {
                "kind": "obs",
                "t": t,
                "stamp": round(stamp - self._t0, 4),
                "clock_offset": clock_offset,
                "ages": ages,
            }
        )

    def event(self, kind: str, stamp: float) -> None:
        """Record a timestamped event row, such as the end of a held command."""
        if self._file is None:
            return
        self._write({"kind": kind, "stamp": round(stamp - self._t0, 4)})

    def _write(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, allow_nan=False, default=str) + "\n"
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(line)
                self._file.flush()
            except (OSError, ValueError):
                self._file = None
