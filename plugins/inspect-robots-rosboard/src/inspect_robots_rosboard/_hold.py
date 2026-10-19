"""Keep the latest command alive between steps: resend it, then send zeros once it is too old.

A robot controller that zeroes its reference after a short timeout otherwise sees each step's
command as a brief pulse whenever the policy is slower than that timeout.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from typing import Any

# (topic, type, fields) of one publish.
Frame = tuple[str, str, dict[str, Any]]
# Due times are sums of periods; this absorbs their rounding.
_EPS = 1e-6


class CommandHold:
    """Resend ``frames`` every ``1 / resend_hz`` s until ``hold_s`` after ``sent_at``.

    At that age ``zero_frames`` go out once and ``on_expire(stamp)`` is called. ``arm`` and
    ``disarm`` are called by the stepping thread; resends happen on a daemon thread.
    """

    def __init__(
        self,
        publish: Callable[[str, str, dict[str, Any]], None],
        *,
        hold_s: float,
        resend_hz: float,
        clock: Callable[[], float],
        on_expire: Callable[[float], None],
        background: bool = True,
    ) -> None:
        self.hold_s = hold_s
        self.period = 1.0 / resend_hz
        self._publish = publish
        self._clock = clock
        self._on_expire = on_expire
        self._background = background
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._frames: tuple[Frame, ...] = ()
        self._zero_frames: tuple[Frame, ...] = ()
        self._deadline = 0.0
        self._next_send = 0.0

    @property
    def armed(self) -> bool:
        """Whether a command is currently being held."""
        with self._lock:
            return bool(self._frames)

    def arm(self, frames: Sequence[Frame], zero_frames: Sequence[Frame], sent_at: float) -> None:
        """Hold ``frames``, already published at ``sent_at``, replacing any held command."""
        with self._lock:
            if self._closed or not frames:
                self._frames = ()
                return
            self._frames, self._zero_frames = tuple(frames), tuple(zero_frames)
            self._deadline = sent_at + self.hold_s
            self._next_send = sent_at + self.period
        self._ensure_thread()
        self._wake.set()

    def disarm(self) -> None:
        """Stop holding without sending zeros; the caller publishes whatever comes next."""
        with self._lock:
            self._frames = ()

    def close(self) -> None:
        """Disarm and end the resend thread."""
        with self._lock:
            self._frames = ()
            self._closed = True
            thread = self._thread
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def tick(self) -> float | None:
        """Resend or expire if due; return seconds until the next due time, None when idle."""
        with self._lock:
            if not self._frames:
                return None
            now = self._clock()
            if now + _EPS >= self._deadline:
                for frame in self._zero_frames:
                    self._send(frame)
                self._frames = ()
                expired = now
            else:
                if now + _EPS >= self._next_send:
                    for frame in self._frames:
                        self._send(frame)
                    self._next_send = max(self._next_send + self.period, now + self.period / 2)
                return max(0.0, min(self._next_send, self._deadline) - now)
        self._on_expire(expired)
        return None

    def _send(self, frame: Frame) -> None:
        # A lost connection is latched by the client and surfaces on the next step.
        with suppress(Exception):
            self._publish(*frame)

    def _ensure_thread(self) -> None:
        if not self._background or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="inspect-robots-rosboard-hold", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
            self._wake.clear()
            timeout = self.tick()
            self._wake.wait(timeout)
