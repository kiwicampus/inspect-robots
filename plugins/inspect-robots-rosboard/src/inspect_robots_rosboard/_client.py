"""Threaded synchronous client for the rosboard websocket protocol.

One receive thread owns ``ClientConnection.recv()``; the calling thread only
sends (the same one-reader/one-writer concurrency contract
``inspect_robots_ros._client`` relies on for ``websockets.sync.client``,
pinned against the same ``>=12`` floor). Topic traffic is reduced to a
latest-value slot per topic, carrying a monotonic receive stamp and a
per-topic sequence number bumped on every received sample.

Much smaller than ``RosbridgeClient``: rosboard has no
advertise/unadvertise, no service-RPC, and no per-subscription
throttle-rate/queue-length/compression knobs, so none of that machinery is
ported. There is also no reconnect: a dead client (a decode failure or a
dropped socket) stays dead, and every later call raises the first latched
error.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from websockets.sync.client import connect as ws_connect

from inspect_robots_rosboard._protocol import (
    RosboardError,
    TopicMessage,
    decode_frame,
    encode_frame,
    parse_incoming,
    publish,
    subscribe,
    unsubscribe,
)

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection


@dataclass(frozen=True)
class TopicSample:
    """The latest payload for a topic with receive time and monotonic sequence."""

    payload: dict[str, Any]
    stamp: float
    seq: int


def _rosboard_socket_url(host: str) -> str:
    """Build rosboard's actual websocket URL from a configured host string.

    Mirrors ``rosboard_client``'s own URL handling: ``ws://``/``wss://``
    prefixed hosts get ``/rosboard/v1`` appended; a bare ``host:port`` string
    infers ``ws://`` (or ``wss://`` for port 443) from the trailing port,
    defaulting to port 80 when the port cannot be parsed.
    """
    if host.startswith("ws://") or host.startswith("wss://"):
        return host + "/rosboard/v1"
    try:
        port = int(host.rsplit(":", 1)[-1])
    except ValueError:
        port = 80
    scheme = "wss://" if port == 443 else "ws://"
    return f"{scheme}{host}/rosboard/v1"


class RosboardClient:
    """Own one rosboard socket, one receiver thread, and latest-value topic caches."""

    def __init__(
        self,
        url: str,
        *,
        connect_timeout_s: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = url
        self.connect_timeout_s = connect_timeout_s
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.RLock()
        self._ws: ClientConnection | None = None
        self._receiver_thread: threading.Thread | None = None
        self._topics: dict[str, TopicSample] = {}
        self._subscriptions: set[str] = set()
        self._latched_error: Exception | None = None
        self._closing = False
        self._closed = False

    @property
    def connected(self) -> bool:
        """Whether a socket has been established and not explicitly closed."""
        with self._lock:
            return self._ws is not None and not self._closed

    @property
    def receiver_alive(self) -> bool:
        """Whether the sole receive thread is still running."""
        with self._lock:
            thread = self._receiver_thread
        return thread is not None and thread.is_alive()

    @property
    def latched_error(self) -> Exception | None:
        """The first asynchronous protocol or connection failure, if any."""
        with self._lock:
            return self._latched_error

    def connect(self) -> None:
        """Connect once and start the sole receive thread; never reconnect a dead client."""
        with self._lock:
            self._raise_latched_locked()
            if self._closed:
                raise RuntimeError("RosboardClient is closed")
            if self._ws is not None:
                return
        socket_url = _rosboard_socket_url(self.url)
        try:
            ws = ws_connect(socket_url, max_size=None, open_timeout=self.connect_timeout_s)
        except Exception as exc:
            raise ConnectionError(f"could not connect to rosboard at {socket_url}: {exc}") from exc
        with self._lock:
            self._ws = ws
            thread = threading.Thread(
                target=self._receive_loop,
                name="inspect-robots-rosboard-receiver",
                daemon=True,
            )
            self._receiver_thread = thread
        thread.start()

    def subscribe(self, topic: str) -> None:
        """Send a subscribe frame and track the topic for close-time unsubscribe."""
        self._send(subscribe(topic))
        with self._lock:
            self._subscriptions.add(topic)

    def unsubscribe(self, topic: str) -> None:
        """Send an unsubscribe frame and drop the close-time cleanup record."""
        self._send(unsubscribe(topic))
        with self._lock:
            self._subscriptions.discard(topic)

    def publish(self, topic: str, topic_type: str, fields: Mapping[str, Any]) -> None:
        """Publish one command message, surfacing any previously latched error first."""
        self._send(publish(topic, topic_type, fields))

    def latest(self, topic: str) -> TopicSample | None:
        """Read a topic's latest slot after surfacing any latched failure."""
        with self._lock:
            self._check_ready_locked()
            return self._topics.get(topic)

    def sequence(self, topic: str) -> int:
        """Read the current per-topic receive sequence, or zero before the first message."""
        sample = self.latest(topic)
        return sample.seq if sample is not None else 0

    def wait_for_sample(self, topic: str, *, after_seq: int = 0, timeout_s: float) -> TopicSample:
        """Wait until a topic slot has a sequence strictly newer than ``after_seq``."""
        deadline = self._clock() + timeout_s
        while True:
            sample = self.latest(topic)
            if sample is not None and sample.seq > after_seq:
                return sample
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError(
                    f"timed out after {timeout_s:g}s waiting for a new message on {topic!r}"
                )
            self._sleep(min(0.01, remaining))

    def close(self) -> None:
        """Best-effort unsubscribe every tracked topic, close, and join; idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closing = True
            ws = self._ws
            subscriptions = tuple(self._subscriptions)
            thread = self._receiver_thread
        if ws is not None:
            for topic in subscriptions:
                self._send_best_effort(ws, unsubscribe(topic))
            with suppress(Exception):
                ws.close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(max(self.connect_timeout_s, 1.0), 5.0))
        with self._lock:
            self._ws = None
            self._subscriptions.clear()
            self._closed = True

    def _send(self, frame: list[Any]) -> None:
        with self._lock:
            self._check_ready_locked()
            ws = self._ws
            assert ws is not None
        try:
            ws.send(encode_frame(frame))
        except Exception as exc:
            error = ConnectionError(
                f"connection to rosboard at {self.url} lost while sending: {exc}"
            )
            self._latch(error)
            raise error from exc

    @staticmethod
    def _send_best_effort(ws: ClientConnection, frame: list[Any]) -> None:
        with suppress(Exception):
            ws.send(encode_frame(frame))

    def _receive_loop(self) -> None:
        with self._lock:
            ws = self._ws
        assert ws is not None
        try:
            while True:
                raw = ws.recv()
                if not isinstance(raw, (str, bytes)):
                    raise RosboardError(
                        "invalid_frame",
                        f"rosboard sent unsupported frame type {type(raw).__name__}",
                    )
                incoming = parse_incoming(decode_frame(raw))
                if isinstance(incoming, TopicMessage):
                    with self._lock:
                        previous = self._topics.get(incoming.topic_name)
                        seq = previous.seq + 1 if previous is not None else 1
                        self._topics[incoming.topic_name] = TopicSample(
                            payload=incoming.payload,
                            stamp=self._clock(),
                            seq=seq,
                        )
        except Exception as exc:
            with self._lock:
                closing = self._closing
            if not closing:
                if isinstance(exc, RosboardError):
                    receive_error: Exception = exc
                else:
                    receive_error = ConnectionError(
                        f"connection to rosboard at {self.url} lost while receiving: {exc}"
                    )
                self._latch(receive_error)

    def _latch(self, error: Exception) -> None:
        with self._lock:
            if self._latched_error is None:
                self._latched_error = error

    def _check_ready_locked(self) -> None:
        self._raise_latched_locked()
        if self._closed:
            raise RuntimeError("RosboardClient is closed")
        if self._ws is None:
            raise ConnectionError(f"not connected to rosboard at {self.url}; call connect() first")

    def _raise_latched_locked(self) -> None:
        if self._latched_error is not None:
            raise self._latched_error
