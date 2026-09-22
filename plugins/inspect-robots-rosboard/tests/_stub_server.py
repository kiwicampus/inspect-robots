"""In-process rosboard-shaped websocket server for adapter tests."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any

from websockets.sync.server import Server, ServerConnection, serve


class StubRosboardServer:
    """Serve the rosboard array-frame protocol used by the plugin on a free local port."""

    def __init__(self) -> None:
        self.frames: list[list[Any]] = []
        self._lock = threading.RLock()
        self._connections: set[ServerConnection] = set()
        self._subscriptions: dict[ServerConnection, set[str]] = {}
        self._server: Server = serve(self._handler, "127.0.0.1", 0, max_size=None)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        """The websocket URL bound to the server's ephemeral port."""
        port = self._server.socket.getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    def frames_of(self, identifier: str) -> list[dict[str, Any]]:
        """Return a snapshot of received ``"m"``-carried payloads (or ``"s"``/``"u"`` payloads)."""
        with self._lock:
            return [dict(frame[1]) for frame in self.frames if frame[0] == identifier]

    def wait_for(
        self, predicate: Callable[[list[list[Any]]], bool], *, timeout_s: float = 2.0
    ) -> None:
        """Wait until a predicate over the recorded frames becomes true."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                snapshot = [list(frame) for frame in self.frames]
            if predicate(snapshot):
                return
            time.sleep(0.005)
        raise TimeoutError("stub rosboard did not receive the expected frame")

    def publish(self, topic: str, topic_type: str, fields: Mapping[str, Any]) -> None:
        """Send a topic message only to connections subscribed to that topic."""
        payload = {
            "_topic_name": topic,
            "_topic_type": topic_type,
            "_time": time.time() * 1000,
            **dict(fields),
        }
        with self._lock:
            targets = [
                connection for connection, topics in self._subscriptions.items() if topic in topics
            ]
        self._send(targets, ["m", payload])

    def send_topics_announcement(self, mapping: Mapping[str, str]) -> None:
        """Send a ``"t"`` topic-announcement frame, which the client must ignore."""
        self._send(self._connected_targets(), ["t", dict(mapping)])

    def send_raw(self, raw: str) -> None:
        """Send an arbitrary raw text frame, for malformed-frame latching tests."""
        for connection in self._connected_targets():
            with suppress(Exception):
                connection.send(raw)

    def drop_connections(self) -> None:
        """Close every client socket to exercise receive-thread death latching."""
        for connection in self._connected_targets():
            with suppress(Exception):
                connection.close()

    def stop(self) -> None:
        """Shut down the websocket server and join its serving thread."""
        with self._lock:
            targets = list(self._connections)
        for connection in targets:
            with suppress(Exception):
                connection.close()
        self._server.shutdown()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise TimeoutError("stub rosboard serving thread did not stop")

    def _handler(self, ws: ServerConnection) -> None:
        with self._lock:
            self._connections.add(ws)
            self._subscriptions[ws] = set()
        try:
            for raw in ws:
                if not isinstance(raw, (str, bytes)):
                    continue
                frame = json.loads(raw)
                if not isinstance(frame, list) or len(frame) != 2:
                    continue
                with self._lock:
                    self.frames.append(frame)
                identifier, payload = frame[0], frame[1]
                if identifier == "s" and isinstance(payload, dict):
                    topic = payload.get("topicName")
                    if isinstance(topic, str):
                        with self._lock:
                            self._subscriptions[ws].add(topic)
                elif identifier == "u" and isinstance(payload, dict):
                    topic = payload.get("topicName")
                    if isinstance(topic, str):
                        with self._lock:
                            self._subscriptions[ws].discard(topic)
        finally:
            with self._lock:
                self._connections.discard(ws)
                self._subscriptions.pop(ws, None)

    def _connected_targets(self) -> list[ServerConnection]:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            with self._lock:
                targets = list(self._connections)
            if targets:
                return targets
            time.sleep(0.005)
        raise TimeoutError("stub rosboard has no connected client")

    @staticmethod
    def _send(targets: list[ServerConnection], frame: list[Any]) -> None:
        encoded = json.dumps(frame, separators=(",", ":"))
        for connection in targets:
            with suppress(Exception):
                connection.send(encoded)
