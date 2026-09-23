"""A synchronous client for the PI-protocol policy-server websocket.

One blocking request in flight at a time (``ws.send()`` then ``ws.recv()``),
matching both Inspect Robots' synchronous rollout loop and the real
``pi_inference_client``'s own ``_timed_request``. Unlike
``inspect_robots_xpolicylab``'s ``PolicyClient`` (which pairs replies by an
explicit ``request_id`` because that protocol allows several in-flight
requests), this protocol needs no such pairing.

``connect()`` performs the ``load`` RPC and caches the resulting
:class:`~inspect_robots_pi_server._spec.PolicySpec`; a dead socket is *not*
transparently reconnected mid-request — the caller (``policy.py``) reconnects
once and retries, replaying ``load()``, mirroring both this plugin
ecosystem's convention (``inspect_robots_xpolicylab``) and
``pi_inference_client``'s own reconnect-and-retry-once behavior.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from websockets.exceptions import WebSocketException
from websockets.sync.client import connect as ws_connect

from inspect_robots_pi_server._msgpack_codec import packb, unpackb
from inspect_robots_pi_server._protocol import PiServerError, parse_response, raise_for_status
from inspect_robots_pi_server._spec import PolicySpec, parse_policy_spec

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection

_CONNECTION_ERRORS = (WebSocketException, OSError, EOFError)
_MAX_FRAME_BYTES = 50 * 1024 * 1024


class PolicyClient:
    """Blocking request/response client for one PI-protocol policy server."""

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        connect_open_timeout_s: float = 360.0,
        connect_max_retries: int = 3,
        request_timeout_s: float = 120.0,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.connect_open_timeout_s = connect_open_timeout_s
        self.connect_max_retries = max(1, connect_max_retries)
        self.request_timeout_s = request_timeout_s
        self._ws: ClientConnection | None = None
        self.spec: PolicySpec | None = None

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def connect(self) -> PolicySpec:
        """Connect (retrying transport failures), send ``load``, cache and return the spec.

        No-op returning the cached spec if already connected. Does not retry
        :class:`PiServerError` (a malformed ``load`` response is not a
        transient failure) or ``TimeoutError``, matching the protocol
        contract's "ProtocolError/TimeoutError not retried" rule.
        """
        if self._ws is not None and self.spec is not None:
            return self.spec
        last_err: Exception | None = None
        for attempt in range(1, self.connect_max_retries + 1):
            try:
                self._ws = ws_connect(
                    self.url,
                    additional_headers={"Authorization": f"Api-Key {self.api_key}"},
                    max_size=_MAX_FRAME_BYTES,
                    compression=None,
                    open_timeout=self.connect_open_timeout_s,
                )
                break
            except (PiServerError, TimeoutError):
                raise
            except Exception as exc:
                last_err = exc
                if attempt < self.connect_max_retries:
                    time.sleep(min(2.0**attempt, 10.0))
        if self._ws is None:
            raise ConnectionError(
                f"could not connect to PI-protocol policy server at {self.url} after "
                f"{self.connect_max_retries} attempts ({last_err}). Confirm the server is "
                "deployed and the URL is reachable."
            ) from last_err
        try:
            payload = self._request("load", {})
        except Exception:
            self._abandon()
            raise
        result = payload.get("result")
        raw_spec = result.get("spec") if isinstance(result, dict) else None
        if not isinstance(raw_spec, str):
            self._abandon()
            raise PiServerError(
                "ProtocolError", "load() response is missing a string 'result.spec' field"
            )
        self.spec = parse_policy_spec(raw_spec)
        return self.spec

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        """One blocking ``infer`` round trip; returns the full response payload.

        The payload's ``result`` field carries ``outputs``/``raw_outputs``;
        ``server_processing_time_ms``/``server_total_time_ms`` are siblings
        of ``result`` at the top level, not nested inside it.
        """
        payload = self._request("infer", request)
        result = payload.get("result")
        if not isinstance(result, dict):
            raise PiServerError(
                "ProtocolError", "infer() response is missing an object 'result' field"
            )
        return payload

    def reset(self) -> None:
        """One blocking ``reset`` round trip. No-op if never connected."""
        if self._ws is None:
            return
        self._request("reset", {})

    def close(self) -> None:
        """Best-effort close; idempotent."""
        ws = self._ws
        self._ws = None
        self.spec = None
        if ws is not None:
            with contextlib.suppress(Exception):
                ws.close()

    def _request(self, api: str, request_payload: dict[str, Any]) -> dict[str, Any]:
        ws = self._ws
        if ws is None:
            raise ConnectionError(f"not connected to {self.url}; call connect() first")
        try:
            ws.send(packb((api, request_payload)))
            deadline = time.monotonic() + self.request_timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out after {self.request_timeout_s:g}s waiting for {api!r}"
                    )
                raw = ws.recv(timeout=remaining)
                if isinstance(raw, str):
                    # A text frame is a transport fault per the protocol contract,
                    # never an in-band error: treat the connection as dead.
                    raise ConnectionError(
                        f"received an unexpected text frame from {self.url} during {api!r}: {raw!r}"
                    )
                decoded = unpackb(raw)
                status, payload = parse_response(decoded)
                raise_for_status(status)
                return payload
        except TimeoutError:
            self._abandon()
            raise
        except ConnectionError:
            self._abandon()
            raise
        except _CONNECTION_ERRORS as exc:
            self._abandon()
            raise ConnectionError(f"connection to {self.url} lost during {api!r}: {exc}") from exc

    def _abandon(self) -> None:
        ws = self._ws
        self._ws = None
        self.spec = None
        if ws is not None:
            with contextlib.suppress(Exception):
                ws.close()
