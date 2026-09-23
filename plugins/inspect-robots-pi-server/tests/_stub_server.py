"""In-process PI-protocol-shaped websocket server for adapter tests."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from typing import Any

from websockets.datastructures import Headers
from websockets.http11 import Request, Response
from websockets.sync.server import Server, ServerConnection, serve

from inspect_robots_pi_server._msgpack_codec import packb, unpackb

_DEFAULT_INPUT_SPEC: dict[str, Any] = {
    "cam_head": [[8, 8, 3], "uint8"],
    "joint_position": [[4], "float32"],
}


class StubPiServer:
    """Serve the PI-protocol array-frame protocol used by the plugin on a free local port."""

    def __init__(
        self,
        *,
        action_dim: int = 4,
        action_horizon: int | None = 2,
        action_keys: tuple[str, ...] = ("actions",),
        input_spec: dict[str, Any] | None = None,
        target_resolution: tuple[int, int] | None = (8, 8),
        image_resolutions: dict[str, tuple[int, int]] | None = None,
        resize_mode: str = "pad",
        interpolation: str = "bilinear",
        require_api_key: str | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.action_keys = list(action_keys)
        self.input_spec = dict(input_spec) if input_spec is not None else dict(_DEFAULT_INPUT_SPEC)
        self.target_resolution = target_resolution
        self.image_resolutions = dict(image_resolutions or {})
        self.resize_mode = resize_mode
        self.interpolation = interpolation
        self._require_api_key = require_api_key

        self._lock = threading.RLock()
        self._requests: list[tuple[str, dict[str, Any]]] = []
        self._connections: set[ServerConnection] = set()
        self._server: Server = serve(
            self._handler,
            "127.0.0.1",
            0,
            max_size=None,
            process_request=self._process_request if require_api_key is not None else None,
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        """The websocket URL bound to the server's ephemeral port."""
        port = self._server.socket.getsockname()[1]
        return f"ws://127.0.0.1:{port}"

    def requests(self) -> list[tuple[str, dict[str, Any]]]:
        """A snapshot of every ``(api, payload)`` request received so far."""
        with self._lock:
            return list(self._requests)

    def wait_for(
        self,
        predicate: Callable[[list[tuple[str, dict[str, Any]]]], bool],
        *,
        timeout_s: float = 2.0,
    ) -> None:
        """Wait until a predicate over the recorded requests becomes true."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate(self.requests()):
                return
            time.sleep(0.005)
        raise TimeoutError("stub pi server did not receive the expected request")

    def stop(self) -> None:
        """Shut down the websocket server and join its serving thread."""
        with self._lock:
            targets = list(self._connections)
        for connection in targets:
            with contextlib.suppress(Exception):
                connection.close()
        self._server.shutdown()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise TimeoutError("stub pi server serving thread did not stop")

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        expected = f"Api-Key {self._require_api_key}"
        if request.headers.get("Authorization") != expected:
            return Response(401, "Unauthorized", Headers(), b"unauthorized")
        return None

    def _spec_payload(self) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "input_spec": self.input_spec,
            "output_spec": {
                key: [[self.action_horizon or 1, self.action_dim], "float32"]
                for key in self.action_keys
            },
            "action_keys": self.action_keys,
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim if self.action_horizon is not None else None,
            "image_preprocess": {
                "target_resolution": list(self.target_resolution)
                if self.target_resolution
                else None,
                "image_resolutions": {k: list(v) for k, v in self.image_resolutions.items()},
                "resize_mode": self.resize_mode,
                "interpolation": self.interpolation,
            },
        }
        return {"result": {"spec": json.dumps(spec)}}

    def _infer_reply(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        inputs = payload.get("inference_input", {}).get("inputs", {})
        instruction = inputs.get("robot_task_string") or inputs.get("prompt")
        if instruction == "__error__":
            return {"success": False, "error_type": "InferenceError", "error_message": "boom"}, {}
        if instruction == "__hang__":
            time.sleep(3600)
            return {"success": True}, {}
        horizon = self.action_horizon or 3
        outputs = {}
        for i, key in enumerate(self.action_keys):
            base = float(i * 100)
            outputs[key] = [
                [base + step + dim * 0.01 for dim in range(self.action_dim)]
                for step in range(horizon)
            ]
        result = {
            "outputs": outputs,
            "raw_outputs": {"actions": outputs[self.action_keys[0]]},
        }
        return {"success": True}, {
            "result": result,
            "server_processing_time_ms": 1.0,
            "server_total_time_ms": 2.0,
        }

    def _handler(self, ws: ServerConnection) -> None:
        with self._lock:
            self._connections.add(ws)
        try:
            for raw in ws:
                if isinstance(raw, str):
                    continue
                decoded = unpackb(raw)
                assert isinstance(decoded, (list, tuple)) and len(decoded) == 2
                api = str(decoded[0])
                payload: dict[str, Any] = dict(decoded[1])
                with self._lock:
                    self._requests.append((api, dict(payload)))

                inputs = (
                    payload.get("inference_input", {}).get("inputs", {}) if api == "infer" else {}
                )
                instruction = inputs.get("robot_task_string") or inputs.get("prompt")

                if api == "load":
                    ws.send(packb(({"success": True}, self._spec_payload())))
                elif api == "reset" or api == "telemetry":
                    ws.send(packb(({"success": True}, {"result": {}})))
                elif api == "infer":
                    if instruction == "__drop__":
                        ws.close()
                        return
                    if instruction == "__text__":
                        ws.send("not msgpack")
                        continue
                    status, response_payload = self._infer_reply(payload)
                    ws.send(packb((status, response_payload)))
                else:
                    ws.send(
                        packb(
                            (
                                {
                                    "success": False,
                                    "error_type": "ProtocolError",
                                    "error_message": f"unknown api {api!r}",
                                },
                                {},
                            )
                        )
                    )
        finally:
            with self._lock:
                self._connections.discard(ws)
