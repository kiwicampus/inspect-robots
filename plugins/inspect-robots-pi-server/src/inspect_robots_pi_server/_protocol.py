"""Build and parse the PI-protocol websocket wire framing.

Every exchange is a 2-tuple: the client sends ``(api: str, payload: dict)``,
the server replies ``(status: dict, payload: dict)`` where ``status`` is
``{"success": bool, "error_type"?: str, "error_message"?: str}``. Both
directions are msgpack-encoded (see :mod:`inspect_robots_pi_server.
_msgpack_codec`); a **text** frame from the server is a transport fault, not
an in-band error, and is the caller's job to treat as a dead connection (see
``_client.py``), not this module's.

Unlike ``inspect_robots_xpolicylab``'s own request/response envelope (which
pairs replies by an explicit ``request_id``), this protocol has no such
pairing: it is a single-in-flight blocking RPC per socket (send, then
receive), confirmed against both the protocol doc and the real
``pi_inference_client`` client's ``_timed_request``.
"""

from __future__ import annotations

from typing import Any


class PiServerError(Exception):
    """A protocol-level failure: a malformed frame, or an in-band server error."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


def build_request(api: str, payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Build the ``(api, payload)`` request tuple to encode and send."""
    return (api, payload)


def parse_response(decoded: object) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a decoded ``(status, payload)`` response tuple's shape.

    Does not raise on ``status["success"] is False``: that is the caller's
    job (``_client.py``), so it can attach request context (which RPC, which
    trial step) to the resulting :class:`PiServerError` before raising.
    """
    if not isinstance(decoded, (list, tuple)) or len(decoded) != 2:
        raise PiServerError("ProtocolError", "response must be a 2-element (status, payload) array")
    status, payload = decoded
    if not isinstance(status, dict):
        raise PiServerError(
            "ProtocolError", f"response status must be an object, got {type(status).__name__}"
        )
    if not isinstance(payload, dict):
        raise PiServerError(
            "ProtocolError", f"response payload must be an object, got {type(payload).__name__}"
        )
    return status, payload


def raise_for_status(status: dict[str, Any]) -> None:
    """Raise :class:`PiServerError` for an in-band ``status.success=False`` reply."""
    if status.get("success", False):
        return
    error_type = status.get("error_type") or "ServerError"
    error_message = status.get("error_message") or "unknown server error"
    raise PiServerError(str(error_type), str(error_message))
