"""Tests for the 2-tuple request/response framing."""

from __future__ import annotations

import pytest

from inspect_robots_pi_server._protocol import (
    PiServerError,
    build_request,
    parse_response,
    raise_for_status,
)


def test_build_request_shape() -> None:
    assert build_request("infer", {"a": 1}) == ("infer", {"a": 1})


def test_parse_response_round_trip() -> None:
    status, payload = parse_response([{"success": True}, {"result": {}}])
    assert status == {"success": True}
    assert payload == {"result": {}}


@pytest.mark.parametrize(
    "decoded",
    [
        {"not": "a tuple"},
        [1, 2, 3],
        ["not a dict", {}],
        [{}, "not a dict"],
    ],
)
def test_parse_response_rejects_malformed_shapes(decoded: object) -> None:
    with pytest.raises(PiServerError, match="must be"):
        parse_response(decoded)


def test_raise_for_status_noop_on_success() -> None:
    raise_for_status({"success": True})


def test_raise_for_status_raises_with_server_fields() -> None:
    with pytest.raises(PiServerError, match="InferenceError: boom"):
        raise_for_status(
            {"success": False, "error_type": "InferenceError", "error_message": "boom"}
        )


def test_raise_for_status_defaults_when_fields_missing() -> None:
    with pytest.raises(PiServerError, match="ServerError: unknown server error"):
        raise_for_status({"success": False})
