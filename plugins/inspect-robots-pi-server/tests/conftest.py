from __future__ import annotations

from collections.abc import Iterator

import pytest
from _stub_server import StubPiServer


@pytest.fixture()
def stub_server() -> Iterator[StubPiServer]:
    server = StubPiServer()
    yield server
    server.stop()
