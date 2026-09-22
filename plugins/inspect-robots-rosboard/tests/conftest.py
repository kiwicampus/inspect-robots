from __future__ import annotations

from collections.abc import Iterator

import pytest
from _stub_server import StubRosboardServer


@pytest.fixture()
def stub_server() -> Iterator[StubRosboardServer]:
    server = StubRosboardServer()
    yield server
    server.stop()
