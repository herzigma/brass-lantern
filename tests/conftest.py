import pytest

from fake_grue import FakeGrue
from brass_lantern.grue_client import GrueClient


@pytest.fixture
def fake():
    return FakeGrue()


@pytest.fixture
def make_grue(fake):
    clients = []

    def _make(**kw):
        c = GrueClient("http://grue.test", transport=fake.transport(), **kw)
        clients.append(c)
        return c

    return _make
