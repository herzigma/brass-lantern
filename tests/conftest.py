import pytest

from fake_grue import FakeGrue
from brass_lantern.grue_client import GrueClient


def pytest_configure(config):
    config.addinivalue_line("markers", "real_probe: run the real memorization probe instead of the stub")


@pytest.fixture(autouse=True)
def stub_memorization_probe(request, monkeypatch):
    """Most tests are about playing: the probe would use up a scripted reply and add a Grue session.

    The stub answers "negative" (so runs stay ranked) without a model call or a session. Tests of the probe
    itself are marked ``real_probe``.
    """
    if request.node.get_closest_marker("real_probe"):
        return

    async def stub(**kwargs):
        return {"status": "negative", "stub": True}

    monkeypatch.setattr("brass_lantern.runner.run_probe", stub)


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
