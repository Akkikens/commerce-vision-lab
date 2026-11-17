import pytest
from fastapi.testclient import TestClient

from cvlab.api import create_app
from cvlab.services import build_stub_services


@pytest.fixture
def services(tmp_path):
    return build_stub_services(review_path=tmp_path / "queue.jsonl")


@pytest.fixture
def client(services):
    return TestClient(create_app(services))
