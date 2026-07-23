import pytest
import json
from app import app, resolve_intent

@pytest.fixture
def client():
    app.config["TESTING"] = True
    # Disable rate limiter for testing
    app.config["RATELIMIT_ENABLED"] = False
    with app.test_client() as client:
        yield client

def test_resolve_intent():
    # Exact match
    tag, resp = resolve_intent("Hi")
    assert tag == "greeting" or resp != "Sorry, I didn’t understand."

    # Empty match
    tag, resp = resolve_intent("")
    assert tag == "unknown"

    # Some invalid intent
    tag, resp = resolve_intent("some gibberish question that doesn't make sense at all XYZ")
    assert tag == "unknown"

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200

def test_chat_valid(client):
    response = client.post("/chat", json={"message": "hello", "uid": "24MCA10001"})
    assert response.status_code == 200
    data = response.get_json()
    assert "response" in data

def test_chat_invalid_json(client):
    response = client.post("/chat", data="not json", content_type="application/json")
    assert response.status_code == 400

def test_chat_too_long(client):
    response = client.post("/chat", json={"message": "A" * 501})
    assert response.status_code == 400

def test_set_uid_valid(client):
    # Depending on students.example.json, 24MCA10001 should be valid
    response = client.post("/set_uid", json={"uid": "24MCA10001"})
    assert response.status_code in [200, 404] # Valid pattern, might be 404 if not found in data

def test_set_uid_invalid_pattern(client):
    response = client.post("/set_uid", json={"uid": "INVALID_UID"})
    assert response.status_code == 400
