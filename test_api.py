import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    client_app = TestClient(app)
    return client_app


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"author": "antonio artur",
                        "date": "2022 mar 28",
                        "local": "brazil"}


def test_inference_class_zero(client):
    r = client.post("/inference", json={
      "age": 39,
      "workclass": "State-gov",
      "fnlgt": 77516,
      "education": "Bachelors",
      "education_num": 13,
      "marital_status": "Never-married",
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "capital_gain": 2174,
      "capital_loss": 0,
      "hours_per_week": 40,
      "native_country": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": " <=50K"}


def test_inference_class_one(client):
    r = client.post("/inference", json={
      "age": 52,
      "workclass": "Self-emp-not-inc",
      "fnlgt": 209642,
      "education": "HS-grad",
      "education_num": 9,
      "marital_status": "Married-civ-spouse",
      "occupation": "Exec-managerial",
      "relationship": "Husband",
      "race": "White",
      "sex": "Male",
      "capital_gain": 10000,
      "capital_loss": 0,
      "hours_per_week": 45,
      "native_country": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": " >50K"}


def test_input_malformed(client):
    r = client.post("/inference", json={
      "age": 39,
      "workclass": "State-gov",
      "fnlgt": 77516,
      "education": "Bachelors",
      "education_num": 13,
      "occupation": "Adm-clerical",
      "relationship": "Not-in-family",
      "race": "White",
      "sex": "Male",
      "capital_gain": 0,
      "hours_per_week": 0,
      "native_country": "United-States"
    })
    assert r.status_code == 422
