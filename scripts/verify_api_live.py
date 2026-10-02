"""Test running the FastAPI server in-process using TestClient and print responses."""

from fastapi.testclient import TestClient
import json
import os

from adlift.api.app import app

with TestClient(app) as client:
    print("\n--- 1. Testing GET /health ---")
    health_resp = client.get("/health")
    print(f"Status: {health_resp.status_code}")
    print(json.dumps(health_resp.json(), indent=2))

    print("\n--- 2. Testing POST /predict ---")
    predict_path = os.path.join("data", "demo", "example_predict_request.json")
    with open(predict_path, "r", encoding="utf-8") as f:
        predict_payload = json.load(f)

    predict_resp = client.post("/predict", json=predict_payload)
    print(f"Status: {predict_resp.status_code}")
    print(json.dumps(predict_resp.json(), indent=2))

    print("\n--- 3. Testing POST /rank ---")
    rank_path = os.path.join("data", "demo", "example_rank_request.json")
    with open(rank_path, "r", encoding="utf-8") as f:
        rank_payload = json.load(f)

    rank_resp = client.post("/rank", json=rank_payload)
    print(f"Status: {rank_resp.status_code}")
    print(json.dumps(rank_resp.json(), indent=2))
