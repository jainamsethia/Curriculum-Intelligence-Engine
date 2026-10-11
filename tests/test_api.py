"""API endpoints on a synthetic 11-course programme (tests/fixtures/courses_small.jsonl; no NMIMS data needed).
Uses the committed thresholds and evaluation results, and the locally cached MiniLM encoder."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from curriculum_engine import data as D
from curriculum_engine import prereq as PQ
from curriculum_engine.api import app, get_engine
from curriculum_engine.api_engine import ApiEngine

FIX = Path(__file__).parent / "fixtures/courses_small.jsonl"


@pytest.fixture(scope="module")
def client():
    recs = D.enrich([json.loads(l) for l in open(FIX, encoding="utf-8")])
    eng = ApiEngine(recs=recs)
    app.dependency_overrides[get_engine] = lambda: eng
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def small_programme(monkeypatch):            # the fixture programme is far below the real coverage rule (50 versions)
    monkeypatch.setattr(PQ, "MIN_VERSIONS", 1)
    monkeypatch.setattr(PQ, "MIN_FAMILIES_PER_SEM", 1)


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["meta"]["course_versions"] == 11 and "POST /check-prerequisites" in r["endpoints"]


def test_compare_two_courses(client):
    r = client.post("/compare-curricula", json={"course_a": {"name": "Deep Learning"}, "course_b": {"id": "nmfx000006"}})
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["overlap"]["confidence"] <= 1 and j["overlap"]["percent"].endswith("%")
    assert any(t["topic_a"] == "Regression" for t in j["common_topics"])
    assert j["meta"]["engine_version"]


def test_compare_needs_a_pair(client):
    assert client.post("/compare-curricula", json={"course_a": {"id": "nmfx000006"}}).status_code == 422
    assert client.post("/compare-curricula", json={"course_a": {"id": "nmfx000006", "name": "x"}, "course_b": {"id": "nmfx000007"}}).status_code == 422


def test_compare_two_years(client):
    r = client.post("/compare-curricula", json={"programme_a": {"programme": "COMP", "academic_year": "2024-25"},
                                                "programme_b": {"programme": "COMP", "academic_year": "2025-26"}})
    assert r.status_code == 200 and r.json()["retained"] == 1          # Machine Learning is in both years


def test_unknown_course_and_programme(client):
    assert client.post("/check-prerequisites", json={"course": {"id": "nm00000000"}}).status_code == 404
    assert client.post("/emerging-topics", json={"programme": "NOPE"}).status_code == 404


def test_map_outcomes(client):
    r = client.post("/map-outcomes", json={"outcomes": ["Explain the layers of the OSI model", "Design a relational database schema"]}).json()
    assert [o["level"] for o in r["outcomes"]] == ["Understand", "Create"]


def test_find_overlaps_excludes_own_course(client):
    r = client.post("/find-overlaps", json={"course": {"name": "Deep Learning"}, "k": 3}).json()
    names = [h["course"]["course_name"] for h in r["results"]]
    assert "Deep Learning" not in names and names[0] == "Machine Learning"
    assert [h["score"] for h in r["results"]] == sorted((h["score"] for h in r["results"]), reverse=True)


def test_check_prerequisites(client):
    missing = client.post("/check-prerequisites", json={"course": {"name": "Software Engineering"}}).json()["result"]
    assert missing["checked"] and [f["phrase"] for f in missing["flags"]] == ["Quantum Field Theory"]
    met = client.post("/check-prerequisites", json={"course": {"name": "Deep Learning"}}).json()["result"]
    assert met["checked"] and met["flags"] == []                       # 'Machine Learning' is taught in semester V
    first = client.post("/check-prerequisites", json={"course": {"name": "Calculus"}}).json()["result"]
    assert not first["checked"] and first["reason"]


def test_detect_redundancy_finds_shared_unit(client):
    r = client.post("/detect-redundancy", json={"programme": "COMP", "k": 5}).json()
    assert {(p["course_a"], p["course_b"]) for p in r["pairs"]} & {("Machine Learning", "Deep Learning"), ("Deep Learning", "Machine Learning")}


def test_raw_text_course(client):
    text = "Linear regression and logistic regression with regularisation.\n\nDecision trees and support vector machines for classification."
    r = client.post("/compare-curricula", json={"course_a": {"text": text}, "course_b": {"name": "Machine Learning"}})
    assert r.status_code == 200 and r.json()["course_a"]["id"].startswith("tx")


def test_emerging_topics(client):
    r = client.post("/emerging-topics", json={"programme": "COMP"}).json()
    assert r["reference_status"].startswith("DRAFT") and all("source" in t for t in r["topics"])
