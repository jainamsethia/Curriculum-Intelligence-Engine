"""Week 7: curriculum modernisation, the Group 11 / Group 4 hand-off contracts, the new endpoints and the UI (synthetic data only)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from curriculum_engine import data as D
from curriculum_engine import integrations as I
from curriculum_engine import modernize as M
from curriculum_engine.api import app, get_engine
from curriculum_engine.api_engine import ApiEngine

FIX = Path(__file__).parent / "fixtures/courses_small.jsonl"
JDS = {"JD1": "Role: ML Engineer\nResponsibilities:\n- Fine-tune large language models with LoRA for customer support.\n"
              "- Evaluate models with automatic metrics and human evaluation.",
       "JD2": "Requirements:\n- Experience with large language models and prompt engineering.\n- Deploy models with quantization."}


@pytest.fixture(scope="module")
def eng():
    return ApiEngine(recs=D.enrich([json.loads(l) for l in open(FIX, encoding="utf-8")]))


@pytest.fixture(scope="module")
def result(eng):
    cs = [r for r in eng.recs if r["course_name"] in ("Machine Learning", "Deep Learning") and r["academic_year"] == "2025-26"]
    return M.run(eng, JDS, cs, "Applied Machine Learning")


def test_requirements_and_phrases():
    reqs = M.requirements(JDS["JD1"])
    assert reqs == ["Fine-tune large language models with LoRA for customer support.",
                    "Evaluate models with automatic metrics and human evaluation."]
    phrases = M.skill_phrases(reqs[0])
    assert "LoRA" in phrases and any("language models" in p for p in phrases)


def test_draft_is_traceable_and_complete(result):
    doc = result["new_curriculum"]
    assert result["traceable"] and doc["units"] and doc["status"].startswith("DRAFT")
    assert all(u["sources"] for u in doc["units"]) and {c["id"] for c in doc["replaces"]} == {"nmfx000006", "nmfx000007"}
    assert any(u["kind"] == "shared topic (deduplicated)" for u in doc["units"])        # both courses teach the same regression unit


def test_group11_contract(result):
    req = result["handoffs"]["group11_request"]
    assert set(req) == {"doc_id", "text"} and I.NoticeRequest(**req)
    resp, mode = I.send_to_group11(req, result["new_curriculum"])                     # no GROUP11_URL here -> mock, and it says so
    assert mode == "mock" and resp["_mock"] and I.NoticeResponse(**{k: v for k, v in resp.items() if not k.startswith("_")})
    assert set(I.email_from_group11(resp)) == {"student", "faculty", "admin"}


def test_group11_unreachable_falls_back(result):
    resp, mode = I.send_to_group11(result["handoffs"]["group11_request"], result["new_curriculum"], url="http://127.0.0.1:9", timeout=2)
    assert mode == "mock (live call failed)" and "failed" in resp["_mock"]


def test_group4_contract(result, eng, tmp_path):
    rows = result["handoffs"]["group4_chunks"]
    assert all(list(r) == ["chunk_id", "page", "section", "part", "text"] for r in rows)
    assert max(len(r["text"].split()) for r in rows) <= 100
    I.export_group4(rows, tmp_path / "chunks.csv")
    a = M.group4_standin(rows, "How many credits does the course carry?", eng.emb)
    assert a["section"] == "Credits and teaching scheme" and "credits" in a["answer"]


def test_modernize_endpoint(eng):
    app.dependency_overrides[get_engine] = lambda: eng
    try:
        c = TestClient(app)
        body = {"jds": [{"id": k, "text": v} for k, v in JDS.items()], "courses": [{"id": "nmfx000006"}, {"id": "nmfx000007"}]}
        r = c.post("/modernize-curriculum", json=body)
        assert r.status_code == 200 and r.json()["traceable"] and r.json()["handoffs"]["group11_request"]["doc_id"]
        assert c.post("/modernize-curriculum", json={**body, "courses": [{"id": "nmfx000006"}]}).status_code == 422
        assert [x["id"] for x in c.get("/courses", params={"q": "learning"}).json()["courses"]][:1] == ["nmfx000007"]
    finally:
        app.dependency_overrides.clear()


def test_ui_renders(eng, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("CE_API_INPROCESS", "1")
    app.dependency_overrides[get_engine] = lambda: eng
    try:
        at = AppTest.from_file(str(Path(__file__).parents[1] / "app/committee_ui.py"), default_timeout=120).run()
        assert not at.exception and at.title[0].value.startswith("Curriculum Intelligence Engine")
        assert len(at.tabs) == 6
    finally:
        app.dependency_overrides.clear()
