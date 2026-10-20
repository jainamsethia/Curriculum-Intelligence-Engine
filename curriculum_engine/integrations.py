"""Clients for other AI University components (Week 7).

Group 11 - University Communications Intelligence (github.com/Ar-Srivas/uni-comms-intelligence, submission/week_6/src/api.py):
    POST {GROUP11_URL}/process-notice  {"doc_id": str, "text": str}
    -> {"doc_id", "title", "category", "audience": [str], "entities": {...}, "summaries": {persona: str}}
Group 4 - University Knowledge Assistant (github.com/vedantkokate2104-ux/SNLP_CHATBOT, Week 6 notebook):
    no HTTP API; its RAGChatbot.ask(question) answers from a passage table (chunk_id, page, section, part, text).
    We hand over our draft syllabus as rows of that table (CSV) and, for the demo, answer from them with a stand-in of their
    offline (extractive) mode.
Set GROUP11_URL to call Group 11's running service; without it the documented mock is used and every reply says so.
"""
import csv
import json
import os
import urllib.request

from pydantic import BaseModel

from curriculum_engine import modernize as M


class NoticeRequest(BaseModel):          # copied from Group 11's api.py (contract)
    doc_id: str
    text: str


class NoticeResponse(BaseModel):         # copied from Group 11's api.py (contract)
    doc_id: str
    title: str
    category: str
    audience: list[str]
    entities: dict
    summaries: dict[str, str]


def send_to_group11(request, doc, url=None, timeout=60):
    """Live call when GROUP11_URL (or url) is set and reachable, otherwise the mock. Returns (response, mode)."""
    req = NoticeRequest(**request)
    url = url or os.environ.get("GROUP11_URL")
    if url:
        try:
            r = urllib.request.Request(url.rstrip("/") + "/process-notice", data=req.model_dump_json().encode(),
                                       headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return NoticeResponse(**json.loads(resp.read())).model_dump(), "live"
        except Exception as e:                                   # unreachable or wrong schema: fall back, and say so
            out = M.group11_mock(req.model_dump(), doc)
            out["_mock"] += f"; live call to {url} failed: {type(e).__name__}"
            return out, "mock (live call failed)"
    out = M.group11_mock(req.model_dump(), doc)
    NoticeResponse(**{k: v for k, v in out.items() if not k.startswith("_")})      # the mock must honour the contract too
    return out, "mock"


def export_group4(rows, path):
    """Write the passage table Group 4's assistant ingests."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chunk_id", "page", "section", "part", "text"])
        w.writeheader()
        w.writerows(rows)
    return path


def email_from_group11(response):
    """A structured email per audience from Group 11's reply (title, category, entities, persona summaries)."""
    ent = response.get("entities", {})
    out = {}
    for persona, summary in response.get("summaries", {}).items():
        out[persona] = {"subject": response.get("title", ""), "to": persona, "category": response.get("category", ""),
                        "body": summary, "deadlines": ent.get("dates_and_deadlines", []), "contact": ent.get("action_roles", [])}
    return out
