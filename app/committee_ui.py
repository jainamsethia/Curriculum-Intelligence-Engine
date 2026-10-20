"""Curriculum committee UI (Week 7). Talks to the Curriculum Intelligence API over HTTP.

    uvicorn curriculum_engine.api:app --port 8000          # terminal 1
    streamlit run app/committee_ui.py                     # terminal 2  (CE_API_URL defaults to http://localhost:8000)

Optional: GROUP11_URL=<Group 11 service> to send the draft syllabus to their /process-notice endpoint live.
Every number shown comes from the API; outputs are drafts for the Board of Studies, never automatic decisions.
"""
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
API = os.environ.get("CE_API_URL", "http://localhost:8000")
DEMO_COURSES = ["nmd6d35385", "nm57ee6fd1", "nm26b2f85a", "nm85e9140a", "nm8b216b85"]
AUTORUN = bool(os.environ.get("CE_UI_AUTORUN"))          # screenshots / unattended demo: run the demo step on load


@st.cache_resource
def client():
    if os.environ.get("CE_API_INPROCESS"):                       # tests: call the FastAPI app in-process
        from fastapi.testclient import TestClient
        from curriculum_engine.api import app
        return TestClient(app)
    import httpx
    return httpx.Client(base_url=API, timeout=600)


def call(method, path, **kw):
    r = client().request(method, path, **kw)
    if r.status_code >= 400:
        st.error(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        st.stop()
    return r.json()


def course_picker(label, key, default_id=None):
    q = st.text_input(f"{label}: search by course name", value="" if default_id else "Machine Learning", key=f"{key}_q")
    hits = call("GET", "/courses", params={"q": q, "limit": 100})["courses"]
    opts = {f"{c['course_name']} | {', '.join(c['programmes']) or 'common'} | {c['academic_year']} | {c['id']}": c["id"] for c in hits}
    if not opts:
        st.warning("no course matches")
        st.stop()
    names = list(opts)
    idx = next((i for i, n in enumerate(names) if n.endswith(default_id or "~")), 0)
    return {"id": opts[st.selectbox(label, names, index=idx, key=f"{key}_s")]}


st.set_page_config(page_title="Curriculum Intelligence Engine", layout="wide")
st.title("Curriculum Intelligence Engine — Group 9")
with st.sidebar:
    h = call("GET", "/health")
    st.success(f"API {h['status']} · engine {h['meta']['engine_version']} · {h['meta']['course_versions']} course versions")
    st.caption(h["meta"]["evaluation_note"])
    st.info("Outputs are drafts for the Board of Studies. Do not use scores to cut or merge courses automatically; "
            "check the evidence shown with every flag.")

tabs = st.tabs(["Modernise curriculum (demo)", "Compare two courses", "Compare two years", "Prerequisites", "Redundancy", "Emerging topics"])

# ------------------------------------------------------------------ 1. the 31 October demo step
with tabs[0]:
    st.subheader("Job descriptions + current NLP syllabi → overlaps, demanded skills, a new structured syllabus")
    jd_files = sorted((ROOT / "demo/jds").glob("JD*.txt"))
    cols = st.columns(len(jd_files) or 1)
    jds = [{"id": f.stem, "text": cols[i].text_area(f.stem, f.read_text(encoding="utf-8"), height=220, key=f"jd_{f.stem}")}
           for i, f in enumerate(jd_files)]
    st.caption("The sample JDs are synthetic, written for the demo; paste real job descriptions to replace them.")
    all_nlp = (call("GET", "/courses", params={"ids": ",".join(DEMO_COURSES)})["courses"]           # the demo syllabi, then the latest 500
               + call("GET", "/courses", params={"q": "", "limit": 500})["courses"])
    opts = {f"{c['course_name']} | {', '.join(c['programmes'])} | {c['academic_year']} | {c['id']}": c["id"] for c in all_nlp}
    chosen = st.multiselect("Current syllabi to modernise", list(opts), default=[k for k, v in opts.items() if v in DEMO_COURSES])
    title = st.text_input("Title of the new course", "Natural Language Processing and Large Language Models")
    if (st.button("Analyse and draft the new syllabus", type="primary") or (AUTORUN and "mod" not in st.session_state)) and len(chosen) >= 2:
        st.session_state["mod"] = call("POST", "/modernize-curriculum",
                                       json={"jds": jds, "courses": [{"id": opts[c]} for c in chosen], "title": title})
    res = st.session_state.get("mod")
    if res:
        st.markdown("#### 1. Overlap between the current syllabi")
        st.dataframe(pd.DataFrame(res["overlaps"])[["course_a", "course_b", "overlap_percent", "score", "confidence"]], use_container_width=True)
        st.markdown("#### 2. Skills the job descriptions ask for (themes named by at least two JDs)")
        th = [t for t in res["skill_themes"] if len(t["jds"]) >= 2]
        st.dataframe(pd.DataFrame([{"skill": t["theme"], "asked for by": ", ".join(t["jds"]), "already taught": t["covered"],
                                    "closest topic": f"{t['closest_units'][0]['topic']} ({t['closest_units'][0]['course']})"} for t in th]),
                     use_container_width=True)
        st.markdown("#### 3. Draft syllabus (every unit names its source)")
        doc = res["new_curriculum"]
        st.warning(doc["status"])
        st.markdown(res["markdown"])
        from curriculum_engine import modernize as M
        buf = io.BytesIO()
        M.to_docx(doc, buf)
        c1, c2, c3 = st.columns(3)
        c1.download_button("Download .docx", buf.getvalue(), "draft_syllabus.docx")
        c2.download_button("Download .md", res["markdown"], "draft_syllabus.md")
        c3.download_button("Download .json", json.dumps(doc, indent=1), "draft_syllabus.json")

        st.markdown("#### 4. Hand-off to Group 11 (University Communications Intelligence)")
        url = st.text_input("Group 11 service URL (empty = documented mock)", os.environ.get("GROUP11_URL", ""))
        if st.button("Send the draft as a notice to Group 11") or AUTORUN:
            from curriculum_engine import integrations as I
            resp, mode = I.send_to_group11(res["handoffs"]["group11_request"], doc, url or None)
            st.caption(f"mode: {mode}")
            st.json(res["handoffs"]["group11_request"], expanded=False)
            for persona, mail in I.email_from_group11(resp).items():
                st.markdown(f"**To: {persona}** — *{mail['subject']}*  \n{mail['body']}")

        st.markdown("#### 5. Hand-off to Group 4 (University Knowledge Assistant)")
        rows = res["handoffs"]["group4_chunks"]
        st.download_button("Download passages for Group 4's knowledge base (CSV)",
                           pd.DataFrame(rows).to_csv(index=False), "group9_new_course_passages.csv")
        q = st.text_input("Student question (answered by our stand-in of Group 4's offline mode)", "What are the prerequisites for the new course?")
        if q:
            from curriculum_engine import modernize as M
            from curriculum_engine.embed import Embedder
            a = M.group4_standin(rows, q, Embedder())
            st.markdown(f"> {a['answer']}  \n*{a['source']} · retrieval confidence {a['confidence']} · {a['_standin']}*")

# ------------------------------------------------------------------ 2-6. Week 6 endpoints
with tabs[1]:
    a, b = st.columns(2)
    with a:
        ca = course_picker("Course A", "ca", DEMO_COURSES[0])
    with b:
        cb = course_picker("Course B", "cb", DEMO_COURSES[3])
    if st.button("Compare", key="cmp"):
        r = call("POST", "/compare-curricula", json={"course_a": ca, "course_b": cb})
        o = r["overlap"]
        st.metric("Unit overlap", o["percent"], help="share of matched units")
        st.write(f"confidence {o['confidence']} (relative) · redundancy warning: {o['redundancy_warning']}")
        st.dataframe(pd.DataFrame(r["common_topics"]), use_container_width=True)
        st.write("Unique to A:", r["unique_to_a"])
        st.write("Unique to B:", r["unique_to_b"])

with tabs[2]:
    p = st.text_input("Programme", "COMP", key="dprog")
    y1, y2 = st.columns(2)
    a1, a2 = y1.text_input("Year A", "2024-25"), y2.text_input("Year B", "2025-26")
    if st.button("Compare years"):
        r = call("POST", "/compare-curricula", json={"programme_a": {"programme": p, "academic_year": a1},
                                                     "programme_b": {"programme": p, "academic_year": a2}})
        st.write(f"retained {r['retained']} · renamed or revised {len(r['renamed_or_revised'])}")
        st.write("Removed:", [x["course_name"] for x in r["removed"]])
        st.write("Added:", [x["course_name"] for x in r["added"]])
        st.caption(r["note"])

with tabs[3]:
    c = course_picker("Course", "pre", "nm1b82f319")
    if st.button("Check prerequisites"):
        r = call("POST", "/check-prerequisites", json={"course": c})["result"]
        st.write("checked" if r["checked"] else f"not checked: {r['reason']}")
        st.dataframe(pd.DataFrame(r["flags"]), use_container_width=True)

with tabs[4]:
    p = st.text_input("Programme", "COMP", key="rprog")
    if st.button("Find redundant units"):
        r = call("POST", "/detect-redundancy", json={"programme": p, "k": 15})
        st.dataframe(pd.DataFrame(r["pairs"]), use_container_width=True)

with tabs[5]:
    p = st.text_input("Programme", "COMP", key="eprog")
    if st.button("Check emerging topics"):
        r = call("POST", "/emerging-topics", json={"programme": p})
        st.caption(r["reference_status"])
        st.dataframe(pd.DataFrame(r["topics"]), use_container_width=True)
