"""Curriculum Intelligence API (FastAPI).

    uvicorn curriculum_engine.api:app --port 8000        # OpenAPI docs at http://localhost:8000/docs

The engine is built once, on first use, from the parsed NMIMS dataset (CE_DATA_DIR) and the committed results/*.json
(thresholds, confidence model, measured precisions). Nothing is sent to external services.
"""
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from curriculum_engine import __version__
from curriculum_engine import schemas as S

app = FastAPI(title="Curriculum Intelligence Engine API", version=__version__,
              description="Course and programme comparison for NMIMS curriculum committees: overlap, redundancy, missing prerequisites, "
                          "curriculum diff, Bloom levels and emerging-topic gaps. Confidence values come from LLM-annotated reference "
                          "labels (no human annotators); they are indicative.")
ENDPOINTS = ["POST /compare-curricula", "POST /map-outcomes", "POST /find-overlaps", "POST /check-prerequisites",
             "POST /detect-redundancy", "POST /emerging-topics", "GET /health"]


@lru_cache(maxsize=1)
def get_engine():
    from curriculum_engine.api_engine import ApiEngine
    return ApiEngine()


def course(eng, ref):
    try:
        return eng.resolve(ref)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0]))


def known_programme(eng, prog):
    if prog not in eng.view:
        raise HTTPException(status_code=404, detail=f"unknown programme {prog!r}; known: {sorted(eng.view)}")


@app.get("/health", response_model=S.Health)
def health(eng=Depends(get_engine)):
    return {"status": "ok", "meta": eng.meta_block(), "endpoints": ENDPOINTS}


@app.post("/compare-curricula", response_model=S.CompareCoursesResponse | S.CompareProgrammesResponse)
def compare_curricula(req: S.CompareRequest, eng=Depends(get_engine)):
    """Two courses -> overlap, common/unique topics, prerequisite check, Bloom levels. Two programmes (or two years of one
    programme) -> courses retained, renamed or revised, removed and added."""
    if req.course_a:
        return eng.compare_courses(course(eng, req.course_a), course(eng, req.course_b))
    for p in (req.programme_a, req.programme_b):
        known_programme(eng, p.programme)
    return eng.compare_programmes(req.programme_a, req.programme_b)


@app.post("/map-outcomes", response_model=S.OutcomesResponse)
def map_outcomes(req: S.OutcomesRequest, eng=Depends(get_engine)):
    if req.course:
        c = course(eng, req.course)
        items = eng.bloom_items([o["text"] for o in c.get("outcomes", [])], [o["k_levels"] for o in c.get("outcomes", [])])
    else:
        items = eng.bloom_items(req.outcomes)
    return {"meta": eng.meta_block(), "outcomes": items}


@app.post("/find-overlaps", response_model=S.FindOverlapsResponse)
def find_overlaps(req: S.FindOverlapsRequest, eng=Depends(get_engine)):
    if req.programme:
        known_programme(eng, req.programme)
    c = course(eng, req.course)
    return {"meta": eng.meta_block(), "query": eng.meta(c), "results": eng.find_overlaps(c, req.k, req.programme)}


@app.post("/check-prerequisites", response_model=S.PrerequisitesResponse)
def check_prerequisites(req: S.PrerequisitesRequest, eng=Depends(get_engine)):
    c = course(eng, req.course)
    return {"meta": eng.meta_block(), "course": eng.meta(c), "result": eng.prereq_v2(c)}


@app.post("/detect-redundancy", response_model=S.RedundancyResponse)
def detect_redundancy(req: S.ProgrammeRequest, eng=Depends(get_engine)):
    known_programme(eng, req.programme)
    n, pairs = eng.detect_redundancy(req.programme, req.academic_year, req.k)
    return {"meta": eng.meta_block(), "programme": req.programme, "academic_year": req.academic_year, "courses_checked": n, "pairs": pairs}


@app.post("/emerging-topics", response_model=S.EmergingResponse)
def emerging_topics(req: S.ProgrammeRequest, eng=Depends(get_engine)):
    known_programme(eng, req.programme)
    return {"meta": eng.meta_block(), "programme": req.programme, **eng.emerging(req.programme)}
