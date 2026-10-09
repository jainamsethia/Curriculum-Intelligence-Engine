"""Step 1 - Data collection.

Pulls course metadata (title, description, human-curated OCW topic tags) from the
MIT Learn search API and the syllabus / calendar / lecture-notes pages from MIT
OpenCourseWare (CC BY-NC-SA 4.0). Raw page HTML is stored as-is; cleaning happens
in preprocess.py.

    python src/collect.py
"""
import json, re, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "ocw"
API = "https://api.learn.mit.edu/api/v1/learning_resources_search/"
UA = {"User-Agent": "Group9-CurriculumIntelligence/0.1 (NLP course project; non-commercial)"}

# Overlapping subject areas (ML / AI / Stats / Math / CS) so pairwise comparison is meaningful.
QUERIES = ["machine learning", "deep learning", "artificial intelligence", "natural language processing",
           "statistics", "probability", "statistical inference", "data science", "algorithms",
           "linear algebra", "optimization", "computer vision", "econometrics", "signal processing",
           "reinforcement learning", "neural networks", "data mining"]
PER_QUERY = 8
KEEP_TOPICS = {"Computer Science", "Mathematics", "Probability and Statistics", "Artificial Intelligence", "Econometrics"}
MAX_OFFERINGS = 2          # keep at most 2 semesters of the same course number
PAGES = ["syllabus", "calendar", "lecture-notes"]


def search(q):
    r = requests.get(API, params=dict(q=q, platform="ocw", resource_type="course", limit=PER_QUERY), headers=UA, timeout=60)
    r.raise_for_status()
    return r.json()["results"]


def page_main_html(url):
    time.sleep(0.5)  # be polite to ocw.mit.edu
    try:
        r = requests.get(url, headers=UA, timeout=60)
    except requests.RequestException:  # some pages redirect-loop or time out: treat as missing
        return ""
    if r.status_code != 200:
        return ""
    main = BeautifulSoup(r.text, "html.parser").find("main")
    return str(main) if main else ""


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    picked, per_number = {}, {}
    for q in QUERIES:
        for c in search(q):
            if c["url"] in picked or not KEEP_TOPICS & set(c.get("ocw_topics") or []):
                continue
            num = c["readable_id"].split("+")[0]
            if per_number.get(num, 0) >= MAX_OFFERINGS:
                continue
            per_number[num] = per_number.get(num, 0) + 1
            picked[c["url"]] = (q, c)
    print(f"{len(picked)} courses selected from {len(QUERIES)} queries")

    for i, (url, (q, c)) in enumerate(picked.items(), 1):
        cid = re.sub(r"[^\w.-]", "_", c["readable_id"])
        out = RAW / f"{cid}.json"
        if out.exists():
            continue
        run = (c.get("runs") or [{}])[0]
        meta = {
            "id": cid, "readable_id": c["readable_id"], "title": c["title"], "url": url,
            "course_numbers": [n["value"] for n in (c.get("course") or {}).get("course_numbers", [])],
            "departments": [d["name"] for d in c.get("departments") or []],
            "level": [l["name"] for l in run.get("level") or []],
            "semester": run.get("semester"), "year": run.get("year"),
            "description": c.get("description") or "", "full_description": c.get("full_description") or "",
            "ocw_topics": c.get("ocw_topics") or [], "found_by_query": q,
            "license": "CC BY-NC-SA 4.0 (MIT OpenCourseWare)",
        }
        pages = {p: page_main_html(f"{url.rstrip('/')}/pages/{p}/") for p in PAGES}
        out.write_text(json.dumps({"meta": meta, "pages": pages}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(picked)}] {cid}: " + ", ".join(p for p, h in pages.items() if h))


if __name__ == "__main__":
    main()
