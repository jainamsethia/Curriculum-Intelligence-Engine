"""Week 7 - the 31 October demo step of Group 9, end to end, plus the checks behind it:
    5 job descriptions (demo/jds, synthetic samples) + the current NLP syllabi of several programmes
    -> overlaps, JD skill themes, coverage, a traceable draft syllabus (JSON, Markdown, Word)
    -> Group 11 hand-off (their /process-notice contract; live if GROUP11_URL is set, otherwise the documented mock)
    -> Group 4 hand-off (their passage-table schema) and answers to the demo questions from a stand-in of their offline mode.
Writes results/week7/*.json (metrics, hand-off payloads, no syllabus text). Outputs that quote NMIMS syllabus text
(draft syllabus .md/.docx, Group 4 passages, full demo result) go to data/week7/: git-ignored, Teams code zip only."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from curriculum_engine import DATA, LABELS, RESULTS, ROOT
from curriculum_engine import emerging as EM
from curriculum_engine import evaluate as E
from curriculum_engine import integrations as I
from curriculum_engine import modernize as M
from curriculum_engine.experiments.common import dump

OUT = "week7"
FULL = DATA / "week7"                                                # quotes syllabus text: never committed
DEMO_COURSES = ["nmd6d35385", "nm57ee6fd1", "nm26b2f85a", "nm85e9140a", "nm8b216b85"]   # latest NLP-related syllabus per programme group
TITLE = "Natural Language Processing and Large Language Models"
QUESTIONS = {"What are the prerequisites for the new course?": "Prerequisites", "Does the course have lab work?": "Lab work",   # -> expected section (substring)
             "How many credits does the course carry?": "Credits and teaching scheme",
             "Which topics on fine-tuning large language models are covered?": "Fine Tuning", "How is the course assessed?": "Assessment and examination"}


def short(sources):
    """'Natural Language Processing (nm57ee6fd1)', ... -> 'NLP ×3, Speech and NLP; JD1, JD3' (report tables)."""
    names = Counter(re.sub(r"\s*\(nm\w+\)$", "", x).replace("Natural Language Processing", "NLP") for x in sources if not x.startswith("JD"))
    jd = [x for x in sources if x.startswith("JD")]
    return "; ".join(filter(None, [", ".join(n + (f" ×{k}" if k > 1 else "") for n, k in names.items()), ", ".join(jd)]))


def jds():
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted((ROOT / "demo/jds").glob("JD*.txt"))}


def demo(ctx, eng):
    courses = [eng.by_id[i] for i in DEMO_COURSES]
    res = M.run(eng, jds(), courses, TITLE)
    doc = res["new_curriculum"]
    FULL.mkdir(parents=True, exist_ok=True)
    (FULL / "new_curriculum.md").write_text(res["markdown"], encoding="utf-8")
    M.to_docx(doc, FULL / "new_curriculum.docx")
    resp, mode = I.send_to_group11(res["handoffs"]["group11_request"], doc)
    dump({"request": res["handoffs"]["group11_request"], "response": resp, "mode": mode, "emails": I.email_from_group11(resp)},
         f"{OUT}/group11_handoff.json")
    rows = res["handoffs"]["group4_chunks"]
    I.export_group4(rows, FULL / "group4_chunks.csv")
    answers = [{**M.group4_standin(rows, q, eng.emb), "expected_section": sec} for q, sec in QUESTIONS.items()]
    for a in answers:
        a["from_expected_section"] = a["expected_section"] in a["section"]
    dump({"note": "answers from Group 9's stand-in for Group 4's offline (extractive) mode over the exported passages",
          "questions": len(answers), "answered_from_expected_section": sum(a["from_expected_section"] for a in answers),
          "answers": answers}, f"{OUT}/group4_handoff.json")
    themes = res["skill_themes"]
    dem = [t for t in themes if len(t["jds"]) >= 2]
    kinds = Counter(u["kind"] for u in doc["units"])
    summary = {"jds": len(jds()), "requirement_statements": sum(len(M.requirements(t)) for t in jds().values()),
               "skill_themes": len(themes), "demanded_themes": len(dem), "demanded_covered": sum(t["covered"] for t in dem),
               "demanded_gaps": sum(not t["covered"] for t in dem), "courses": len(courses),
               "programmes": doc["programmes"], "identical_pairs": sum(o["overlap_percent"] == "100%" for o in res["overlaps"]),
               "pairs": len(res["overlaps"]), "shared_topics": len(res["shared_topics"]),
               "units": len(doc["units"]), "units_shared": kinds.get("shared topic (deduplicated)", 0),
               "units_existing_demanded": kinds.get("existing unit, demanded by industry", 0), "units_new": kinds.get("new unit from job descriptions", 0),
               "total_hours": doc["total_hours"], "outcomes": len(doc["outcomes"]), "traceable": res["traceable"],
               "group4_passages": len(rows), "group11_mode": mode, "endpoints": len(__import__("curriculum_engine.api", fromlist=["ENDPOINTS"]).ENDPOINTS),
               "overlap_table": {"columns": ["Course A", "Course B", "Unit overlap", "Score", "Confidence"],
                                 "rows": [[o["course_a"], o["course_b"], o["overlap_percent"], o["score"], o["confidence"]] for o in res["overlaps"]]},
               "theme_table": {"columns": ["Skill theme (from JDs)", "Asked for by", "Already taught?", "Closest unit topic (course)"],
                               "rows": [[t["theme"], ", ".join(t["jds"]), "yes" if t["covered"] else "gap",
                                         f"{t['closest_units'][0]['topic'][:45]} ({short([t['closest_units'][0]['course']])})"] for t in dem]},
               "unit_table": {"columns": ["#", "Unit of the draft syllabus", "Kind", "Hours", "Source"],
                              "rows": [[i, u["title"][:60], u["kind"], u["hours"], short(u["sources"])] for i, u in enumerate(doc["units"], 1)]}}
    (FULL / "demo_result.json").write_text(json.dumps({k: v for k, v in res.items() if k not in ("markdown", "handoffs")}, indent=1,
                                                      ensure_ascii=False, default=str), encoding="utf-8")
    return dump(summary, f"{OUT}/summary.json"), res


def contracts(res):
    """The hand-offs must match the other groups' schemas exactly; the draft must be traceable."""
    g11 = json.loads((RESULTS / OUT / "group11_handoff.json").read_text(encoding="utf-8"))
    ok_req = I.NoticeRequest(**g11["request"]) is not None
    ok_resp = I.NoticeResponse(**{k: v for k, v in g11["response"].items() if not k.startswith("_")}) is not None
    rows = list(csv.DictReader(open(FULL / "group4_chunks.csv", encoding="utf-8")))
    return dump({"group11_request_schema": ok_req, "group11_response_schema": ok_resp,
                 "group4_columns": list(rows[0]) == ["chunk_id", "page", "section", "part", "text"],
                 "group4_max_words": max(len(r["text"].split()) for r in rows), "group4_rows": len(rows),
                 "draft_traceable": res["traceable"]}, f"{OUT}/contracts.json")


def emerging_threshold(ctx, eng):
    """Coverage threshold for topic vs units: Week 3 setting (prerequisite topic threshold) vs the validation-tuned unit-match
    threshold, on the 67 emerging-topic decisions labelled in Week 3 (these labels also exposed the problem: optimistic)."""
    sims = {f"{g['topic']}|{g['programme']}": g["best_similarity"] for g in EM.coverage(ctx.view, ctx.emb, 0.0)}
    rows = [r for r in csv.DictReader(open(LABELS / "flags_llm.csv", encoding="utf-8")) if r["kind"].startswith("emerging") and r["status"] != "uncertain"]
    taught = np.array([(r["kind"] == "emerging_covered") == (r["consensus"] == "correct") for r in rows])
    s = np.array([sims[r["key"]] for r in rows])
    out = {"labelled_decisions": len(rows), "really_taught": int(taught.sum())}
    for name, t in (("week3_threshold", eng.th["prereq_topic"]), ("unit_match_threshold", eng.th["unit_match"])):
        p = s >= t
        out[name] = {"threshold": E.r3(t), "accuracy": E.r3((p == taught).mean()),
                     "gap_precision": E.r3((~p & ~taught).sum() / max((~p).sum(), 1)), "gap_recall": E.r3((~p & ~taught).sum() / max((~taught).sum(), 1))}
    return dump(out, f"{OUT}/emerging_threshold.json")


def coverage_items(res):
    """Every skill-theme decision (covered / gap) as a claim for the LLM-annotated reference labels (same protocol as the flags)."""
    from curriculum_engine.labelling import ANN, write_items
    if (LABELS / "jdskills_llm.csv").exists():
        return
    rows, items = [], []
    for t in res["skill_themes"]:
        ev = "; ".join(f"{u['course']} - {u['unit']}: {u['topic']}" for u in t["closest_units"])
        claim = (f"CLAIM: the selected NLP syllabi already teach the skill '{t['theme']}'." if t["covered"] else
                 f"CLAIM: the selected NLP syllabi do not teach the skill '{t['theme']}' (a gap).")
        txt = (f"{claim}\nHow the job descriptions phrase it: {'; '.join(t['phrases'])}\n"
               f"Closest topics in the current syllabi: {ev}\nQUESTION: Is the claim correct, judging from these closest topics?")
        rows.append({"item": f"JS{len(rows) + 1:03d}", "kind": "covered" if t["covered"] else "gap", "theme": t["theme"],
                     "demanded": len(t["jds"]) >= 2})
        items.append({"item": rows[-1]["item"], "full": txt, "compact": txt})
    ANN.mkdir(parents=True, exist_ok=True)
    with open(ANN / "jdskills_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    write_items("jdskills", items, parts=1)


def coverage_metrics():
    f = LABELS / "jdskills_llm.csv"
    if not f.exists():
        return dump({"note": "skill-coverage labels not available yet"}, f"{OUT}/coverage_precision.json")
    rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["status"] != "uncertain"]
    res = {}
    for scope, rs in (("all_themes", rows), ("demanded_themes", [r for r in rows if r["demanded"] == "True"])):
        for kind in ("gap", "covered"):
            ok = [r["consensus"] == "correct" for r in rs if r["kind"] == kind]
            res.setdefault(scope, {})[kind] = {"n": len(ok), "precision": E.r3(np.mean(ok)) if ok else None}
        res[scope]["accuracy"] = E.r3(np.mean([r["consensus"] == "correct" for r in rs])) if rs else None
    allr = list(csv.DictReader(open(f, encoding="utf-8")))
    res["labelled"], res["uncertain"] = len(allr), sum(r["status"] == "uncertain" for r in allr)
    res["needed_pass3"] = sum(r["pass1"] != r["pass2"] for r in allr)
    return dump(res, f"{OUT}/coverage_precision.json")


def fairness():
    """Responsible-AI evidence: how much parsed data and how good the results are per programme (from Week 5 / dataset stats)."""
    from curriculum_engine import prereq as PQ
    pp = json.loads((RESULTS / "week5/per_programme.json").read_text(encoding="utf-8"))["table"]
    versions = json.loads((RESULTS / "dataset/stats.json").read_text(encoding="utf-8"))["versions_per_programme"]
    rows, bloom = [], []
    for prog, pairs, auc, tagged, acc, cases, rec in pp["rows"]:
        if prog == "COMMON":
            continue
        n = versions.get(prog, 0)
        rows.append([prog, n, "yes" if n >= PQ.MIN_VERSIONS else "no (too little parsed data)", auc, acc, rec])
        if tagged >= 10 and acc != "n/a":
            bloom.append((float(acc), prog))
    excluded = sorted(p for p, n in versions.items() if n < PQ.MIN_VERSIONS)
    return dump({"excluded_from_prerequisite_checks": excluded, "bloom_accuracy_min": min(bloom), "bloom_accuracy_max": max(bloom),
                 "table": {"columns": ["Programme", "Parsed course versions", "Prerequisite checks", "Overlap ROC-AUC", "Bloom accuracy", "Removal recall"],
                           "rows": rows}}, f"{OUT}/fairness.json")


def architecture_figure():
    """Where Group 9 sits in the AI University application, as used in the 31 October demo chain (results/week7/fig_architecture.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from curriculum_engine.api import ENDPOINTS
    boxes = {"data": (0.02, 0.62, "NMIMS semester\nsyllabus booklets\n(parsed, not public)"), "jds": (0.02, 0.22, "Job descriptions\n(JD1-JD5)"),
             "engine": (0.30, 0.38, "Group 9 Curriculum Intelligence Engine\nparser - families - embeddings\noverlap - redundancy - prerequisites\ndiff - Bloom - emerging - modernise"),
             "api": (0.30, 0.06, f"FastAPI service\n{len(ENDPOINTS)} endpoints, JSON + confidence"),
             "ui": (0.66, 0.06, "Committee UI (Streamlit)\nBoard of Studies reviews drafts"),
             "g11": (0.70, 0.70, "Group 11 Communications\nPOST /process-notice\n-> emails to faculty / students"),
             "g4": (0.70, 0.40, "Group 4 Knowledge Assistant\npassage table -> answers\nprerequisites, labs, credits")}
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for k, (x, y, t) in boxes.items():
        w, h = (0.34, 0.28) if k == "engine" else (0.24, 0.22) if k in ("g11", "g4") else (0.22, 0.2) if k in ("data", "jds") else (0.28, 0.16)
        col = "#DCE9F7" if k in ("engine", "api", "ui") else "#FCE9D8" if k in ("g11", "g4") else "#EEEEEE"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=col, ec="#555555"))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=7.5)
        boxes[k] = (x, y, w, h)
    def arrow(a, b, label, dy=0.0):
        xa, ya, wa, ha = boxes[a]; xb, yb, wb, hb = boxes[b]
        p1, p2 = (xa + wa, ya + ha / 2 + dy), (xb, yb + hb / 2)
        if b == "api":
            p1, p2 = (xa + wa / 2, ya), (xb + wb / 2, yb + hb)
        if a == "api":
            p1, p2 = (xa + wa, ya + ha / 2), (xb, yb + hb / 2)
        ax.annotate("", p2, p1, arrowprops=dict(arrowstyle="->", color="#333333"))
        ax.text((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 + 0.02, label, fontsize=6.5, ha="center", color="#333333")
    arrow("data", "engine", "courses"); arrow("jds", "engine", "skills"); arrow("engine", "api", "")
    arrow("api", "ui", "HTTP"); arrow("engine", "g11", "draft syllabus notice", 0.08); arrow("engine", "g4", "new-course passages", -0.02)
    ax.set_title("Group 9 in the AI University application (31 October demo chain: Group 9 -> Group 11, Group 9 -> Group 4)", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / OUT / "fig_architecture.png", dpi=170)
    plt.close(fig)


def run(ctx):
    from curriculum_engine.api_engine import ApiEngine
    eng = ApiEngine(recs=ctx.recs)
    print("  week7.demo", flush=True)
    summary, res = demo(ctx, eng)
    contracts(res)
    emerging_threshold(ctx, eng)
    coverage_items(res)
    coverage_metrics()
    fairness()
    architecture_figure()
    ctx.emb.save()
