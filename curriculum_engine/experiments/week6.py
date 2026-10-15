"""Week 6 - (1) prerequisite checker v2 vs v1 on the same split and the same reference labels (plus a fresh labelled sample),
(2) API examples generated from real calls (docs/examples/). Writes results/week6/*.json."""
import csv, json
from collections import Counter

import numpy as np

from curriculum_engine import LABELS, RESULTS, ROOT, SEED
from curriculum_engine import data as D
from curriculum_engine import evaluate as E
from curriculum_engine import prereq as PQ
from curriculum_engine.experiments.common import dump

OUT = "week6"
OWN_GRID = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, None)          # None = own-content filter off


def exact_key(prog, fam, phrase):
    return (prog, fam, D.norm_name(phrase))


def same_phrase(a, b):
    a, b = D.norm_name(a), D.norm_name(b)
    return bool(a) and bool(b) and (a in b or b in a)


class OwnChecker:
    """Does the course itself teach the phrase? (then it is course content, not a prerequisite)."""

    def __init__(self, emb, tau):
        from curriculum_engine.overlap import chunk_vectors
        self.emb, self.tau, self.cache, self.cv = emb, tau, {}, chunk_vectors

    def __call__(self, phrase, c):
        if self.tau is None:
            return False
        if c["id"] not in self.cache:
            self.cache[c["id"]] = self.cv(self.emb, c)
        U = self.cache[c["id"]]
        return bool(len(U)) and float((U @ self.emb([phrase])[0]).max()) >= self.tau


def checker(ctx, th, version, own_tau=None, single_alias=True):
    """f(prog, st, recs, families, log) -> flags, for v1 (Week 3) or v2."""
    chk = PQ.TopicChecker(ctx.reps, ctx.emb, th["prereq_topic"])
    own = OwnChecker(ctx.emb, own_tau)

    def run(prog, st, recs, families, log=None):
        res = PQ.EmbeddingResolver(recs, ctx.emb, th["prereq_resolve"])
        if version == 1:
            return PQ.check_programme(prog, ctx.view, st, res, ctx.versions, chk, families=families)
        names = {}
        for r in sorted(recs, key=lambda r: r["id"]):
            names.setdefault(r["family"], r["course_name"])
        return PQ.check_programme_v2(prog, ctx.view, st, res, ctx.versions, chk, own, names, families=families, log=log, single_alias=single_alias)
    return run


def tests(ctx, run, split):
    """Week 3 removal / intact / rename cases (same rule), with phrase matching that tolerates v2's cleaned phrases."""
    cases = PQ.removal_cases(ctx.recs, ctx.view, ctx.st, ctx.versions, ctx.families(split))
    hit = lambda fl, c: any(same_phrase(f["phrase"], c["phrase"]) for f in fl)
    intact, removed, renamed = [], [], []
    for c in cases:
        fam = {c["family"]}
        intact.append(hit(run(c["programme"], ctx.st, ctx.recs, fam), c))
        removed.append(hit(run(c["programme"], PQ.without_family(ctx.st, c["programme"], c["removed_family"]), ctx.recs, fam), c))
        recs2 = [dict(r, course_name="Principles of " + r["course_name"]) if r["family"] == c["removed_family"] else r for r in ctx.recs]
        renamed.append(hit(run(c["programme"], ctx.st, recs2, fam), c))
    g = [c["family"] for c in cases]
    return {"cases": len(cases), "removal_recall": E.mean_ci(removed, g), "intact_false_alarm": E.mean_ci(intact, g),
            "rename_false_alarm": E.mean_ci(renamed, g)}


def all_flags(ctx, run, split, log=None):
    fams = ctx.families(split)
    return [f for p in sorted(ctx.view) for f in run(p, ctx.st, ctx.recs, fams, log)]


def tune_own(ctx, th):
    """Own-content threshold on validation cases only. The label-free objective (recall - mean false alarm) cannot reward this
    filter, so the rule is: the most aggressive threshold (lowest) that keeps the validation objective at its maximum.
    (First attempt, v2a: ties went to 'off'; revised after that run, disclosed in the report.)"""
    rows = []
    for tau in OWN_GRID:
        r = tests(ctx, checker(ctx, th, 2, tau), "val")
        rows.append({"own_threshold": tau, "objective": E.r3(r["removal_recall"]["value"] - (r["intact_false_alarm"]["value"] + r["rename_false_alarm"]["value"]) / 2),
                     **{k: r[k]["value"] for k in ("removal_recall", "intact_false_alarm", "rename_false_alarm")}})
    top = max(r["objective"] for r in rows)
    numeric = [r["own_threshold"] for r in rows if r["objective"] == top and r["own_threshold"] is not None]
    return (min(numeric) if numeric else None), rows


def same_label_eval(v2_flags):
    """The 20 reference-labelled Week 3 core flags: which ones v2 still raises, and the precision among those."""
    rows = [r for r in csv.DictReader(open(LABELS / "flags_llm.csv", encoding="utf-8")) if r["kind"] == "prerequisite" and r["method"] == "core"]
    out = []
    for r in rows:
        prog, fam, phrase = r["key"].split("|", 2)
        kept = any(f["programme"] == prog and f["family"] == fam and same_phrase(f["phrase"], phrase) for f in v2_flags)
        out.append({"key": r["key"], "label": r["consensus"], "still_flagged_by_v2": kept})
    kept = [o for o in out if o["still_flagged_by_v2"]]
    labelled = {(o["key"].split("|", 2)[0], o["key"].split("|", 2)[1]) for o in out}
    return {"labelled_flags": len(out), "v1_precision": E.r3(np.mean([o["label"] == "correct" for o in out])),
            "v2_still_flagged": len(kept), "v2_precision_on_those": E.r3(np.mean([o["label"] == "correct" for o in kept])) if kept else None,
            "correct_flags_dropped_by_v2": sum(o["label"] == "correct" and not o["still_flagged_by_v2"] for o in out),
            "wrong_flags_dropped_by_v2": sum(o["label"] != "correct" and not o["still_flagged_by_v2"] for o in out),
            "items": out, "labelled_courses": len(labelled)}


def fresh_sample(ctx, th, v1_test, v2_test, v1_val, v2_val):
    """New items for the same labelling protocol: v2 test flags not in the labelled sample, and a validation sample of v1 and v2."""
    from curriculum_engine.experiments import week3 as W3
    from curriculum_engine.labelling import ANN, write_items
    labelled = {exact_key(*r["key"].split("|", 2)): r["key"] for r in csv.DictReader(open(LABELS / "flags_llm.csv", encoding="utf-8"))
                if r["kind"] == "prerequisite" and r["status"] != "uncertain"}
    rng = np.random.default_rng(SEED)
    pick = lambda fl, k: [fl[i] for i in sorted(rng.permutation(len(fl))[:k])]
    groups = [("test", "v2", v2_test), ("val", "v1", pick(v1_val, 20)), ("val", "v2", pick(v2_val, 20))]
    chk = PQ.TopicChecker(ctx.reps, ctx.emb, th["prereq_topic"])
    names = {}
    for r in ctx.recs:
        names.setdefault(r["family"], r["course_name"])
    rows, items, seen = [], [], {}
    for split, method, fl in groups:
        for f in fl:
            key = f"{f['programme']}|{f['family']}|{f['phrase']}"
            if split == "test" and exact_key(f["programme"], f["family"], f["phrase"]) in labelled:   # identical flag: reuse its label
                rows.append({"item": "L:" + labelled[exact_key(f["programme"], f["family"], f["phrase"])], "split": split, "method": method, "key": key})
                continue
            if key in seen:                                # the same flag raised by v1 and v2 is labelled once
                rows.append({**rows[seen[key]], "method": method, "item": rows[seen[key]]["item"]})
                continue
            seen[key] = len(rows)
            rows.append({"item": f"PV{len(items) + 1:03d}", "split": split, "method": method, "key": key})
            full, compact = W3.prereq_item(ctx, f, chk, names)
            items.append({"item": rows[-1]["item"], "full": full, "compact": compact})
    uniq = {r["item"]: r for r in rows if not r["item"].startswith("L:")}
    with open(ANN / "prereqv2_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["item", "split", "method", "key"])
        w.writeheader(); w.writerows(uniq.values())
    with open(LABELS / "prereqv2_membership.csv", "w", newline="", encoding="utf-8") as fh:   # which method raised which labelled item
        w = csv.DictWriter(fh, fieldnames=["item", "split", "method", "key"])
        w.writeheader(); w.writerows(rows)
    write_items("prereqv2", items, parts=1)
    return rows


def fresh_eval():
    """Precision on the fresh sample (LLM-annotated reference labels, same protocol as the Week 3 flags)."""
    f, m = LABELS / "prereqv2_llm.csv", LABELS / "prereqv2_membership.csv"
    if not f.exists():
        return {"note": "fresh sample not labelled yet"}
    lab = {r["item"]: r for r in csv.DictReader(open(f, encoding="utf-8"))}
    lab |= {"L:" + r["key"]: r for r in csv.DictReader(open(LABELS / "flags_llm.csv", encoding="utf-8")) if r["kind"] == "prerequisite"}
    res = {}
    for r in csv.DictReader(open(m, encoding="utf-8")):
        l = lab[r["item"]]
        if l["status"] != "uncertain":
            res.setdefault(f"{r['split']}_{r['method']}", []).append((l["consensus"] == "correct", r["key"].split("|")[1]))
    return {k: (lambda m: {"n": m["n"], "precision": m["value"], "ci95": m["ci95"]})(E.mean_ci([a for a, _ in v], [g for _, g in v]))
            for k, v in sorted(res.items())}


def fresh_agreement():
    """Pass 1 vs pass 2 on the fresh sample (both Claude, different wording: not independent) and how many items needed pass 3."""
    from curriculum_engine.labelling import agreement
    f = LABELS / "prereqv2_llm.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    for r in rows:
        for k in ("pass1", "pass2", "pass3"):
            r[k] = None if r[k] in ("", "None") else r[k]
    a = agreement(rows)
    return {"items": a["n_items"], "both_passes_agree": sum(r["pass1"] == r["pass2"] for r in rows if r["pass1"] is not None),
            "percent_agreement": a["percent_agreement"], "needed_pass3": sum(r["pass1"] != r["pass2"] for r in rows)}


def prerequisites_v2(ctx):
    th = json.loads((RESULTS / "week3/thresholds.json").read_text(encoding="utf-8"))
    own_tau, grid = tune_own(ctx, th)
    v1, v2 = checker(ctx, th, 1), checker(ctx, th, 2, own_tau)
    v2a = checker(ctx, th, 2, None, single_alias=False)               # first attempt (pooled alias words, own-content filter off)
    log = {}
    v1_test, v2_test = all_flags(ctx, v1, "test"), all_flags(ctx, v2, "test", log)
    v2a_test = all_flags(ctx, v2a, "test")
    v1_val, v2_val = all_flags(ctx, v1, "val"), all_flags(ctx, v2, "val")
    if not (LABELS / "prereqv2_membership.csv").exists():
        fresh_sample(ctx, th, v1_test, v2_test, v1_val, v2_val)
    res = {"own_content_threshold_from_val": own_tau, "tuning_on_val": grid,
           "filters_on_test_phrases": log, "flags_test": {"v1": len(v1_test), "v2": len(v2_test)},
           "flags_by_type_test": {"v1": dict(Counter(f["type"] for f in v1_test)), "v2": dict(Counter(f["type"] for f in v2_test))},
           "tests_test": {"v1": tests(ctx, v1, "test"), "v2": tests(ctx, v2, "test")},
           "same_labels": same_label_eval(v2_test), "fresh_sample": fresh_eval(), "fresh_sample_agreement": fresh_agreement(),
           "v2a_first_attempt": {"flags_test": len(v2a_test), "tests_test": tests(ctx, v2a, "test"), "same_labels": same_label_eval(v2a_test),
                                 "why_revised": "pooling words of all earlier course names met too many removed prerequisites (removal recall fell)"},
           "bias_note": "v2's rules were designed from test-split failure cases (F05-F07), so v2 precision on the same test labels is optimistic"}
    dump(v2_test, f"{OUT}/prerequisite_flags_v2.json")
    return dump(res, f"{OUT}/prerequisites_v2.json")


def fusion_reference(ctx):
    """Validation-pair scores: the fusion turns a pair's TF-IDF and MaxSim into percentiles of these (Week 4 selection)."""
    from curriculum_engine.experiments import week2 as W2, week4 as W4
    val = ctx.pair_labels("val")
    return dump({"pairs": len(val), "tfidf_cosine": sorted(round(float(x), 6) for x in W2.score_pairs(ctx, val, ctx.tfidf.cosine)),
                 "maxsim_minilm": sorted(round(float(x), 6) for x in W2.score_pairs(ctx, val, W4.soft_with(ctx.emb)))}, f"{OUT}/fusion_reference.json")


def bloom_confidence(ctx):
    """Confidence of a Bloom prediction = precision of that predicted level against faculty tags on the test split."""
    from curriculum_engine import bloom as B
    t = B.tagged_outcomes(ctx.recs, ctx.split)["test"]
    by = {}
    for x, l, _ in t:
        p = B.lexicon_level(x) or 3
        by.setdefault(B.BLOOM[p], []).append(p == l)
    return dump({**{k: E.r3(np.mean(v)) for k, v in sorted(by.items())}, "_n": {k: len(v) for k, v in sorted(by.items())}}, f"{OUT}/bloom_confidence.json")


def api_examples(ctx):
    """One real request/response per endpoint (lists shortened) -> docs/examples/, and the OpenAPI schema -> docs/openapi.json."""
    from fastapi.testclient import TestClient
    from curriculum_engine.api import app
    ex = json.loads((RESULTS / "week3/examples.json").read_text(encoding="utf-8"))["examples"][0]
    flagged = json.loads((RESULTS / OUT / "prerequisite_flags_v2.json").read_text(encoding="utf-8"))[0]
    pre_course = next(r for r in sorted(ctx.recs, key=lambda r: r["id"]) if r["family"] == flagged["family"] and flagged["programme"] in r["programmes"])
    calls = {
        "compare_courses": ("/compare-curricula", {"course_a": {"id": ex["course_a"]["id"]}, "course_b": {"id": ex["course_b"]["id"]}}),
        "compare_programmes": ("/compare-curricula", {"programme_a": {"programme": "COMP", "academic_year": "2024-25"},
                                                      "programme_b": {"programme": "COMP", "academic_year": "2025-26"}}),
        "map_outcomes": ("/map-outcomes", {"outcomes": ["Design a relational database schema for a given problem", "Explain the layers of the OSI model",
                                                        "Construct the curves using surveying instruments"]}),
        "find_overlaps": ("/find-overlaps", {"course": {"id": ex["course_a"]["id"]}, "k": 5}),
        "check_prerequisites": ("/check-prerequisites", {"course": {"id": pre_course["id"]}}),
        "detect_redundancy": ("/detect-redundancy", {"programme": "COMP", "k": 5}),
        "emerging_topics": ("/emerging-topics", {"programme": "COMP"}),
    }
    out_dir = ROOT / "docs/examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = TestClient(app)
    shorten = lambda v: [shorten(x) for x in v[:5]] if isinstance(v, list) else {k: shorten(x) for k, x in v.items()} if isinstance(v, dict) else v
    status = {}
    for name, (path, body) in calls.items():
        r = client.post(path, json=body)
        status[name] = r.status_code
        (out_dir / f"{name}.json").write_text(json.dumps({"request": {"method": "POST", "path": path, "body": body},
                                                          "status": r.status_code, "response": shorten(r.json())}, indent=1, ensure_ascii=False), encoding="utf-8")
    h = client.get("/health")
    status["health"] = h.status_code
    (out_dir / "health.json").write_text(json.dumps({"request": {"method": "GET", "path": "/health"}, "status": h.status_code, "response": h.json()},
                                                    indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "docs/openapi.json").write_text(json.dumps(app.openapi(), indent=1), encoding="utf-8")
    cc = json.loads((out_dir / "compare_courses.json").read_text(encoding="utf-8"))["response"]
    excerpt = {"request": calls["compare_courses"][1], "response": {k: cc[k] for k in ("course_a", "course_b", "overlap")} |
               {"common_topics": cc["common_topics"][:3], "unique_to_a": cc["unique_to_a"][:3], "prerequisites_a": cc["prerequisites_a"],
                "meta": {k: cc["meta"][k] for k in ("engine_version", "encoder", "split_hash")}}}
    from curriculum_engine import __version__
    paths = {("POST", path) for path, _ in calls.values()} | {("GET", "/health")}       # /compare-curricula has two examples
    return dump({"status_codes": status, "all_ok": all(v == 200 for v in status.values()), "endpoints": len(paths), "example_calls": len(status),
                 "engine_version": __version__,
                 "compare_courses_excerpt": excerpt}, f"{OUT}/api_examples.json")


def run(ctx):
    print("  week6.prerequisites_v2", flush=True)
    prerequisites_v2(ctx)
    fusion_reference(ctx)
    bloom_confidence(ctx)
    print("  week6.api_examples", flush=True)
    api_examples(ctx)
    ctx.emb.save()
