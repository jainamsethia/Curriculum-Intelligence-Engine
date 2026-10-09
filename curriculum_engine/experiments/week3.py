"""Week 3 - the core semantic model on the same split, metrics and synthetic cases as the Week 2 baselines.
Thresholds are tuned on validation only and saved to results/week3/thresholds.json (the engine reads them)."""
import json
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from curriculum_engine import RESULTS
from curriculum_engine import diff as DF
from curriculum_engine import evaluate as E
from curriculum_engine import prereq as PQ
from curriculum_engine import redundancy as RD
from curriculum_engine.data import chunks
from curriculum_engine.experiments import week2 as W2
from curriculum_engine.experiments.common import dump
from curriculum_engine.overlap import chunk_vectors, maxsim

OUT = "week3"


def vecs(ctx, c):
    return chunk_vectors(ctx.emb, c)


def soft(ctx):
    return lambda a, b: maxsim(vecs(ctx, a), vecs(ctx, b), 1.0)["soft"]


def coverage(ctx, tau):
    return lambda a, b: maxsim(vecs(ctx, a), vecs(ctx, b), tau)["overlap"]


def max_unit(ctx):
    def f(a, b):
        A, B = vecs(ctx, a), vecs(ctx, b)
        return float((A @ B.T).max()) if len(A) and len(B) else 0.0
    return f


def unit_matrix(ctx):
    f = max_unit(ctx)
    return lambda cs: np.array([[f(a, b) for b in cs] for a in cs])


# ------------------------------------------------------------------ thresholds (validation only)
def tune(ctx):
    val = ctx.pair_labels("val")
    yv = np.array([p["y"] for p in val])
    sv = W2.score_pairs(ctx, val, soft(ctx))
    grid = [round(float(x), 2) for x in np.arange(0.5, 0.91, 0.05)]
    rho = {t: spearmanr(yv, W2.score_pairs(ctx, val, coverage(ctx, t)))[0] for t in grid}
    th = {"unit_match": max(grid, key=lambda t: rho[t]),
          "any_overlap": E.best_threshold(yv >= 1, sv), "substantial_overlap": E.best_threshold(yv == 2, sv),
          "unit_duplicate": E.best_threshold(yv >= 1, W2.score_pairs(ctx, val, max_unit(ctx)))}
    # prerequisite thresholds: best (removal recall - mean false alarm) on the validation removal / rename cases (label-free)
    best = None
    for tr in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        for tc in (0.4, 0.45, 0.5, 0.55, 0.6):
            chk = PQ.TopicChecker(ctx.reps, ctx.emb, tc)
            r = W2.prereq_tests(ctx, lambda recs, tr=tr: PQ.EmbeddingResolver(recs, ctx.emb, tr), lambda c, st, chk=chk: chk, split="val")
            obj = r["removal_recall"]["value"] - (r["intact_false_alarm"]["value"] + r["rename_false_alarm"]["value"]) / 2
            if best is None or obj > best[0]:
                best = (obj, tr, tc, r)
    th |= {"prereq_resolve": best[1], "prereq_topic": best[2]}
    dump({"thresholds": th, "unit_match_spearman_on_val": {str(t): E.r3(v) for t, v in rho.items()},
          "prereq_val_tests": best[3]}, f"{OUT}/tuning.json")
    return dump(th, f"{OUT}/thresholds.json")


# ------------------------------------------------------------------ capability 1
def overlap(ctx, th):
    val, test = ctx.pair_labels("val"), ctx.pair_labels("test")
    yv, yt = np.array([p["y"] for p in val]), np.array([p["y"] for p in test])
    gt = [p["family_a"] for p in test]
    res = {"n_val": len(val), "n_test": len(test), "methods": {}}
    for name, fn in {"tfidf_cosine": ctx.tfidf.cosine, "maxsim_soft": soft(ctx), "maxsim_coverage": coverage(ctx, th["unit_match"])}.items():
        sv, st = W2.score_pairs(ctx, val, fn), W2.score_pairs(ctx, test, fn)
        res["methods"][name] = {"test": E.ranking(yt, st, gt), "at_val_threshold": E.prf(yt >= 1, st, E.best_threshold(yv >= 1, sv), gt)}
    vp = []
    for f in sorted(ctx.families("test")):
        vs = sorted([r for r in ctx.recs if r["family"] == f], key=lambda r: r["id"])[:4]
        vp += [(a, b) for i, a in enumerate(vs) for b in vs[i + 1:]]
    s = soft(ctx)
    vs_scores = np.array([s(a, b) for a, b in vp])
    st = W2.score_pairs(ctx, test, s)
    res["version_pairs"] = {"n": len(vp), "maxsim_median": E.r3(np.median(vs_scores)),
                            "cross_course_median_by_label": {str(k): E.r3(np.median(st[yt == k])) if (yt == k).any() else None for k in (0, 1, 2)}}
    names = {"tfidf_cosine": "TF-IDF cosine (Week 2 baseline)", "maxsim_soft": "Chunk MaxSim, soft (core)", "maxsim_coverage": "Chunk MaxSim, unit coverage F"}
    res["table"] = {"columns": ["Method", "Spearman [95% CI]", "ROC-AUC [95% CI]", "Avg. precision", "P / R / F1 at val threshold"],
                    "rows": [[names[m], W2._ci(v["test"]["spearman"]), W2._ci(v["test"]["roc_auc"]), W2._v(v["test"]["avg_precision"]["value"]),
                              f"{W2._v(v['at_val_threshold']['precision'])} / {W2._v(v['at_val_threshold']['recall'])} / {W2._v(v['at_val_threshold']['f1'])}"]
                             for m, v in res["methods"].items()]}
    return dump(res, f"{OUT}/overlap.json")


# ------------------------------------------------------------------ capability 2
def unit_flags(ctx, fams, thr):
    flags = []
    for p in sorted(ctx.view):
        cs = RD.programme_courses(ctx.view, p, fams)
        V = [vecs(ctx, c) for c in cs]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if cs[i]["family"] == cs[j]["family"] or not len(V[i]) or not len(V[j]):
                    continue
                M = V[i] @ V[j].T
                a, b = np.unravel_index(M.argmax(), M.shape)
                if M[a, b] >= thr:
                    flags.append({"programme": p, "course_a": cs[i]["course_name"], "unit_a": chunks(cs[i])[a][0],
                                  "course_b": cs[j]["course_name"], "unit_b": chunks(cs[j])[b][0], "similarity": round(float(M[a, b]), 3),
                                  "id_a": cs[i]["id"], "id_b": cs[j]["id"], "ua": int(a), "ub": int(b)})
    return sorted(flags, key=lambda f: (-f["similarity"], f["programme"], f["id_a"], f["id_b"]))


def redundancy(ctx, th):
    fams = ctx.families("test")
    flags = unit_flags(ctx, fams, th["unit_duplicate"])
    res = {"method": "unit-level near-duplicates (MiniLM MaxSim) within a programme", "threshold_from_val": E.r3(th["unit_duplicate"]),
           "flags_test": len(flags), "flags_by_programme": dict(Counter(f["programme"] for f in flags)),
           "top_flags": [{k: f[k] for k in ("programme", "course_a", "unit_a", "course_b", "unit_b", "similarity")} for f in flags[:8]],
           "synthetic_injection": W2.redundancy_eval(ctx, fams, unit_matrix(ctx), th["unit_duplicate"])}
    dump(flags, f"{OUT}/redundancy_flags.json")
    return dump(res, f"{OUT}/redundancy.json")


# ------------------------------------------------------------------ capability 3
def prerequisites(ctx, th):
    fams = ctx.families("test")
    chk = PQ.TopicChecker(ctx.reps, ctx.emb, th["prereq_topic"])
    resolver = PQ.EmbeddingResolver(ctx.recs, ctx.emb, th["prereq_resolve"])
    flags = [f for p in sorted(ctx.view) for f in PQ.check_programme(p, ctx.view, ctx.st, resolver, ctx.versions, chk, families=fams)]
    w2 = json.loads((RESULTS / "week2/prerequisites.json").read_text(encoding="utf-8"))
    res = {"method": "embedding resolution + topic check over earlier semesters", "thresholds": {k: th[k] for k in ("prereq_resolve", "prereq_topic")},
           "phrases_checked": w2["phrases_checked"], "flags": len(flags), "flags_by_type": dict(Counter(f["type"] for f in flags)),
           "flag_rate_per_phrase": E.r3(len(flags) / max(1, w2["phrases_checked"])),
           "examples": [{k: f[k] for k in ("programme", "course", "semester", "phrase", "type")} for f in flags[:6]],
           "tests": W2.prereq_tests(ctx, lambda recs: PQ.EmbeddingResolver(recs, ctx.emb, th["prereq_resolve"]), lambda c, st: chk)}
    dump(flags, f"{OUT}/prerequisite_flags.json")
    return dump(res, f"{OUT}/prerequisites.json")


# ------------------------------------------------------------------ capability 4
def diff(ctx, th):
    s = soft(ctx)

    def sim(a, b):
        v = s(a, b)
        return v if v >= th["substantial_overlap"] else None

    method = lambda A, B: DF.diff_by_content(A, B, sim)
    summ, per = W2.diff_eval(ctx, method)
    ren, _ = W2.diff_eval(ctx, method, rename=True)
    return dump({"method": "name match, then content match (MaxSim) for renamed courses", **summ, "rename_test": ren, "per_year_pair": per},
                f"{OUT}/diff.json")


# ------------------------------------------------------------------ capability 7 + examples
def topic_model(ctx):
    from curriculum_engine.topics import TopicModel
    tm = TopicModel(ctx.emb).fit([c for f, c in sorted(ctx.reps.items()) if ctx.split[f] == "train"])
    return dump({"k": tm.k, "fitted_on": "units of training courses", "mean_coherence": E.r3(np.mean([t["coherence"] for t in tm.info])),
                 "topics": tm.info}, f"{OUT}/topics.json")


def examples(ctx, th):
    from curriculum_engine.engine import Engine
    eng = Engine(th, ctx.recs)
    test = sorted(ctx.pair_labels("test"), key=lambda p: (-p["y"], p["item"]))
    pick = [p for p in test if p["y"] == 2][:1] + [p for p in test if p["y"] == 1][:1]
    out = [eng.compare(p["a"], p["b"], k=6) for p in pick]
    ctx.emb.save()
    ex = out[0] if out else {}                  # the substantially overlapping pair
    return dump({"examples": out, "excerpt": {k: ex[k] for k in ("course_a", "course_b", "overlap", "common_topics", "unique_to_a", "unique_to_b") if k in ex}},
                f"{OUT}/examples.json")


def summary(ctx):
    j = {k: json.loads((RESULTS / OUT / f"{k}.json").read_text(encoding="utf-8")) for k in ("overlap", "redundancy", "prerequisites", "diff")}
    j["bloom"] = json.loads((RESULTS / "week2/bloom.json").read_text(encoding="utf-8"))
    t = W2.summary_rows(j["overlap"], j["redundancy"], j["prerequisites"], j["diff"], j["bloom"],
                        {"overlap": "maxsim_soft", "overlap_name": "Chunk MaxSim (MiniLM)", "bloom": "lexicon"})
    w2 = json.loads((RESULTS / "week2/summary.json").read_text(encoding="utf-8"))["table"]
    prog = {"columns": ["Capability", "Metric (test split)", "Week 2 baseline", "Week 3 core"],
            "rows": [[a[0], a[2], a[3], b[3]] for a, b in zip(w2["rows"], t["rows"])]}
    fp = json.loads((RESULTS / OUT / "flag_precision.json").read_text(encoding="utf-8"))
    for kind, label in (("redundancy", "2 Redundancy"), ("prerequisite", "3 Prerequisites")):
        if kind in fp:
            prog["rows"].append([label, f"precision of flags (reference labels, sample of {K_FLAGS})",
                                 W2._v(fp[kind].get("baseline", {}).get("precision")), W2._v(fp[kind].get("core", {}).get("precision"))])
    for kind, label, metric in (("emerging_gap", "6 Emerging topics", "precision of flagged gaps"), ("topic_label", "7 Topic labels", "share of labels accepted")):
        if kind in fp:
            prog["rows"].append([label, metric, "-", W2._v(fp[kind]["core"]["precision"])])
    return dump({"table": t, "progress": prog}, f"{OUT}/summary.json")


def emerging(ctx, th):
    from curriculum_engine import emerging as EM
    cov = EM.coverage(ctx.view, ctx.emb, th["prereq_topic"])
    by = {}
    for g in cov:
        by.setdefault(g["programme"], {"checked": 0, "gaps": 0})
        by[g["programme"]]["checked"] += 1
        by[g["programme"]]["gaps"] += not g["covered"]
    return dump({"status": EM.REFERENCE["status"], "threshold": E.r3(th["prereq_topic"]), "verified_topics": len(EM.topics()),
                 "unverified_topics": [t["topic"] for t in EM.topics(False) if not t["verified"]],
                 "pairs_checked": len(cov), "gaps": sum(not g["covered"] for g in cov), "by_programme": by,
                 "table": {"columns": ["Programme", "Topics checked", "Gaps flagged"], "rows": [[p, v["checked"], v["gaps"]] for p, v in sorted(by.items())]},
                 "gaps_list": [{k: g[k] for k in ("topic", "programme", "best_similarity")} for g in cov if not g["covered"]]}, f"{OUT}/emerging.json")


def run(ctx):
    print("  week3.tune", flush=True)
    th = tune(ctx)
    for fn in (overlap, redundancy, prerequisites, diff):
        print(f"  week3.{fn.__name__}", flush=True)
        fn(ctx, th)
    print("  week3.topics", flush=True)
    topic_model(ctx)
    emerging(ctx, th)
    print("  week3.examples", flush=True)
    examples(ctx, th)
    build_flag_items(ctx, th)
    flag_metrics(ctx)
    summary(ctx)
    ctx.emb.save()


# ------------------------------------------------------------------ flags for LLM-annotated reference labels (precision)
K_FLAGS = 20


def baseline_redundancy_top(ctx, fams, k=K_FLAGS):
    out = []
    for p in sorted(ctx.view):
        cs = RD.programme_courses(ctx.view, p, fams)
        if len(cs) >= 2:
            out += [(s, p, cs[i], cs[j]) for i, j, s in RD.course_pairs(ctx.tfidf.matrix(cs), cs, -1.0)]
    out.sort(key=lambda x: (-x[0], x[1], x[2]["id"], x[3]["id"]))
    return out[:k]


def _units(c, n=6, words=18):
    from curriculum_engine.labelling import _cut
    return "; ".join(f"{t}: {_cut(x, words)}" for t, x in chunks(c)[:n])


def build_flag_items(ctx, th):
    """~150 flags from baseline and core detectors (test split) + topic labels, rendered without any score."""
    from curriculum_engine import emerging as EM
    from curriculum_engine.labelling import _cut, write_items
    fams, rng = ctx.families("test"), np.random.default_rng(7)
    rows, items = [], []

    def add(kind, method, key, full, compact):
        rows.append({"item": f"FL{len(rows) + 1:03d}", "kind": kind, "method": method, "key": key})
        items.append({"item": rows[-1]["item"], "full": full, "compact": compact})

    q_red = "QUESTION: Is the warning correct, i.e. do these two courses of one programme teach the same content twice (redundant content)?"
    for s, p, a, b in baseline_redundancy_top(ctx, fams):
        txt = f"WARNING (redundancy): two courses of programme {p} may repeat content.\nCOURSE A: {a['course_name']}\n  units: {_units(a)}\nCOURSE B: {b['course_name']}\n  units: {_units(b)}\n{q_red}"
        add("redundancy", "baseline", f"{a['id']}|{b['id']}", txt, txt)
    for f in json.loads((RESULTS / OUT / "redundancy_flags.json").read_text(encoding="utf-8"))[:K_FLAGS]:
        a, b = ctx.by_id[f["id_a"]], ctx.by_id[f["id_b"]]
        ua, ub = chunks(a)[f["ua"]], chunks(b)[f["ub"]]
        txt = (f"WARNING (redundancy): a unit of one course seems to be taught again in another course of programme {f['programme']}.\n"
               f"COURSE A: {a['course_name']}\n  unit: {ua[0]}: {_cut(ua[1], 60)}\nCOURSE B: {b['course_name']}\n  unit: {ub[0]}: {_cut(ub[1], 60)}\n"
               "QUESTION: Is the warning correct, i.e. do these two units teach substantially the same content?")
        add("redundancy", "core", f"{a['id']}|{b['id']}", txt, txt)

    names = {}
    for r in ctx.recs:
        names.setdefault(r["family"], r["course_name"])
    q_pre = "QUESTION: Is the warning correct, i.e. is this prerequisite really NOT covered by any course taught in an earlier semester of the programme?"
    base = [f for p in sorted(ctx.view) for f in PQ.check_programme(p, ctx.view, ctx.st, PQ.ExactResolver(ctx.recs), ctx.versions, families=fams)]
    core = json.loads((RESULTS / OUT / "prerequisite_flags.json").read_text(encoding="utf-8"))
    chk = PQ.TopicChecker(ctx.reps, ctx.emb, th["prereq_topic"])
    for method, fl in (("baseline", base), ("core", core)):
        for i in sorted(rng.permutation(len(fl))[:K_FLAGS]):
            f = fl[i]
            before = sorted({names[x] for x in PQ.earlier(ctx.st, f["programme"], f["semester"])})
            q = ctx.emb([f["phrase"]])[0]
            near = sorted(((float((chk.U[x] @ q).max()), names[x]) for x in PQ.earlier(ctx.st, f["programme"], f["semester"]) if x in chk.U and len(chk.U[x])), reverse=True)[:5]
            full = (f"WARNING (missing prerequisite): course '{f['course']}' (programme {f['programme']}, semester {f['semester']}) lists the prerequisite "
                    f"'{f['phrase']}', which the tool could not find among the courses of earlier semesters.\n"
                    f"Courses taught in earlier semesters of this programme: {'; '.join(before)}\n{q_pre}")
            compact = (f"WARNING (missing prerequisite): course '{f['course']}' (semester {f['semester']}) needs '{f['phrase']}'.\n"
                       f"Most related earlier courses: {'; '.join(n for _, n in near)}\n{q_pre}")
            add("prerequisite", method, f"{f['programme']}|{f['family']}|{f['phrase']}", full, compact)

    # every topic-programme decision is labelled (gap or covered), so both gap precision and missed gaps can be measured
    for g in EM.coverage(ctx.view, ctx.emb, th["prereq_topic"]):
        ev = "; ".join(f"{e['course']} - {e['unit']}" for e in g["evidence"])
        claim = (f"CLAIM: programme {g['programme']} does not teach '{g['topic']}' (an emerging-topic gap)." if not g["covered"] else
                 f"CLAIM: programme {g['programme']} already teaches '{g['topic']}'.")
        txt = (f"{claim}\nThe programme's units closest to this topic: {ev}\n"
               "QUESTION: Is the claim correct, judging from these closest units?")
        add("emerging_gap" if not g["covered"] else "emerging_covered", "core", f"{g['topic']}|{g['programme']}", txt, txt)

    for t in json.loads((RESULTS / OUT / "topics.json").read_text(encoding="utf-8"))["topics"]:
        txt = (f"TOPIC LABEL produced by a topic model: '{t['name']}' (top terms: {', '.join(t['terms'])}).\n"
               f"Example units in this topic: {'; '.join(t['examples'])}\n"
               "QUESTION: Is the label correct, i.e. do the example units form one coherent topic that the label describes?")
        add("topic_label", "core", str(t["topic"]), txt, txt)

    from curriculum_engine.labelling import ANN
    import csv
    ANN.mkdir(parents=True, exist_ok=True)
    with open(ANN / "flags_sample.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    write_items("flags", items, parts=2)
    return rows


def flag_metrics(ctx):
    """Precision of each detector's flags against the LLM-annotated reference labels (uncertain items excluded)."""
    import csv
    from curriculum_engine import LABELS
    f = LABELS / "flags_llm.csv"
    if not f.exists():
        return dump({"note": "flag labels not available yet"}, f"{OUT}/flag_precision.json")
    rows = [r for r in csv.DictReader(open(f, encoding="utf-8"))]
    res = {}
    for kind in ("redundancy", "prerequisite", "emerging_gap", "emerging_covered", "topic_label"):
        for method in ("baseline", "core"):
            rs = [r for r in rows if r["kind"] == kind and r["method"] == method]
            ok = [r["consensus"] == "correct" for r in rs if r["status"] != "uncertain"]
            if rs:
                res.setdefault(kind, {})[method] = {"flags_labelled": len(rs), "uncertain": sum(r["status"] == "uncertain" for r in rs),
                                                    "precision": E.r3(np.mean(ok)) if ok else None}
    # missed gaps: 'covered' claims judged wrong are real gaps the detector did not flag
    sure = lambda k, v: sum(1 for r in rows if r["kind"] == k and r["status"] != "uncertain" and r["consensus"] == v)
    real_gaps = sure("emerging_gap", "correct") + sure("emerging_covered", "wrong")
    res["emerging_gap_recall"] = E.r3(sure("emerging_gap", "correct") / real_gaps) if real_gaps else None
    res["emerging_real_gaps"] = real_gaps
    from curriculum_engine.labelling import agreement
    for r in rows:
        for k in ("pass1", "pass2", "pass3"):
            r[k] = None if r[k] in ("", "None") else r[k]
    res["agreement_pass1_vs_pass2"] = agreement(rows)
    return dump(res, f"{OUT}/flag_precision.json")
