"""Week 2 - dataset statistics, EDA and the baselines of capabilities 1-5 on the test split.
Writes results/week2/{eda,labels,overlap,redundancy,prerequisites,diff,bloom}.json and fig_eda.png."""
import csv, re
from collections import Counter

import numpy as np

from curriculum_engine import LABELS, RESULTS
from curriculum_engine import bloom as B
from curriculum_engine import data as D
from curriculum_engine import diff as DF
from curriculum_engine import evaluate as E
from curriculum_engine import prereq as PQ
from curriculum_engine import redundancy as RD
from curriculum_engine.experiments.common import dump
from curriculum_engine.labelling import agreement
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

OUT = "week2"


# ------------------------------------------------------------------ EDA
def eda(ctx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = ctx.recs
    progs = [p for p, _ in ctx.versions.most_common() if ctx.versions[p] >= 50]
    ays = sorted({r["academic_year"] for r in recs})
    M = np.array([[sum(r["academic_year"] == a and p in r["programmes"] for r in recs) for a in ays] for p in progs])
    words = np.array([len(D.course_text(r).split()) for r in recs])
    has_pre = [bool(PQ.split_prereq(r["prerequisites"])) for r in recs]
    per_prog = {p: {"versions": int(ctx.versions[p]),
                    "with_prerequisite": round(float(np.mean([h for r, h in zip(recs, has_pre) if p in r["programmes"]])), 3),
                    "with_outcomes": round(float(np.mean([bool(r["outcomes"]) for r in recs if p in r["programmes"]])), 3),
                    "units_per_course": round(float(np.mean([len(r["units"]) for r in recs if p in r["programmes"]])), 1)} for p in progs}
    fam = Counter(r["family"] for r in recs)
    res = {"programmes_shown": progs, "academic_years": ays, "versions_by_programme_and_year": M.tolist(),
           "words_per_course": {"median": int(np.median(words)), "p10": int(np.percentile(words, 10)), "p90": int(np.percentile(words, 90))},
           "units_per_course_median": float(np.median([len(r["units"]) for r in recs])),
           "outcomes_per_course_median": float(np.median([len(r["outcomes"]) for r in recs])),
           "with_prerequisite_phrase": int(sum(has_pre)), "with_prerequisite_phrase_share": round(float(np.mean(has_pre)), 3),
           "outcomes_with_faculty_tag_share": round(sum(bool(o["k_levels"]) for r in recs for o in r["outcomes"]) / sum(len(r["outcomes"]) for r in recs), 3),
           "versions_per_family": {"1": sum(v == 1 for v in fam.values()), "2-3": sum(2 <= v <= 3 for v in fam.values()),
                                   "4-9": sum(4 <= v <= 9 for v in fam.values()), "10+": sum(v >= 10 for v in fam.values())},
           "per_programme": {"columns": ["Programme", "Versions", "With prerequisite", "With outcomes", "Units / course"],
                             "rows": [[p, v["versions"], f"{v['with_prerequisite']:.0%}", f"{v['with_outcomes']:.0%}", v["units_per_course"]]
                                      for p, v in per_prog.items()]}}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    im = ax[0].imshow(M, cmap="Blues", aspect="auto")
    ax[0].set_xticks(range(len(ays)), ays, fontsize=7, rotation=30); ax[0].set_yticks(range(len(progs)), progs, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax[0].text(j, i, M[i, j], ha="center", va="center", fontsize=6, color="white" if M[i, j] > M.max() / 2 else "black")
    ax[0].set_title("Parsed course versions by programme and academic year", fontsize=8)
    ax[1].hist(np.clip(words, 0, 1500), bins=30, color="#4C72B0")
    ax[1].set_title("Words per course (objectives + outcomes + units)", fontsize=8); ax[1].tick_params(labelsize=7)
    fig.tight_layout()
    (RESULTS / OUT).mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / OUT / "fig_eda.png", dpi=150)
    plt.close(fig)
    return dump(res, f"{OUT}/eda.json")


# ------------------------------------------------------------------ labels
def labels_summary(ctx):
    f = LABELS / "pairs_llm.csv"
    rows = list(csv.DictReader(open(f, encoding="utf-8"))) if f.exists() else []
    for r in rows:
        for k in ("pass1", "pass2", "pass3"):
            r[k] = None if r[k] in ("", "None") else r[k]
    by = Counter((r["split"], r["consensus"]) for r in rows if r["status"] != "uncertain")
    phi = [r for r in rows if r.get("pass2_by") == "Phi-3"]
    clb = [r for r in rows if r.get("pass2_by") == "Claude (wording B)"]
    res = {"file": "labels/pairs_llm.csv", "items": len(rows),
           "pass2_by": {"phi3": len(phi), "claude_b": len(clb)},
           # reported separately: Claude vs Phi-3 is cross-model; Claude vs Claude (wording B) is the same model family, not independent
           "agreement_claude_vs_phi3": agreement(phi) if phi else None, "agreement_claude_vs_claude": agreement(clb) if clb else None,
           "resolved_by_pass3": sum(r["status"] == "majority" for r in rows), "uncertain_excluded": sum(r["status"] == "uncertain" for r in rows),
           "by_split": {s: {str(k): by[(s, str(k))] for k in (0, 1, 2)} for s in ("val", "test")},
           "strata": dict(Counter(r["stratum"] for r in rows))}
    return dump(res, f"{OUT}/labels.json")


# ------------------------------------------------------------------ capability 1: overlap
def _tokens(c):
    return {w for w in re.findall(r"[a-z]{3,}", D.course_text(c).lower()) if w not in ENGLISH_STOP_WORDS}


def jaccard(a, b):
    A, Bt = _tokens(a), _tokens(b)
    return len(A & Bt) / max(1, len(A | Bt))


def score_pairs(ctx, pairs, fn):
    return np.array([fn(p["a"], p["b"]) for p in pairs])


def overlap(ctx):
    val, test = ctx.pair_labels("val"), ctx.pair_labels("test")
    res = {"n_val": len(val), "n_test": len(test), "methods": {}}
    if not val or not test:
        return dump({**res, "note": "no reference labels yet"}, f"{OUT}/overlap.json")
    yv, yt = np.array([p["y"] for p in val]), np.array([p["y"] for p in test])
    gt = [p["family_a"] for p in test]
    for name, fn in {"tfidf_cosine": ctx.tfidf.cosine, "word_jaccard": jaccard}.items():
        sv, st = score_pairs(ctx, val, fn), score_pairs(ctx, test, fn)
        thr = E.best_threshold(yv >= 1, sv)
        res["methods"][name] = {"test": E.ranking(yt, st, gt), "at_val_threshold": E.prf(yt >= 1, st, thr, gt)}
    # versions of one course are a separate task: how do they score compared with labelled cross-course pairs?
    vp = []
    for f in sorted(ctx.families("test")):
        vs = sorted([r for r in ctx.recs if r["family"] == f], key=lambda r: r["id"])[:4]
        vp += [(a, b) for i, a in enumerate(vs) for b in vs[i + 1:]]
    vs_scores = np.array([ctx.tfidf.cosine(a, b) for a, b in vp])
    st = score_pairs(ctx, test, ctx.tfidf.cosine)
    res["version_pairs"] = {"n": len(vp), "tfidf_median": E.r3(np.median(vs_scores)),
                            "cross_course_median_by_label": {str(k): E.r3(np.median(st[yt == k])) if (yt == k).any() else None for k in (0, 1, 2)},
                            "share_above_overlap_threshold": E.r3(np.mean(vs_scores >= res["methods"]["tfidf_cosine"]["at_val_threshold"]["threshold"]))}
    res["table"] = {"columns": ["Method", "Spearman [95% CI]", "ROC-AUC [95% CI]", "Avg. precision", "P / R / F1 at val threshold"],
                    "rows": [[{"tfidf_cosine": "TF-IDF cosine (baseline)", "word_jaccard": "Word-set Jaccard (trivial)"}[m],
                              f"{v['test']['spearman']['value']:.3f} [{v['test']['spearman']['ci95'][0]:.2f}, {v['test']['spearman']['ci95'][1]:.2f}]",
                              f"{v['test']['roc_auc']['value']:.3f} [{v['test']['roc_auc']['ci95'][0]:.2f}, {v['test']['roc_auc']['ci95'][1]:.2f}]",
                              f"{v['test']['avg_precision']['value']:.3f}",
                              f"{v['at_val_threshold']['precision']} / {v['at_val_threshold']['recall']} / {v['at_val_threshold']['f1']}"]
                             for m, v in res["methods"].items()]}
    return dump(res, f"{OUT}/overlap.json")


# ------------------------------------------------------------------ capability 2: redundancy
def rank_of_pair(scores_fn, courses, ix, iy):
    """Rank (1 = highest) of pair (ix, iy) among all pairs of different families of one programme."""
    S = scores_fn(courses)
    target = S[ix, iy]
    n = sum(1 for i in range(len(courses)) for j in range(i + 1, len(courses))
            if courses[i]["family"] != courses[j]["family"] and S[i, j] > target)
    return n + 1, float(target)


def tfidf_matrix(ctx):
    return lambda cs: (lambda T: (T @ T.T).toarray())(ctx.tfidf.matrix(cs))


def injected(ctx, fams, n=60):
    """Synthetic redundancy cases between clearly unrelated courses of one programme (TF-IDF cosine < 0.1 before injection)."""
    cases = []
    for c in RD.inject_cases(ctx.view, fams, n=4 * n, seed=1):
        orig = [x for x in [c["x"]] + [ctx.view[c["programme"]][c["y"]["family"]]]]
        if ctx.tfidf.cosine(*orig) < 0.1:
            cases.append(c)
        if len(cases) == n:
            break
    return cases


def redundancy_eval(ctx, fams, scorer, thr, k=20):
    hits, det = [], []
    for c in injected(ctx, fams):
        cs = [c["y"] if x["family"] == c["y"]["family"] else x for x in RD.programme_courses(ctx.view, c["programme"], fams)]
        ix = next(i for i, x in enumerate(cs) if x["family"] == c["x"]["family"])
        iy = next(i for i, x in enumerate(cs) if x["family"] == c["y"]["family"])
        r, s = rank_of_pair(scorer, cs, ix, iy)
        hits.append(r <= k); det.append(s >= thr)
    return {"cases": len(hits), f"recall_at_{k}": E.r3(np.mean(hits)), "detected_at_threshold": E.r3(np.mean(det))}


def redundancy(ctx):
    val = ctx.pair_labels("val")
    thr = E.best_threshold(np.array([p["y"] == 2 for p in val]), score_pairs(ctx, val, ctx.tfidf.cosine)) if val else 0.5
    fams = ctx.families("test")
    flags = []
    for p in sorted(ctx.view):
        cs = RD.programme_courses(ctx.view, p, fams)
        if len(cs) >= 2:
            flags += [(p, cs[i]["course_name"], cs[j]["course_name"], round(s, 3)) for i, j, s in RD.course_pairs(ctx.tfidf.matrix(cs), cs, thr)]
    res = {"method": "course-level TF-IDF cosine within a programme", "threshold_from_val_substantial_overlap": E.r3(thr),
           "flags_test": len(flags), "flags_by_programme": dict(Counter(f[0] for f in flags)),
           "top_flags": [{"programme": f[0], "course_a": f[1], "course_b": f[2], "tfidf": f[3]} for f in flags[:8]],
           "synthetic_injection": redundancy_eval(ctx, fams, tfidf_matrix(ctx), thr)}
    return dump(res, f"{OUT}/redundancy.json")


# ------------------------------------------------------------------ capability 3: missing prerequisites
def prereq_tests(ctx, make_resolver, topic_covered=None, split="test"):
    """Removal recall, false alarms on intact programmes and after renaming the prerequisite course (label-free)."""
    fams = ctx.families(split)
    res0 = make_resolver(ctx.recs)
    cases = PQ.removal_cases(ctx.recs, ctx.view, ctx.st, ctx.versions, fams)

    def flagged(st, resolver, c):
        fl = PQ.check_programme(c["programme"], ctx.view, st, resolver, ctx.versions, topic_covered(ctx, st) if topic_covered else None,
                                families={c["family"]})
        return any(f["phrase"] == c["phrase"] for f in fl)

    intact = [flagged(ctx.st, res0, c) for c in cases]
    removed = [flagged(PQ.without_family(ctx.st, c["programme"], c["removed_family"]), res0, c) for c in cases]
    renamed = []
    for c in cases:
        recs2 = [dict(r, course_name="Principles of " + r["course_name"]) if r["family"] == c["removed_family"] else r for r in ctx.recs]
        renamed.append(flagged(ctx.st, make_resolver(recs2), c))
    g = [c["family"] for c in cases]
    return {"cases": len(cases), "removal_recall": E.mean_ci(removed, g), "intact_false_alarm": E.mean_ci(intact, g),
            "rename_false_alarm": E.mean_ci(renamed, g)}


def prerequisites(ctx):
    fams = ctx.families("test")
    resolver = PQ.ExactResolver(ctx.recs)
    flags = [f for p in sorted(ctx.view) for f in PQ.check_programme(p, ctx.view, ctx.st, resolver, ctx.versions, families=fams)]
    checked = sum(1 for p in ctx.view for f, c in ctx.view[p].items() if f in fams and not c["common"] and c["sem"] and c["sem"] >= 2
                  and PQ.covered(ctx.st, ctx.versions, p, c["sem"]))
    phrases = sum(len(PQ.split_prereq(c["prerequisites"])) for p in ctx.view for f, c in ctx.view[p].items() if f in fams
                  and not c["common"] and c["sem"] and c["sem"] >= 2 and PQ.covered(ctx.st, ctx.versions, p, c["sem"]))
    res = {"method": "regex split + exact course-name match", "programmes_checked": sorted({p for p in ctx.view if ctx.versions[p] >= PQ.MIN_VERSIONS}),
           "course_programme_pairs_checked": checked, "phrases_checked": phrases, "flags": len(flags), "flags_by_type": dict(Counter(f["type"] for f in flags)),
           "flag_rate_per_phrase": E.r3(len(flags) / max(1, phrases)),
           "examples": [{k: f[k] for k in ("programme", "course", "semester", "phrase", "type")} for f in flags[:6]],
           "tests": prereq_tests(ctx, PQ.ExactResolver)}
    return dump(res, f"{OUT}/prerequisites.json")


# ------------------------------------------------------------------ capability 4: curriculum diff
def renamed(B, share=0.2, seed=0):
    """Synthetic renaming: a share of the later year's courses get a new title ('Principles of ...'), content unchanged."""
    rng = np.random.default_rng(seed)
    return {f: (dict(c, course_name="Principles of " + c["course_name"]) if rng.random() < share else c) for f, c in sorted(B.items())}


def diff_eval(ctx, method, rename=False):
    tp = p = t = 0
    per = []
    for k, (prog, a, b) in enumerate(DF.year_pairs(ctx.recs)):
        A, Bs = DF.snapshot(ctx.recs, prog, a), DF.snapshot(ctx.recs, prog, b)
        if rename:
            Bs = renamed(Bs, seed=k)
        x = DF.score(method(A, Bs), DF.truth(A, Bs))
        tp, p, t = tp + x[0], p + x[1], t + x[2]
        per.append({"programme": prog, "from": a, "to": b, "courses_from": len(A), "courses_to": len(Bs), "tp": x[0], "claimed": x[1], "true": x[2]})
    return {"year_pairs": len(per), "precision": E.r3(tp / p) if p else None, "recall": E.r3(tp / t) if t else None, "true_changes": t, "claimed_changes": p}, per


def diff(ctx):
    summ, per = diff_eval(ctx, DF.diff_by_name)
    ren, _ = diff_eval(ctx, DF.diff_by_name, rename=True)
    return dump({"method": "exact course-name set difference", **summ, "rename_test": ren, "per_year_pair": per}, f"{OUT}/diff.json")


# ------------------------------------------------------------------ capability 5: Bloom
def bloom(ctx):
    data = B.tagged_outcomes(ctx.recs, ctx.split)
    major = Counter(l for _, l, _ in data["train"]).most_common(1)[0][0]
    t = data["test"]
    pred = [B.lexicon_level(x) or major for x, _, _ in t]
    res = {"method": "hand-written verb lexicon (majority level when no verb matches)", "train_outcomes": len(data["train"]),
           "val_outcomes": len(data["val"]), "test_outcomes": len(t), "verb_found_share": E.r3(np.mean([B.lexicon_level(x) is not None for x, _, _ in t])),
           "majority_level": B.BLOOM[major], "lexicon": E.classification([l for _, l, _ in t], pred, [f for _, _, f in t]),
           "majority_class": E.classification([l for _, l, _ in t], [major] * len(t), [f for _, _, f in t])}
    return dump(res, f"{OUT}/bloom.json")


def run(ctx):
    for fn in (eda, labels_summary, overlap, redundancy, prerequisites, diff, bloom, summary):
        print(f"  week2.{fn.__name__}", flush=True)
        fn(ctx)


def _v(x, f=".3f"):
    return "n/a" if x is None else format(x, f)


def _ci(m):
    return f"{_v(m['value'])} [{_v(m['ci95'][0], '.2f')}, {_v(m['ci95'][1], '.2f')}]"


def summary_rows(o, r, p, d, b, label):
    """One row per capability: (capability, method, metric, value). Shared by the Week 2-5 summaries."""
    rows = []
    if o.get("methods"):
        m = o["methods"][label["overlap"]]
        rows += [["1 Overlap", label["overlap_name"], "Spearman [95% CI]", _ci(m["test"]["spearman"])],
                 ["", "", "ROC-AUC any overlap [95% CI]", _ci(m["test"]["roc_auc"])],
                 ["", "", "F1 at validation threshold", _v(m["at_val_threshold"]["f1"])]]
    s = r["synthetic_injection"]
    rows += [["2 Redundancy", r["method"], "injected near-duplicates found in top 20", _v(s["recall_at_20"])]]
    t = p["tests"]
    rows += [["3 Prerequisites", p["method"], "removal recall [95% CI]", _ci(t["removal_recall"])],
             ["", "", "false alarms, intact programme", _ci(t["intact_false_alarm"])],
             ["", "", "false alarms after renaming the course", _ci(t["rename_false_alarm"])]]
    rows += [["4 Curriculum diff", d["method"], "precision / recall of changes", f"{_v(d['precision'])} / {_v(d['recall'])}"]]
    if "rename_test" in d:
        rows += [["", "", "precision / recall with renamed courses", f"{_v(d['rename_test']['precision'])} / {_v(d['rename_test']['recall'])}"]]
    m = b[label["bloom"]]
    rows += [["5 Bloom level", b["method"], "accuracy [95% CI]", _ci(m["accuracy"])],
             ["", "", "within one level / macro-F1", f"{_v(m['within_one']['value'])} / {_v(m['macro_f1'])}"]]
    return {"columns": ["Capability", "Method", "Metric (test split)", "Result"], "rows": rows}


def summary(ctx):
    import json
    j = {k: json.loads((RESULTS / OUT / f"{k}.json").read_text(encoding="utf-8")) for k in ("overlap", "redundancy", "prerequisites", "diff", "bloom")}
    t = summary_rows(j["overlap"], j["redundancy"], j["prerequisites"], j["diff"], j["bloom"],
                     {"overlap": "tfidf_cosine", "overlap_name": "TF-IDF cosine", "bloom": "lexicon"})
    return dump({"table": t}, f"{OUT}/summary.json")
