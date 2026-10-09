"""Week 5 - evaluation and error analysis of the final system (methods selected on validation in Weeks 3-4):
held-out test results vs the proposal targets, per-programme breakdown, threshold sensitivity, robustness to perturbed input,
and failure cases (selected automatically; root causes in reports/failure_notes.json).
Writes results/week5/*.json and figures."""
import copy, csv, json
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import spearmanr

from curriculum_engine import LABELS, RESULTS, ROOT, SEED
from curriculum_engine import bloom as B
from curriculum_engine import evaluate as E
from curriculum_engine import perturb as PT
from curriculum_engine import prereq as PQ
from curriculum_engine.experiments import week2 as W2
from curriculum_engine.experiments import week4 as W4
from curriculum_engine.experiments.common import dump

OUT = "week5"


def J(path):
    return json.loads((RESULTS / f"{path}.json").read_text(encoding="utf-8"))


def final_overlap_fn(ctx):
    """The overlap score selected on validation in Week 4 (a callable on two course records)."""
    sel = J("week4/overlap")["selected_on_validation"]
    if sel == "fusion_logistic":
        cm = J("week4/confidence_model")
        fns = {"tfidf_cosine": ctx.tfidf.cosine, "maxsim_minilm": W4.soft_with(ctx.emb), "max_unit_minilm": W4.max_unit_with(ctx.emb)}
        w, b = np.array(cm["coef"]), cm["intercept"]
        return sel, lambda a, c: float(1 / (1 + np.exp(-(np.dot(w, [fns[f](a, c) for f in cm["features"]]) + b))))
    if sel == "fusion_rank_tfidf_maxsim":
        # rank average for a single pair: each score becomes its percentile among the validation pairs' scores, then averaged
        val = ctx.pair_labels("val")
        t, m = ctx.tfidf.cosine, W4.soft_with(ctx.emb)
        ref_t, ref_m = np.sort(W2.score_pairs(ctx, val, t)), np.sort(W2.score_pairs(ctx, val, m))
        pct = lambda ref, x: float(np.searchsorted(ref, x, side="right")) / len(ref)
        return "fusion of TF-IDF and MaxSim (validation percentiles)", lambda a, c: (pct(ref_t, t(a, c)) + pct(ref_m, m(a, c))) / 2
    fns = {"tfidf_cosine": ctx.tfidf.cosine, "maxsim_minilm": W4.soft_with(ctx.emb), "meanpool_minilm": W4.meanvec_with(ctx.emb),
           "max_unit_minilm": W4.max_unit_with(ctx.emb)}
    if sel == "maxsim_bge_base":
        from curriculum_engine.embed import BGE_BASE, Embedder
        fns[sel] = W4.soft_with(Embedder(BGE_BASE))
    return sel, fns[sel]


# ------------------------------------------------------------------ 1. final results vs targets
def final(ctx, fn, name):
    val, test = ctx.pair_labels("val"), ctx.pair_labels("test")
    yv, yt = np.array([p["y"] for p in val]), np.array([p["y"] for p in test])
    sv, st = W2.score_pairs(ctx, val, fn), W2.score_pairs(ctx, test, fn)
    thr = E.best_threshold(yv >= 1, sv)
    gt = [p["family_a"] for p in test]
    ov = {"method": name, "test": E.ranking(yt, st, gt), "at_val_threshold": E.prf(yt >= 1, st, thr, gt)}
    tg = J("plan/targets")
    fp = J("week3/flag_precision")
    w3p, w4b, w3d, w3r = J("week3/prerequisites"), J("week4/bloom"), J("week3/diff"), J("week3/redundancy")
    pre = w3p["tests"]
    bl = w4b["improved_test"] if w4b["vs_lexicon_accuracy"]["verdict"] == "improved" else w4b["lexicon"]
    def g(d, *keys):
        for k in keys:
            d = d.get(k) if isinstance(d, dict) else None
        return d

    rows =[("1 Overlap", "Spearman", ov["test"]["spearman"]["value"], tg["overlap"]["spearman"], ">="),
            ("", "ROC-AUC (any overlap)", ov["test"]["roc_auc"]["value"], tg["overlap"]["roc_auc"], ">="),
            ("", "F1 at validation threshold", ov["at_val_threshold"]["f1"], tg["overlap"]["f1"], ">="),
            ("2 Redundancy", "precision of top flags (reference labels)", g(fp, "redundancy", "core", "precision"), tg["redundancy"]["precision_at_20"], ">="),
            ("", "injected near-duplicates in top 20", w3r["synthetic_injection"]["recall_at_20"], tg["redundancy"]["synthetic_recall_at_20"], ">="),
            ("3 Prerequisites", "removal recall", pre["removal_recall"]["value"], tg["prerequisites"]["removal_recall"], ">="),
            ("", "precision of flags (reference labels)", g(fp, "prerequisite", "core", "precision"), tg["prerequisites"]["flag_precision"], ">="),
            ("", "false alarms after renaming", pre["rename_false_alarm"]["value"], tg["prerequisites"]["rename_false_alarm_max"], "<="),
            ("4 Curriculum diff", "precision (real year pairs)", w3d["precision"], tg["diff"]["precision"], ">="),
            ("", "recall (real year pairs)", w3d["recall"], tg["diff"]["recall"], ">="),
            ("5 Bloom", "accuracy vs faculty tags", bl["accuracy"]["value"], tg["bloom"]["accuracy"], ">="),
            ("", "within one level", bl["within_one"]["value"], tg["bloom"]["within_one"], ">="),
            ("6 Emerging topics", "precision of flagged gaps", g(fp, "emerging_gap", "core", "precision"), tg["emerging"]["gap_precision"], ">="),
            ("7 Topic labels", "share accepted", g(fp, "topic_label", "core", "precision"), tg["topics"]["accepted_share"], ">=")]
    table = {"columns": ["Capability", "Metric (test split)", "Result", "Target", "Met?"],
             "rows": [[c, m, W2._v(v), f"{op} {t}", "n/a" if v is None else ("yes" if (v >= t if op == ">=" else v <= t) else "no")] for c, m, v, t, op in rows]}
    met = Counter(r[4] for r in table["rows"])
    return dump({"overlap": ov, "threshold": E.r3(thr), "targets_met": met.get("yes", 0), "targets_missed": met.get("no", 0),
                 "targets_total": len(rows), "table": table}, f"{OUT}/final.json")


# ------------------------------------------------------------------ 2. per programme
def per_programme(ctx, fn):
    test = ctx.pair_labels("test")
    st = W2.score_pairs(ctx, test, fn)
    by = defaultdict(list)
    for p, s in zip(test, st):
        for prog in set(p["a"]["programmes"]) | set(p["b"]["programmes"]) or {"COMMON"}:
            by[prog].append((p["y"], s))
    data = B.tagged_outcomes(ctx.recs, ctx.split)
    fam_prog = defaultdict(set)
    for r in ctx.recs:
        fam_prog[r["family"]] |= set(r["programmes"]) or {"COMMON"}
    bl = defaultdict(list)
    for x, l, f in data["test"]:
        for prog in fam_prog[f]:
            bl[prog].append((B.lexicon_level(x) or 3) == l)
    th = J("week3/thresholds")
    chk = PQ.TopicChecker(ctx.reps, ctx.emb, th["prereq_topic"])
    res0 = PQ.EmbeddingResolver(ctx.recs, ctx.emb, th["prereq_resolve"])
    cases = PQ.removal_cases(ctx.recs, ctx.view, ctx.st, ctx.versions, ctx.families("test"))
    pq = defaultdict(list)
    for c in cases:
        fl = PQ.check_programme(c["programme"], ctx.view, PQ.without_family(ctx.st, c["programme"], c["removed_family"]), res0, ctx.versions, chk, families={c["family"]})
        pq[c["programme"]].append(any(f["phrase"] == c["phrase"] for f in fl))
    rows = []
    for prog in sorted(set(by) | set(bl) | set(pq)):
        y = np.array([a for a, _ in by[prog]]); s = np.array([b for _, b in by[prog]])
        auc = E.r3(E.auc_any(y, s)) if len(set(y >= 1)) == 2 and len(y) >= 8 else None
        rows.append([prog, len(y), W2._v(auc), len(bl[prog]), W2._v(np.mean(bl[prog]) if bl[prog] else None),
                     len(pq[prog]), W2._v(np.mean(pq[prog]) if pq[prog] else None)])
    return dump({"table": {"columns": ["Programme", "Labelled pairs", "Overlap ROC-AUC", "Tagged outcomes", "Bloom accuracy",
                                       "Prerequisite cases", "Removal recall"], "rows": rows},
                 "note": "a pair counts for every programme of either course; ROC-AUC only with at least 8 pairs of both classes"}, f"{OUT}/per_programme.json")


# ------------------------------------------------------------------ 3. threshold sensitivity
def sensitivity(ctx, fn, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    val, test = ctx.pair_labels("val"), ctx.pair_labels("test")
    yv, yt = np.array([p["y"] >= 1 for p in val]), np.array([p["y"] >= 1 for p in test])
    sv, st = W2.score_pairs(ctx, val, fn), W2.score_pairs(ctx, test, fn)
    chosen = E.best_threshold(yv, sv)
    grid = np.quantile(np.concatenate([sv, st]), np.linspace(0.05, 0.97, 30))
    curve = [{"threshold": E.r3(t), **{k: E.prf(yt, st, t)[k] for k in ("precision", "recall", "f1")}} for t in grid]
    th = J("week3/thresholds")
    chk = PQ.TopicChecker(ctx.reps, ctx.emb, th["prereq_topic"])
    pre = []
    for tr in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        r = W2.prereq_tests(ctx, lambda recs, tr=tr: PQ.EmbeddingResolver(recs, ctx.emb, tr), lambda c, st_: chk)
        pre.append({"resolve_threshold": tr, "removal_recall": r["removal_recall"]["value"], "intact_false_alarm": r["intact_false_alarm"]["value"],
                    "rename_false_alarm": r["rename_false_alarm"]["value"]})
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    for k, c in (("precision", "#4C72B0"), ("recall", "#DD8452"), ("f1", "#55A868")):
        ax[0].plot([c_["threshold"] for c_ in curve], [c_[k] or 0 for c_ in curve], label=k, color=c)
    ax[0].axvline(chosen, ls="--", color="grey"); ax[0].set_title(f"Overlap ({name}): test P/R/F1 vs threshold (dashed: chosen on validation)", fontsize=8)
    ax[0].legend(fontsize=7); ax[0].tick_params(labelsize=7)
    for k, c in (("removal_recall", "#55A868"), ("intact_false_alarm", "#C44E52"), ("rename_false_alarm", "#8172B3")):
        ax[1].plot([p["resolve_threshold"] for p in pre], [p[k] for p in pre], marker="o", label=k.replace("_", " "), color=c)
    ax[1].axvline(th["prereq_resolve"], ls="--", color="grey"); ax[1].set_title("Prerequisites: test rates vs resolution threshold", fontsize=8)
    ax[1].legend(fontsize=7); ax[1].tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(RESULTS / OUT / "fig_sensitivity.png", dpi=150); plt.close(fig)
    best = max(curve, key=lambda c: c["f1"] or 0)
    return dump({"overlap_curve": curve, "chosen_on_validation": E.r3(chosen), "test_f1_at_chosen": E.prf(yt, st, chosen)["f1"],
                 "best_test_f1_any_threshold": best["f1"], "best_test_threshold": best["threshold"], "prerequisite_curve": pre}, f"{OUT}/sensitivity.json")


# ------------------------------------------------------------------ 4. robustness
def _perturbed(c, how, seed):
    c = copy.deepcopy(c)
    if how == "typos":
        for u in c["units"]: u["text"] = PT.typos(u["text"], 0.05, seed)
        c["outcomes"] = [dict(o, text=PT.typos(o["text"], 0.05, seed)) for o in c["outcomes"]]
    elif how == "word_drop":
        for u in c["units"]: u["text"] = PT.word_drop(u["text"], 0.15, seed)
    elif how == "shuffled_units":
        rng = np.random.default_rng(seed); c["units"] = [c["units"][i] for i in rng.permutation(len(c["units"]))]
    elif how == "truncated_half":
        c["units"] = c["units"][: max(1, len(c["units"]) // 2)]
    elif how == "missing_sections":
        c["outcomes"], c["objectives"] = [], ""
    elif how == "paraphrased":
        c["units"] = [dict(u, text=paraphrase(u["text"])) for u in c["units"]]
    return c


_PARA = {}


def paraphrase(text):
    """Phi-3 paraphrase of a unit (cached in data/interim/paraphrases.json); the original text if the LLM is unavailable."""
    from curriculum_engine import DATA, llm
    f = DATA / "interim/paraphrases.json"
    if not _PARA and f.exists():
        _PARA.update(json.loads(f.read_text(encoding="utf-8")))
    if text not in _PARA:
        if not llm.available():
            return text
        _PARA[text] = llm.generate("Rewrite this syllabus unit in different words, keeping every topic. Reply with the rewritten text only.\n\n" + text[:1200],
                                   num_predict=220, num_ctx=2048)
        f.write_text(json.dumps(_PARA, ensure_ascii=False), encoding="utf-8")
    return _PARA[text]


def robustness(ctx, fn, name, with_paraphrase):
    test = ctx.pair_labels("test")
    yt = np.array([p["y"] for p in test])
    gt = [p["family_a"] for p in test]
    kinds = ["typos", "word_drop", "shuffled_units", "truncated_half", "missing_sections"] + (["paraphrased"] if with_paraphrase else [])
    methods = {name: fn, "tfidf_cosine": ctx.tfidf.cosine}
    base = {m: W2.score_pairs(ctx, test, f) for m, f in methods.items()}
    res, rows = {}, []
    for k in kinds:
        pairs = test if k != "paraphrased" else test[:40]
        idx = np.arange(len(pairs))
        res[k] = {}
        for m, f in methods.items():
            s = np.array([f(p["a"], _perturbed(p["b"], k, SEED + i)) for i, p in enumerate(pairs)])
            r = E.ranking(yt[idx], s, [gt[i] for i in idx])
            r0 = E.ranking(yt[idx], base[m][idx], [gt[i] for i in idx])
            res[k][m] = {"n_pairs": len(pairs), "spearman": r["spearman"]["value"], "spearman_clean": r0["spearman"]["value"],
                         "roc_auc": r["roc_auc"]["value"], "roc_auc_clean": r0["roc_auc"]["value"],
                         "mean_score_change": E.r3(np.mean(s - base[m][idx]))}
        rows.append([k.replace("_", " "), res[k][name]["n_pairs"],
                     f"{W2._v(res[k][name]['spearman_clean'])} → {W2._v(res[k][name]['spearman'])}",
                     f"{W2._v(res[k]['tfidf_cosine']['spearman_clean'])} → {W2._v(res[k]['tfidf_cosine']['spearman'])}",
                     f"{W2._v(res[k][name]['roc_auc_clean'])} → {W2._v(res[k][name]['roc_auc'])}",
                     f"{W2._v(res[k]['tfidf_cosine']['roc_auc_clean'])} → {W2._v(res[k]['tfidf_cosine']['roc_auc'])}"])
    # same course vs different courses: self-similarity of a perturbed copy against version pairs and labelled cross-course pairs
    selfsim = {k: E.r3(np.median([fn(c, _perturbed(c, k, SEED)) for f, c in sorted(ctx.reps.items()) if ctx.split[f] == "test"][:60]))
               for k in kinds if k != "paraphrased"}
    return dump({"perturbed_side": "course B of every labelled test pair", "results": res, "self_similarity_median": selfsim,
                 "table": {"columns": ["Perturbation", "Pairs", f"Spearman {name}", "Spearman TF-IDF", f"ROC-AUC {name}", "ROC-AUC TF-IDF"], "rows": rows}},
                f"{OUT}/robustness.json")


# ------------------------------------------------------------------ 5. failure cases
def failures(ctx, fn, name):
    """At least ten failure cases, selected by fixed rules (no cherry-picking): the three highest-scoring test pairs labelled 'no overlap'
    that pass the threshold and the three lowest-scoring overlapping pairs below it; the first three prerequisite flags, two redundancy
    flags and one emerging-topic gap the reference labels call wrong; the first topic the detector calls covered although the labels
    see a gap; and the two Bloom errors furthest from the faculty tag. Root causes and fixes: reports/failure_notes.json."""
    out = []
    test = ctx.pair_labels("test")
    st = W2.score_pairs(ctx, test, fn)
    thr = J(f"{OUT}/final")["threshold"]
    fp = sorted([(s, p) for s, p in zip(st, test) if p["y"] == 0 and s >= thr], key=lambda x: -x[0])[:3]
    fn_ = sorted([(s, p) for s, p in zip(st, test) if p["y"] >= 1 and s < thr], key=lambda x: x[0])[:3]
    for kind, lst, truth in (("overlap false positive", fp, "no overlap"), ("overlap false negative", fn_, None)):
        for s, p in lst:
            out.append({"key": f"overlap|{p['item']}", "type": kind, "input": f"{p['course_a']} vs {p['course_b']}",
                        "prediction": f"score {s:.3f} (threshold {thr:.3f})", "truth": truth or {1: "partial overlap", 2: "substantial overlap"}[p["y"]],
                        "evidence": p["reason_pass1"]})
    flags = list(csv.DictReader(open(LABELS / "flags_llm.csv", encoding="utf-8"))) if (LABELS / "flags_llm.csv").exists() else []
    fam_name = {}
    for c in ctx.recs:
        fam_name.setdefault(c["family"], c["course_name"])

    def describe(kind, key):
        k = key.split("|")
        if kind == "redundancy":
            return f"{ctx.by_id[k[0]]['course_name']} vs {ctx.by_id[k[1]]['course_name']}"
        if kind == "prerequisite":
            return f"{fam_name.get(k[1], k[1])} ({k[0]}) needs '{k[2]}'"
        return f"'{k[0]}' in programme {k[1]}"

    for kind, k, what, truth in (("prerequisite", 3, "flagged as missing", "reference labels: warning is wrong"),
                                 ("redundancy", 2, "flagged as redundant", "reference labels: warning is wrong"),
                                 ("emerging_gap", 1, "flagged as a gap", "reference labels: warning is wrong"),
                                 ("emerging_covered", 1, "judged as already taught", "reference labels: a real gap")):
        for r in [r for r in flags if r["kind"] == kind and r["method"] == "core" and r["consensus"] == "wrong"][:k]:
            out.append({"key": f"{kind}|{r['key']}", "type": {"emerging_covered": "missed emerging-topic gap"}.get(kind, f"{kind.replace('_', ' ')} flag judged wrong"),
                        "input": describe(kind if kind != "emerging_covered" else "emerging_gap", r["key"]),
                        "prediction": what, "truth": truth, "evidence": r["reason_pass1"]})
    data = B.tagged_outcomes(ctx.recs, ctx.split)["test"]
    errs = sorted([(abs((B.lexicon_level(x) or 3) - l), x, l) for x, l, _ in data if (B.lexicon_level(x) or 3) != l], key=lambda e: (-e[0], e[1]))[:2]
    for d, x, l in errs:
        p = B.lexicon_level(x) or 3
        out.append({"key": f"bloom|{x[:40]}", "type": "Bloom level error", "input": x[:140], "prediction": B.BLOOM[p], "truth": f"faculty tag {B.BLOOM[l]}",
                    "evidence": f"first verb: {B.first_word(x)}"})
    notes = json.loads((ROOT / "reports/failure_notes.json").read_text(encoding="utf-8")) if (ROOT / "reports/failure_notes.json").exists() else {}
    for i, f in enumerate(out, 1):
        f["id"] = f"F{i:02d}"
        f |= notes.get(f["key"], {"category": "to analyse", "root_cause": "to analyse", "fix_or_limitation": "to analyse"})
    table = {"columns": ["#", "Type", "Input", "Prediction / truth", "Category", "Root cause", "Fix or known limitation"],
             "rows": [[f["id"], f["type"], f["input"][:120], f"{f['prediction']} / {f['truth']}", f["category"], f["root_cause"], f["fix_or_limitation"]] for f in out]}
    return dump({"n": len(out), "selection_rules": failures.__doc__.split("\n", 1)[1].strip(), "cases": out, "table": table,
                 "by_category": dict(Counter(f["category"] for f in out))}, f"{OUT}/failures.json")


def run(ctx, with_paraphrase=False):          # paraphrase variant not run (time)
    (RESULTS / OUT).mkdir(parents=True, exist_ok=True)
    name, fn = final_overlap_fn(ctx)
    for step in (lambda: final(ctx, fn, name), lambda: per_programme(ctx, fn), lambda: sensitivity(ctx, fn, name),
                 lambda: robustness(ctx, fn, name, with_paraphrase), lambda: failures(ctx, fn, name)):
        step()
        print("  week5 step done", flush=True)
    ctx.emb.save()
