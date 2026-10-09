"""Week 4 - ablations and improvements tied to the core objective, each compared with the Week 3 core on the same test items
with paired 95% intervals (resampling families). Choices are made on validation; the test split is only reported.
Writes results/week4/{overlap,redundancy,bloom,summary,confidence_model}.json (bge-base prerequisite re-tuning not run: time)."""
import json
from collections import Counter

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from curriculum_engine import RESULTS, SEED
from curriculum_engine import bloom as B
from curriculum_engine import evaluate as E
from curriculum_engine import prereq as PQ
from curriculum_engine.embed import BGE_BASE, Embedder
from curriculum_engine.experiments import week2 as W2
from curriculum_engine.experiments.common import dump
from curriculum_engine.overlap import chunk_vectors, maxsim

OUT = "week4"


def soft_with(emb):
    return lambda a, b: maxsim(chunk_vectors(emb, a), chunk_vectors(emb, b), 1.0)["soft"]


def max_unit_with(emb):
    def f(a, b):
        A, Bm = chunk_vectors(emb, a), chunk_vectors(emb, b)
        return float((A @ Bm.T).max()) if len(A) and len(Bm) else 0.0
    return f


def meanvec_with(emb):
    def v(c):
        E_ = chunk_vectors(emb, c)
        m = E_.mean(0) if len(E_) else np.zeros(E_.shape[1] if E_.ndim == 2 else 384)
        return m / (np.linalg.norm(m) + 1e-9)
    return lambda a, b: float(v(a) @ v(b))


def _scores(ctx, fns, pairs):
    return np.column_stack([W2.score_pairs(ctx, pairs, f) for f in fns])


# ------------------------------------------------------------------ capability 1
def overlap(ctx, th, bge):
    val, test = ctx.pair_labels("val"), ctx.pair_labels("test")
    yv, yt = np.array([p["y"] for p in val]), np.array([p["y"] for p in test])
    gt = [p["family_a"] for p in test]
    base_fns = {"tfidf_cosine": ctx.tfidf.cosine, "maxsim_minilm": soft_with(ctx.emb), "meanpool_minilm": meanvec_with(ctx.emb),
                "maxsim_bge_base": soft_with(bge), "max_unit_minilm": max_unit_with(ctx.emb)}
    SV, ST = _scores(ctx, base_fns.values(), val), _scores(ctx, base_fns.values(), test)
    cols = list(base_fns)
    scores = {m: (SV[:, i], ST[:, i]) for i, m in enumerate(cols)}
    # fusion 1: average of rank-normalised TF-IDF and MaxSim (no training)
    rn = lambda x: rankdata(x) / len(x)
    scores["fusion_rank_tfidf_maxsim"] = tuple((rn(S[:, 0]) + rn(S[:, 1])) / 2 for S in (SV, ST))
    # fusion 2: logistic regression on validation labels -> P(any overlap); its validation score is cross-validated
    feats = [0, 1, 4]
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(SV[:, feats], yv >= 1)
    cv = np.zeros(len(yv))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(SV, yv >= 1):
        cv[te] = LogisticRegression(C=1.0, max_iter=2000).fit(SV[tr][:, feats], yv[tr] >= 1).predict_proba(SV[te][:, feats])[:, 1]
    scores["fusion_logistic"] = (cv, lr.predict_proba(ST[:, feats])[:, 1])
    dump({"features": [cols[i] for i in feats], "coef": lr.coef_[0].round(4).tolist(), "intercept": round(float(lr.intercept_[0]), 4),
          "fitted_on": "validation reference labels (any overlap)", "encoder": "all-MiniLM-L6-v2"}, f"{OUT}/confidence_model.json")
    res = {"n_val": len(val), "n_test": len(test), "methods": {}}
    core = scores["maxsim_minilm"][1]
    for m, (sv, st) in scores.items():
        res["methods"][m] = {"val_spearman": E.r3(spearmanr(yv, sv)[0]), "test": E.ranking(yt, st, gt),
                             "at_val_threshold": E.prf(yt >= 1, st, E.best_threshold(yv >= 1, sv), gt),
                             "vs_core_spearman": E.paired(E.spearman, yt, core, st, gt), "vs_core_auc": E.paired(E.auc_any, yt, core, st, gt)}
    res["selected_on_validation"] = max(res["methods"], key=lambda m: res["methods"][m]["val_spearman"])
    res["confidence_calibration_test"] = E.calibration(yt >= 1, scores["fusion_logistic"][1])
    names = {"tfidf_cosine": "TF-IDF cosine (Week 2)", "maxsim_minilm": "MaxSim, MiniLM (Week 3 core)", "meanpool_minilm": "Mean-pooled course vector, MiniLM",
             "maxsim_bge_base": "MaxSim, bge-base encoder", "max_unit_minilm": "Best single unit pair, MiniLM",
             "fusion_rank_tfidf_maxsim": "Fusion: rank average TF-IDF + MaxSim", "fusion_logistic": "Fusion: logistic regression (probability)"}
    res["table"] = {"columns": ["Method", "Val Spearman", "Test Spearman", "Δ Spearman vs core [95% CI]", "Test ROC-AUC", "Δ AUC vs core [95% CI]", "Verdict"],
                    "rows": [[names[m], W2._v(v["val_spearman"]), W2._v(v["test"]["spearman"]["value"]),
                              f"{W2._v(v['vs_core_spearman']['difference'], '+.3f')} [{W2._v(v['vs_core_spearman']['ci95'][0], '.2f')}, {W2._v(v['vs_core_spearman']['ci95'][1], '.2f')}]",
                              W2._v(v["test"]["roc_auc"]["value"]),
                              f"{W2._v(v['vs_core_auc']['difference'], '+.3f')} [{W2._v(v['vs_core_auc']['ci95'][0], '.2f')}, {W2._v(v['vs_core_auc']['ci95'][1], '.2f')}]",
                              "(core)" if m == "maxsim_minilm" else v["vs_core_spearman"]["verdict"]]
                             for m, v in res["methods"].items()]}
    return dump(res, f"{OUT}/overlap.json")


# ------------------------------------------------------------------ capability 2
def redundancy(ctx, th, bge):
    val = ctx.pair_labels("val")
    yv = np.array([p["y"] >= 1 for p in val])
    fams = ctx.families("test")
    out = {}
    for name, emb in (("minilm", ctx.emb), ("bge_base", bge)):
        f = max_unit_with(emb)
        thr = E.best_threshold(yv, W2.score_pairs(ctx, val, f))
        out[name] = {"threshold_from_val": E.r3(thr),
                     "synthetic_injection": W2.redundancy_eval(ctx, fams, lambda cs, f=f: np.array([[f(a, b) for b in cs] for a in cs]), thr)}
    return dump({"method": "unit-level near-duplicates; encoder ablation", **out}, f"{OUT}/redundancy.json")


# ------------------------------------------------------------------ capability 5
def bloom(ctx):
    data = B.tagged_outcomes(ctx.recs, ctx.split)
    tr, va, te = data["train"], data["val"], data["test"]
    major = Counter(l for _, l, _ in tr).most_common(1)[0][0]
    vmap = B.learn_verb_map([x for x, _, _ in tr], [l for _, l, _ in tr])
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(ctx.emb([x for x, _, _ in tr]), [l for _, l, _ in tr])

    def predict(items, order):
        out = []
        P = clf.predict(ctx.emb([x for x, _, _ in items])) if items else []
        for (x, _, _), p in zip(items, P):
            look = {"lexicon": B.lexicon_level(x), "learned": vmap.get(B.first_word(x))}
            out.append(next((look[k] for k in order if look[k]), int(p)))
        return out

    orders = {"lexicon_then_learned_then_classifier": ("lexicon", "learned"), "learned_then_lexicon_then_classifier": ("learned", "lexicon")}
    val_acc = {k: float(np.mean(np.array(predict(va, o)) == [l for _, l, _ in va])) for k, o in orders.items()}
    pick = max(val_acc, key=val_acc.get)
    y, g = [l for _, l, _ in te], [f for _, _, f in te]
    base = [B.lexicon_level(x) or major for x, _, _ in te]
    new = predict(te, orders[pick])
    acc = lambda yy, pp: float(np.mean(np.asarray(yy) == np.asarray(pp)))
    return dump({"baseline": "verb lexicon (Week 2)", "improved": pick, "val_accuracy": {k: E.r3(v) for k, v in val_acc.items()},
                 "lexicon": E.classification(y, base, g), "improved_test": E.classification(y, new, g),
                 "vs_lexicon_accuracy": E.paired(acc, y, base, new, g)}, f"{OUT}/bloom.json")


# ------------------------------------------------------------------ summary
def summary(ctx):
    j = {k: json.loads((RESULTS / OUT / f"{k}.json").read_text(encoding="utf-8")) for k in ("overlap", "redundancy", "bloom")}
    w2 = {k: json.loads((RESULTS / "week2" / f"{k}.json").read_text(encoding="utf-8")) for k in ("redundancy", "prerequisites")}
    w3p = json.loads((RESULTS / "week3/prerequisites.json").read_text(encoding="utf-8"))["tests"]
    o = j["overlap"]["methods"]
    sel = j["overlap"]["selected_on_validation"]
    rows = [["1 Overlap", "Spearman", W2._v(o["tfidf_cosine"]["test"]["spearman"]["value"]), W2._v(o["maxsim_minilm"]["test"]["spearman"]["value"]),
             f"{W2._v(o[sel]['test']['spearman']['value'])} ({sel})", o[sel]["vs_core_spearman"]["verdict"] if sel != "maxsim_minilm" else "core kept"],
            ["", "ROC-AUC any overlap", W2._v(o["tfidf_cosine"]["test"]["roc_auc"]["value"]), W2._v(o["maxsim_minilm"]["test"]["roc_auc"]["value"]),
             W2._v(o[sel]["test"]["roc_auc"]["value"]), o[sel]["vs_core_auc"]["verdict"] if sel != "maxsim_minilm" else "core kept"],
            ["2 Redundancy", "injected found in top 20", W2._v(w2["redundancy"]["synthetic_injection"]["recall_at_20"]),
             W2._v(j["redundancy"]["minilm"]["synthetic_injection"]["recall_at_20"]), f"{W2._v(j['redundancy']['bge_base']['synthetic_injection']['recall_at_20'])} (bge-base)", "encoder ablation"],
            ["3 Prerequisites", "removal recall / rename false alarm",
             f"{W2._v(w2['prerequisites']['tests']['removal_recall']['value'])} / {W2._v(w2['prerequisites']['tests']['rename_false_alarm']['value'])}",
             f"{W2._v(w3p['removal_recall']['value'])} / {W2._v(w3p['rename_false_alarm']['value'])}", "Week 3 model kept",
             "bge-base prerequisite re-tuning not run (time)"],
            ["5 Bloom", "accuracy", W2._v(j["bloom"]["lexicon"]["accuracy"]["value"]), W2._v(j["bloom"]["lexicon"]["accuracy"]["value"]),
             W2._v(j["bloom"]["improved_test"]["accuracy"]["value"]), j["bloom"]["vs_lexicon_accuracy"]["verdict"]]]
    return dump({"table": {"columns": ["Capability", "Metric (test)", "Week 2 baseline", "Week 3 core", "Week 4 improved", "Verdict (95% CI of difference)"],
                           "rows": rows}}, f"{OUT}/summary.json")


def run(ctx):
    th = json.loads((RESULTS / "week3/thresholds.json").read_text(encoding="utf-8"))
    bge = Embedder(BGE_BASE)
    for fn in (overlap, redundancy):
        print(f"  week4.{fn.__name__}", flush=True)
        fn(ctx, th, bge)
        bge.save()
    print("  week4.bloom", flush=True)
    bloom(ctx)
    summary(ctx)
    ctx.emb.save()
