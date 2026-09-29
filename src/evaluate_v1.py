"""Week 3 - evaluation of pipeline v1 against the Week 2 baseline (TF-IDF).

  A. Same-course retrieval : given a course (title, code, prerequisites hidden), rank all other courses; the other versions of the
                             same course (other years / programmes) should come first.
  B. Gold-pair overlap     : distinct-course pairs labelled 0/1/2 by an LLM annotator panel (data/annotation/nmims_gold_pairs.csv).
  C. Unit alignment        : recover known unit correspondences between two versions of a course from unit descriptions alone.
  D. Bloom's taxonomy      : outcome text -> level, against the faculty K/L tags printed in the syllabi.
  E. Prerequisites         : extraction statistics + panel-graded precision of the prerequisite -> course matching.
  F. Week 2 continuity     : embeddings vs TF-IDF on the MIT OCW gold pairs used in Week 2.

Course-level train/test split grouped by course name (seed 42). TF-IDF IDF, the unit threshold and the Bloom classifier are fit on
train courses only. Writes results/metrics_v1.json, figures, and data/processed/calibration.json (used by pipeline.py).

    python src/evaluate_v1.py
"""
import csv, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, KFold
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P
from pipeline import text_of

ROOT = Path(__file__).resolve().parents[1]
RES, ANN = ROOT / "results", ROOT / "data/annotation"
SEED = 42
BLUE, ORANGE, GREEN, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"   # dataviz reference palette
TAU_PREREQ_RESOLVE, TAU_PREREQ_COVER = 0.75, 0.5      # 0.75 chosen from the prerequisite audit (see E); 0.5 = phrase found in the other course


def r3(x):
    return round(float(x), 3)


def tfidf_model():
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)


# ------------------------------------------------------------------ A. retrieval
def retrieval(courses, train_keys, emb, tau):
    # keep one record per (course, identical text): re-used syllabi would make retrieval trivially easy
    seen, recs = set(), []
    for c in courses:
        t = (c["course_key"], re.sub(r"\W+", "", text_of(c).lower()))
        if len(t[1]) > 100 and t not in seen:
            seen.add(t); recs.append(c)
    fam = defaultdict(list)
    for i, c in enumerate(recs):
        fam[c["course_key"]].append(i)
    keys = np.array([c["course_key"] for c in recs])
    queries = [i for i, c in enumerate(recs) if c["course_key"] not in train_keys and len(fam[c["course_key"]]) >= 2]

    tf = tfidf_model().fit([text_of(c) for c in recs if c["course_key"] in train_keys])
    T = tf.transform([text_of(c) for c in recs])
    S_tfidf = (T @ T.T).toarray()
    V = np.stack([P.course_vec(emb, c) for c in recs])
    S_doc = V @ V.T
    Vt = emb([text_of(c) for c in recs])                        # ablation: one truncated embedding of the concatenated text
    S_trunc = Vt @ Vt.T
    U = [emb([t for _, t in P.unit_texts(c)]) for c in recs]
    S_soft = np.zeros_like(S_doc); S_hard = np.zeros_like(S_doc)
    for i in queries:                                           # unit alignment only for the query rows
        for j in range(len(recs)):
            if i != j:
                al = P.align(U[i], U[j], tau)
                S_soft[i, j], S_hard[i, j] = al["soft"], al["overlap"]
    methods = {"TF-IDF cosine (Week 2 baseline)": S_tfidf, "Embedding, truncated concatenation": S_trunc,
               "Embedding, mean of chunks": S_doc, "Unit alignment, soft coverage": S_soft,
               "Unit alignment, overlap fraction": S_hard + 1e-3 * S_doc}
    # 'heavily revised': even the closest other version of the query has TF-IDF similarity < 0.6
    hard_q = {i for i in queries if max(S_tfidf[i, j] for j in fam[keys[i]] if j != i) < 0.6}
    out = {}
    for name, S in methods.items():
        rr, r1, r5, ap, hrr, hr1 = [], [], [], [], [], []
        for i in queries:
            s = S[i].copy(); s[i] = -np.inf
            order = np.argsort(-s)
            rel = np.isin(order, [j for j in fam[keys[i]] if j != i])
            hits = np.flatnonzero(rel)
            rr.append(1 / (hits[0] + 1)); r1.append(bool(rel[0])); r5.append(bool(rel[:5].any()))
            ap.append(np.mean([(k + 1) / (h + 1) for k, h in enumerate(hits)]))
            if i in hard_q:
                hrr.append(rr[-1]); hr1.append(r1[-1])
        out[name] = {"MRR": r3(np.mean(rr)), "Recall@1": r3(np.mean(r1)), "Recall@5": r3(np.mean(r5)), "MAP": r3(np.mean(ap)),
                     "MRR_heavily_revised": r3(np.mean(hrr)), "Recall@1_heavily_revised": r3(np.mean(hr1))}
    return {"records": len(recs), "queries": len(queries), "heavily_revised_queries": len(hard_q), "results": out}


# ------------------------------------------------------------------ B. gold pairs
def cv_logistic(X, y, folds=5):
    """5-fold cross-validated probabilities of a standardised logistic regression."""
    cv = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=SEED).split(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        cv[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return cv


def gold_eval(courses, emb, tfidf, tau, cal):
    f = ANN / "nmims_gold_pairs.csv"
    if not f.exists():
        return None, None
    gold = list(csv.DictReader(open(f, encoding="utf-8")))
    by_id = {c["id"]: c for c in courses}
    y = np.array([float(g["gold_label"]) for g in gold])
    yb = (y >= 1).astype(int)
    A = [by_id[g["id_a"]] for g in gold]; B = [by_id[g["id_b"]] for g in gold]
    Ta, Tb = tfidf.transform([text_of(c) for c in A]), tfidf.transform([text_of(c) for c in B])
    tf = np.asarray(Ta.multiply(Tb).sum(1)).ravel()
    doc, soft, hard = [], [], []
    for a, b in zip(A, B):
        al = P.align(emb([t for _, t in P.unit_texts(a)]), emb([t for _, t in P.unit_texts(b)]), tau)
        doc.append(float(P.course_vec(emb, a) @ P.course_vec(emb, b))); soft.append(al["soft"]); hard.append(al["overlap"])
    doc, soft, hard = map(np.array, (doc, soft, hard))
    feats = {"tfidf_cosine": tf, "doc_cosine": doc, "unit_overlap": hard, "soft_coverage": soft}
    X_emb = np.c_[doc, hard, soft]
    X_all = np.c_[tf, doc, hard, soft]
    cv_emb, cv_all = cv_logistic(X_emb, yb), cv_logistic(X_all, yb)
    mu, sd = X_all.mean(0), X_all.std(0) + 1e-9                  # final model on all gold pairs, used by pipeline.py
    full = LogisticRegression(C=1.0, max_iter=2000).fit((X_all - mu) / sd, yb)
    cal["logit"] = {"features": P.FEATURES, "mean": mu.tolist(), "scale": sd.tolist(), "coef": full.coef_[0].tolist(), "intercept": float(full.intercept_[0])}
    scores = {"TF-IDF cosine (Week 2 baseline)": tf, "Embedding doc cosine (mean of chunks)": doc, "Unit alignment, soft coverage": soft,
              "Unit alignment, overlap fraction": hard, "Calibrated, embedding features only (5-fold CV)": cv_emb,
              "Calibrated, TF-IDF + embedding features (5-fold CV)": cv_all}
    res = {}
    for k, s in scores.items():
        res[k] = {"spearman": r3(spearmanr(s, y).statistic), "roc_auc": r3(roc_auc_score(yb, s)), "avg_precision": r3(average_precision_score(yb, s))}
        if k.startswith("Calibrated"):
            pred = s >= 0.5
            tp = int((pred & (yb == 1)).sum())
            res[k].update({"precision@0.5": r3(tp / max(pred.sum(), 1)), "recall@0.5": r3(tp / max(yb.sum(), 1)), "f1@0.5": r3(f1_score(yb, pred))})
    # where do the methods go wrong? list the gold-negative pairs with the highest TF-IDF / embedding score
    order_e = np.argsort(-doc)
    fp = [{"a": gold[i]["course_a"], "b": gold[i]["course_b"], "tfidf": r3(tf[i]), "embedding": r3(doc[i]), "gold": int(y[i])} for i in order_e[:60] if y[i] == 0][:6]
    out = {"n_pairs": len(y), "label_counts": {str(int(k)): int((y == k).sum()) for k in (0, 1, 2)}, "positive_rate": r3(yb.mean()),
           "chance_avg_precision": r3(yb.mean()), "results": res, "highest_scoring_gold_negatives": fp}
    g = ANN / "nmims_gold_annotators.csv"
    if g.exists():
        rows = list(csv.DictReader(open(g, encoding="utf-8")))
        k = [c for c in rows[0] if c.startswith("annotator_")]
        out["annotator_agreement"] = {"n_annotators": len(k), "exact_all_agree": r3(np.mean([len({r[a] for a in k}) == 1 for r in rows])),
                                      "pairwise_weighted_kappa": [r3(cohen_kappa_score([int(r[a]) for r in rows], [int(r[b]) for r in rows], weights="linear"))
                                                                  for i, a in enumerate(k) for b in k[i + 1:]]}
    return out, (tf, doc, soft, hard, cv_all, y)


# ------------------------------------------------------------------ C. unit alignment
def unit_alignment(courses, train_keys, emb, max_pairs_per_family=6):
    """Two versions of the same course: units with the same title are the known correspondences. Match on the descriptions only
    (title removed) and check that each unit finds its counterpart among all units of the other version."""
    rng = random.Random(SEED)
    norm_title = lambda u: re.sub(r"[^a-z0-9]+", "", u["title"].lower())

    def body(u):
        t = u["text"]
        if u["title"] and t.lower().startswith(u["title"].lower()):
            t = t[len(u["title"]):]
        return t.strip(" :.-–")

    tf = tfidf_model().fit([b for c in courses if c["course_key"] in train_keys for b in map(body, c["units"]) if b])
    fam = defaultdict(dict)
    for c in courses:
        if c["course_key"] not in train_keys:
            fam[c["course_key"]].setdefault(re.sub(r"\W+", "", text_of(c).lower()), c)
    pairs = []
    for k, d in fam.items():
        vs = list(d.values())
        cand = [(a, b) for i, a in enumerate(vs) for b in vs[i + 1:]]
        rng.shuffle(cand)
        pairs += cand[:max_pairs_per_family]
    rr = {"TF-IDF": {"all": [], "revised": []}, "Embedding": {"all": [], "revised": []}}
    chance, n_pairs = [], 0
    paired = Counter()                     # top-1 outcomes per revised unit: who is right when they disagree
    for a, b in pairs:
        ta, tb = defaultdict(list), defaultdict(list)
        for i, u in enumerate(a["units"]):
            if len(u["title"].split()) >= 2:
                ta[norm_title(u)].append(i)
        for j, u in enumerate(b["units"]):
            if len(u["title"].split()) >= 2:
                tb[norm_title(u)].append(j)
        common = [t for t in ta if t in tb and len(ta[t]) == 1 and len(tb[t]) == 1]
        cand_b = [j for j, u in enumerate(b["units"]) if len(body(u).split()) >= 5]
        if len(cand_b) < 3 or not common:
            continue
        n_pairs += 1
        Bb = [body(b["units"][j]) for j in cand_b]
        Eb, Tb = emb(Bb), tf.transform(Bb)
        for t in common:
            i, j = ta[t][0], tb[t][0]
            qa = body(a["units"][i])
            if j not in cand_b or len(qa.split()) < 5:
                continue
            pos = cand_b.index(j)
            revised = re.sub(r"\W+", "", qa.lower()) != re.sub(r"\W+", "", Bb[pos].lower())
            ranks = {}
            for name, sc in (("Embedding", Eb @ emb([qa])[0]), ("TF-IDF", (Tb @ tf.transform([qa]).T).toarray().ravel())):
                ranks[name] = 1 + int((sc > sc[pos]).sum()) + 0.5 * (int((sc == sc[pos]).sum()) - 1)   # ties count half
                rr[name]["all"].append(1 / ranks[name])
                if revised:
                    rr[name]["revised"].append(1 / ranks[name])
            if revised:
                paired[("emb_right" if ranks["Embedding"] == 1 else "emb_wrong") + "_" + ("tfidf_right" if ranks["TF-IDF"] == 1 else "tfidf_wrong")] += 1
            chance.append(1 / len(cand_b))
    n_rev = len(rr["TF-IDF"]["revised"])
    top1 = lambda x: r3(np.mean([v >= 1.0 for v in x]))
    from scipy.stats import binomtest
    b, c = paired["emb_right_tfidf_wrong"], paired["emb_wrong_tfidf_right"]        # exact McNemar test on the discordant units
    return {"version_pairs": n_pairs, "aligned_units": len(chance), "units_with_revised_text": n_rev, "chance_MRR_approx": r3(np.mean(chance)),
            "revised_units_top1_paired": dict(paired), "mcnemar_exact_p": r3(binomtest(b, b + c, 0.5).pvalue) if b + c else None,
            "results": {m: {"top1_all": top1(v["all"]), "MRR_all": r3(np.mean(v["all"])), "top1_revised_units": top1(v["revised"]),
                            "MRR_revised_units": r3(np.mean(v["revised"]))} for m, v in rr.items()}}


# ------------------------------------------------------------------ D. Bloom
def bootstrap_ci(correct, n=2000):
    rng = np.random.RandomState(SEED)
    m = [rng.choice(correct, len(correct)).mean() for _ in range(n)]
    return [r3(np.percentile(m, 2.5)), r3(np.percentile(m, 97.5))]


def bloom_eval(courses, train_keys, emb):
    def collect(keep):
        X, y, g = P.bloom_training_data([c for c in courses if keep(c["course_key"])])
        return X, np.array(y), g
    Xtr, ytr, gtr = collect(lambda k: k in train_keys)
    Xte, yte, _ = collect(lambda k: k not in train_keys)
    tr_set = set(Xtr)
    keep = [i for i, t in enumerate(Xte) if t not in tr_set]                     # no outcome text seen in training
    Xte, yte = [Xte[i] for i in keep], yte[keep]
    prior = Counter(ytr).most_common(1)[0][0]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(Xtr)
    clf = P.BloomClassifier(emb).fit(Xtr, ytr, gtr)
    pred = {"Majority class": np.full(len(yte), prior),
            "Hand-written verb lexicon": np.array([P.bloom_rule(t) or prior for t in Xte]),
            "Learned verb map only (train majority level per verb)": np.array([clf.vmap[P.first_word(t)][0] if P.first_word(t) in clf.vmap else prior for t in Xte]),
            "TF-IDF + logistic regression": LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(tf.transform(Xtr), ytr).predict(tf.transform(Xte)),
            "Embedding + logistic regression": LogisticRegression(C=4, max_iter=3000, class_weight="balanced").fit(emb(Xtr), ytr).predict(emb(Xte)),
            "Pipeline v1 (verb look-ups, embedding fallback)": np.array([p["level"] for p in clf.predict(Xte)])}
    out = {}
    for k, p in pred.items():
        out[k] = {"accuracy": r3((p == yte).mean()), "accuracy_95ci": bootstrap_ci((p == yte).astype(float)), "within_1_level": r3((abs(p - yte) <= 1).mean()),
                  "macro_f1": r3(f1_score(yte, p, average="macro", labels=sorted(set(yte)), zero_division=0))}
    how = Counter(p["method"] for p in clf.predict(Xte))
    final = pred["Pipeline v1 (verb look-ups, embedding fallback)"]
    wrong = final != yte
    dist = Counter(int(d) for d in abs(final - yte)[wrong])
    conf = Counter((P.BLOOM[int(t)], P.BLOOM[int(q)]) for t, q in zip(yte[wrong], final[wrong])).most_common(3)
    return {"train_outcomes": len(Xtr), "test_outcomes": len(Xte), "test_label_distribution": {str(k): int(v) for k, v in sorted(Counter(yte).items())},
            "lookup_order_chosen_by_train_cv": list(clf.order), "train_cv_evidence": clf.cv,
            "pipeline_method_usage_on_test": dict(how), "pipeline_errors": {"n_errors": int(wrong.sum()), "by_level_distance": {str(k): v for k, v in sorted(dist.items())},
                                                                          "most_common_confusions": [{"faculty_tag": t, "predicted": q, "count": n} for (t, q), n in conf]},
            "results": out}, (yte, pred["Pipeline v1 (verb look-ups, embedding fallback)"])


# ------------------------------------------------------------------ E. prerequisites
def prereq_stats(pipe):
    cat, cal = pipe.catalog, pipe.cal
    n = Counter()
    for c in pipe.courses:
        ph = P.split_prereq(c.get("prerequisites", ""))
        n["courses"] += 1
        if not ph:
            n["no_prerequisite_listed"] += 1
            continue
        n["courses_with_prerequisites"] += 1
        for p in P.extract_prereqs(c, cat, cal):
            n["phrases"] += 1
            n["resolved_to_a_course"] += p["resolved_course"] is not None
            n["schedule_ok"] += p.get("offered_before") is True
            n["schedule_violation"] += p.get("offered_before") is False
    return dict(n)


def prereq_audit():
    f = ANN / "prereq_audit.csv"
    if not f.exists():
        return None
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    ok = lambda rs: r3(sum(r["verdict"] in ("correct", "acceptable") for r in rs) / max(len(rs), 1))
    bands = {}
    for lo, hi in ((0, .6), (.6, .75), (.75, .9), (.9, 1.01)):
        b = [r for r in rows if lo <= float(r["similarity"]) < hi]
        bands[f"{lo}-{min(hi, 1)}"] = {"n": len(b), "precision_correct_or_acceptable": ok(b), "correct": sum(r["verdict"] == "correct" for r in b)}
    conf = [r for r in rows if float(r["similarity"]) >= TAU_PREREQ_RESOLVE]
    return {"n_items": len(rows), "verdicts": dict(Counter(r["verdict"] for r in rows)), "graders_agree": r3(np.mean([r["agree"] == "1" for r in rows])),
            "precision_all_candidates_at_0.5": ok(rows), "precision_at_chosen_threshold": {"threshold": TAU_PREREQ_RESOLVE, "n": len(conf), "precision": ok(conf)},
            "by_similarity_band": bands,
            "note": "threshold chosen from this audit, so the precision at the chosen threshold is optimistic"}


# ------------------------------------------------------------------ F. MIT continuity
def mit_continuity(emb):
    mit = {}
    for l in open(ROOT / "data/processed/courses.jsonl", encoding="utf-8"):
        d = json.loads(l); mit[d["id"]] = d
    gold = [g for g in csv.DictReader(open(ANN / "gold_pairs.csv", encoding="utf-8")) if g["gold_label"].strip()]
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True).fit([d["clean_text"] for d in mit.values() if d["split"] == "train"])

    def chunks(d):
        sents = re.split(r"(?<=[.!?])\s+", d["description"] + " " + d["syllabus_text"])
        groups = [" ".join(sents[i:i + 3]) for i in range(0, len(sents), 3)]
        return [g for g in groups if len(g.split()) > 4][:40] + d["lecture_topics"][:60]

    y = np.array([float(g["gold_label"]) for g in gold])
    tfc, doc, soft = [], [], []
    for g in gold:
        a, b = mit[g["course_a"]], mit[g["course_b"]]
        tfc.append(float(tf.transform([a["clean_text"]]).multiply(tf.transform([b["clean_text"]])).sum()))
        Ea, Eb = emb(chunks(a)), emb(chunks(b))
        va, vb = Ea.mean(0), Eb.mean(0)
        doc.append(float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb))))
        soft.append(P.align(Ea, Eb, 0.5)["soft"])
    yb = (y >= 1).astype(int)
    res = {k: {"spearman": r3(spearmanr(s, y).statistic), "roc_auc": r3(roc_auc_score(yb, s)), "avg_precision": r3(average_precision_score(yb, s))}
           for k, s in {"TF-IDF cosine (Week 2 baseline)": tfc, "Embedding doc cosine": doc, "Unit alignment, soft coverage": soft}.items()}
    return {"n_pairs": len(y), "results": res}


# ------------------------------------------------------------------ figures
def figures(gold_arr, bloom_arr, retr, align_res):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False})
    if gold_arr is not None:
        tf, doc, soft, hard, cv, y = gold_arr
        fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.6), dpi=160)
        for ax, s, name in ((axs[0], tf, "TF-IDF cosine (Week 2)"), (axs[1], doc, "Embedding cosine (v1)"), (axs[2], cv, "Calibrated hybrid P(overlap), CV")):
            rng = np.random.RandomState(0)
            for lab, col in ((0, BLUE), (1, GREEN), (2, ORANGE)):
                m = y == lab
                if m.any():
                    ax.scatter(y[m] + rng.uniform(-0.14, 0.14, m.sum()), s[m], s=16, c=col, edgecolors="white", linewidths=0.5, zorder=3, label=f"label {lab} (n={m.sum()})")
            ax.set_xticks([0, 1, 2], ["none", "partial", "substantial"]); ax.set_xlabel("Gold overlap label"); ax.set_ylabel(name, fontsize=8.5)
            ax.set_title(f"Spearman {spearmanr(s, y).statistic:.2f}, AUC {roc_auc_score((y >= 1).astype(int), s):.2f}", fontsize=9, loc="left", color=INK)
            ax.grid(axis="y", color="#ecebe6", lw=0.6, zorder=0)
        fig.tight_layout(); fig.savefig(RES / "fig_gold_scores.png"); plt.close(fig)
    yte, yp = bloom_arr
    cm = confusion_matrix(yte, yp, labels=[1, 2, 3, 4, 5, 6])
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=160)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    names = [P.BLOOM[i] for i in range(1, 7)]
    ax.set_xticks(range(6), names, rotation=35, ha="right", fontsize=8); ax.set_yticks(range(6), names, fontsize=8)
    for i in range(6):
        for j in range(6):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Faculty tag (K/L)"); ax.set_title("Bloom level, held-out courses", fontsize=9.5, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(RES / "fig_bloom_confusion.png"); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10.4, 3.4), dpi=160)
    names = list(retr["results"]); w = 0.36
    for k, (m, col) in enumerate((("Recall@1", BLUE), ("MRR_heavily_revised", ORANGE))):
        axs[0].barh(np.arange(len(names)) + (k - 0.5) * w, [retr["results"][n][m] for n in names], height=w, color=col, zorder=3,
                    label="Recall@1 (all queries)" if m == "Recall@1" else "MRR, heavily revised versions")
    axs[0].set_yticks(range(len(names)), names, fontsize=7.5); axs[0].invert_yaxis(); axs[0].set_xlim(0, 1.5)
    axs[0].grid(axis="x", color="#ecebe6", lw=0.6, zorder=0); axs[0].legend(frameon=False, fontsize=7.5, loc="center right")
    axs[0].set_title("Same-course retrieval (test courses)", fontsize=9.5, loc="left", color=INK)
    ms = list(align_res["results"])
    for k, (m, col, lab) in enumerate((("top1_all", BLUE, "all aligned units"), ("top1_revised_units", ORANGE, "units whose text was revised"))):
        axs[1].bar(np.arange(len(ms)) + (k - 0.5) * 0.36, [align_res["results"][x][m] for x in ms], width=0.36, color=col, zorder=3, label=lab)
    axs[1].set_xticks(range(len(ms)), ms); axs[1].set_ylim(0, 1.05); axs[1].grid(axis="y", color="#ecebe6", lw=0.6, zorder=0)
    axs[1].legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2); axs[1].set_ylabel("top-1 accuracy")
    axs[1].set_title("Unit alignment between course versions", fontsize=9.5, loc="left", color=INK)
    fig.tight_layout(); fig.savefig(RES / "fig_retrieval_alignment.png"); plt.close(fig)


def main():
    RES.mkdir(exist_ok=True)
    pipe = P.Pipeline()
    courses, emb = pipe.courses, pipe.emb
    keys = sorted({c["course_key"] for c in courses})
    tr_i, _ = next(GroupShuffleSplit(1, test_size=0.3, random_state=SEED).split(keys, groups=keys))
    train_keys = {keys[i] for i in tr_i}
    # unit-match threshold: 99th percentile of the best match between units of two different courses of the SAME department
    # (train courses only, no labels). Inside a domain almost every unit has some loosely related unit in another course, so a
    # cross-department threshold (95th pct = 0.36) would call nearly everything 'common'.
    rng = np.random.RandomState(SEED)
    dept_of = lambda c: (re.search(r"7\d\d([A-Z]{2})", c["code"]) or [None, None])[1]
    by_dept = defaultdict(list)
    for c in {c["course_key"]: c for c in courses if c["course_key"] in train_keys and P.unit_texts(c)}.values():
        if dept_of(c):
            by_dept[dept_of(c)].append(c)
    pools = [v for v in by_dept.values() if len(v) >= 2]
    best = []
    for _ in range(3000):
        pool = pools[rng.randint(len(pools))]
        a, b = rng.choice(len(pool), 2, replace=False)
        best += P.align(emb([t for _, t in P.unit_texts(pool[a])]), emb([t for _, t in P.unit_texts(pool[b])]), 1)["best_a"].tolist()
    tau = float(np.quantile(best, 0.99))
    cal = {**P.DEFAULT_CAL, "tau_unit": round(tau, 3), "tau_prereq_resolve": TAU_PREREQ_RESOLVE, "tau_prereq_cover": TAU_PREREQ_COVER,
           "tfidf_fit_courses": sorted(train_keys),
           "tau_unit_source": "99th percentile of the best unit match between two different courses of the same department (train courses)"}
    print("tau_unit", round(tau, 3), "from", len(best), "unit comparisons", flush=True)
    results = {"split": {"train_courses": len(train_keys), "test_courses": len(keys) - len(train_keys)}, "tau_unit": cal["tau_unit"]}
    results["A_retrieval"] = retrieval(courses, train_keys, emb, tau); print("A done", flush=True)
    tfidf_train = tfidf_model().fit([text_of(c) for c in courses if c["course_key"] in train_keys])
    results["B_gold_pairs"], gold_arr = gold_eval(courses, emb, tfidf_train, tau, cal); print("B done", flush=True)
    results["C_unit_alignment"] = unit_alignment(courses, train_keys, emb); print("C done", flush=True)
    results["D_bloom"], bloom_arr = bloom_eval(courses, train_keys, emb); print("D done", flush=True)
    P.CALIB.write_text(json.dumps(cal, indent=1))
    pipe.cal = P.load_calibration()
    results["E_prerequisites"] = {"extraction": prereq_stats(pipe), "panel_audit": prereq_audit()}
    results["F_mit_continuity"] = mit_continuity(emb)
    emb.save()
    if gold_arr is not None:
        figures(gold_arr, bloom_arr, results["A_retrieval"], results["C_unit_alignment"])
    (RES / "metrics_v1.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
