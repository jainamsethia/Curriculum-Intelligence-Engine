"""Step 6 - Baseline models + initial evaluation.

  A. Course-overlap scoring  : TF-IDF + cosine similarity  (vs. word-overlap Jaccard as a trivial baseline)
  B. Topic classification    : TF-IDF + one-vs-rest logistic regression (vs. dummy prior baseline)
  C. Rule-based extraction   : topic-lexicon keyword matching + regex prerequisites -> JSON comparison report

IDF, the decision threshold and C are fit on TRAIN courses/pairs only; all numbers are reported on TEST.

    python src/baseline.py
"""
import csv, json, re
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, f1_score, hamming_loss,
                             precision_recall_fscore_support, precision_recall_curve, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OVERLAP_THRESHOLD = 0.30   # must match preprocess.py (silver label cut-off)

# Curriculum topic lexicon (baseline "ontology"): canonical topic -> regex over lower-cased text.
# ponytail: hand-written lexicon, misses paraphrases; week 3 replaces it with embedding/BERTopic topics.
TOPIC_LEXICON = {
    "Probability": r"probabilit(y|ies)|random variables?",
    "Statistical Inference": r"statistical inference|estimat(ion|ors?)|maximum likelihood|confidence intervals?",
    "Hypothesis Testing": r"hypothesis test\w*|p-values?|significance tests?",
    "Bayesian Methods": r"bayes(ian)?",
    "Regression": r"regression",
    "Classification": r"classif(ication|iers?)",
    "Clustering": r"cluster(ing|s)?|k-means",
    "Dimensionality Reduction": r"dimensionality reduction|principal component|\bpca\b|singular value|\bsvd\b",
    "Neural Networks": r"neural net(work)?s?|perceptrons?|backprop\w*",
    "Deep Learning": r"deep learning|convolutional|\bcnns?\b|recurrent|\brnns?\b|\blstms?\b",
    "Transformers & LLMs": r"transformers?|attention mechanisms?|large language models?|\bllms?\b|\bgpt\b|\bbert\b",
    "Generative Models": r"generative (models?|adversarial)|\bgans?\b|variational autoencoders?|diffusion models?",
    "Natural Language Processing": r"natural language|\bnlp\b|parsing|language model(s|ing)?|machine translation",
    "Computer Vision": r"computer vision|machine vision|image (processing|recognition)|object recognition",
    "Reinforcement Learning": r"reinforcement learning|markov decision|\bmdps?\b|q-learning|policy gradient",
    "Kernel Methods & SVM": r"kernels?|support vector",
    "Ensemble & Tree Methods": r"boosting|bagging|random forests?|decision trees?|ensembles?",
    "Learning Theory": r"learning theory|generalization|vc[- ]dimension|\bpac\b|rademacher",
    "Graphical Models": r"graphical models?|bayesian networks?|markov random fields?|belief propagation|sum-product",
    "Hidden Markov Models": r"hidden markov|\bhmms?\b",
    "Markov Chains & Stochastic Processes": r"markov chains?|stochastic process(es)?|random walks?|poisson process\w*",
    "Monte Carlo & Simulation": r"monte carlo|simulations?|\bmcmc\b",
    "Linear Algebra": r"linear algebra|matri(x|ces)|eigen\w*|vector spaces?",
    "Calculus & Differential Equations": r"calculus|differential equations?",
    "Optimization": r"optimi[sz]ation|gradient descent|convex|linear programming|lagrang\w*",
    "Numerical Methods": r"numerical (methods?|analysis|linear algebra)|finite differences?|iterative methods?",
    "Algorithms & Complexity": r"algorithms?|np-complete\w*|asymptotic|complexity",
    "Data Structures": r"data structures?|hash(ing| tables?)|heaps?|binary search trees?",
    "Graph Algorithms": r"graph algorithms?|shortest paths?|minimum spanning|network flows?|max(imum)?[- ]flow",
    "Dynamic Programming": r"dynamic programming",
    "Discrete Mathematics": r"combinatori\w*|discrete mathematics|graph theory",
    "Information Theory": r"information theory|entropy|mutual information",
    "Signal Processing": r"signal processing|fourier|z-transforms?|sampling theorem|digital filters?",
    "Time Series": r"time series|autoregressive|\barima?\b|forecasting",
    "Econometrics & Causal Inference": r"econometric\w*|instrumental variables?|panel data|causal|treatment effects?",
    "Data Analysis & Analytics": r"data mining|analytics|data analysis|data science",
    "Search & Planning": r"search algorithms?|heuristic search|planning|constraint satisfaction",
    "Logic & Knowledge Representation": r"propositional logic|first-order logic|knowledge representation",
    "Robotics & Control": r"robot\w*|control systems?|feedback control|motion planning",
    "Game Theory": r"game theory|nash equilibri\w*",
    "Distributed & Parallel Computing": r"distributed (algorithms?|systems?|computing)|parallel (computing|algorithms?)",
    "Databases": r"databases?|\bsql\b",
    "Finance": r"financ\w*|portfolios?|asset pricing|option pricing",
    "Healthcare": r"health ?care|clinical|medical",
    "Ethics & Fairness": r"ethic\w*|fairness",
    "Programming Tools (Python/R/Julia/MATLAB)": r"python|julia|matlab|\br\b (programming|language|software)",
}
LEXICON = {t: re.compile(rf"\b(?:{p})", re.I) for t, p in TOPIC_LEXICON.items()}
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"   # dataviz reference palette


def topics_in(text):
    """{topic: mention count} for every lexicon topic found in the text."""
    return {t: n for t, rx in LEXICON.items() if (n := len(rx.findall(text)))}


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    return float(t[np.argmax(f[:-1])])


def pair_metrics(y, score, silver, thr):
    pred = (score >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"spearman_vs_silver_score": round(spearmanr(score, silver).statistic, 3),
            "roc_auc": round(roc_auc_score(y, score), 3), "avg_precision": round(average_precision_score(y, score), 3),
            "threshold_from_train": round(thr, 3), "precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "accuracy": round(float((pred == y).mean()), 3)}


def compare(a, b, sim, thr):
    """Structured curriculum comparison report (baseline version of POST /compare-curricula)."""
    sim = float(sim)
    ta, tb = topics_in(a["text"]), topics_in(b["text"])
    pa, pb = topics_in(a["prerequisites_text"]), topics_in(b["prerequisites_text"])
    # topics a course's prerequisites assume that the other course does not teach (matters if that one precedes it)
    gaps = [{"course": y["title"], "assumes": t, "not_taught_in": x["title"]}
            for x, y, py, tx in ((a, b, pb, ta), (b, a, pa, tb)) for t in py if t not in tx]
    return {
        "course_a": f'{a["course_number"]} {a["title"]} ({a["offering"]})',
        "course_b": f'{b["course_number"]} {b["title"]} ({b["offering"]})',
        "overlap_score": round(sim, 3), "overlap_percent": f"{sim:.0%}",
        "redundancy_warning": sim >= thr,
        "common_topics": [{"topic": t, "mentions_a": ta[t], "mentions_b": tb[t]} for t in ta if t in tb],
        "unique_to_a": [t for t in ta if t not in tb], "unique_to_b": [t for t in tb if t not in ta],
        "prerequisites": {"a": {"text": a["prerequisites_text"][:300], "course_refs": a["prereq_course_refs"]},
                          "b": {"text": b["prerequisites_text"][:300], "course_refs": b["prereq_course_refs"]}},
        "missing_prerequisites": gaps,
        "a_is_listed_prerequisite_of_b": a["course_number"] in b["prereq_course_refs"],
        "b_is_listed_prerequisite_of_a": b["course_number"] in a["prereq_course_refs"],
        "method": "baseline: TF-IDF cosine + topic lexicon + regex prerequisites",
    }


def main():
    RES.mkdir(exist_ok=True)
    docs = [json.loads(l) for l in open(ROOT / "data/processed/courses.jsonl", encoding="utf-8")]
    idx = {d["id"]: i for i, d in enumerate(docs)}
    train = [d for d in docs if d["split"] == "train"]
    test = [d for d in docs if d["split"] == "test"]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True)
    vec.fit([d["clean_text"] for d in train])                 # IDF from train only
    X = vec.transform([d["clean_text"] for d in docs])
    S = (X @ X.T).toarray()                                   # rows are L2-normalised => cosine
    sets = [set(d["clean_text"].split()) for d in docs]
    jac = lambda i, j: len(sets[i] & sets[j]) / len(sets[i] | sets[j])

    # sanity: a course compared with itself is a full overlap with nothing unique
    self_rep = compare(docs[0], docs[0], S[0, 0], 0.5)
    assert abs(self_rep["overlap_score"] - 1) < 1e-6 and not self_rep["unique_to_a"] and not self_rep["unique_to_b"]

    # ---------------- A. pairwise overlap vs silver labels ----------------
    pairs = list(csv.DictReader(open(ROOT / "data/annotation/pairs_silver.csv", encoding="utf-8")))
    for p in pairs:
        i, j = idx[p["course_a"]], idx[p["course_b"]]
        p["tfidf_cosine"], p["jaccard"] = float(S[i, j]), jac(i, j)
    get = lambda split, k: np.array([float(p[k]) for p in pairs if p["split"] == split])
    y_tr, y_te = get("train", "silver_label").astype(int), get("test", "silver_label").astype(int)
    overlap = {"n_test_pairs": len(y_te), "n_test_positive": int(y_te.sum()),
               "chance_avg_precision": round(float(y_te.mean()), 3)}
    thr = {}
    for m in ("jaccard", "tfidf_cosine"):
        thr[m] = best_f1_threshold(y_tr, get("train", m))
        overlap[m] = pair_metrics(y_te, get("test", m), get("test", "silver_score"), thr[m])

    # gold pairs (0/1/2, see README); gold "overlap" = label >= 1
    man = [r for r in csv.DictReader(open(ROOT / "data/annotation/gold_pairs.csv", encoding="utf-8")) if r["gold_label"].strip()]
    if len(man) >= 10:
        gold = np.array([float(r["gold_label"]) for r in man])
        yg = (gold >= 1).astype(int)
        ij = [(idx[r["course_a"]], idx[r["course_b"]]) for r in man]
        scores = {"tfidf_cosine": np.array([S[i, j] for i, j in ij]), "jaccard": np.array([jac(i, j) for i, j in ij]),
                  "silver_score": np.array([float(r["silver_score"]) for r in man])}
        g = {"n_pairs": len(man), "n_gold_overlap": int(yg.sum()),
             "label_counts": {str(k): int((gold == k).sum()) for k in (0, 1, 2)},
             "annotator": sorted({r["annotator"] for r in man}),
             "note": "single annotator; pairs are stratified by silver score, not a random sample"}
        for m, sc in scores.items():
            g[m] = {"spearman_vs_gold": round(spearmanr(sc, gold).statistic, 3),
                    "roc_auc": round(roc_auc_score(yg, sc), 3), "avg_precision": round(average_precision_score(yg, sc), 3)}
        for m, pred in (("tfidf_cosine", scores["tfidf_cosine"] >= thr["tfidf_cosine"]),
                        ("silver_score", scores["silver_score"] >= OVERLAP_THRESHOLD)):
            pr, rc, f1, _ = precision_recall_fscore_support(yg, pred.astype(int), average="binary", zero_division=0)
            g[m].update({"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3)})
        overlap["manual_gold"] = g

    with open(RES / "pair_scores_test.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["course_a", "course_b", "title_a", "title_b", "silver_score", "silver_label", "tfidf_cosine", "jaccard"]
        w = csv.writer(f); w.writerow(cols + ["tfidf_pred"])
        for p in sorted((p for p in pairs if p["split"] == "test"), key=lambda p: -p["tfidf_cosine"]):
            w.writerow([p[c] if not isinstance(p[c], float) else round(p[c], 4) for c in cols] + [int(p["tfidf_cosine"] >= thr["tfidf_cosine"])])

    # ---------------- B. multi-label topic classification ----------------
    mlb = MultiLabelBinarizer().fit([d["labels"] for d in docs])
    Xtr, Xte = X[[idx[d["id"]] for d in train]], X[[idx[d["id"]] for d in test]]
    Ytr, Yte = mlb.transform([d["labels"] for d in train]), mlb.transform([d["labels"] for d in test])
    grid = GridSearchCV(OneVsRestClassifier(LogisticRegression(class_weight="balanced", max_iter=2000)),
                        {"estimator__C": [1, 10, 100]}, scoring="f1_micro", cv=GroupKFold(3))
    grid.fit(Xtr, Ytr, groups=[d["course_number"] for d in train])
    top2 = np.isin(np.arange(Ytr.shape[1]), np.argsort(-Ytr.sum(0))[:2]).astype(int)
    preds = {"always_top2_frequent_labels": np.tile(top2, (len(test), 1)), "tfidf_logreg": grid.predict(Xte)}
    present = np.flatnonzero(Yte.sum(0))                      # macro-F1 only over labels that occur in test
    topic = {"labels": list(mlb.classes_), "n_train": len(train), "n_test": len(test),
             "chosen_C (3-fold grouped CV on train)": grid.best_params_["estimator__C"]}
    for name, P in preds.items():
        topic[name] = {"micro_f1": round(f1_score(Yte, P, average="micro", zero_division=0), 3),
                       "macro_f1": round(f1_score(Yte, P, labels=present, average="macro", zero_division=0), 3),
                       "samples_f1": round(f1_score(Yte, P, average="samples", zero_division=0), 3),
                       "hamming_loss": round(hamming_loss(Yte, P), 3)}
    P = preds["tfidf_logreg"]
    rep = classification_report(Yte, P, target_names=mlb.classes_, output_dict=True, zero_division=0)
    topic["per_label_tfidf_logreg"] = {k: {m: round(v[m], 3) for m in ("precision", "recall", "f1-score", "support")}
                                       for k, v in rep.items() if k in mlb.classes_}
    with open(RES / "topic_predictions_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id", "title", "gold_labels", "predicted_labels"])
        for d, row in zip(test, P):
            w.writerow([d["id"], d["title"], "; ".join(d["labels"]), "; ".join(mlb.classes_[row.astype(bool)])])

    # ---------------- C. rule-based extraction + example reports ----------------
    tc = [len(topics_in(d["text"])) for d in docs]
    extraction = {"lexicon_topics": len(TOPIC_LEXICON), "avg_topics_per_course": round(float(np.mean(tc)), 2),
                  "courses_with_>=3_topics": int(sum(n >= 3 for n in tc)),
                  "courses_with_prerequisites_found": sum(bool(d["prerequisites_text"]) for d in docs),
                  "courses_with_prereq_course_refs": sum(bool(d["prereq_course_refs"]) for d in docs),
                  "courses_with_outcomes_found": sum(bool(d["outcomes"]) for d in docs)}
    tp = sorted(((i, j) for i, j in combinations([idx[d["id"]] for d in test], 2)), key=lambda ij: -S[ij])
    showcase = tp[:3] + [tp[len(tp) // 2], tp[-1]]           # 3 most similar, a median and the least similar test pair
    ml = [i for i, d in enumerate(docs) if re.search(r"^(introduction to )?machine learning$", d["title"], re.I)]
    dl = [i for i, d in enumerate(docs) if re.search(r"deep learning", d["title"], re.I)]
    if ml and dl:
        showcase.insert(0, (ml[0], dl[0]))                   # the proposal's own example: ML vs Deep Learning
    reports = [compare(docs[i], docs[j], S[i, j], thr["tfidf_cosine"]) for i, j in showcase]
    (RES / "example_reports.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics = {"overlap_scoring": overlap, "topic_classification": topic, "rule_based_extraction": extraction,
               "tfidf": {"vocab_size": len(vec.vocabulary_), "ngram_range": [1, 2], "min_df": 2, "max_df": 0.8,
                         "sublinear_tf": True, "fit_on": "train courses only"}}
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---------------- figures ----------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
    te = [p for p in pairs if p["split"] == "test"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for lab, col, name in ((0, BLUE, "Silver: no overlap"), (1, ORANGE, "Silver: overlap")):
        pts = [p for p in te if int(p["silver_label"]) == lab]
        ax.scatter([float(p["silver_score"]) for p in pts], [p["tfidf_cosine"] for p in pts], s=22, c=col,
                   edgecolors="white", linewidths=0.8, label=f"{name} (n={len(pts)})", zorder=3)
    ax.axhline(thr["tfidf_cosine"], color=INK2, lw=1, ls="--")
    ax.text(1.0, thr["tfidf_cosine"], f" threshold from train = {thr['tfidf_cosine']:.2f}", color=INK2,
            fontsize=8, va="bottom", ha="right")
    ax.set(xlabel="Silver overlap score (IDF-weighted Jaccard of OCW topic tags)", ylabel="TF-IDF cosine similarity")
    ax.set_title(f"Test pairs: TF-IDF cosine vs silver overlap (Spearman ρ = {overlap['tfidf_cosine']['spearman_vs_silver_score']})",
                 fontsize=10, color=INK, loc="left")
    ax.grid(color="#ecebe6", lw=0.6, zorder=0); ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(RES / "fig_similarity_vs_silver.png"); plt.close(fig)

    lab = sorted(((k, v) for k, v in topic["per_label_tfidf_logreg"].items() if v["support"]),  # skip labels absent from test
                 key=lambda kv: kv[1]["f1-score"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    y = np.arange(len(lab))
    ax.barh(y, [v["f1-score"] for _, v in lab], height=0.6, color=BLUE, zorder=3)
    for k, (_, v) in enumerate(lab):
        ax.text(v["f1-score"] + 0.01, k, f'{v["f1-score"]:.2f} (n={int(v["support"])})', va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [k for k, _ in lab], fontsize=8); ax.set_xlim(0, 1.15); ax.set_xlabel("Test F1")
    ax.set_title("Topic classification (TF-IDF + logistic regression): F1 per label", fontsize=10, color=INK, loc="left")
    ax.grid(axis="x", color="#ecebe6", lw=0.6, zorder=0)
    fig.tight_layout(); fig.savefig(RES / "fig_topic_f1_per_label.png"); plt.close(fig)

    print(json.dumps({k: metrics[k] for k in ("overlap_scoring", "rule_based_extraction")}, indent=2))
    print(json.dumps({k: v for k, v in topic.items() if k != "per_label_tfidf_logreg"}, indent=2))


if __name__ == "__main__":
    main()
