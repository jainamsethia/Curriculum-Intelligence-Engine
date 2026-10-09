"""Topic map v1: which topics grow or shrink across academic years?

Every syllabus unit is embedded (all-MiniLM-L6-v2), the units are clustered with k-means, each cluster is named by class-based TF-IDF
(the BERTopic recipe without the dependency), and the share of units per academic year is compared between an early and a late period.
Identical unit texts are counted once per academic year, so a syllabus shared by several programmes is not counted several times.

    python src/topic_trends.py
"""
import csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
K, SEED = 40, 42
# computing programmes only: the archive holds different programmes in different years, which would otherwise masquerade as topic trends
COMPUTING = re.compile(r"comput|information technology|artificial|data scien|cyber|\bIT\b|\bAI\b|\bDS\b|CSBS|CSE|\bCE\b", re.I)
EARLY, LATE = ("2021-22", "2022-23"), ("2025-26", "2026-27")
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"


def main():
    pipe = P.Pipeline()
    seen, units = set(), []
    for c in pipe.courses:
        if not COMPUTING.search(re.split(r"(?i)\b(?:except|excluding)\b", c["program"])[0]):      # 'All programmes except CSBS' is not a computing programme
            continue
        for u in c["units"]:
            key = (c["academic_year"], re.sub(r"\W+", "", u["text"].lower()))
            if len(u["text"].split()) >= 8 and key not in seen and c["academic_year"] >= "2021-22":
                seen.add(key); units.append((c["academic_year"], c["course_name"], u))
    E = pipe.emb([u["text"] for _, _, u in units])
    pipe.emb.save()
    lab = KMeans(K, n_init=3, random_state=SEED).fit_predict(E)
    docs = [" ".join(u["text"] for (_, _, u), l in zip(units, lab) if l == k) for k in range(K)]
    cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=3, max_df=0.5, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b")
    X = cv.fit_transform(docs).toarray().astype(float)
    tf = X / np.maximum(X.sum(1, keepdims=True), 1)
    ctfidf = tf * np.log(1 + X.sum() / np.maximum(X.sum(0), 1) / K)            # class-based TF-IDF
    terms = np.array(cv.get_feature_names_out())
    names = [", ".join(terms[np.argsort(-ctfidf[k])[:5]]) for k in range(K)]
    ays = sorted({a for a, _, _ in units})
    cnt = {a: Counter(l for (aa, _, _), l in zip(units, lab) if aa == a) for a in ays}
    share = {a: np.array([cnt[a][k] for k in range(K)]) / max(sum(cnt[a].values()), 1) for a in ays}
    early = np.mean([share[a] for a in EARLY], 0)
    late = np.mean([share[a] for a in LATE], 0)
    rows = []
    for k in range(K):
        ex = Counter(cn for (a, cn, _), l in zip(units, lab) if l == k and a in LATE).most_common(3) or Counter(cn for (_, cn, _), l in zip(units, lab) if l == k).most_common(3)
        rows.append({"topic": k, "label": names[k], "units_total": int((lab == k).sum()), "share_early_pct": round(100 * early[k], 2), "share_late_pct": round(100 * late[k], 2),
                     "change_pct_points": round(100 * (late[k] - early[k]), 2), "example_courses": "; ".join(c for c, _ in ex)})
    rows.sort(key=lambda r: -r["change_pct_points"])
    with open(ROOT / "results/topic_trends.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"units_clustered": len(units), "clusters": K, "early": list(EARLY), "late": list(LATE),
               "units_per_year": {a: sum(cnt[a].values()) for a in ays}, "growing": rows[:8], "shrinking": rows[-8:][::-1]}
    (ROOT / "results/topic_trends.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    # figure: 8 most growing and 8 most shrinking topics
    sel = rows[:8] + rows[-8:]
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=160)
    y = np.arange(len(sel))[::-1]
    ax.barh(y, [r["change_pct_points"] for r in sel], color=[BLUE if r["change_pct_points"] > 0 else ORANGE for r in sel], height=0.62, zorder=3)
    ax.set_yticks(y, [r["label"][:58] for r in sel], fontsize=7.5)
    ax.axvline(0, color="#c9c8c2", lw=1)
    ax.set_xlabel(f"Change in share of syllabus units, {EARLY[0]}/{EARLY[1][-2:]} to {LATE[0]}/{LATE[1][-2:]} (percentage points)")
    ax.set_title("Topics growing and shrinking in NMIMS computing-programme syllabi" + chr(10) + "(k-means clusters of unit embeddings, labelled by class-based TF-IDF)", fontsize=9.5, loc="left", color=INK)
    ax.grid(axis="x", color="#ecebe6", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(ROOT / "results/fig_topic_trends.png"); plt.close(fig)
    print("units:", len(units), "| per year:", summary["units_per_year"])
    print("GROWING"); [print(f'  {r["change_pct_points"]:+.2f}pp  {r["share_early_pct"]:.2f}->{r["share_late_pct"]:.2f}  {r["label"]}  [{r["example_courses"][:70]}]') for r in rows[:10]]
    print("SHRINKING"); [print(f'  {r["change_pct_points"]:+.2f}pp  {r["share_early_pct"]:.2f}->{r["share_late_pct"]:.2f}  {r["label"]}  [{r["example_courses"][:70]}]') for r in rows[-8:][::-1]]


if __name__ == "__main__":
    main()
