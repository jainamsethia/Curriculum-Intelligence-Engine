"""Week 4 - one figure comparing baseline and improved systems on the four tasks (reads results/w4_*.json).

    python src/w4_figures.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
BASE, IMPR, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"      # baseline = blue, improved = orange


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def panel(ax, rows, title, xlabel, xmax=None):
    """rows: (label, value, lo, hi, is_improved); drawn top to bottom."""
    y = list(range(len(rows)))[::-1]
    for yi, (lab, v, lo, hi, imp) in zip(y, rows):
        ax.barh(yi, v, color=IMPR if imp else BASE, height=0.62, zorder=3)
        if lo is not None:
            ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color=INK2, capsize=2, lw=1, zorder=4)
        ax.text((hi if hi is not None else v) + 0.01 * (xmax or 1), yi, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, xmax or max(r[3] or r[1] for r in rows) * 1.15)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9, loc="left", color=INK)
    ax.grid(axis="x", color="#ecebe6", lw=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axs = plt.subplots(2, 2, figsize=(11.2, 7.6), dpi=160)
    r = load("w4_retrieval.json")
    if r:
        t = r["test"]
        key = lambda *parts: next((k for k in t if all(x in k for x in parts)), None)
        picks = [("TF-IDF cosine (Week 2)", key("TF-IDF"), False), ("BM25, whole course", key("BM25, whole"), False), ("BM25, best unit", key("BM25, best"), False),
                 ("MiniLM-L6, best unit", key("MiniLM-L6 (Week 3", "best unit"), True), ("bge-small, best unit", key("bge-small", "best unit"), True),
                 ("nomic-embed, best unit", key("nomic", "best unit"), True), ("bge-base, best unit", key("bge-base", "best unit"), True),
                 ("MiniLM domain-adapted, best unit", key("domain-adapted", "best unit"), True),
                 ("Hybrid: BM25 + bge-base (validation-selected)", key("Hybrid: ", "bge-base"), True), ("Hybrid: BM25 + adapted MiniLM", key("Hybrid: ", "domain-adapted"), True),
                 ("Hybrid + cross-encoder rerank", key("rerank"), True)]
        rows = [(lab, t[k]["MRR@10"], t[k]["MRR@10_95ci"][0], t[k]["MRR@10_95ci"][1], imp) for lab, k, imp in picks if k]
        panel(axs[0, 0], rows, "1. Course search: outcome to course (MRR@10)", "MRR@10, test courses (95% CI over courses)", 1.0)
    b = load("w4_bloom.json")
    if b:
        order = ["Majority class", "Rule-based verb lexicon (Week 3)", "Week 3 pipeline classifier", "TF-IDF + logistic regression", "Fine-tuned transformer (text)",
                 "Fine-tuned transformer (text + rule hint)", "Phi-3 zero-shot", "Phi-3 few-shot (12 examples)"]
        rows = [(s.replace(" (Week 3)", ""), b["systems"][s]["accuracy"], b["systems"][s]["accuracy_95ci"][0], b["systems"][s]["accuracy_95ci"][1],
                 s.startswith(("Fine", "Phi"))) for s in order if s in b["systems"]]
        panel(axs[0, 1], rows, f"2. Bloom level, {b['n_outcomes']} outcomes (accuracy)", "accuracy, grouped 5-fold CV (95% CI over courses)", 1.0)
    p = load("w4_prereq_extraction.json")
    if p:
        rows = [(s, v["overall"]["f1"], None, None, s.startswith(("Phi", "spaCy"))) for s, v in p["systems"].items()]
        panel(axs[1, 0], rows, f"3. Prerequisite extraction, {p['n_items']} strings (F1)", "phrase-level F1", 1.0)
    s = load("w4_summarization.json")
    if s:
        rows = [(k.replace("Extractive: ", "Extractive, ").replace("Template: ", "Template, "), v["rougeL"], v["rougeL_95ci"][0], v["rougeL_95ci"][1], k.startswith("Phi")) for k, v in s["systems"].items()]
        panel(axs[1, 1], rows, f"4. Course summaries, {s['n_courses']} courses (ROUGE-L)", "ROUGE-L F1 (95% CI over courses)", None)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASE), plt.Rectangle((0, 0), 1, 1, color=IMPR)]
    fig.legend(handles, ["baseline", "candidate improvement"], loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RES / "fig_w4_comparison.png")
    print("wrote results/fig_w4_comparison.png")


if __name__ == "__main__":
    main()
