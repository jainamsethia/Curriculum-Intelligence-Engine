"""Week 4, task 2 - Bloom's taxonomy of learning outcomes: rule-based -> fine-tuned transformer -> LLM.

Ground truth = the faculty's own K/L tag printed next to the outcome in the syllabus (single-level tags, identical texts kept once).
Every outcome is predicted by models that never saw its course: 5-fold cross-validation grouped by course name, over ALL tagged outcomes
(the Week 3 held-out split had only 98).

Systems: majority class; hand-written verb lexicon (rule-based, Week 3); Week 3 pipeline classifier; TF-IDF + logistic regression;
fine-tuned MiniLM transformer (text only, and text + rule level as a hint); Phi-3 through Ollama, zero-shot and few-shot.

    python src/w4_bloom.py [--skip-llm] [--seeds 2]
"""
import csv, json, math, random, re, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
import improved as I
import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
SEED, BASE = 42, "sentence-transformers/all-MiniLM-L6-v2"
CKPT = ROOT / "data/interim/w4_bloom_ckpt.json"      # per-fold predictions; a restart resumes where it stopped
DEFS = """Classify a university course learning outcome into one level of Bloom's revised taxonomy:
1 Remember (recall facts: define, list, identify, state, name)
2 Understand (explain, describe, summarize, classify, interpret)
3 Apply (use, implement, solve, compute, demonstrate, execute)
4 Analyze (analyze, compare, differentiate, examine, debug, break down)
5 Evaluate (evaluate, assess, justify, critique, validate, judge)
6 Create (design, develop, formulate, construct, propose, synthesize)
Answer with the level number only."""


def finetune_predict(Xtr, ytr, Xte, seed, rule_hint, epochs=8, lr=5e-5, bs=16):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=6)
    prep = (lambda t: f"level {P.bloom_rule(t) or 0} . {t}") if rule_hint else (lambda t: t)
    Xtr, Xte = [prep(t) for t in Xtr], [prep(t) for t in Xte]
    ytr = np.array(ytr) - 1
    cnt = np.bincount(ytr, minlength=6).astype(float)
    w = torch.tensor(np.where(cnt > 0, (cnt.sum() / (6 * np.maximum(cnt, 1))) ** 0.5, 0.0), dtype=torch.float)   # softened class weights
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(Xtr) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    model.train()
    for _ in range(epochs):
        idx = np.random.permutation(len(Xtr))
        for b in range(0, len(idx), bs):
            j = idx[b:b + bs]
            enc = tok([Xtr[i] for i in j], padding=True, truncation=True, max_length=48, return_tensors="pt")
            loss = lossf(model(**enc).logits, torch.tensor(ytr[j]))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(Xte), 64):
            enc = tok(Xte[b:b + 64], padding=True, truncation=True, max_length=48, return_tensors="pt")
            out += model(**enc).logits.argmax(-1).tolist()
    return np.array(out) + 1


def llm_predict(texts, shots=None):
    """Level from the local LLM; a reply without a level digit counts as a wrong answer (0)."""
    prefix = DEFS + "\n\n" + "".join(f'Outcome: "{t}"\nLevel: {l}\n\n' for t, l in (shots or []))
    preds = []
    for t in texts:
        r = I.ollama_generate(prefix + f'Outcome: "{t}"\nLevel:', num_predict=4)
        m = re.search(r"[1-6]", r)
        preds.append(int(m.group()) if m else 0)
    return np.array(preds)


def main():
    skip_llm = "--skip-llm" in sys.argv
    n_seeds = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 1
    courses = P.load_courses()
    X, y, g = P.bloom_training_data(courses)
    y, g = np.array(y), np.array(g)
    print(f"{len(X)} tagged outcomes from {len(set(g))} courses; level counts {dict(sorted(Counter(y.tolist()).items()))}", flush=True)
    emb = P.Embedder()
    preds = {k: np.zeros(len(X), int) for k in ("Majority class", "Rule-based verb lexicon (Week 3)", "Week 3 pipeline classifier", "TF-IDF + logistic regression",
                                              "Fine-tuned transformer (text)", "Fine-tuned transformer (text + rule hint)", "Phi-3 zero-shot", "Phi-3 few-shot (12 examples)")}
    ft_runs = {"Fine-tuned transformer (text)": [], "Fine-tuned transformer (text + rule hint)": []}
    t0 = time.time()
    ck = json.loads(CKPT.read_text()) if CKPT.exists() else {}

    def cached(fold, key, fn):
        """Run fn() once per (fold, key); the result is saved so that a restarted job skips finished work."""
        d = ck.setdefault(str(fold), {})
        if key not in d:
            d[key] = [int(v) for v in fn()]
            CKPT.parent.mkdir(parents=True, exist_ok=True)
            CKPT.write_text(json.dumps(ck))
        return np.array(d[key])

    for fold, (tr, te) in enumerate(GroupKFold(5).split(X, y, g)):
        Xtr, ytr, gtr, Xte = [X[i] for i in tr], y[tr], g[tr], [X[i] for i in te]
        preds["Majority class"][te] = Counter(ytr.tolist()).most_common(1)[0][0]
        preds["Rule-based verb lexicon (Week 3)"][te] = [P.bloom_rule(t) or Counter(ytr.tolist()).most_common(1)[0][0] for t in Xte]
        preds["Week 3 pipeline classifier"][te] = [p["level"] for p in P.BloomClassifier(emb).fit(Xtr, ytr, gtr).predict(Xte)]
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(Xtr)
        preds["TF-IDF + logistic regression"][te] = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(tf.transform(Xtr), ytr).predict(tf.transform(Xte))
        for name, hint in (("Fine-tuned transformer (text)", False), ("Fine-tuned transformer (text + rule hint)", True)):
            runs = [cached(fold, f"{name}|seed{s}", lambda s=s, hint=hint: finetune_predict(Xtr, ytr, Xte, SEED + s, hint)) for s in range(n_seeds)]
            ft_runs[name].append((te, runs))
            preds[name][te] = [Counter(col).most_common(1)[0][0] for col in zip(*[r.tolist() for r in runs])]     # majority vote over seeds
        print(f"fold {fold + 1}/5 classical + fine-tuned done ({time.time() - t0:.0f}s)", flush=True)
        if not skip_llm:
            rng = random.Random(SEED + fold)
            shots = []
            for lvl in range(1, 7):
                pool = [i for i in range(len(Xtr)) if ytr[i] == lvl]
                shots += [(Xtr[i], lvl) for i in rng.sample(pool, min(2, len(pool)))]
            rng.shuffle(shots)
            preds["Phi-3 zero-shot"][te] = cached(fold, "Phi-3 zero-shot", lambda: llm_predict(Xte))
            preds["Phi-3 few-shot (12 examples)"][te] = cached(fold, "Phi-3 few-shot", lambda: llm_predict(Xte, shots))
            print(f"fold {fold + 1}/5 LLM done ({time.time() - t0:.0f}s)", flush=True)
    if skip_llm:
        for k in ("Phi-3 zero-shot", "Phi-3 few-shot (12 examples)"):
            preds.pop(k)
    res = {"n_outcomes": len(X), "n_courses": int(len(set(g))), "level_counts": {str(k): v for k, v in sorted(Counter(y.tolist()).items())}, "seeds": n_seeds, "systems": {}}
    base_c = preds["Rule-based verb lexicon (Week 3)"] == y
    from scipy.stats import binomtest
    for k, p in preds.items():
        c = p == y
        b, d = int((c & ~base_c).sum()), int((~c & base_c).sum())
        res["systems"][k] = {"accuracy": round(float(c.mean()), 3), "accuracy_95ci": I.cluster_bootstrap(c.astype(float), g), "within_1_level": round(float((abs(p - y) <= 1).mean()), 3),
                             "macro_f1": round(float(f1_score(y, p, average="macro", labels=list(range(1, 7)), zero_division=0)), 3),
                             "gain_over_lexicon_95ci": I.cluster_bootstrap(c.astype(float) - base_c.astype(float), g),
                             "right_where_lexicon_wrong": b, "wrong_where_lexicon_right": d, "mcnemar_exact_p": round(binomtest(b, b + d, 0.5).pvalue, 3) if b + d else None}
    # spread over seeds for the fine-tuned models (accuracy of each seed over all folds)
    for name, folds in ft_runs.items():
        accs = []
        for s in range(n_seeds):
            p = np.zeros(len(X), int)
            for te, runs in folds:
                p[te] = runs[s]
            accs.append(round(float((p == y).mean()), 3))
        res["systems"][name]["accuracy_per_seed"] = accs
    (ROOT / "results/w4_bloom.json").write_text(json.dumps(res, indent=1))
    with open(ROOT / "results/w4_bloom_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["outcome", "course", "faculty_level"] + list(preds))
        for i in range(len(X)):
            w.writerow([X[i], g[i], y[i]] + [int(p[i]) for p in preds.values()])
    for k, v in res["systems"].items():
        print(f'{v["accuracy"]:.3f} {v["accuracy_95ci"]} within1 {v["within_1_level"]} macroF1 {v["macro_f1"]} vs lexicon +{v["right_where_lexicon_wrong"]}/-{v["wrong_where_lexicon_right"]} p={v["mcnemar_exact_p"]} | {k}')


if __name__ == "__main__":
    main()
