"""Week 4 - exploratory follow-up to w4_bloom.py: does a simple vote between Bloom classifiers beat the verb lexicon?

Reads results/w4_bloom_predictions.csv (every outcome predicted by models that never saw its course). The vote (lexicon, fine-tuned transformer
with rule hint, Phi-3 zero-shot; ties go to the lexicon) was chosen before looking at its score but is one of several possible combinations,
so it is reported as exploratory. The 'oracle' (an outcome counts as right if any of the three systems is right) is an upper bound, not a model.

    python src/w4_bloom_vote.py
"""
import csv, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import improved as I

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = list(csv.DictReader(open(ROOT / "results/w4_bloom_predictions.csv", encoding="utf-8")))
    y, g = np.array([int(r["faculty_level"]) for r in rows]), np.array([r["course"] for r in rows])
    col = lambda k: np.array([int(r[k]) for r in rows])
    lex, ft, z = col("Rule-based verb lexicon (Week 3)"), col("Fine-tuned transformer (text + rule hint)"), col("Phi-3 zero-shot")

    def vote(*ps):
        out = []
        for vals in zip(*ps):
            c = Counter(vals).most_common()
            out.append(c[0][0] if len(c) == 1 or c[0][1] > c[1][1] else vals[0])
        return np.array(out)

    base = (lex == y).astype(float)
    res = {}
    for name, p in {"vote(lexicon, transformer + rule hint, Phi-3 zero-shot)": vote(lex, ft, z),
                    "oracle (any of the three right)": np.where((lex == y) | (ft == y) | (z == y), y, lex)}.items():
        c = (p == y).astype(float)
        res[name] = {"accuracy": round(float(c.mean()), 3), "accuracy_95ci": I.cluster_bootstrap(c, g), "gain_over_lexicon": round(float(c.mean() - base.mean()), 3),
                     "gain_95ci": I.cluster_bootstrap(c - base, g)}
    res["agreement_lexicon_vs_phi3_zero_shot"] = round(float((lex == z).mean()), 3)
    res["agreement_lexicon_vs_transformer_hint"] = round(float((lex == ft).mean()), 3)
    (ROOT / "results/w4_bloom_vote.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
