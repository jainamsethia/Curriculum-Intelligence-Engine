"""Text perturbations for synthetic tests and robustness checks (seeded, deterministic)."""
import numpy as np

from curriculum_engine import SEED


def typos(text, rate=0.05, seed=SEED):
    """Swap two neighbouring letters inside `rate` of the words."""
    rng = np.random.default_rng(seed)
    out = []
    for w in text.split():
        if len(w) > 3 and rng.random() < rate:
            i = int(rng.integers(1, len(w) - 2))
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        out.append(w)
    return " ".join(out)


def word_drop(text, rate=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    ws = text.split()
    kept = [w for w in ws if rng.random() >= rate]
    return " ".join(kept or ws[:1])


def noisy(text, seed=SEED):
    """The default injected copy: word drop + typos (a lightly edited re-use of a unit)."""
    return typos(word_drop(text, 0.15, seed), 0.05, seed)


def truncate(text, keep=0.5):
    ws = text.split()
    return " ".join(ws[: max(1, int(len(ws) * keep))])
