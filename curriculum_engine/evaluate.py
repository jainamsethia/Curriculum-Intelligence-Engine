"""Metrics with 95% intervals that resample whole course families (items of one course are correlated)."""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from curriculum_engine import SEED

N_BOOT = 1000


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)


def _boot(fn, groups, n=N_BOOT):
    """Percentile CI of fn(index array), resampling groups with replacement."""
    groups = np.asarray(groups)
    if not len(groups):
        return [None, None]
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        try:
            v = fn(idx)
        except ValueError:                      # a resample with one class only
            continue
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [r3(np.percentile(vals, 2.5)), r3(np.percentile(vals, 97.5))] if vals else [None, None]


def mean_ci(values, groups):
    v = np.asarray(values, float)
    return {"value": r3(v.mean()) if len(v) else None, "ci95": _boot(lambda i: v[i].mean(), groups) if len(v) else [None, None], "n": int(len(v))}


def ranking(y, s, groups):
    """y in {0,1,2}; s = score. Spearman with the 0/1/2 label, ROC-AUC / AP for 'any overlap' (y>=1) and for 'substantial' (y=2)."""
    y, s = np.asarray(y), np.asarray(s, float)
    out = {"n": int(len(y)), "n_any_overlap": int((y >= 1).sum()), "n_substantial": int((y == 2).sum())}
    if not len(y):
        return {**out, **{k: {"value": None, "ci95": [None, None]} for k in ("spearman", "roc_auc", "avg_precision", "roc_auc_substantial")}}
    for name, fn in {"spearman": lambda i: spearmanr(y[i], s[i])[0],
                     "roc_auc": lambda i: roc_auc_score(y[i] >= 1, s[i]),
                     "avg_precision": lambda i: average_precision_score(y[i] >= 1, s[i]),
                     "roc_auc_substantial": lambda i: roc_auc_score(y[i] == 2, s[i])}.items():
        idx = np.arange(len(y))
        try:
            v = fn(idx)
        except ValueError:
            v = None
        out[name] = {"value": r3(v), "ci95": _boot(fn, groups)}
    return out


def best_threshold(y_bin, s):
    """Threshold that maximises F1 (chosen on validation only)."""
    y_bin, s = np.asarray(y_bin, bool), np.asarray(s, float)
    best = (-1.0, 0.5)
    for t in np.unique(s):
        p = s >= t
        tp = (p & y_bin).sum()
        f = 2 * tp / (p.sum() + y_bin.sum()) if p.sum() + y_bin.sum() else 0.0
        if f > best[0]:
            best = (f, float(t))
    return best[1]


def prf(y_bin, s, thr, groups=None):
    y_bin, p = np.asarray(y_bin, bool), np.asarray(s, float) >= thr

    def f1(i):
        tp = (p[i] & y_bin[i]).sum()
        return 2 * tp / (p[i].sum() + y_bin[i].sum()) if p[i].sum() + y_bin[i].sum() else np.nan

    tp = int((p & y_bin).sum())
    out = {"threshold": r3(thr), "precision": r3(tp / p.sum()) if p.sum() else None, "recall": r3(tp / y_bin.sum()) if y_bin.sum() else None,
           "f1": r3(f1(np.arange(len(p)))), "tp": tp, "fp": int((p & ~y_bin).sum()), "fn": int((~p & y_bin).sum())}
    if groups is not None:
        out["f1_ci95"] = _boot(f1, groups)
    return out


def classification(y, pred, groups):
    """Accuracy, macro-F1 and within-one-level accuracy for ordinal labels (Bloom levels)."""
    from sklearn.metrics import f1_score
    y, pred = np.asarray(y), np.asarray(pred)
    ok, near = (y == pred).astype(float), (np.abs(y - pred) <= 1).astype(float)
    return {"accuracy": mean_ci(ok, groups), "within_one": mean_ci(near, groups),
            "macro_f1": r3(f1_score(y, pred, average="macro", labels=sorted(set(y)))), "n": int(len(y))}


def paired(metric, y, s_base, s_new, groups):
    """Difference metric(new) - metric(base) on the same items, 95% CI by resampling families, and a verdict:
    'improved' if the interval is above zero, 'worse' if below, otherwise 'no clear difference'."""
    y, a, b = np.asarray(y), np.asarray(s_base, float), np.asarray(s_new, float)
    fn = lambda i: metric(y[i], b[i]) - metric(y[i], a[i])
    v, ci = fn(np.arange(len(y))), _boot(fn, groups)
    verdict = "improved" if ci[0] is not None and ci[0] > 0 else "worse" if ci[1] is not None and ci[1] < 0 else "no clear difference"
    return {"difference": r3(v), "ci95": ci, "verdict": verdict}


def spearman(y, s):
    return spearmanr(y, s)[0]


def auc_any(y, s):
    return roc_auc_score(np.asarray(y) >= 1, s)


def calibration(y_bin, p, bins=5):
    """Brier score and expected calibration error of predicted probabilities."""
    y_bin, p = np.asarray(y_bin, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = sum(abs(p[idx == b].mean() - y_bin[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    return {"brier": r3(np.mean((p - y_bin) ** 2)), "ece": r3(ece)}
