"""Capability 2 - redundant content inside one programme.
Baseline (Week 2): course pairs of the same programme whose TF-IDF cosine passes a validation-tuned threshold.
Core (Week 3): unit-level near-duplicates (a unit of one course that another course of the programme also teaches)."""
import copy

import numpy as np

from curriculum_engine.perturb import noisy


def programme_courses(view, prog, families):
    return [c for f, c in sorted(view[prog].items()) if f in families]


def course_pairs(T, courses, thr):
    """Baseline flags: (i, j, cosine) for course pairs with TF-IDF cosine >= thr, highest first."""
    S = (T @ T.T).toarray()
    out = [(i, j, float(S[i, j])) for i in range(len(courses)) for j in range(i + 1, len(courses))
           if courses[i]["family"] != courses[j]["family"] and S[i, j] >= thr]
    return sorted(out, key=lambda x: -x[2])


def inject_cases(view, families, n=60, seed=0):
    """Synthetic redundancy: copy one unit of course X (lightly edited: word drop + typos) into an unrelated course Y of the same
    programme. Returns (programme, X, Y_modified, injected unit index in Y)."""
    rng = np.random.default_rng(seed)
    progs = [p for p in sorted(view) if len(programme_courses(view, p, families)) >= 6]
    cases = []
    while len(cases) < n:
        p = progs[int(rng.integers(len(progs)))]
        cs = [c for c in programme_courses(view, p, families) if len(c["units"]) >= 2]
        x, y = (cs[int(i)] for i in rng.choice(len(cs), 2, replace=False))
        u = x["units"][int(rng.integers(len(x["units"])))]
        y2 = copy.deepcopy(y)
        y2["units"].append({"title": u["title"], "text": noisy(u["text"], seed=len(cases)), "hours": u["hours"]})
        cases.append({"programme": p, "x": x, "y": y2, "unit_x": x["units"].index(u), "unit_y": len(y2["units"]) - 1})
    return cases
