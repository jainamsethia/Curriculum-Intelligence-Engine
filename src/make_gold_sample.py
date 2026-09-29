"""Sample NMIMS course pairs for gold annotation, and render them for annotators.

One representative (latest academic year) per course name; pairs of *different* courses only.
Strata, so that no single method decides which pairs get labelled:
    top TF-IDF pairs | top embedding pairs | disagreements both ways | same-department random | random
Writes data/annotation/nmims_pairs_sample.csv (no labels) and data/interim/gold_batches/<annotator>_<n>.txt.

    python src/make_gold_sample.py
"""
import csv, json, random, re
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
sys_path = Path(__file__).parent
import sys; sys.path.insert(0, str(sys_path))
from pipeline import Embedder, course_vec, load_courses, parts

ROOT = Path(__file__).resolve().parents[1]
SEED, N_ANNOTATORS, BATCH = 7, 3, 15


def render(c, width=230):
    """Compact text an annotator reads: name, prerequisites, objectives, outcomes, units (title + start of text)."""
    lines = [f"COURSE: {c['course_name']}  (credits {c.get('credits')}, program: {c['program'][:70]}, sem {c['semester']}, AY {c['academic_year']})",
             f"Prerequisites: {c['prerequisites'][:200] or '-'}", f"Objectives: {c['objectives'][:500] or '-'}", "Outcomes:"]
    lines += [f"  - {o['text'][:200]}" for o in c["outcomes"][:8]]
    lines += ["Units:"] + [f"  {i + 1}. {(u['title'] or '')[:70]} :: {u['text'][:width]}" for i, u in enumerate(c["units"][:10])]
    return "\n".join(lines)


def dept(c):
    m = re.search(r"7\d\d([A-Z]{2})", c.get("code", "")) or re.search(r"^[A-Z]{2}([A-Z]{2})\d", c.get("code", ""))
    return m.group(1) if m else ""


def main():
    rng = random.Random(SEED)
    courses = load_courses()
    rep = {}
    for c in sorted(courses, key=lambda c: c["academic_year"]):
        if parts(c):
            rep[c["course_key"]] = c            # sorted ascending, so the latest year wins
    reps = list(rep.values())
    n = len(reps)
    emb = Embedder()
    V = np.stack([course_vec(emb, c) for c in reps])
    emb.save()
    T = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True).fit_transform(
        [" ".join(parts(c)) for c in reps])
    Se, St = V @ V.T, (T @ T.T).toarray()
    iu = np.triu_indices(n, 1)
    pe, pt = Se[iu], St[iu]
    ok = np.array([reps[i]["course_name"].lower() != reps[j]["course_name"].lower() for i, j in zip(*iu)])
    depts = [dept(c) for c in reps]
    same_dept = np.array([bool(depts[i]) and depts[i] == depts[j] for i, j in zip(*iu)], dtype=bool)

    def pick(mask, k, order=None):
        idx = np.flatnonzero(mask & ok)
        if order is not None:
            idx = idx[np.argsort(-order[idx])][: 4 * k]      # candidates: the top of the ranking, then sample k of them
        return rng.sample(list(idx), min(k, len(idx)))

    rank_e, rank_t = pe.argsort().argsort() / len(pe), pt.argsort().argsort() / len(pt)
    everything = np.ones(len(pe), bool)
    chosen = {}
    for name, idxs in [("top_embedding", pick(everything, 26, pe)), ("top_tfidf", pick(everything, 26, pt)),
                       ("emb_high_tfidf_low", pick(everything, 16, rank_e - rank_t)), ("tfidf_high_emb_low", pick(everything, 16, rank_t - rank_e)),
                       ("same_dept_random", pick(same_dept, 24)), ("random", pick(everything, 12))]:
        for i in idxs:
            chosen.setdefault(int(i), name)
    rows = []
    for i, stratum in chosen.items():
        a, b = reps[iu[0][i]], reps[iu[1][i]]
        rows.append({"pair_id": f"p{len(rows):03d}", "id_a": a["id"], "id_b": b["id"], "course_a": a["course_name"], "course_b": b["course_name"], "stratum": stratum})
    rng.shuffle(rows)
    for k, r in enumerate(rows):
        r["pair_id"] = f"p{k:03d}"
    out = ROOT / "data/annotation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "nmims_pairs_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by_id = {c["id"]: c for c in courses}
    bdir = ROOT / "data/interim/gold_batches"
    bdir.mkdir(parents=True, exist_ok=True)
    for a in range(N_ANNOTATORS):                                # each annotator: own order and own A/B orientation
        r2 = random.Random(SEED + 100 + a)
        order = rows[:]
        r2.shuffle(order)
        for b0 in range(0, len(order), BATCH):
            txt = []
            for r in order[b0:b0 + BATCH]:
                x, y = (r["id_a"], r["id_b"]) if r2.random() < 0.5 else (r["id_b"], r["id_a"])
                txt.append(f"##### PAIR {r['pair_id']}\n=== COURSE 1 ===\n{render(by_id[x])}\n=== COURSE 2 ===\n{render(by_id[y])}\n")
            (bdir / f"a{a}_b{b0 // BATCH}.txt").write_text("\n".join(txt), encoding="utf-8")
    print(len(rows), "pairs;", {s: sum(r["stratum"] == s for r in rows) for s in dict.fromkeys(r["stratum"] for r in rows)},
          "|", N_ANNOTATORS, "annotators x", -(-len(rows) // BATCH), "batches")


def prereq_sample(n_items=90, n_graders=2, batch=30):
    """Sample resolved prerequisite phrases for a grader panel: does the phrase really refer to the course it was matched to?"""
    from pipeline import Pipeline
    rng = random.Random(SEED)
    p = Pipeline()
    cal = {"tau_prereq": 0.5}
    items = []
    for c in p.courses:
        for r in __import__("pipeline").extract_prereqs(c, p.catalog, cal):
            if r["resolved_course"]:
                items.append({"course": c["course_name"], "prereq_text": c["prerequisites"][:250], "phrase": r["phrase"],
                              "resolved_course": r["resolved_course"], "similarity": r["similarity"]})
    uniq = {(i["phrase"].lower(), i["resolved_course"]): i for i in items}       # one item per distinct (phrase, resolution)
    pool = list(uniq.values())
    pool.sort(key=lambda i: i["similarity"])
    strata = [pool[k::4] for k in range(4)]                                        # spread over the similarity range
    rows = [x for s in strata for x in rng.sample(s, min(n_items // 4, len(s)))]
    rng.shuffle(rows)
    for k, r in enumerate(rows):
        r["item_id"] = f"q{k:03d}"
    out = ROOT / "data/annotation"
    with open(out / "prereq_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "course", "prereq_text", "phrase", "resolved_course", "similarity"]); w.writeheader(); w.writerows(rows)
    bdir = ROOT / "data/interim/prereq_batches"
    bdir.mkdir(parents=True, exist_ok=True)
    for g in range(n_graders):
        order = rows[:]
        random.Random(SEED + 200 + g).shuffle(order)
        for b0 in range(0, len(order), batch):
            txt = [f"##### ITEM {r['item_id']}\nA course named \"{r['course']}\" lists this prerequisite text: \"{r['prereq_text']}\"\n"
                   f"Extracted phrase: \"{r['phrase']}\"\nSystem matched it to the course: \"{r['resolved_course']}\"\n" for r in order[b0:b0 + batch]]
            (bdir / f"g{g}_b{b0 // batch}.txt").write_text("\n".join(txt), encoding="utf-8")
    print(len(rows), "prerequisite items;", n_graders, "graders x", -(-len(rows) // batch), "batches")


if __name__ == "__main__":
    main()
    prereq_sample()
