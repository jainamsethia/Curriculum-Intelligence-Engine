"""LLM-annotated reference labels (no human annotators were used; results are indicative).

    sample            stratified cross-course pairs -> data/annotation/pairs_items.jsonl + packets for the Claude passes
    phi3 <kind>       pass 2: Phi-3 via Ollama, prompt wording B, checkpointed per item (kind = pairs | flags)
    merge <kind>      pass 1 + pass 2 (+ pass 3 on disagreements) -> labels/<kind>_llm.csv (2-of-3 majority, else 'uncertain')
    spotcheck         labels/human_spot_check.csv: 10 consensus pairs per team member, empty label column (optional)

Labellers never see model scores: packets hold course text only. Pass 1 and pass 3 are Claude subagents with fresh contexts
and different prompt wordings (A, C); pass 2 is Phi-3 (wording B).
"""
import csv, json, re, sys
from collections import Counter

import numpy as np

from curriculum_engine import DATA, LABELS, SEED
from curriculum_engine import data as D
from curriculum_engine import llm

ANN, PASSES = DATA / "annotation", DATA / "annotation/passes"
MEMBERS = ["J048", "J049", "J050", "J052", "J053"]
STOP = set("and of in for the to with a an i ii iii iv v vi lab laboratory introduction fundamentals basics principles applied".split())

PROMPT_B = {
    "pairs": ("You are reviewing two university course syllabi for a curriculum committee. Decide how much of their CONTENT "
              "(the topics actually taught) is the same.\n"
              "0 = different subjects, or they share only generic skills or words.\n"
              "1 = some units or topics are clearly shared, but most of the content differs.\n"
              "2 = most units teach the same topics; one course largely repeats the other.\n"
              'Reply with JSON only: {"label": 0 or 1 or 2, "reason": "one short sentence"}\n\n'),
    "flags": ("You are checking an automatic warning produced by a curriculum-analysis tool. Read the evidence and decide "
              "whether the warning is correct.\n"
              'Reply with JSON only: {"label": "correct" or "wrong", "reason": "one short sentence"}\n\n'),
}
LABELS_OF = {"pairs": {0, 1, 2}, "flags": {"correct", "wrong"}}


# ------------------------------------------------------------------ rendering (no scores, no ids)
def _cut(t, n):
    w = (t or "").split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def render(c, compact=False):
    progs = ", ".join(c["programmes"]) or ("all programmes" if c["common"] else "-")
    lines = [f"{c['course_name']} | programme: {progs} | semester: {c.get('semester') or '-'} | AY {c['academic_year']}",
             "Objectives: " + _cut(c.get("objectives"), 40 if compact else 80)]
    outs = c.get("outcomes", [])[: 4 if compact else 8]
    if outs:
        lines.append("Outcomes: " + " / ".join(_cut(o["text"], 18 if compact else 30) for o in outs))
    lines.append("Units:")
    for i, u in enumerate(c.get("units", [])[: 8 if compact else 12], 1):
        lines.append(f"  {i}. {u['title'] + ': ' if u['title'] else ''}{_cut(u['text'], 22 if compact else 70)}")
    return "\n".join(lines)


# ------------------------------------------------------------------ pair sampling
def _name_tokens(c):
    return {w for w in re.findall(r"[a-z]+", c["course_name"].lower()) if w not in STOP and len(w) > 2}


def sample_pairs(n_val=80, n_test=120):
    from curriculum_engine.embed import Embedder
    from curriculum_engine.overlap import Tfidf, chunk_vectors, maxsim
    recs = D.load()
    sp, reps = D.split(recs), D.representatives(recs)
    tf, emb = Tfidf([c for f, c in reps.items() if sp[f] == "train"]), Embedder()
    rng = np.random.default_rng(SEED)
    rows, items = [], []
    for split_name, n in (("val", n_val), ("test", n_test)):
        cs = [c for f, c in reps.items() if sp[f] == split_name]
        T = tf.matrix(cs)
        TT = (T @ T.T).toarray()
        V = [chunk_vectors(emb, c) for c in cs]
        cand = [(i, j, TT[i, j], maxsim(V[i], V[j], 1.0)["soft"]) for i in range(len(cs)) for j in range(i + 1, len(cs))]
        t = np.array([x[2] for x in cand]); e = np.array([x[3] for x in cand])
        pt = t.argsort().argsort() / (len(t) - 1); pe = e.argsort().argsort() / (len(e) - 1)     # percentile ranks
        s = np.maximum(pt, pe)
        same = np.array([bool(set(cs[i]["programmes"]) & set(cs[j]["programmes"])) for i, j, _, _ in cand])
        near = np.array([len(_name_tokens(cs[i]) & _name_tokens(cs[j])) / max(1, len(_name_tokens(cs[i]) | _name_tokens(cs[j]))) >= 0.34
                         for i, j, _, _ in cand])
        pools = {"high": s >= 0.97, "mid": (s >= 0.85) & (s < 0.97), "low": s < 0.85,
                 "disagreement": (np.abs(pt - pe) >= 0.4) & (s >= 0.9), "near_name": near, "random": np.ones(len(cand), bool)}
        quota = {"high": 0.3, "mid": 0.3, "disagreement": 0.1, "near_name": 0.1, "low": 0.1, "random": 0.1}
        taken = set()
        for stratum, q in quota.items():
            k = round(q * n)
            idx = np.flatnonzero(pools[stratum])
            if stratum in ("high", "mid", "low"):              # 60% same programme, 40% cross-programme
                parts = [(idx[same[idx]], round(0.6 * k)), (idx[~same[idx]], k - round(0.6 * k))]
            else:
                parts = [(idx, k)]
            for pool, kk in parts:
                pool = [x for x in rng.permutation(pool) if x not in taken][:kk]
                for x in pool:
                    taken.add(x)
                    i, j = cand[x][:2]
                    rows.append({"split": split_name, "stratum": stratum, "pair_type": "same_programme" if same[x] else "cross_programme",
                                 "id_a": cs[i]["id"], "id_b": cs[j]["id"], "family_a": cs[i]["family"], "family_b": cs[j]["family"],
                                 "course_a": cs[i]["course_name"], "course_b": cs[j]["course_name"],
                                 "tfidf": round(float(cand[x][2]), 4), "maxsim_soft": round(float(cand[x][3]), 4)})
    emb.save()
    by_id = {c["id"]: c for c in recs}
    for k, r in enumerate(rows, 1):
        r["item"] = f"PR{k:03d}"
        a, b = by_id[r["id_a"]], by_id[r["id_b"]]
        items.append({"item": r["item"], "full": f"COURSE A\n{render(a)}\n\nCOURSE B\n{render(b)}",
                      "compact": f"COURSE A\n{render(a, True)}\n\nCOURSE B\n{render(b, True)}"})
    ANN.mkdir(parents=True, exist_ok=True)
    with open(ANN / "pairs_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    write_items("pairs", items)
    return rows


def write_items(kind, items, parts=3):
    """items.jsonl for Phi-3 and plain-text packets (course text only) for the Claude passes."""
    with open(ANN / f"{kind}_items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    pk = ANN / "packets"
    pk.mkdir(parents=True, exist_ok=True)
    size = -(-len(items) // parts)
    for p in range(parts):
        chunk = items[p * size:(p + 1) * size]
        (pk / f"{kind}_part{p + 1}.txt").write_text("\n\n".join(f"===== {it['item']} =====\n{it['full']}" for it in chunk), encoding="utf-8")


# ------------------------------------------------------------------ pass 2: Phi-3
def run_phi3(kind):
    items = [json.loads(l) for l in open(ANN / f"{kind}_items.jsonl", encoding="utf-8")]
    out = PASSES / f"pass2_{kind}.jsonl"
    PASSES.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["item"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    for n, it in enumerate(items, 1):
        if it["item"] in done:
            continue
        raw = llm.generate(PROMPT_B[kind] + it["compact"] + "\n\nJSON:", fmt="json", num_predict=80, num_ctx=2048)
        try:
            d = json.loads(raw)
            lab = d.get("label")
            lab = int(lab) if kind == "pairs" and str(lab).strip() in {"0", "1", "2"} else str(lab).strip().lower()
        except Exception:
            d, lab = {}, None
        rec = {"item": it["item"], "label": lab if lab in LABELS_OF[kind] else None, "reason": str(d.get("reason", ""))[:300], "raw": raw[:300]}
        with open(out, "a", encoding="utf-8") as f:                 # checkpoint per item
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"{n}/{len(items)} {it['item']} -> {rec['label']}", flush=True)


# ------------------------------------------------------------------ merge
def _load_pass(kind, p):
    res = {}
    for f in sorted(PASSES.glob(f"pass{p}_{kind}*.json*")):
        txt = f.read_text(encoding="utf-8")
        recs = [json.loads(l) for l in txt.splitlines() if l.strip()] if f.suffix == ".jsonl" else json.loads(txt)
        for r in recs:
            lab = r.get("label")
            if kind == "pairs" and lab is not None and str(lab) in {"0", "1", "2"}:
                lab = int(lab)
            res[r["item"]] = (lab if lab in LABELS_OF[kind] else None, r.get("reason", ""), labeller(f.name, p))
    return res


def labeller(fname, p):
    """Who produced a pass file: pass 1 = Claude wording A, pass 3 = Claude wording C; pass 2 = Phi-3 (.jsonl) or Claude wording B."""
    if p == 2:
        return "Claude (wording B)" if "claudeB" in fname else "Phi-3"
    return {1: "Claude (wording A)", 3: "Claude (wording C)"}[p]


def consensus(p1, p2, p3):
    votes = Counter(x for x in (p1, p2, p3) if x is not None)
    lab, c = votes.most_common(1)[0] if votes else (None, 0)
    return (lab, "agreed" if p1 == p2 and p1 is not None else "majority") if c >= 2 else (None, "uncertain")


def merge(kind):
    meta = {r["item"]: r for r in csv.DictReader(open(ANN / f"{kind}_sample.csv", encoding="utf-8"))}
    P = {p: _load_pass(kind, p) for p in (1, 2, 3)}
    rows = []
    for item, m in meta.items():
        (l1, r1, _), (l2, r2, by2), (l3, r3, _) = (P[p].get(item, (None, "", "")) for p in (1, 2, 3))
        lab, status = consensus(l1, l2, l3)
        keep = {k: v for k, v in m.items() if k not in ("tfidf", "maxsim_soft") and not k.startswith("_")}
        rows.append({**keep, "pass1": l1, "pass2": l2, "pass3": l3, "pass1_by": "Claude (wording A)", "pass2_by": by2,
                     "pass3_by": "Claude (wording C)" if l3 is not None else "", "consensus": lab, "status": status,
                     "reason_pass1": r1, "reason_pass2": r2, "reason_pass3": r3})
    LABELS.mkdir(exist_ok=True)
    with open(LABELS / f"{kind}_llm.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return rows


def disagreements(kind):
    """Items where pass 1 and pass 2 differ (or pass 2 failed) -> packet for pass 3."""
    P1, P2 = _load_pass(kind, 1), _load_pass(kind, 2)
    items = [json.loads(l) for l in open(ANN / f"{kind}_items.jsonl", encoding="utf-8")]
    todo = [it for it in items if P1.get(it["item"], (None,))[0] is None or P1[it["item"]][0] != P2.get(it["item"], (None,))[0]]
    (ANN / "packets" / f"{kind}_pass3_items.json").write_text(json.dumps([it["item"] for it in todo]), encoding="utf-8")
    (ANN / "packets" / f"{kind}_pass3.txt").write_text("\n\n".join(f"===== {it['item']} =====\n{it['full']}" for it in todo), encoding="utf-8")
    return [it["item"] for it in todo]


def agreement(rows):
    """LLM-vs-LLM agreement between pass 1 (Claude) and pass 2 (Phi-3), on items both labelled."""
    from sklearn.metrics import cohen_kappa_score
    both = [(r["pass1"], r["pass2"]) for r in rows if r["pass1"] not in (None, "", "None") and r["pass2"] not in (None, "", "None")]
    a = [str(x) for x, _ in both]; b = [str(y) for _, y in both]
    numeric = all(x in {"0", "1", "2"} for x in a + b)
    return {"n_items": len(rows), "n_both_labelled": len(both),
            "percent_agreement": round(float(np.mean([x == y for x, y in zip(a, b)])), 3) if both else None,
            "cohen_kappa": round(float(cohen_kappa_score(a, b, weights="quadratic" if numeric else None)), 3) if len(set(a + b)) > 1 else None,
            "kappa_weights": "quadratic" if numeric else "none",
            "status": dict(Counter(r["status"] for r in rows)),
            "consensus": {str(k): v for k, v in sorted(Counter(str(r["consensus"]) for r in rows if r["status"] != "uncertain").items())}}


def spot_check():
    rows = [r for r in csv.DictReader(open(LABELS / "pairs_llm.csv", encoding="utf-8")) if r["status"] != "uncertain"]
    rng = np.random.default_rng(SEED)
    pick = [rows[i] for i in rng.permutation(len(rows))[:10 * len(MEMBERS)]]
    with open(LABELS / "human_spot_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "member", "course_a", "course_b", "human_label_0_1_2", "notes"])
        for k, r in enumerate(pick):
            w.writerow([r["item"], MEMBERS[k // 10], r["course_a"], r["course_b"], "", ""])


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample":
        rows = sample_pairs()
        print(len(rows), Counter((r["split"], r["stratum"]) for r in rows))
    elif cmd == "phi3":
        run_phi3(sys.argv[2])
    elif cmd == "disagreements":
        print(disagreements(sys.argv[2]))
    elif cmd == "merge":
        print(json.dumps(agreement(merge(sys.argv[2])), indent=1))
    elif cmd == "spotcheck":
        spot_check()
