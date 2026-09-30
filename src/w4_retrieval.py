"""Week 4, task 1 - keyword search -> vector retrieval ("TF-IDF -> sentence transformer").

Question: which course teaches this learning outcome?  A query is one outcome statement; the collection has one entry per course
(all its distinct unit descriptions, across versions). Outcomes, objectives, title and code are NOT indexed, so nothing leaks.
Relevant = any course containing that identical outcome text (near-duplicate courses are merged into clusters first).

Systems: TF-IDF cosine (the Week 2 method), BM25 (keyword search), dense encoders (MiniLM, bge, nomic; mean vector or best unit),
hybrid (reciprocal rank fusion), hybrid + cross-encoder rerank, and a domain-adapted MiniLM (w4_finetune_encoder.py).
Selection of aggregators / fusion inputs uses a validation split of the training courses; numbers are reported on test courses.

    python src/w4_retrieval.py [--models minilm,bge-small,nomic,bge-base,ft]
"""
import hashlib, json, pickle, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, str(Path(__file__).parent))
import improved as I
import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
RES, INTERIM = ROOT / "results", ROOT / "data/interim"
SEED, MAX_Q_PER_FAMILY, MIN_Q_WORDS, CLUSTER_COS, TOP_K_RERANK = 42, 8, 6, 0.85, 10
BGE_Q = "Represent this sentence for searching relevant passages: "
MODELS = {  # key -> (label, embedder factory, query prefix, document prefix)
    "minilm": ("MiniLM-L6 (Week 3 encoder)", lambda: P.Embedder("sentence-transformers/all-MiniLM-L6-v2"), "", ""),
    "bge-small": ("bge-small-en-v1.5", lambda: P.Embedder("BAAI/bge-small-en-v1.5"), BGE_Q, ""),
    "nomic": ("nomic-embed-text", lambda: I.OllamaEmbedder("nomic-embed-text"), "search_query: ", "search_document: "),
    "bge-base": ("bge-base-en-v1.5", lambda: P.Embedder("BAAI/bge-base-en-v1.5"), BGE_Q, ""),
    "ft": ("MiniLM-L6 domain-adapted (ours)", lambda: P.Embedder(str(INTERIM / "w4_minilm_ft")), "", ""),
}
norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()


def build():
    courses = P.load_courses()
    keys = sorted({c["course_key"] for c in courses})
    tr_i, _ = next(GroupShuffleSplit(1, test_size=0.3, random_state=SEED).split(keys, groups=keys))   # same split as Week 3
    train = sorted(keys[i] for i in tr_i)
    test = set(keys) - set(train)
    perm = np.random.RandomState(SEED).permutation(len(train))
    val = {train[i] for i in perm[:100]}                       # validation courses (never used to train anything)
    fit = set(train) - val
    units, outs = defaultdict(dict), defaultdict(dict)
    for c in courses:
        for u in c["units"]:
            units[c["course_key"]].setdefault(norm(u["text"]), u["text"])
        for o in c["outcomes"]:
            outs[c["course_key"]].setdefault(norm(o["text"]), o["text"])
    fams = sorted(k for k, d in units.items() if len(d) >= 2 and sum(len(t.split()) for t in d.values()) >= 40)
    chunks, chunk_fam, starts = [], [], []
    for i, f in enumerate(fams):
        starts.append(len(chunks))
        for t in units[f].values():
            chunks.append(t); chunk_fam.append(i)
    # merge near-duplicate courses (renamed or re-numbered copies) so they are not counted as errors of each other
    docs = [" ".join(units[f].values()) for f in fams]
    T = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit_transform(docs)
    sim = (T @ T.T).toarray()
    parent = list(range(len(fams)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(*np.where(np.triu(sim >= CLUSTER_COS, 1))):
        parent[find(i)] = find(j)
    cluster = np.array([find(i) for i in range(len(fams))])
    who = defaultdict(set)                                       # outcome text -> courses that contain it
    for i, f in enumerate(fams):
        for n in outs[f]:
            who[n].add(i)
    queries, qsplit, qfam, qrel = [], [], [], []
    for i, f in enumerate(fams):
        split = "test" if f in test else "val" if f in val else None
        if split is None:
            continue
        cand = sorted((n for n in outs[f] if len(n.split()) >= MIN_Q_WORDS), key=lambda n: hashlib.md5(n.encode()).hexdigest())[:MAX_Q_PER_FAMILY]
        for n in cand:
            queries.append(outs[f][n]); qsplit.append(split); qfam.append(i); qrel.append({int(cluster[j]) for j in who[n]})
    return dict(fams=fams, docs=docs, chunks=chunks, chunk_fam=np.array(chunk_fam), starts=np.array(starts), cluster=cluster, queries=queries,
                qsplit=np.array(qsplit), qfam=np.array(qfam), qrel=qrel, n_train=len(fit), n_val=len(val), n_test=len(test),
                outs=outs, fit=fit, val=val, test=test)


def first_rank(S, D):
    order = np.argsort(-S, axis=1, kind="stable")[:, :300]
    r = np.full(len(S), 10 ** 6)
    for i in range(len(S)):
        hit = np.flatnonzero(np.isin(D["cluster"][order[i]], list(D["qrel"][i])))
        if len(hit):
            r[i] = hit[0] + 1
    return r


def metrics(r):
    return {"MRR@10": float(np.mean(np.where(r <= 10, 1 / r, 0))), "Recall@1": float(np.mean(r <= 1)), "Recall@5": float(np.mean(r <= 5)),
            "Recall@10": float(np.mean(r <= 10)), "nDCG@10": float(np.mean(np.where(r <= 10, 1 / np.log2(r + 1), 0)))}


def best_unit(sims, starts):
    return np.maximum.reduceat(sims, starts, axis=1)


def main():
    want = (sys.argv[sys.argv.index("--models") + 1].split(",") if "--models" in sys.argv else ["minilm", "bge-small", "nomic"])
    D = build()
    nq, nf = len(D["queries"]), len(D["fams"])
    print(f"{nf} courses ({len(set(D['cluster']))} clusters after merging near-duplicates), {len(D['chunks'])} unit chunks, "
          f"queries: val {int((D['qsplit'] == 'val').sum())}, test {int((D['qsplit'] == 'test').sum())}", flush=True)
    scores = {}                                                         # system -> (nq x nf) score matrix
    # ---- keyword systems
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    Xd = tf.fit_transform(D["docs"])
    scores["TF-IDF cosine (Week 2 baseline)"] = (tf.transform(D["queries"]) @ Xd.T).toarray()
    bm_doc = BM25Okapi([I.tokens(d) for d in D["docs"]])
    scores["BM25, whole course"] = np.stack([bm_doc.get_scores(I.tokens(q)) for q in D["queries"]])
    bm_chunk = BM25Okapi([I.tokens(c) for c in D["chunks"]])
    scores["BM25, best unit"] = best_unit(np.stack([bm_chunk.get_scores(I.tokens(q)) for q in D["queries"]]), D["starts"])
    print("keyword systems done", flush=True)
    # ---- dense systems
    sims_by_model = {}
    for key in want:
        if key == "ft" and not (INTERIM / "w4_minilm_ft").exists():
            print("no fine-tuned encoder yet, skipping", flush=True); continue
        label, make, qp, dp = MODELS[key]
        try:
            enc = make()
            Ec = enc([dp + t for t in D["chunks"]]); Eq = enc([qp + q for q in D["queries"]])
            enc.save()
        except Exception as e:
            print(f"{label}: skipped ({str(e)[:100]})", flush=True); continue
        fv = np.stack([Ec[D["chunk_fam"] == i].mean(0) for i in range(nf)]); fv /= np.linalg.norm(fv, axis=1, keepdims=True)
        sims = Eq @ Ec.T
        sims_by_model[key] = sims
        scores[f"{label} / course vector"] = Eq @ fv.T
        scores[f"{label} / best unit"] = best_unit(sims, D["starts"])
        print(f"{label} done", flush=True)
    R = {s: first_rank(S, D) for s, S in scores.items()}
    val_m, test_m = {}, {}
    for s, r in R.items():
        val_m[s] = metrics(r[D["qsplit"] == "val"])
    lex = [s for s in scores if s.startswith(("TF-IDF", "BM25"))]
    dense = [s for s in scores if s not in lex and "domain-adapted" not in s]
    best_lex = max(lex, key=lambda s: val_m[s]["MRR@10"])
    best_dense = max(dense, key=lambda s: val_m[s]["MRR@10"])
    # ---- hybrid + rerank
    scores[f"Hybrid: {best_lex} + {best_dense} (RRF)"] = hyb = I.rrf([scores[best_lex], scores[best_dense]])
    hyb_name = f"Hybrid: {best_lex} + {best_dense} (RRF)"
    ft_dense = [s for s in scores if "domain-adapted" in s]
    if ft_dense:
        best_ft = max(ft_dense, key=lambda s: val_m.get(s, metrics(first_rank(scores[s], D)[D["qsplit"] == "val"]))["MRR@10"])
        scores[f"Hybrid: {best_lex} + {best_ft} (RRF)"] = I.rrf([scores[best_lex], scores[best_ft]])
    # cross-encoder rerank of the hybrid's top candidates (test queries only; the setting is fixed, not tuned)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=256)
    cache_path = INTERIM / "w4_ce_cache.pkl"
    ce_cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {}
    dkey = next(k for k in want if MODELS[k][0] in best_dense)
    sims = sims_by_model[dkey]
    rer = np.array(hyb, dtype=float)
    test_idx = np.flatnonzero(D["qsplit"] == "test")
    for n_done, i in enumerate(test_idx):
        cand = np.argsort(-hyb[i])[:TOP_K_RERANK]
        pairs, owner = [], []
        for f in cand:
            lo = D["starts"][f]; hi = D["starts"][f + 1] if f + 1 < nf else len(D["chunks"])
            for ci in lo + np.argsort(-sims[i, lo:hi])[:2]:
                pairs.append((D["queries"][i], D["chunks"][ci])); owner.append(f)
        keys_ = [hashlib.md5((a + "||" + b).encode()).hexdigest() for a, b in pairs]
        todo = [k for k in dict.fromkeys(keys_) if k not in ce_cache]
        if todo:
            pmap = {k: p for k, p in zip(keys_, pairs)}
            pred = ce.predict([pmap[k] for k in todo], batch_size=32, show_progress_bar=False)
            ce_cache.update(dict(zip(todo, map(float, pred))))
        fam_score = defaultdict(lambda: -1e9)
        for f, k in zip(owner, keys_):
            fam_score[f] = max(fam_score[f], ce_cache[k])
        rer[i, :] = hyb[i] * 1e-3
        for f, sc in fam_score.items():
            rer[i, f] = 10 + sc
        if n_done % 100 == 0:
            print(f"  rerank {n_done}/{len(test_idx)}", flush=True)
            cache_path.write_bytes(pickle.dumps(ce_cache))
    cache_path.write_bytes(pickle.dumps(ce_cache))
    scores[f"Hybrid + cross-encoder rerank (top {TOP_K_RERANK})"] = rer
    R = {s: first_rank(S, D) for s, S in scores.items()}
    # ---- report (test courses only); rerank exists only for test queries
    is_t = D["qsplit"] == "test"
    out = {"setup": {"courses": nf, "clusters": int(len(set(D["cluster"]))), "unit_chunks": len(D["chunks"]), "fit_courses": D["n_train"], "val_courses": D["n_val"],
                     "test_courses": D["n_test"], "queries": {"val": int((~is_t).sum()), "test": int(is_t.sum())}, "cluster_cosine": CLUSTER_COS,
                     "selected_on_validation": {"keyword": best_lex, "dense": best_dense}},
           "validation_MRR@10": {s: round(v["MRR@10"], 3) for s, v in val_m.items()}, "test": {}}
    base_r = R["TF-IDF cosine (Week 2 baseline)"][is_t]
    for s, r in R.items():
        rt = r[is_t]
        m = metrics(rt)
        rr = np.where(rt <= 10, 1 / rt, 0)
        out["test"][s] = {**{k: round(v, 3) for k, v in m.items()}, "MRR@10_95ci": I.cluster_bootstrap(rr, D["qfam"][is_t]),
                          "MRR@10_gain_over_TFIDF_95ci": I.cluster_bootstrap(rr - np.where(base_r <= 10, 1 / base_r, 0), D["qfam"][is_t])}
    RES.mkdir(exist_ok=True)
    (RES / "w4_retrieval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"selected": out["setup"]["selected_on_validation"], "queries": out["setup"]["queries"]}))
    for s, v in sorted(out["test"].items(), key=lambda kv: -kv[1]["MRR@10"]):
        print(f'{v["MRR@10"]:.3f} [{v["MRR@10_95ci"][0]:.3f},{v["MRR@10_95ci"][1]:.3f}] R@1 {v["Recall@1"]:.3f} R@5 {v["Recall@5"]:.3f} R@10 {v["Recall@10"]:.3f} | {s}')


if __name__ == "__main__":
    main()
