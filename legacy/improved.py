"""Week 4 - components of the improved approach, shared by the experiments and by pipeline v2.

    OllamaEmbedder   nomic-embed-text through a local Ollama server (same call interface as pipeline.Embedder)
    ollama_generate  local LLM call (Phi-3 by default), deterministic
    tokens / BM25    keyword search
    rrf              reciprocal rank fusion (hybrid retrieval)
    cluster_bootstrap  confidence intervals that resample courses, not queries
"""
import hashlib, json, pickle, re, time, urllib.request
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data/interim"
OLLAMA = "http://localhost:11434"


def _post(path, body, timeout=900):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class OllamaEmbedder:
    """Unit-length embeddings from a local Ollama embedding model, cached on disk by text hash."""

    def __init__(self, model="nomic-embed-text", batch=32):
        self.name, self.batch = model, batch
        self.path = INTERIM / f"emb_ollama_{re.sub(r'[^A-Za-z0-9]+', '_', model)}.pkl"
        self.cache = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.dirty = False

    def __call__(self, texts):
        texts = list(texts)
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        miss = list({k: t for k, t in zip(keys, texts) if k not in self.cache}.items())
        for i in range(0, len(miss), self.batch):
            chunk = miss[i:i + self.batch]
            emb = self._embed([t for _, t in chunk])
            emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
            self.cache.update({k: e for (k, _), e in zip(chunk, emb)})
            self.dirty = True
        return np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, 768), np.float32)

    def _embed(self, texts):
        """One batch; if the server rejects it (it did once under memory pressure), retry item by item with shorter text."""
        try:
            return np.array(_post("/api/embed", {"model": self.name, "input": [t[:3000] for t in texts]})["embeddings"], np.float32)
        except Exception:
            out = []
            for t in texts:
                for cut in (3000, 1200, 500):
                    try:
                        out.append(np.array(_post("/api/embed", {"model": self.name, "input": [t[:cut]]})["embeddings"][0], np.float32))
                        break
                    except Exception:
                        if cut == 500:
                            raise
                        time.sleep(2)
            return np.stack(out)

    def save(self):
        if self.dirty:
            INTERIM.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(pickle.dumps(self.cache))
            self.dirty = False


def ollama_generate(prompt, model="phi3", num_predict=120, temperature=0.0, fmt=None, system=None, retries=2):
    """One deterministic completion from the local LLM. Returns the text; raises after `retries` failures."""
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature, "num_predict": num_predict, "seed": 7}}
    if fmt:
        body["format"] = fmt
    if system:
        body["system"] = system
    for attempt in range(retries + 1):
        try:
            return _post("/api/generate", body)["response"].strip()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(3)


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in ENGLISH_STOP_WORDS]


def rrf(score_matrices, k=60):
    """Reciprocal rank fusion of several (queries x items) score matrices."""
    out = 0
    for S in score_matrices:
        ranks = np.argsort(np.argsort(-S, axis=1), axis=1)
        out = out + 1.0 / (k + ranks + 1)
    return out


def cluster_bootstrap(values, groups, n=2000, seed=0):
    """95% CI of the mean of `values` (one per query), resampling whole groups (courses) because queries of a course are correlated."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    s, c = np.bincount(inv, weights=values), np.bincount(inv).astype(float)
    rng = np.random.RandomState(seed)
    idx = rng.randint(len(uniq), size=(n, len(uniq)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


# ---------------------------------------------------------------- hybrid course search (keyword + vector)
class HybridIndex:
    """Course search over unit descriptions: BM25 over whole courses and dense embeddings, fused with reciprocal rank fusion.

    The defaults reproduce the system that validation selected in w4_retrieval.py (BM25 on the whole course + bge-base course vectors).
    One entry per course name; `dense_agg` is 'mean' (course vector) or 'max' (best unit). `mode` is 'keyword', 'dense' or 'hybrid'."""

    def __init__(self, courses, embedder, query_prefix="", doc_prefix="", dense_agg="mean"):
        from collections import defaultdict
        from rank_bm25 import BM25Okapi
        units, latest = defaultdict(dict), {}
        for c in sorted(courses, key=lambda c: c["academic_year"]):
            latest[c["course_key"]] = c
            for u in c["units"]:
                units[c["course_key"]].setdefault(re.sub(r"\W+", " ", u["text"].lower()).strip(), u["text"])
        self.keys = sorted(k for k, d in units.items() if d)
        self.chunks, self.owner, starts = [], [], []
        for i, k in enumerate(self.keys):
            starts.append(len(self.chunks))
            for t in units[k].values():
                self.chunks.append(t); self.owner.append(i)
        self.starts, self.names, self.agg = np.array(starts), [latest[k]["course_name"] for k in self.keys], dense_agg
        self.example = [latest[k]["id"] for k in self.keys]
        self.bm25 = BM25Okapi([tokens(" ".join(units[k].values())) for k in self.keys])
        self.emb, self.qp = embedder, query_prefix
        self.E = embedder([doc_prefix + t for t in self.chunks])
        fv = np.stack([self.E[np.array(self.owner) == i].mean(0) for i in range(len(self.keys))])
        self.F = fv / np.linalg.norm(fv, axis=1, keepdims=True)

    def search(self, query, k=5, mode="hybrid"):
        qv = self.emb([self.qp + query])[0]
        chunk_sims = self.E @ qv
        dense = np.maximum.reduceat(chunk_sims, self.starts) if self.agg == "max" else self.F @ qv
        keyword = self.bm25.get_scores(tokens(query))
        S = {"keyword": keyword[None], "dense": dense[None], "hybrid": rrf([keyword[None], dense[None]])}[mode][0]
        out = []
        for i in np.argsort(-S)[:k]:
            lo = self.starts[i]; hi = self.starts[i + 1] if i + 1 < len(self.starts) else len(self.chunks)
            unit = lo + int(np.argmax(chunk_sims[lo:hi]))
            out.append({"course": self.names[i], "score": round(float(S[i]), 4), "best_unit": " ".join(self.chunks[unit].split()[:30]), "example_id": self.example[i]})
        return out


# ---------------------------------------------------------------- LLM-assisted summary of a comparison report
def _content_stems(text):
    from nltk.stem import PorterStemmer
    st = PorterStemmer().stem
    return {st(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in ENGLISH_STOP_WORDS}


def comparison_facts(r):
    a, b, o = r["course_a"], r["course_b"], r["overlap"]
    lines = [f"Course A: {a['course_name']} ({a['academic_year']}, {a['program'][:60]})", f"Course B: {b['course_name']} ({b['academic_year']}, {b['program'][:60]})",
             f"Unit overlap: {o['percent']}" + (f"; probability that reviewers call the pair overlapping: {round(100 * o['confidence'])}%" if o["confidence"] is not None else ""),
             f"Redundancy warning: {'yes' if o['redundancy_warning'] else 'no'}"]
    if r["common_topics"]:
        lines.append("Common topics: " + "; ".join(t["topic_a"] for t in r["common_topics"][:6]))
    if r["unique_to_a"]:
        lines.append("Only in A: " + "; ".join(r["unique_to_a"][:6]))
    if r["unique_to_b"]:
        lines.append("Only in B: " + "; ".join(r["unique_to_b"][:6]))
    if r["missing_prerequisites"]:
        lines.append("Prerequisites not covered: " + "; ".join(m["prerequisite"] for m in r["missing_prerequisites"][:4]))
    return "\n".join(lines)


def template_comparison_summary(r):
    o, a, b = r["overlap"], r["course_a"]["course_name"], r["course_b"]["course_name"]
    s = f"{a} and {b} share {o['percent']} of their units"
    s += f" (probability of being judged overlapping: {round(100 * o['confidence'])}%)." if o["confidence"] is not None else "."
    if r["common_topics"]:
        s += " Common topics: " + ", ".join(t["topic_a"] for t in r["common_topics"][:4]) + "."
    if r["unique_to_a"]:
        s += f" Only in {a}: " + ", ".join(r["unique_to_a"][:3]) + "."
    if r["unique_to_b"]:
        s += f" Only in {b}: " + ", ".join(r["unique_to_b"][:3]) + "."
    return s + (" Redundancy warning raised." if o["redundancy_warning"] else "")


def llm_comparison_summary(r, model="phi3", max_unsupported=0.2):
    """3-sentence committee summary written by the local LLM from the report's facts. If more than `max_unsupported` of its content words
    are absent from the facts, or a percentage differs from the report, the template summary is returned instead."""
    facts = comparison_facts(r)
    prompt = ("You summarise a comparison of two university courses for a curriculum committee.\nFacts:\n" + facts +
              "\n\nWrite exactly 3 short sentences: how much the courses overlap, which topics they share or lack, and whether the committee should act. "
              "Use only the facts above; do not invent topics or numbers.")
    text = re.sub(r"\s+", " ", ollama_generate(prompt, model=model, num_predict=130)).strip()
    allowed = {str(round(100 * r["overlap"]["confidence"])) if r["overlap"]["confidence"] is not None else "", r["overlap"]["percent"].rstrip("%")}
    pct_ok = all(p in allowed for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", text))
    src = _content_stems(facts) | _content_stems("course courses overlap topic topics unit units share shared committee review redundancy prerequisite prerequisites recommend consider "
                                                 "covered cover covers similar different content curriculum should action likely significant partial substantial")
    stems = _content_stems(text)
    unsupported = len(stems - src) / max(len(stems), 1)
    ok = pct_ok and unsupported <= max_unsupported
    return {"text": text if ok else template_comparison_summary(r), "method": "llm" if ok else "template (llm output failed the faithfulness check)",
            "llm_text": text, "unsupported_term_rate": round(unsupported, 3), "percentages_consistent": pct_ok}
