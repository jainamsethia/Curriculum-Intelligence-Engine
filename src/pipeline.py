"""Week 3 - Curriculum Intelligence pipeline v1 (the core NLP component).

    text/PDF/DOCX/parsed syllabus -> structured course -> sentence-transformer embeddings
        -> overlap score + common / unique topics + redundancy flag          (semantic similarity)
        -> prerequisite extraction + resolution + missing-prerequisite check (information extraction)
        -> outcome -> Bloom's-taxonomy level                                 (classification)
        -> JSON report with confidence scores

CLI:
    python src/pipeline.py compare --a nm0012 --b nm0345 [--out report.json]
    python src/pipeline.py compare --a syllabus_a.pdf --b syllabus_b.docx
    python src/pipeline.py outcomes --course nm0012
    python src/pipeline.py search "transformers and large language models" -k 5 [--mode hybrid|dense|keyword|course-vector]
    python src/pipeline.py compare --a nm59a8f694 --b nm7677637d --summary      # adds an LLM summary (local Ollama, phi3)
"""
import argparse, hashlib, json, pickle, re, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA, INTERIM = ROOT / "data/processed/nmims_courses.jsonl", ROOT / "data/interim"
CALIB = ROOT / "data/processed/calibration.json"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BLOOM = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyze", 5: "Evaluate", 6: "Create"}
ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
# defaults used until evaluate_v1.py has written data/processed/calibration.json
# Week 4: encoder behind Pipeline.search (chosen on validation courses in w4_retrieval.py)
SEARCH_ENCODER = "BAAI/bge-base-en-v1.5"          # validation-selected dense model with the course-vector aggregator
SEARCH_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_CAL = {"tau_unit": 0.65, "tau_prereq_resolve": 0.75, "tau_prereq_cover": 0.5, "logit": None, "tfidf_fit_courses": None}


# ---------------------------------------------------------------- embeddings
class Embedder:
    """Sentence-transformer with an on-disk cache keyed by text hash (re-runs and evaluation are then instant)."""

    def __init__(self, model=MODEL):
        self.name, self._m = model, None
        self.path = INTERIM / f"emb_{re.sub(r'[^A-Za-z0-9]+', '_', model)}.pkl"
        self.cache = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.dirty = False

    def __call__(self, texts):
        texts = list(texts)
        keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
        miss = {k: t for k, t in zip(keys, texts) if k not in self.cache}
        if miss:
            if self._m is None:
                from sentence_transformers import SentenceTransformer
                self._m = SentenceTransformer(self.name)
            vec = self._m.encode(list(miss.values()), batch_size=64, normalize_embeddings=True, show_progress_bar=len(miss) > 800)
            self.cache.update(zip(miss, vec.astype(np.float32)))
            self.dirty = True
        return np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, 384), np.float32)

    def save(self):
        if self.dirty:
            INTERIM.mkdir(parents=True, exist_ok=True)
            self.path.write_bytes(pickle.dumps(self.cache))
            self.dirty = False


def unit_label(u):
    return u["title"] or " ".join(u["text"].split()[:8])


def parts(c, which=("objectives", "outcomes", "units")):
    """The text chunks a course is made of, each short enough for the 256-token encoder window."""
    out = []
    if "objectives" in which and c.get("objectives"):
        out.append(c["objectives"])
    if "outcomes" in which:
        out += [o["text"] for o in c.get("outcomes", [])]
    if "units" in which:
        out += [u["text"] for u in c.get("units", [])]
    return out


def text_of(c):
    """Lexical view of a course: objectives + outcomes + units. Title, code and prerequisites are left out so they cannot leak identity."""
    return " ".join(parts(c))


def unit_texts(c):
    """Units are the 'topics' of a course; a course without units (e.g. a project) falls back to its outcomes."""
    us = [(unit_label(u), u["text"]) for u in c.get("units", [])]
    return us or [(o["text"][:60], o["text"]) for o in c.get("outcomes", [])] or ([("objectives", c["objectives"])] if c.get("objectives") else [])


def course_vec(emb, c, which=("objectives", "outcomes", "units")):
    """Course embedding = normalised mean of its chunk embeddings (avoids truncating a long syllabus to 256 tokens)."""
    E = emb(parts(c, which))
    if not len(E):
        return np.zeros(384, np.float32)
    v = E.mean(0)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------- overlap
def align(A, B, tau):
    """Bidirectional soft alignment of two unit-embedding matrices (rows L2-normalised)."""
    if not len(A) or not len(B):
        return {"S": np.zeros((len(A), len(B))), "best_a": np.zeros(len(A)), "best_b": np.zeros(len(B)), "cov_a": 0.0, "cov_b": 0.0, "overlap": 0.0, "soft": 0.0}
    S = A @ B.T
    ab, ba = S.max(1), S.max(0)
    ha, hb = float((ab >= tau).mean()), float((ba >= tau).mean())
    f = 2 * ha * hb / (ha + hb) if ha + hb else 0.0
    return {"S": S, "best_a": ab, "best_b": ba, "cov_a": ha, "cov_b": hb, "overlap": f, "soft": float((ab.mean() + ba.mean()) / 2)}


def load_calibration():
    return {**DEFAULT_CAL, **(json.loads(CALIB.read_text()) if CALIB.exists() else {})}


FEATURES = ["tfidf_cosine", "doc_cosine", "unit_overlap", "soft_coverage"]


def calibrated(cal, feat):
    """Probability that annotators would judge the pair as overlapping (logistic model fitted on the gold pairs)."""
    lg = cal.get("logit")
    if not lg:
        return None
    x = (np.array([feat[k] for k in lg["features"]]) - lg["mean"]) / lg["scale"]
    return float(1 / (1 + np.exp(-(np.dot(lg["coef"], x) + lg["intercept"]))))


# ---------------------------------------------------------------- prerequisites
NONE_PREREQ = re.compile(r"^\W*(nil|na|n/?a|none|not applicable|no prerequisites?|-+)\W*$", re.I)
LEAD = re.compile(r"^(?:(?:basic|fundamental|elementary|working|good|sound|prior|general|introductory)\s+)*"
                  r"(?:knowledge|understanding|concepts?|familiarity|awareness|proficiency|grasp|idea|basics|fundamentals|principles)"
                  r"(?:\s+(?:of|in|about|with|on))?\s+|^(?:basics|fundamentals|introduction)\s+(?:of|to)\s+", re.I)


def split_prereq(text):
    """'Physics (BTME01003), Calculus and Linear Algebra' -> ['Physics', 'Calculus', 'Linear Algebra']."""
    t = re.sub(r"\(([^)]*\d[^)]*)\)", " ", text or "")          # drop course codes in brackets
    if not t.strip() or NONE_PREREQ.match(t.strip()):
        return []
    out = []
    for p in re.split(r"[;,\n]|\band\b|&|\bas well as\b|\bor\b|/", t):
        p = LEAD.sub("", re.sub(r"\s+", " ", p).strip(" .:-–"))
        if 2 < len(p) <= 90 and not NONE_PREREQ.match(p):
            out.append(p)
    return list(dict.fromkeys(out))


def prog_key(p):
    return re.sub(r"[^a-z0-9]+", "", (p or "").lower())


def sem_min(s):
    v = [ROMAN[x] for x in re.findall(r"\b(?:VIII|VII|VI|IV|IX|V|III|II|I|X)\b", (s or "").upper())]
    return min(v) if v else None


class Catalog:
    """All known courses, with name + content embeddings so a prerequisite phrase can be resolved to a course."""

    def __init__(self, emb, courses):
        self.emb, self.courses = emb, courses
        names = sorted({c["course_name"] for c in courses})
        self.names = names
        self.name_vec = emb(names)
        rep = {}                                                   # one representative record per course name
        for c in courses:
            rep.setdefault(c["course_name"], c)
        self.rep = [rep[n] for n in names]
        self.content_vec = np.stack([course_vec(emb, r) for r in self.rep])

    def resolve(self, phrase, tau):
        q = self.emb([phrase])[0]
        sn, sc = self.name_vec @ q, self.content_vec @ q
        s = np.maximum(sn, 0.8 * sc + 0.1)                          # content match is looser than a name match
        i = int(s.argmax())
        return (self.names[i], float(s[i])) if s[i] >= tau else (None, float(s[i]))


def extract_prereqs(course, cat, cal):
    """Prerequisite phrases of a course, each resolved to a catalogue course when the match is confident (audited: 87% precise at 0.75)."""
    out = []
    for ph in split_prereq(course.get("prerequisites", "")):
        name, sim = cat.resolve(ph, cal["tau_prereq_resolve"])
        r = {"phrase": ph, "resolved_course": name, "similarity": round(sim, 3)}
        if name:                                                   # schedule check inside the same programme only
            mine, prog = sem_min(course.get("semester")), prog_key(course.get("program"))
            same = [sem_min(c["semester"]) for c in cat.courses if c["course_name"] == name and prog_key(c["program"]) == prog and sem_min(c["semester"])]
            r["offered_before"] = (min(same) < mine) if same and mine else None
        out.append(r)
    return out


def sequencing_warnings(course, cat, cal):
    return [{"course": course["course_name"], "prerequisite": p["phrase"], "resolved_course": p["resolved_course"],
             "reason": "prerequisite course is scheduled in the same or a later semester of this programme"}
            for p in extract_prereqs(course, cat, cal) if p.get("offered_before") is False]


def missing_prereqs(target, other, cat, cal, emb):
    """Prerequisite phrases of `target` that neither resolve to a catalogue course nor are covered by `other`'s content."""
    O = emb([t for _, t in unit_texts(other)] + [o["text"] for o in other.get("outcomes", [])])
    res = []
    for p in extract_prereqs(target, cat, cal):
        cov = float((O @ emb([p["phrase"]])[0]).max()) if len(O) else 0.0
        if p["resolved_course"] is None and cov < cal["tau_prereq_cover"]:
            res.append({"course": target["course_name"], "prerequisite": p["phrase"], "best_match_in_other": round(cov, 3),
                        "reason": "not taught in the other course and not confidently matched to a course in the catalogue"})
    return res


# ---------------------------------------------------------------- Bloom's taxonomy
BLOOM_VERBS = {
    1: "define list recall recognize recognise name state label memorize repeat retrieve enumerate cite reproduce know mention recite tell identify",
    2: "explain describe summarize summarise discuss interpret classify illustrate paraphrase understand comprehend distinguish outline review translate express relate convert compare",
    3: "apply use implement solve execute demonstrate compute calculate operate perform employ utilize utilise practice show simulate install configure write draw sketch program select measure",
    4: "analyze analyse examine contrast categorize categorise investigate deduce infer diagnose experiment organize organise survey decompose debug troubleshoot differentiate test model",
    5: "evaluate assess justify critique judge defend recommend appraise validate verify argue support prioritize prioritise rate optimize optimise conclude decide select",
    6: "design develop create formulate construct build compose devise plan propose synthesize synthesise invent generate integrate assemble architect engineer author produce",
}
VERB_LEVEL = {}
for lvl, words in BLOOM_VERBS.items():
    for w in words.split():
        VERB_LEVEL.setdefault(w, lvl)          # a verb listed at several levels keeps its lowest
SKIP_WORDS = set("to be able will would can could students student the a an and or of in on for with also should".split())


def first_word(text):
    """The action verb an outcome starts with ('Students will be able to effectively design ...' -> 'design')."""
    for w in re.findall(r"[a-z]+", text.lower())[:7]:
        if w not in SKIP_WORDS and not w.endswith("ly"):
            return w
    return None


def bloom_rule(text):
    """Level of the first hand-listed Bloom verb among the first 5 words; None if none matches."""
    for w in re.findall(r"[a-z]+", text.lower())[:5]:
        if w in VERB_LEVEL:
            return VERB_LEVEL[w]
    return None


def bloom_features(emb, texts, rule_scale=3.0):
    E = emb(texts)
    R = np.zeros((len(texts), 7), np.float32)
    for i, t in enumerate(texts):
        R[i, bloom_rule(t) or 0] = rule_scale         # column 0 = no verb matched
    return np.hstack([E, R])


def learn_verb_map(texts, levels, min_count=2):
    """verb -> (level faculty tagged it with most often, share of that level) for verbs seen at least min_count times."""
    by = defaultdict(Counter)
    for t, l in zip(texts, levels):
        w = first_word(t)
        if w:
            by[w][l] += 1
    return {w: (c.most_common(1)[0][0], c.most_common(1)[0][1] / sum(c.values())) for w, c in by.items() if sum(c.values()) >= min_count}


class BloomClassifier:
    """Outcome text -> Bloom level, trained on faculty-tagged (K/L) outcomes.
    Two verb look-ups are tried in an order chosen by grouped cross-validation on the training courses only:
      learned map = the level faculty most often tagged the outcome's action verb with;  lexicon = a hand-written Bloom verb list.
    Outcomes with no known verb fall back to logistic regression on the sentence embedding."""

    def __init__(self, emb):
        self.emb, self.clf, self.vmap, self.order, self.cv = emb, None, {}, ("learned", "lexicon"), None

    def fit(self, texts, levels, groups=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        self.vmap = learn_verb_map(texts, levels)
        self.clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(bloom_features(self.emb, texts), levels)
        if groups is not None and len(set(groups)) >= 10:                     # choose the look-up order without touching test data
            hit, n = {"learned": 0, "lexicon": 0}, 0
            for tr, te in GroupKFold(5).split(texts, levels, groups):
                vm = learn_verb_map([texts[i] for i in tr], [levels[i] for i in tr])
                for i in te:
                    w, r = first_word(texts[i]), bloom_rule(texts[i])
                    if w in vm and r:
                        n += 1
                        hit["learned"] += vm[w][0] == levels[i]
                        hit["lexicon"] += r == levels[i]
            self.cv = {"outcomes_where_both_lookups_apply": int(n), "learned_correct": int(hit["learned"]), "lexicon_correct": int(hit["lexicon"])}
            self.order = ("learned", "lexicon") if hit["learned"] > hit["lexicon"] else ("lexicon", "learned")
        return self

    def predict(self, texts):
        P = self.clf.predict_proba(bloom_features(self.emb, texts))
        out = []
        for t, p in zip(texts, P):
            w, r = first_word(t), bloom_rule(t)
            found = {"learned": (self.vmap[w][0], self.vmap[w][1]) if w in self.vmap else None, "lexicon": (r, 0.5) if r else None}
            first = next((k for k in self.order if found[k]), None)
            if first:
                lvl, conf = found[first]
                how = "learned verb map" if first == "learned" else "verb lexicon"
            else:
                i = int(p.argmax())
                lvl, conf, how = int(self.clf.classes_[i]), float(p[i]), "embedding classifier"
            out.append({"outcome": t, "level": int(lvl), "name": BLOOM[int(lvl)], "confidence": round(float(conf), 3), "method": how})
        return out


def bloom_training_data(courses):
    """Single-level tagged outcomes; identical texts are kept once (many syllabi are re-used across years/branches).
    Returns texts, levels and the course each came from (for grouped cross-validation)."""
    seen, X, y, g = set(), [], [], []
    for c in courses:
        for o in c.get("outcomes", []):
            if len(o["k_levels"]) == 1 and o["text"] not in seen:
                seen.add(o["text"]); X.append(o["text"]); y.append(o["k_levels"][0]); g.append(c["course_key"])
    return X, y, g


# ---------------------------------------------------------------- the pipeline
class Pipeline:
    def __init__(self, courses=None, model=MODEL):
        self.emb = Embedder(model)
        self.courses = courses if courses is not None else load_courses()
        self.by_id = {c["id"]: c for c in self.courses}
        self.cal = load_calibration()
        self._cat = self._bloom = self._tfidf = self._index = None

    # lazily built helpers -------------------------------------------------
    @property
    def catalog(self):
        if self._cat is None:
            self._cat = Catalog(self.emb, self.courses)
        return self._cat

    @property
    def bloom(self):
        if self._bloom is None:
            self._bloom = BloomClassifier(self.emb).fit(*bloom_training_data(self.courses))
        return self._bloom

    @property
    def tfidf(self):
        """Lexical model. Fitted on the same (training) courses that evaluate_v1.py used, so calibration and inference agree."""
        if self._tfidf is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            fit = set(self.cal.get("tfidf_fit_courses") or [])
            docs = [text_of(c) for c in self.courses if not fit or c["course_key"] in fit] or [text_of(c) for c in self.courses]
            self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.8, sublinear_tf=True).fit(docs)
        return self._tfidf

    def get(self, ref):
        """A course id from the dataset, or a path to a .pdf/.docx/.txt syllabus ('booklet.pdf::Discrete Mathematics' picks one course)."""
        if ref in self.by_id:
            return self.by_id[ref]
        path, _, name = ref.partition("::")
        return course_from_file(Path(path), name)

    def pair_features(self, a, b):
        cal = self.cal
        al = align(self.emb([t for _, t in unit_texts(a)]), self.emb([t for _, t in unit_texts(b)]), cal["tau_unit"])
        Ta, Tb = self.tfidf.transform([text_of(a)]), self.tfidf.transform([text_of(b)])
        feat = {"tfidf_cosine": float(Ta.multiply(Tb).sum()), "doc_cosine": float(course_vec(self.emb, a) @ course_vec(self.emb, b)),
                "unit_overlap": al["overlap"], "soft_coverage": al["soft"]}
        return feat, al

    # public API ------------------------------------------------------------
    def compare(self, a, b, max_topics=12, summary=False):
        cal = self.cal
        ua, ub = unit_texts(a), unit_texts(b)
        feat, al = self.pair_features(a, b)
        conf = calibrated(cal, feat)
        S = al["S"]
        common = []
        for i in np.argsort(-al["best_a"]):
            j = int(S[i].argmax())
            if S[i, j] >= cal["tau_unit"]:
                common.append({"topic_a": ua[i][0], "topic_b": ub[j][0], "similarity": round(float(S[i, j]), 3)})
        uniq_a = [ua[i][0] for i in range(len(ua)) if al["best_a"][i] < cal["tau_unit"]]
        uniq_b = [ub[j][0] for j in range(len(ub)) if al["best_b"][j] < cal["tau_unit"]]
        meta = lambda c: {k: c.get(k, "") for k in ("id", "course_name", "code", "program", "semester", "academic_year")}
        redundant = (conf >= 0.8) if conf is not None else al["overlap"] >= 0.5
        report = {
            "course_a": meta(a), "course_b": meta(b),
            "overlap": {"score": round(al["overlap"], 3), "percent": f"{al['overlap']:.0%}",
                        "confidence": None if conf is None else round(conf, 3),
                        "similarity": {k: round(v, 3) for k, v in feat.items()},
                        "coverage_of_a_in_b": round(al["cov_a"], 3), "coverage_of_b_in_a": round(al["cov_b"], 3),
                        "redundancy_warning": bool(redundant)},
            "common_topics": common[:max_topics], "unique_to_a": uniq_a[:max_topics], "unique_to_b": uniq_b[:max_topics],
            "prerequisites": {"a": extract_prereqs(a, self.catalog, cal), "b": extract_prereqs(b, self.catalog, cal)},
            "missing_prerequisites": missing_prereqs(a, b, self.catalog, cal, self.emb) + missing_prereqs(b, a, self.catalog, cal, self.emb),
            "sequencing_warnings": sequencing_warnings(a, self.catalog, cal) + sequencing_warnings(b, self.catalog, cal),
            "outcome_bloom": {"a": self.map_outcomes(a), "b": self.map_outcomes(b)},
            "method": f"{self.emb.name} unit alignment (tau={cal['tau_unit']:.2f}) + TF-IDF, logistic calibration" + ("" if conf is not None else " [uncalibrated: run evaluate_v1.py]"),
        }
        if summary:                                   # Week 4: LLM-assisted committee summary with a faithfulness guard (needs a local Ollama + phi3)
            import improved
            report["summary"] = improved.llm_comparison_summary(report)
        return report

    def map_outcomes(self, c):
        """Bloom level for every learning outcome of a course (the faculty K/L tag is kept alongside when the syllabus has one)."""
        outs = c.get("outcomes", [])
        if not outs:
            return []
        pred = self.bloom.predict([o["text"] for o in outs])
        for p, o in zip(pred, outs):
            if o["k_levels"]:
                p["faculty_tag"] = o["k_levels"]
        return pred

    def search(self, query, k=5, mode="hybrid"):
        """Week 4 course search over unit descriptions: BM25 + dense embeddings fused by reciprocal rank fusion (mode: hybrid, dense, keyword).
        Evaluated against TF-IDF / BM25 / single encoders in w4_retrieval.py."""
        if self._index is None:
            import improved
            enc = Embedder(SEARCH_ENCODER)
            self._index = improved.HybridIndex(self.courses, enc, SEARCH_QUERY_PREFIX)
            enc.save()
        return self._index.search(query, k, mode)

    def similar(self, query, k=5):
        """Semantic search: courses (one per name) whose content is closest to a free-text query."""
        q = self.emb([query])[0]
        s = self.catalog.content_vec @ q
        return [{"course": self.catalog.names[i], "similarity": round(float(s[i]), 3), "example_id": self.catalog.rep[i]["id"]}
                for i in np.argsort(-s)[:k]]


# ---------------------------------------------------------------- input handling
def load_courses(path=DATA):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def course_from_file(path, name=""):
    """PDF/DOCX/TXT -> course record. NMIMS-template syllabi are parsed into units/outcomes (the first course of a booklet, or the one
    whose name contains `name`); anything else falls back to paragraphs so the comparison still works (no prerequisites/outcomes then)."""
    sys.path.insert(0, str(Path(__file__).parent))
    import parse_nmims as P
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        import fitz
        pages = [p.get_text() for p in fitz.open(path)]
    elif suffix == ".docx":
        xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
        pages = [re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))]
    else:
        pages = [path.read_text(encoding="utf-8", errors="ignore")]
    for block in P.split_blocks(P.clean_pages(pages)):
        rec = P.parse_block(block, str(path))
        if rec and name.lower() in rec["course_name"].lower():
            rec["id"] = path.stem
            return rec
    if name:
        raise SystemExit(f"no course named like '{name}' found in {path}")
    paras = [P.norm(p) for p in re.split(r"\n\s*\n|\n(?=[A-Z0-9][^\n]{0,80}\n)", "\n".join(pages)) if len(P.norm(p).split()) >= 8]
    return {"id": path.stem, "course_name": path.stem, "code": "", "program": "", "semester": "", "academic_year": "",
            "prerequisites": "", "objectives": "", "outcomes": [],
            "units": [{"title": "", "text": p, "hours": None} for p in paras[:60]]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.add_argument("--a", required=True); c.add_argument("--b", required=True); c.add_argument("--out")
    c.add_argument("--summary", action="store_true", help="add an LLM-written summary (needs Ollama with phi3)")
    o = sub.add_parser("outcomes"); o.add_argument("--course", required=True)
    s = sub.add_parser("search"); s.add_argument("query"); s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", choices=["hybrid", "dense", "keyword", "course-vector"], default="hybrid", help="course-vector = the Week 3 method")
    args = ap.parse_args()
    p = Pipeline()
    if args.cmd == "compare":
        res = p.compare(p.get(args.a), p.get(args.b), summary=args.summary)
    elif args.cmd == "outcomes":
        res = p.map_outcomes(p.get(args.course))
    else:
        res = p.similar(args.query, args.k) if args.mode == "course-vector" else p.search(args.query, args.k, args.mode)
    p.emb.save()
    text = json.dumps(res, indent=2, ensure_ascii=False)
    if getattr(args, "out", None):
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
