"""Steps 2-5 - Cleaning, preprocessing, (silver) annotation and train/test split.

Reads data/raw/ocw/*.json (+ optional own documents in data/raw/local/*.txt|*.docx|*.pdf) and writes
    data/processed/courses.jsonl, courses.csv   one row per course in the common schema
    data/annotation/pairs_silver.csv            every course pair with a silver overlap score/label
    data/annotation/gold_pairs.csv              stratified test pairs to label (created once, never overwritten)
    results/dataset_stats.json

    python src/preprocess.py
"""
import csv, hashlib, itertools, json, math, random, re, unicodedata, zipfile
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup
import spacy
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
RAW, LOCAL = ROOT / "data/raw/ocw", ROOT / "data/raw/local"
PROC, ANN, RES = ROOT / "data/processed", ROOT / "data/annotation", ROOT / "results"
SEED, TEST_SIZE, MIN_WORDS = 42, 0.3, 50

# Syllabus sections that carry course content vs. logistics. Checked in this order.
KEEP_HEAD = re.compile(r"prereq|descr|overview|content|topic|outline|objective|goal|outcome|purpose|about|learn|scope|introduction")
DROP_HEAD = re.compile(r"meeting|grad|exam|quiz|problem set|homework|assignment|polic|collaborat|honest|integrity|attend|late|"
                       r"text|book|reading|software|staff|lecturer|instructor|team|evaluat|requirement|recitation|format|"
                       r"participat|laptop|matlab|makeup|extension|habit|reference|project|calendar|video|tutorial|office|contact")
OUTCOME_HEAD = re.compile(r"objective|goal|outcome|purpose|learn")
OUTCOME_SENT = re.compile(r"\b(students?|you) (will|should|can) (be able to|learn|understand|gain|develop)|by the end of (the|this)|upon (successful )?completion", re.I)
NOISE_CELL = re.compile(r"^(no (class|lecture)|holiday|.*\b(quiz|exam|midterm|review session|due|out)\b.*|student holiday.*)$", re.I)
COURSE_REF = re.compile(r"\b(?:\d{1,2}[A-Z]?|[A-Z]{2,4})\.[S\d]\d{1,3}[A-Z]{0,2}\b")
# OCW top-level categories: too broad to count as evidence of overlap.
GENERIC_TAGS = {"Mathematics", "Engineering", "Science", "Social Science", "Business", "Humanities",
                "Fine Arts", "Health and Medicine", "Energy", "Teaching and Education"}
MIN_LABEL_SUPPORT = 8      # a topic tag becomes a classification label if >= 8 courses carry it
OVERLAP_THRESHOLD = 0.30   # silver label: weighted tag-Jaccard >= this => "overlapping pair" (picked by inspecting pairs)


def clean(s):
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"https?://\S+|\S+@\S+", " ", s)
    s = re.sub(r"\(\s*(PDF|MP4|ZIP|TXT|XLS)[^)]*\)", " ", s, flags=re.I)  # OCW file-type markers
    s = re.sub(r"\b(courtesy of|image by|used with permission)[^.]*\.?", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def sections(html):
    """Split a page's <main> into (heading, text) blocks; tables are handled separately."""
    soup = BeautifulSoup(html, "html.parser")
    for t in soup.find_all(["table", "tr"]):
        t.decompose()
    out, head, buf = [], "", []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        if el.name.startswith("h"):
            out.append((head, clean(" ".join(buf)))); head, buf = el.get_text(" ", strip=True).lower(), []
        elif not el.find_parent("li"):  # nested <li>/<p> text is already inside the parent <li>
            buf.append(el.get_text(" ", strip=True))
    out.append((head, clean(" ".join(buf))))
    return [(h, t) for h, t in out if t]


def keep(head):
    return bool(KEEP_HEAD.search(head)) or not DROP_HEAD.search(head)


def table_topics(html):
    """TOPICS / SUBJECTS column of the calendar & lecture-notes tables.
    OCW nests tables inside <p>, so the parser detaches <tr>s from <table>: walk rows in document order."""
    topics, col = [], None
    for tr in BeautifulSoup(html, "html.parser").find_all("tr"):
        cells = [clean(c.get_text(" ", strip=True)) for c in tr.find_all(["th", "td"])]
        if tr.find("th"):  # header row starts a new table
            col = next((i for i, h in enumerate(cells) if re.search(r"topic|subject", h, re.I)), None)
        elif col is not None and col < len(cells):
            cell = re.sub(r"^(lecture|lec|session|ses|week)\s*\d+\s*[:.-]?\s*", "", cells[col], flags=re.I)
            if len(cell) > 2 and not NOISE_CELL.match(cell):
                topics.append(cell)
    return topics


def from_raw(d):
    m, pages = d["meta"], d["pages"]
    secs = [(h, t) for h, t in sections(pages.get("syllabus", "")) if keep(h)]
    lecture_topics = [t for p in ("calendar", "lecture-notes", "syllabus") for t in table_topics(pages.get(p, ""))]
    if not lecture_topics:  # pages without a topics table: fall back to their kept text blocks
        secs += [(h, t) for p in ("calendar", "lecture-notes") for h, t in sections(pages.get(p, "")) if keep(h)]
    lecture_topics = list(dict.fromkeys(lecture_topics))  # de-duplicate, keep order
    prereq = " ".join(t for h, t in secs if "prereq" in h)
    body = [t for h, t in secs if "prereq" not in h]
    return {
        "id": m["id"], "source": "MIT OCW", "url": m["url"], "license": m["license"],
        "course_number": (m["course_numbers"] or [m["readable_id"].split("+")[0]])[0],
        "title": m["title"], "department": (m["departments"] or [""])[0], "level": "/".join(m["level"]),
        "offering": m["readable_id"].split("+")[-1],
        "description": clean(BeautifulSoup(m["full_description"] or m["description"], "html.parser").get_text(" ")),
        "prerequisites_text": prereq, "outcomes_raw": [t for h, t in secs if OUTCOME_HEAD.search(h)],
        "syllabus_text": " ".join(body), "lecture_topics": lecture_topics, "ocw_topics": m["ocw_topics"],
    }


def from_local(path):
    """Own course outlines/handbooks (e.g. NMIMS docs). No topic tags => unlabeled, excluded from evaluation."""
    if path.suffix == ".pdf":
        import fitz  # PyMuPDF
        text = " ".join(page.get_text() for page in fitz.open(path))
    elif path.suffix == ".docx":
        xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf8")
        text = " ".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", xml))
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
    text = clean(text)
    m = re.search(r"prerequisites?\s*:?(.{0,400}?)(?=\.\s|$)", text, re.I)
    return {"id": f"local_{path.stem}", "source": "local", "url": "", "license": "own/institutional",
            "course_number": path.stem, "title": path.stem.replace("_", " "), "department": "", "level": "",
            "offering": "", "description": "", "prerequisites_text": m.group(1).strip() if m else "",
            "outcomes_raw": [], "syllabus_text": text, "lecture_topics": [], "ocw_topics": []}


def main():
    for p in (PROC, ANN, RES, LOCAL):
        p.mkdir(parents=True, exist_ok=True)
    docs = [from_raw(json.loads(f.read_text(encoding="utf-8"))) for f in sorted(RAW.glob("*.json"))]
    docs += [from_local(f) for f in sorted(LOCAL.glob("*")) if f.suffix in (".txt", ".docx", ".pdf")]
    n_raw = len(docs)

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    seen, kept = set(), []
    for d in docs:
        d["text"] = ". ".join(x for x in [d["title"], d["description"], d["prerequisites_text"], d["syllabus_text"],
                                          ". ".join(d["lecture_topics"])] if x)
        d["n_words"] = len(d["text"].split())
        h = hashlib.md5(d["text"].lower().encode()).hexdigest()
        if d["n_words"] < MIN_WORDS or h in seen:  # quality filter + exact-duplicate removal
            continue
        seen.add(h)
        doc = nlp(d["text"])
        d["clean_text"] = " ".join(t.lemma_.lower() for t in doc
                                   if t.is_alpha and not t.is_stop and len(t) > 1)
        d["outcomes"] = [s.text.strip() for o in d.pop("outcomes_raw") for s in nlp(o).sents] + \
                        [s.text.strip() for s in doc.sents if OUTCOME_SENT.search(s.text)]
        d["outcomes"] = list(dict.fromkeys(d["outcomes"]))
        d["prereq_course_refs"] = sorted(set(COURSE_REF.findall(d["prerequisites_text"])) - {d["course_number"]})
        kept.append(d)

    # --- annotation: human-curated OCW tags -> classification labels + pairwise silver overlap ---
    labeled = [d for d in kept if d["ocw_topics"]]
    specific = lambda d: set(d["ocw_topics"]) - GENERIC_TAGS
    df = Counter(t for d in labeled for t in specific(d))
    label_set = sorted(t for t, c in df.items() if c >= MIN_LABEL_SUPPORT)
    idf = {t: math.log(len(labeled) / c) for t, c in df.items()}
    for d in kept:
        d["labels"] = [t for t in label_set if t in d["ocw_topics"]]

    # --- course-level split, grouped by course number so two offerings never straddle train/test ---
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr, _ = next(gss.split(labeled, groups=[d["course_number"] for d in labeled]))
    tr = {labeled[i]["id"] for i in tr}
    for d in kept:
        d["split"] = "unlabeled" if not d["ocw_topics"] else ("train" if d["id"] in tr else "test")

    with open(PROC / "courses.jsonl", "w", encoding="utf-8") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    cols = ["id", "split", "course_number", "title", "department", "level", "offering", "n_words", "labels",
            "ocw_topics", "prereq_course_refs", "prerequisites_text", "description", "lecture_topics", "url"]
    with open(PROC / "courses.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(cols)
        for d in kept:
            w.writerow(["; ".join(d[c]) if isinstance(d[c], list) else d[c] for c in cols])

    pairs = []
    for a, b in itertools.combinations(labeled, 2):
        A, B = specific(a), specific(b)
        score = sum(idf[t] for t in A & B) / (sum(idf[t] for t in A | B) or 1)
        same = a["course_number"] == b["course_number"]  # same course, different semester = known overlap
        score = 1.0 if same else score
        split = a["split"] if a["split"] == b["split"] else "cross"
        pairs.append({"course_a": a["id"], "course_b": b["id"], "title_a": a["title"], "title_b": b["title"],
                      "split": split, "same_course": int(same), "shared_tags": "; ".join(sorted(A & B)),
                      "silver_score": round(score, 4), "silver_label": int(score >= OVERLAP_THRESHOLD)})
    with open(ANN / "pairs_silver.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pairs[0].keys()); w.writeheader(); w.writerows(pairs)

    # stratified sample of test pairs for gold labelling (0/1/2); deterministic, and never overwrites existing labels
    rng = random.Random(SEED)
    test_pairs = [p for p in pairs if p["split"] == "test"]
    bins = [[p for p in test_pairs if lo <= p["silver_score"] < hi] for lo, hi in [(0, .15), (.15, .35), (.35, .6), (.6, 1.01)]]
    sample = [p for b in bins for p in rng.sample(b, min(15, len(b)))]
    if not (ANN / "gold_pairs.csv").exists():
        with open(ANN / "gold_pairs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["course_a", "title_a", "course_b", "title_b", "silver_score", "gold_label", "annotator", "rationale"])
            for p in sample:
                w.writerow([p["course_a"], p["title_a"], p["course_b"], p["title_b"], p["silver_score"], "", "", ""])

    by = lambda k: dict(Counter(d[k] for d in kept).most_common())
    stats = {
        "raw_documents": n_raw, "after_cleaning": len(kept), "dropped_short_or_duplicate": n_raw - len(kept),
        "split_sizes": by("split"), "departments": by("department"),
        "words_per_doc": {"mean": round(sum(d["n_words"] for d in kept) / len(kept), 1),
                          "min": min(d["n_words"] for d in kept), "max": max(d["n_words"] for d in kept)},
        "vocab_size_after_preprocessing": len({w for d in kept for w in d["clean_text"].split()}),
        "coverage": {k: sum(bool(d[k]) for d in kept) for k in ("description", "prerequisites_text", "outcomes", "lecture_topics")},
        "classification_labels": {t: df[t] for t in label_set},
        "pairs": dict(Counter(p["split"] for p in pairs)),
        "positive_pairs": dict(Counter(p["split"] for p in pairs if p["silver_label"])),
        "gold_pairs": len(sample),
    }
    (RES / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
