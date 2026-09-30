"""Week 4, task 3 - prerequisite extraction: rule-based -> NER-style model -> LLM-assisted.

Input: the free-text 'Pre-requisite' field of a syllabus. Output: the list of prerequisite subjects.
Systems: (1) the Week 3 regex splitter, (2) spaCy noun-chunk extraction (a classic NER-style span model),
(3) few-shot Phi-3 through Ollama returning JSON. Evaluated on 60 strings labelled in data/annotation/prereq_extraction_gold.json
(single LLM annotator). A predicted phrase matches a gold phrase when their content-word sets overlap (Jaccard >= 0.5), one-to-one.

    python src/w4_prereq_extraction.py
"""
import json, re, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import improved as I
import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6"}
GENERIC = set("basic basics knowledge fundamental fundamentals concepts concept understanding skills skill of the a an in to and with on for".split())

FEW_SHOT = [
    ("Engineering Mathematics - I, II (BTAB01001, BTAB02001), Physics", ["Engineering Mathematics - I", "Engineering Mathematics - II", "Physics"]),
    ("Nil", []),
    ("Objectives: To understand the concepts of sorting and searching and to design efficient programs.", []),
    ("Basic knowledge of programming and data structures", ["programming", "data structures"]),
    ("Software Engineering, Database Management Systems", ["Software Engineering", "Database Management Systems"]),
    ("Students should have completed the courses Operating Systems and Computer Networks", ["Operating Systems", "Computer Networks"]),
]
INSTRUCTION = ('You extract the prerequisite subjects from the "Pre-requisite" field of a university syllabus. '
               'Answer only with JSON: {"prerequisites": [list of short subject or course names]}. '
               'One entry per subject or course. Keep official course names whole (for example "Computer Organization and Architecture"). '
               'Expand "Mathematics I and II" into separate entries. Drop course codes in brackets and words like "basic knowledge of". '
               'If the text names no subject (for example "Nil", "NA", a list of course objectives, or an eligibility statement), answer {"prerequisites": []}.')


def toks(phrase):
    t = re.sub(r"[^a-z0-9]+", " ", phrase.lower().replace("&", " and ")).split()
    return frozenset(ROMAN.get(w, w) for w in t if w not in GENERIC)


def match(pred, gold, thr=0.5):
    pairs = []
    for i, p in enumerate(map(toks, pred)):
        for j, g in enumerate(map(toks, gold)):
            if p and g and len(p & g) / len(p | g) >= thr:
                pairs.append((len(p & g) / len(p | g), i, j))
    used_p, used_g, tp = set(), set(), 0
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_p and j not in used_g:
            used_p.add(i); used_g.add(j); tp += 1
    return tp


def rule_based(text):
    return P.split_prereq(text)


_nlp = None


def spacy_chunks(text):
    """Noun chunks of a general-purpose English pipeline (en_core_web_sm), minus pronouns and generic head words."""
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    doc = _nlp(re.sub(r"\([^)]*\d[^)]*\)", " ", text))
    out = []
    for ch in doc.noun_chunks:
        words = [t.text for t in ch if not t.is_stop and not t.is_punct]
        if ch.root.pos_ in ("NOUN", "PROPN") and words and toks(" ".join(words)):
            out.append(" ".join(words))
    return list(dict.fromkeys(out))


def llm_extract(text):
    shots = "\n".join(f'Text: "{t}"\nJSON: {json.dumps({"prerequisites": g})}' for t, g in FEW_SHOT)
    prompt = f'{INSTRUCTION}\n\n{shots}\nText: "{text[:700]}"\nJSON:'
    raw = I.ollama_generate(prompt, num_predict=160, fmt="json")
    try:
        v = json.loads(raw)["prerequisites"]
        return [str(x) for x in v] if isinstance(v, list) else []
    except Exception:
        return None                                              # unparseable output counts as a failure (no prediction)


def score(items, preds):
    by = defaultdict(lambda: [0, 0, 0, 0, 0])                    # tp, fp, fn, gold-empty items, false alarms on them
    for it in items:
        p = preds[it["id"]] or []
        tp = match(p, it["gold"])
        for key in ("all", it["kind"]):
            b = by[key]; b[0] += tp; b[1] += len(p) - tp; b[2] += len(it["gold"]) - tp
            if not it["gold"]:
                b[3] += 1; b[4] += bool(p)
    res = {}
    for k, (tp, fp, fn, ne, fa) in by.items():
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        res[k] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 3), "gold_phrases": tp + fn,
                  "false_alarm_rate_on_empty_gold": round(fa / ne, 3) if ne else None}
    return res


def per_item(items, preds):
    rows = []
    for it in items:
        p = preds[it["id"]] or []
        tp = match(p, it["gold"])
        rows.append((tp, len(p) - tp, len(it["gold"]) - tp))
    return np.array(rows)


def micro_f1(a):
    tp, fp, fn = a.sum(0)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def bootstrap_gain(base, other, n=2000, seed=0):
    """95% CI of the micro-F1 difference (other - base), resampling the evaluation strings."""
    idx = np.random.RandomState(seed).randint(len(base), size=(n, len(base)))
    d = np.array([micro_f1(other[i]) - micro_f1(base[i]) for i in idx])
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def main():
    gold = json.load(open(ROOT / "data/annotation/prereq_extraction_gold.json", encoding="utf-8"))
    items = gold["items"]
    preds, secs, failures = {}, {}, {}
    for name, fn in (("Rule-based splitter (Week 3)", rule_based), ("spaCy noun chunks (NER-style)", spacy_chunks), ("Phi-3 few-shot (LLM-assisted)", llm_extract)):
        t0, bad = time.time(), 0
        ck = ROOT / "data/interim/w4_prereq_llm_ckpt.json"
        out = json.loads(ck.read_text(encoding="utf-8")) if name.startswith("Phi") and ck.exists() else {}      # resumable LLM pass
        for k, it in enumerate(items):
            if it["id"] not in out:
                out[it["id"]] = fn(it["text"])
                if name.startswith("Phi"):
                    ck.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            bad += out[it["id"]] is None
            if name.startswith("Phi") and k % 10 == 0:
                print(f"  {name}: {k}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
        preds[name], secs[name], failures[name] = out, round((time.time() - t0) / len(items), 3), bad
        print(name, "done", flush=True)
    res = {"n_items": len(items), "n_gold_empty": sum(not i["gold"] for i in items), "systems": {}}
    for name, p in preds.items():
        s = score(items, p)
        res["systems"][name] = {"overall": s.pop("all"), "by_kind": s, "seconds_per_item": secs[name], "unparseable_outputs": failures[name]}
        res["systems"][name]["f1_gain_over_rule_based_95ci"] = bootstrap_gain(per_item(items, preds["Rule-based splitter (Week 3)"]), per_item(items, p))
    res["examples"] = [{"id": it["id"], "text": it["text"][:140], "gold": it["gold"], **{n[:12]: preds[n][it["id"]] for n in preds}} for it in items
                       if it["id"] in ("r14", "r15", "r11", "r12", "r01", "r56", "r09", "r44")]
    (ROOT / "results/w4_prereq_extraction.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for n, v in res["systems"].items():
        o = v["overall"]
        print(f'{n}: P {o["precision"]} R {o["recall"]} F1 {o["f1"]} | false alarms on empty-gold {o["false_alarm_rate_on_empty_gold"]} | {v["seconds_per_item"]}s/item | unparseable {v["unparseable_outputs"]}')


if __name__ == "__main__":
    main()
