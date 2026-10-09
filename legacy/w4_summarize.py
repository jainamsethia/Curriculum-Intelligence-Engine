"""Week 4, task 4 - course summarisation: basic (extractive / template) -> LLM-assisted (Phi-3 via Ollama).

Reference = the faculty-written 'Course Objective' paragraph of the syllabus. It is never shown to any system: every system receives the same
input (course name, unit topics, learning outcomes). Test courses only (the Week 3 split); latest version of each course.

Metrics: ROUGE-1/2/L F1 against the objective, embedding cosine to the objective (all-MiniLM-L6-v2), summary length, and an 'unsupported term'
rate (share of the summary's content words that appear nowhere in the input) as a hallucination proxy.

    python src/w4_summarize.py [--n 60] [--skip-full]
"""
import hashlib, json, re, sys, time
from pathlib import Path
import numpy as np
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, str(Path(__file__).parent))
import improved as I
import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
stem = PorterStemmer().stem
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def content_stems(text):
    return {stem(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in ENGLISH_STOP_WORDS}


def clip(text, n=70):
    text = re.sub(r"\s+", " ", text.strip().strip('"'))
    return " ".join(text.split()[:n])


def label(u):
    return u["title"] or " ".join(u["text"].split()[:6])


def lead_titles(c):
    return "This course covers " + ", ".join(label(u) for u in c["units"][:6]) + "."


def first_outcomes(c):
    return " ".join(o["text"].rstrip(".") + "." for o in c["outcomes"][:2])


def centroid_extractive(c, emb):
    cands = [" ".join(u["text"].split()[:30]) for u in c["units"]]
    E = emb(cands)
    centre = E.mean(0)
    top = sorted(np.argsort(-(E @ centre))[:2])
    return " ".join(cands[i] for i in top)


def prompt_titles(c):
    units = "; ".join(label(u) for u in c["units"][:9])
    outs = "\n".join(f"- {o['text']}" for o in c["outcomes"][:5])
    return (f"You write the course objective paragraph of a university syllabus.\nCourse: {c['course_name']}\nUnits: {units}\nLearning outcomes:\n{outs}\n\n"
            "Write 2 sentences (at most 50 words) describing what the course aims to teach. Use only the information above and do not add topics that are not listed. Start with \"The course\".")


def prompt_full(c):
    units = "\n".join(f"- {label(u)}: {' '.join(u['text'].split()[:45])}" for u in c["units"][:8])
    outs = "\n".join(f"- {o['text']}" for o in c["outcomes"][:5])
    return (f"You write the course objective paragraph of a university syllabus.\nCourse: {c['course_name']}\nUnits:\n{units}\nLearning outcomes:\n{outs}\n\n"
            "Write 2 sentences (at most 50 words) describing what the course aims to teach. Use only the information above and do not add topics that are not listed. Start with \"The course\".")


def source_text(c):
    return " ".join([c["course_name"]] + [u["text"] for u in c["units"]] + [o["text"] for o in c["outcomes"]])


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40
    courses = P.load_courses()
    keys = sorted({c["course_key"] for c in courses})
    tr_i, _ = next(GroupShuffleSplit(1, test_size=0.3, random_state=SEED).split(keys, groups=keys))
    test = set(keys) - {keys[i] for i in tr_i}
    latest = {}
    for c in sorted(courses, key=lambda c: c["academic_year"]):
        if c["course_key"] in test and len(c["units"]) >= 4 and len(c["outcomes"]) >= 3 and 25 <= len(c["objectives"].split()) <= 120:
            latest[c["course_key"]] = c
    pool = sorted(latest.values(), key=lambda c: hashlib.md5(c["course_key"].encode()).hexdigest())[:n]
    print(f"{len(pool)} test courses with a 25-120 word objective, 4+ units, 3+ outcomes", flush=True)
    emb = P.Embedder()
    systems = {"Template: first unit titles": lambda c: lead_titles(c), "Extractive: first two outcomes": lambda c: first_outcomes(c),
               "Extractive: centroid sentences": lambda c: centroid_extractive(c, emb),
               "Phi-3 (unit titles + outcomes)": lambda c: clip(I.ollama_generate(prompt_titles(c), num_predict=90))}
    if "--skip-full" not in sys.argv:
        systems["Phi-3 (full unit text + outcomes)"] = lambda c: clip(I.ollama_generate(prompt_full(c), num_predict=90))
    outs, secs = {}, {}
    ckpt = ROOT / "data/interim/w4_summaries_ckpt.json"            # generated texts per system: a restarted job resumes mid-system
    saved = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    for name, fn in systems.items():
        t0 = time.time(); outs[name] = list(saved.get(name, {}).get("texts", []))[:len(pool)]
        n0 = len(outs[name])
        for k in range(n0, len(pool)):
            outs[name].append(fn(pool[k]))
            if name.startswith("Phi") and k % 5 == 0:
                saved[name] = {"texts": outs[name], "sec_per_course": saved.get(name, {}).get("sec_per_course")}
                ckpt.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
                print(f"  {name}: {k + 1}/{len(pool)} ({time.time() - t0:.0f}s)", flush=True)
        secs[name] = round((time.time() - t0) / max(len(pool) - n0, 1), 2) if len(pool) > n0 else saved.get(name, {}).get("sec_per_course") or 0.0
        saved[name] = {"texts": outs[name], "sec_per_course": secs[name]}
        ckpt.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        print(name, "done", secs[name], "s/course", flush=True)
    ref = [c["objectives"] for c in pool]
    Eref = emb(ref)
    per = {}
    for name, texts in outs.items():
        r = [scorer.score(rf, tx) for rf, tx in zip(ref, texts)]
        Es = emb(texts)
        unsup = []
        for c, tx in zip(pool, texts):
            s, src = content_stems(tx), content_stems(source_text(c))
            unsup.append(len(s - src) / max(len(s), 1))
        per[name] = {"rouge1": [x["rouge1"].fmeasure for x in r], "rouge2": [x["rouge2"].fmeasure for x in r], "rougeL": [x["rougeL"].fmeasure for x in r],
                     "embedding_cosine": (Es * Eref).sum(1).tolist(), "words": [len(t.split()) for t in texts], "unsupported_terms": unsup}
    groups = [c["course_key"] for c in pool]
    base = "Extractive: first two outcomes"
    res = {"n_courses": len(pool), "reference": "faculty 'Course Objective' paragraph", "seconds_per_course": secs, "systems": {}}
    for name, m in per.items():
        row = {k: round(float(np.mean(v)), 3) for k, v in m.items()}
        row["rougeL_95ci"] = I.cluster_bootstrap(m["rougeL"], groups)
        row["rougeL_gain_over_best_baseline_95ci"] = I.cluster_bootstrap(np.array(m["rougeL"]) - np.array(per[base]["rougeL"]), groups)
        row["embedding_cosine_gain_over_best_baseline_95ci"] = I.cluster_bootstrap(np.array(m["embedding_cosine"]) - np.array(per[base]["embedding_cosine"]), groups)
        res["systems"][name] = row
    res["examples"] = [{"course": pool[i]["course_name"], "reference": ref[i], **{k: v[i] for k, v in outs.items()}} for i in range(min(4, len(pool)))]
    (ROOT / "results/w4_summarization.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    for k, v in res["systems"].items():
        print(f'R1 {v["rouge1"]:.3f} R2 {v["rouge2"]:.3f} RL {v["rougeL"]:.3f} {v["rougeL_95ci"]} | cos {v["embedding_cosine"]:.3f} | words {v["words"]:.0f} | unsupported {v["unsupported_terms"]:.3f} | {k}')


if __name__ == "__main__":
    main()
