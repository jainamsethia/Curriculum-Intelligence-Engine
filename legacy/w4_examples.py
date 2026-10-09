"""Week 4 - examples of the improved components on real courses -> results/w4_examples.json

  * course search, four ways: keyword (BM25), dense, hybrid, and the Week 3 course-vector method
  * comparison reports with an LLM-written committee summary (falls back to a template when the faithfulness guard trips)

    python src/w4_examples.py     (needs Ollama with phi3 running for the summaries)
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import improved as I
import pipeline as P
from examples_v1 import pick

ROOT = Path(__file__).resolve().parents[1]
QUERIES = ["transformers and large language models", "blockchain and smart contracts", "cloud computing and DevOps", "protecting a network from attacks", "regression and classification with scikit-learn"]


def main():
    pipe = P.Pipeline()
    out = {"search": {}, "comparisons": {}}
    for q in QUERIES:
        out["search"][q] = {m: pipe.search(q, 3, m) for m in ("keyword", "dense", "hybrid")}
        out["search"][q]["course-vector (Week 3)"] = pipe.similar(q, 3)
    cases = {"same course, two versions": (pick(pipe, "Probability and Statistics", "2021-22", "Computer"), pick(pipe, "Probability and Statistics", "2025-26")),
             "related but distinct courses": (pick(pipe, "Machine Learning"), pick(pipe, "Deep Learning")),
             "course and its prerequisite": (pick(pipe, "Operating Systems"), pick(pipe, "Distributed Computing"))}
    for name, (a, b) in cases.items():
        t0 = time.time()
        r = pipe.compare(a, b, summary=True)
        out["comparisons"][name] = {"course_a": r["course_a"]["course_name"] + " " + r["course_a"]["academic_year"], "course_b": r["course_b"]["course_name"] + " " + r["course_b"]["academic_year"],
                                    "overlap": r["overlap"]["percent"], "confidence": r["overlap"]["confidence"], "template_summary": I.template_comparison_summary(r),
                                    "summary": r["summary"], "seconds": round(time.time() - t0, 1)}
        print(name, "->", r["summary"]["method"], "| unsupported", r["summary"]["unsupported_term_rate"], flush=True)
    pipe.emb.save()
    (ROOT / "results/w4_examples.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote results/w4_examples.json")


if __name__ == "__main__":
    main()
