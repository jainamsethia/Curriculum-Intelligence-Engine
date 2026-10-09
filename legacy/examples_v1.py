"""Write results/example_reports_v1.json: pipeline v1 on a few real course pairs, outcome mapping and semantic search.

    python src/examples_v1.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P

ROOT = Path(__file__).resolve().parents[1]


def pick(pipe, name, ay=None, program_has=None):
    """Latest (or given academic year) record with this course name."""
    rs = [c for c in pipe.courses if c["course_name"].lower() == name.lower() and (ay is None or c["academic_year"] == ay)
          and (program_has is None or program_has.lower() in c["program"].lower())]
    return sorted(rs, key=lambda c: c["academic_year"])[-1]


def main():
    pipe = P.Pipeline()
    cases = {
        "same course, two syllabus versions (2021-22 vs 2025-26)": (pick(pipe, "Probability and Statistics", "2021-22", "Computer"), pick(pipe, "Probability and Statistics", "2025-26")),
        "related but distinct courses": (pick(pipe, "Machine Learning"), pick(pipe, "Deep Learning")),
        "course and its prerequisite": (pick(pipe, "Operating Systems"), pick(pipe, "Distributed Computing")),
    }
    out = {"comparisons": {k: pipe.compare(a, b) for k, (a, b) in cases.items()},
           "outcome_bloom_mapping": {"course": pick(pipe, "Deep Learning")["course_name"], "outcomes": pipe.map_outcomes(pick(pipe, "Deep Learning"))},
           "semantic_search": {q: pipe.similar(q, 5) for q in ["transformers and large language models", "blockchain and smart contracts", "cloud computing and DevOps"]}}
    pipe.emb.save()
    (ROOT / "results/example_reports_v1.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for k, r in out["comparisons"].items():
        o = r["overlap"]
        print(f"{k}: overlap {o['percent']} (confidence {o['confidence']}), common {len(r['common_topics'])}, unique a/b {len(r['unique_to_a'])}/{len(r['unique_to_b'])}, "
              f"missing prereq {len(r['missing_prerequisites'])}, sequencing {len(r['sequencing_warnings'])}, redundancy={o['redundancy_warning']}")


if __name__ == "__main__":
    main()
