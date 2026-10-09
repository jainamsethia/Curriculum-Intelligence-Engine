"""One entry point that rebuilds everything from the NMIMS zip:

    python -m curriculum_engine.run data      parse (if needed) -> results/dataset/{parse_stats,stats,split}.json
    python -m curriculum_engine.run week2     baselines            -> results/week2/
    python -m curriculum_engine.run all       every step in order

Environment: NMIMS_ZIP (default ./B TECH.zip), CE_DATA_DIR (default ./data; NMIMS-derived, never committed).
"""
import json, sys
from collections import Counter

from curriculum_engine import RESULTS, SEED
from curriculum_engine import data as D


def dump(obj, path):
    path = RESULTS / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False), encoding="utf-8")


def step_data():
    if not D.COURSES.exists() or "--reparse" in sys.argv:
        from curriculum_engine import parse
        parse.main(parse.ZIP)
    recs = D.load()
    sp = D.split(recs)
    fam_versions = Counter(r["family"] for r in recs)
    progs = Counter(p for r in recs for p in r["programmes"])
    st = D.structure(recs)
    dump({
        "course_versions": len(recs), "families": len(fam_versions),
        "families_with_2plus_versions": sum(v >= 2 for v in fam_versions.values()),
        "programmes": len(progs), "versions_per_programme": dict(progs.most_common()),
        "common_courses": sum(r["common"] for r in recs), "open_electives": sum(r["open_elective"] for r in recs),
        "semester_known": sum(bool(r["sem"]) for r in recs),
        "families_per_programme_semester": {p: {str(s): len(f) for s, f in sorted(v.items())} for p, v in st.items()},
    }, "dataset/stats.json")
    fam_split = Counter(sp.values())
    ver_split = Counter(sp[r["family"]] for r in recs)
    dump({"seed": SEED, "fractions": D.FRACTIONS, "unit": "course family (same code or same name)",
          "hash": D.split_hash(sp), "families": dict(fam_split), "versions": dict(ver_split)}, "dataset/split.json")
    return recs


def step_week(n):
    def fn():
        import importlib
        from curriculum_engine.experiments.common import Ctx
        importlib.import_module(f"curriculum_engine.experiments.week{n}").run(Ctx())
    return fn


STEPS = {"data": step_data, "week2": step_week(2), "week3": step_week(3), "week4": step_week(4), "week5": step_week(5)}


def main(target):
    if target == "all":
        for name, fn in STEPS.items():
            print(f"== {name}", flush=True)
            fn()
    else:
        STEPS[target]()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
