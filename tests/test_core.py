"""Core logic with tiny synthetic inputs: curriculum diff, prerequisite checks, perturbations (no NMIMS data)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from curriculum_engine import diff as DF
from curriculum_engine import perturb as PT
from curriculum_engine import prereq as PQ


def course(fam, name, sem=3, pre="", progs=("COMP",), common=False):
    return {"id": "nm" + fam, "family": fam, "course_name": name, "sem": sem, "semester": str(sem), "prerequisites": pre,
            "programmes": list(progs), "common": common, "academic_year": "2025-26", "units": [], "outcomes": [], "objectives": ""}


def test_diff_by_name_and_content():
    A = {"f1": course("f1", "Data Structures"), "f2": course("f2", "Operating Systems")}
    B = {"f1": course("f1", "Data Structures"), "f3": course("f3", "Principles of Operating Systems")}
    assert DF.diff_by_name(A, B) == {"retained": [("f1", "f1")], "removed": ["f2"], "added": ["f3"]}
    same_content = lambda a, b: 0.9 if "Operating" in a["course_name"] and "Operating" in b["course_name"] else None
    d = DF.diff_by_content(A, B, same_content)
    assert d["removed"] == [] and d["added"] == [] and ("f2", "f3") in d["retained"]
    assert DF.score(d, DF.truth(A, B)) == (0, 0, 2)


def test_prerequisite_check_flags_only_missing_course():
    recs = [course("f1", "Programming in C", sem=1), course("f2", "Data Structures", sem=3, pre="Programming in C, Discrete Mathematics")]
    st = {"COMP": defaultdict(set, {1: {"f1"}, 3: {"f2"}, 2: {"x1", "x2", "x3"}})}
    st["COMP"][1] |= {"y1", "y2", "y3"}
    view = {"COMP": {r["family"]: r for r in recs}}
    flags = PQ.check_programme("COMP", view, st, PQ.ExactResolver(recs), {"COMP": 100})
    assert [(f["phrase"], f["type"]) for f in flags] == [("Discrete Mathematics", "unresolved")]
    st2 = PQ.without_family(st, "COMP", "f1")
    flags2 = PQ.check_programme("COMP", view, st2, PQ.ExactResolver(recs), {"COMP": 100})
    assert ("Programming in C", "not_offered_earlier") in [(f["phrase"], f["type"]) for f in flags2]


def test_coverage_rule_skips_sparse_programmes():
    st = {"IBM": defaultdict(set, {1: {"a"}, 2: {"b"}})}
    assert not PQ.covered(st, {"IBM": 1}, "IBM", 3)


def test_perturbations_are_deterministic_and_bounded():
    t = "Network layers OSI and TCP IP models encapsulation routing protocols"
    assert PT.noisy(t, seed=3) == PT.noisy(t, seed=3)
    assert len(PT.word_drop(t, 0.15).split()) <= len(t.split())
    assert PT.truncate(t, 0.5).split() == t.split()[:5]
