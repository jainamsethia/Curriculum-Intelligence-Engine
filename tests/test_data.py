"""Parser regression, programme mapping, course families and split leakage (synthetic data: runs on a clean clone)."""
from pathlib import Path

from curriculum_engine import data as D
from curriculum_engine import parse as P

FIX = Path(__file__).parent / "fixtures"


def block(text):
    return [(l, "2025-26") for l in text.splitlines()]


def test_empty_prerequisite_does_not_swallow_objectives():
    rec = P.parse_block(block((FIX / "syllabus_block.txt").read_text()), "3RD YEAR/COMPUTER/SEM V/x.pdf")
    assert rec["prerequisites"] == ""
    assert rec["objectives"].startswith("The course would enable")
    assert [o["k_levels"] for o in rec["outcomes"]] == [[2], [3]]
    assert [u["hours"] for u in rec["units"]] == [8, 7]


def test_filled_prerequisite_is_kept():
    t = (FIX / "syllabus_block.txt").read_text().replace("Pre-requisite:\n", "Pre-requisite: Computer Networks\n")
    assert P.parse_block(block(t), "x.pdf")["prerequisites"] == "Computer Networks"


def test_no_grading_tables_in_content_fields():
    t = (FIX / "syllabus_block.txt").read_text() + "Details of Term work:\nAttendance 10 marks, ICA 40 marks\n"
    rec = P.parse_block(block(t), "x.pdf")
    assert "Attendance" not in " ".join([rec["text_books"], rec["lab_work"]] + [u["text"] for u in rec["units"]])


def test_programme_mapping():
    cases = {"I T": "IT", "COMPUTER": "COMP", "CSE Data Science (311)": "CSE-DS", "DATA SCIENCES": "DS", "MECHATORNICS": "MECHATRONICS",
             "ARTIFICAL INTELLIGENCE": "AI", "Computer Science & Business Systems": "CSBS", "COMMON FOR ALL BRANCH": "COMMON",
             "Open Elective-I": "OPEN", "CSE CYBERSECURITY": "CSE-CYBER", "ELETRICAL": "ELECTRICAL"}
    assert {k: D.programme_of(k) for k in cases} == cases


def test_families_join_same_code_or_same_name():
    recs = [{"id": "nm01", "course_key": "networks", "code": "C1"}, {"id": "nm02", "course_key": "computernetworks", "code": "C1"},
            {"id": "nm03", "course_key": "computernetworks", "code": ""}, {"id": "nm04", "course_key": "algebra", "code": "C9"}]
    f = D._families(recs)
    assert f["nm01"] == f["nm02"] == f["nm03"] != f["nm04"]


def test_split_keeps_families_together():
    recs = [{"family": f"f{i % 37}"} for i in range(400)]
    sp = D.split(recs)
    assert set(sp.values()) == {"train", "val", "test"}
    assert len(sp) == 37                              # one entry per family: all versions share one side


def test_semester_numbers():
    assert D.sem_min("V/ VI") == 5 and D.sem_min("I / II") == 1 and D.sem_min("VII") == 7 and D.sem_min("") is None
