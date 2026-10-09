"""Capability 3 - missing prerequisites, checked against the programme structure (which course sits in which semester).

A course's free-text prerequisite field is split into phrases; each phrase is resolved to a course family.
Flags (per programme):
  not_offered_earlier       the resolved prerequisite course is not taught in an earlier semester of this programme
  unresolved                (baseline only) the phrase matches no course name, so it cannot be checked
  assumed_topic_not_taught  (core) the phrase matches no course and no unit of an earlier-semester course covers it
Coverage-aware: programmes with too little parsed data, or a missing earlier semester, are not checked (scanned booklets).
"""
import re
from collections import defaultdict

import numpy as np

from curriculum_engine import data as D

NONE_PREREQ = re.compile(r"^\W*(nil|na|n/?a|none|not applicable|no prerequisites?|-+)\W*$", re.I)
LEAD = re.compile(r"^(?:(?:basic|fundamental|elementary|working|good|sound|prior|general|introductory)\s+)*"
                  r"(?:knowledge|understanding|concepts?|familiarity|awareness|proficiency|grasp|idea|basics|fundamentals|principles)"
                  r"(?:\s+(?:of|in|about|with|on))?\s+|^(?:basics|fundamentals|introduction)\s+(?:of|to)\s+", re.I)
MIN_VERSIONS, MIN_FAMILIES_PER_SEM = 50, 3          # coverage rule: enough parsed data in the programme and in every earlier semester


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


class ExactResolver:
    """Baseline: a phrase names a course only if the normalised strings are identical."""
    name = "exact course-name match"

    def __init__(self, recs):
        self.by_name = {}
        for r in sorted(recs, key=lambda r: r["id"]):
            self.by_name.setdefault(D.norm_name(r["course_name"]), r["family"])

    def resolve(self, phrase, prefer=()):
        return self.by_name.get(D.norm_name(phrase))


def programme_view(recs):
    """programme -> family -> its latest version listed in that programme (common first-year courses count for every programme)."""
    progs = sorted({p for r in recs for p in r["programmes"]})
    view = {p: {} for p in progs}
    for r in sorted(recs, key=lambda r: (r["academic_year"], r["id"])):
        for p in (progs if r["common"] else r["programmes"]):
            view[p][r["family"]] = r
    return view


def covered(st, versions, prog, sem):
    return versions.get(prog, 0) >= MIN_VERSIONS and all(len(st[prog].get(s, ())) >= MIN_FAMILIES_PER_SEM for s in range(1, sem))


def earlier(st, prog, sem):
    return set().union(*[st[prog].get(s, set()) for s in range(1, sem)]) if sem and sem > 1 else set()


def check_programme(prog, view, st, resolver, versions, topic_covered=None, families=None):
    """All prerequisite flags for one programme. topic_covered(phrase, earlier_families) -> bool enables the core topic check."""
    flags = []
    for fam, c in sorted(view[prog].items()):
        if families is not None and fam not in families:
            continue
        if c["common"] or not c["sem"] or c["sem"] < 2 or not covered(st, versions, prog, c["sem"]):
            continue
        before = earlier(st, prog, c["sem"])
        for ph in split_prereq(c.get("prerequisites", "")):
            d = resolver.resolve(ph, before)
            if d == fam:
                continue
            base = {"programme": prog, "family": fam, "course": c["course_name"], "semester": c["sem"], "phrase": ph, "resolved_family": d}
            if d is not None:
                if d not in before:
                    flags.append({**base, "type": "not_offered_earlier"})
            elif topic_covered is None:
                flags.append({**base, "type": "unresolved"})
            elif not topic_covered(ph, before):
                flags.append({**base, "type": "assumed_topic_not_taught"})
    return flags


def removal_cases(recs, view, st, versions, families):
    """Label-free test cases, built by a rule that is neither method: a phrase whose normalised text contains (or is contained in)
    the name of exactly one family taught earlier in the programme. Removing that family must raise a flag."""
    names = {}
    for r in recs:
        names.setdefault(r["family"], D.norm_name(r["course_name"]))
    cases = []
    for prog in view:
        for fam, c in sorted(view[prog].items()):
            if fam not in families or c["common"] or not c["sem"] or c["sem"] < 2 or not covered(st, versions, prog, c["sem"]):
                continue
            before = earlier(st, prog, c["sem"])
            for ph in split_prereq(c.get("prerequisites", "")):
                k = D.norm_name(ph)
                hits = [d for d in before if d != fam and len(k) >= 5 and len(names[d]) >= 5 and (k in names[d] or names[d] in k)]
                if len(hits) == 1:
                    cases.append({"programme": prog, "family": fam, "phrase": ph, "removed_family": hits[0]})
    return cases


def without_family(st, prog, fam):
    """A copy of the programme structure with one family removed from every semester of one programme."""
    st2 = {p: v for p, v in st.items()}
    st2[prog] = defaultdict(set, {s: (f - {fam}) for s, f in st[prog].items()})
    return st2


class EmbeddingResolver:
    """Core: a phrase resolves to the course whose name embedding is closest, if cosine >= tau (tuned on validation cases)."""
    name = "embedding match to course names + topic check over earlier semesters"

    def __init__(self, recs, emb, tau):
        pairs = sorted({(r["course_name"], r["family"]) for r in recs})
        self.names, self.fams, self.emb, self.tau = [n for n, _ in pairs], np.array([f for _, f in pairs]), emb, tau
        self.N = emb(self.names)

    def best(self, phrase):
        s = self.N @ self.emb([phrase])[0]
        i = int(s.argmax())
        return self.fams[i], self.names[i], float(s[i])

    def resolve(self, phrase, prefer=()):
        """Courses taught earlier in the programme (`prefer`) win when they pass the threshold; otherwise the whole catalogue."""
        s = self.N @ self.emb([phrase])[0]
        if prefer:
            m = np.isin(self.fams, list(prefer)) & (s >= self.tau)
            if m.any():
                return str(self.fams[np.flatnonzero(m)[s[m].argmax()]])
        i = int(s.argmax())
        return str(self.fams[i]) if s[i] >= self.tau else None


class TopicChecker:
    """Is a prerequisite topic taught in an earlier semester? Max cosine between the phrase and any unit of an earlier course."""

    def __init__(self, reps, emb, tau):
        from curriculum_engine.overlap import chunk_vectors
        self.U = {f: chunk_vectors(emb, c) for f, c in reps.items()}
        self.emb, self.tau = emb, tau

    def best(self, phrase, before):
        q = self.emb([phrase])[0]
        return max((float((self.U[f] @ q).max()) for f in before if f in self.U and len(self.U[f])), default=0.0)

    def __call__(self, phrase, before):
        return self.best(phrase, before) >= self.tau
