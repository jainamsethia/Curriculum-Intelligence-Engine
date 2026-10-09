"""Scoring, thresholds, label consensus and the report checker's wording rules."""
import numpy as np

from curriculum_engine import evaluate as E
from curriculum_engine.bloom import lexicon_level
from curriculum_engine.labelling import consensus
from curriculum_engine.overlap import maxsim
from curriculum_engine.prereq import split_prereq


def test_best_threshold_and_prf():
    y = np.array([0, 0, 1, 1, 1]); s = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    t = E.best_threshold(y, s)
    assert t == 0.3
    m = E.prf(y, s, t)
    assert (m["precision"], m["recall"], m["f1"]) == (1.0, 1.0, 1.0)


def test_ranking_perfect_and_ci_shape():
    y = np.array([0, 0, 1, 2] * 5); s = y + np.linspace(0, 0.1, 20)
    r = E.ranking(y, s, groups=np.arange(20) // 2)
    assert r["roc_auc"]["value"] == 1.0 and len(r["roc_auc"]["ci95"]) == 2


def test_maxsim_identical_and_disjoint():
    A = np.eye(3)
    assert maxsim(A, A, 0.9)["overlap"] == 1.0
    assert maxsim(A[:1], A[1:], 0.9)["overlap"] == 0.0


def test_consensus_two_of_three():
    assert consensus(1, 1, None) == (1, "agreed")
    assert consensus(0, 1, 1) == (1, "majority")
    assert consensus(0, 1, 2) == (None, "uncertain")
    assert consensus(2, None, 2) == (2, "majority")


def test_prerequisite_split():
    assert split_prereq("Physics (BTME01003), Calculus and Linear Algebra") == ["Physics", "Calculus", "Linear Algebra"]
    assert split_prereq("NIL") == [] and split_prereq("- NA") == []
    assert split_prereq("Basic knowledge of Computer Networks") == ["Computer Networks"]


def test_bloom_lexicon():
    assert lexicon_level("Design a database schema for a library") == 6
    assert lexicon_level("Explain the OSI model") == 2
    assert lexicon_level("Apply Laplace transforms to circuits") == 3
    assert lexicon_level("Students will be able to apply it") is None      # baseline only reads the first 5 words
