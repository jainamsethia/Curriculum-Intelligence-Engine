"""Shared state for the experiments: courses, split, programme view and structure, TF-IDF, embeddings, reference labels."""
import csv, json
from collections import Counter
from functools import cached_property

from curriculum_engine import LABELS, RESULTS
from curriculum_engine import data as D
from curriculum_engine import prereq as PQ


def dump(obj, path):
    path = RESULTS / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    return obj


class Ctx:
    @cached_property
    def recs(self):
        return D.load()

    @cached_property
    def by_id(self):
        return {r["id"]: r for r in self.recs}

    @cached_property
    def split(self):
        return D.split(self.recs)

    def families(self, *splits):
        return {f for f, s in self.split.items() if s in splits}

    @cached_property
    def reps(self):
        return D.representatives(self.recs)

    @cached_property
    def view(self):
        return PQ.programme_view(self.recs)

    @cached_property
    def st(self):
        return D.structure(self.recs)

    @cached_property
    def versions(self):
        return Counter(p for r in self.recs for p in r["programmes"])

    @cached_property
    def tfidf(self):
        from curriculum_engine.overlap import Tfidf
        return Tfidf([c for f, c in self.reps.items() if self.split[f] == "train"])

    @cached_property
    def emb(self):
        from curriculum_engine.embed import Embedder
        return Embedder()

    def pair_labels(self, split=None):
        """Reference-labelled pairs (consensus only; 'uncertain' items are excluded)."""
        f = LABELS / "pairs_llm.csv"
        if not f.exists():
            return []
        rows = [r for r in csv.DictReader(open(f, encoding="utf-8")) if r["status"] != "uncertain" and r["consensus"] not in ("", "None")]
        return [{**r, "y": int(r["consensus"]), "a": self.by_id[r["id_a"]], "b": self.by_id[r["id_b"]]}
                for r in rows if split is None or r["split"] == split]
