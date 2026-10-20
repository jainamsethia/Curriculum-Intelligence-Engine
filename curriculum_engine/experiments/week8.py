"""Week 8 - final results of v1.0 against the success criteria fixed in Proposal v2, and a summary of every label set.
Reads results/*.json and labels/*.csv only (no model runs). Final testing (clean clone, latency) is tools/final_test.py."""
import csv
import json
import operator
from collections import Counter

from curriculum_engine import LABELS, RESULTS
from curriculum_engine.experiments.common import dump

OUT = "week8"


def get(path):
    """'week6/prerequisites_v2:tests_test.v2.removal_recall.value' -> value"""
    f, _, keys = path.partition(":")
    d = json.loads((RESULTS / f"{f}.json").read_text(encoding="utf-8"))
    for k in keys.split("."):
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


# rows of results/week5/final.json whose value changed after Week 5 (same metric, final component of v1.0)
CHANGED = {("3 Prerequisites", "removal recall"): ("week6/prerequisites_v2:tests_test.v2.removal_recall.value", "checker v2 (Week 6)"),
           ("3 Prerequisites", "precision of flags (reference labels)"): ("week6/prerequisites_v2:fresh_sample.test_v2.precision", "checker v2 (Week 6)"),
           ("3 Prerequisites", "false alarms after renaming"): ("week6/prerequisites_v2:tests_test.v2.rename_false_alarm.value", "checker v2 (Week 6)"),
           ("6 Emerging topics", "precision of flagged gaps"): ("week7/emerging_threshold:unit_match_threshold.gap_precision", "threshold 0.55 (Week 7)")}
OPS = {">=": operator.ge, "<=": operator.le}


def final_table():
    w5 = get("week5/final:table")
    rows, missed, cap = [], [], ""
    for name, metric, value, target, _ in w5["rows"]:
        cap = name or cap
        src, version = CHANGED.get((cap, metric), (None, "unchanged since Week 5"))
        final = get(src) if src else float(value)
        op, t = target.split()
        ok = OPS[op](final, float(t))
        rows.append([name, metric, value, f"{final:.3f}", target, "yes" if ok else "no", version])
        missed += [] if ok else [f"{cap}: {metric}"]
    rows.append(["8 Curriculum modernisation", "taught / gap accuracy on JD skill themes (reference labels)", "–",
                 f"{get('week7/coverage_precision:all_themes.accuracy'):.3f}", "no target set", "–", "new in Week 7"])
    met = sum(r[5] == "yes" for r in rows)
    with_target = sum(r[5] in ("yes", "no") for r in rows)
    return dump({"targets_met": met, "targets_total": with_target, "targets_missed": with_target - met,
                 "missed": missed,
                 "week5_targets_met": get("week5/final:targets_met"),
                 "table": {"columns": ["Capability", "Metric (test split)", "Week 5", "Final v1.0", "Target", "Met?", "Final component"],
                           "rows": rows}}, f"{OUT}/final.json")


def labels_summary():
    """Every LLM-annotated reference label set: items, settled by passes 1+2, by pass 3, uncertain, who labelled pass 2."""
    names = {"pairs_llm": "course pairs (overlap 0/1/2)", "flags_llm": "detector flags (Week 3)",
             "prereqv2_llm": "prerequisite v2 flags (Week 6)", "jdskills_llm": "JD skill coverage (Week 7)"}
    rows, tot = [], Counter()
    for f, what in names.items():
        recs = list(csv.DictReader(open(LABELS / f"{f}.csv", encoding="utf-8")))
        st, by = Counter(r["status"] for r in recs), Counter(r["pass2_by"] for r in recs)
        rows.append([what, len(recs), st["agreed"], st["majority"], st["uncertain"], ", ".join(f"{k} ({v})" for k, v in by.items())])
        tot.update({"items": len(recs), "agreed": st["agreed"], "pass3": st["majority"], "uncertain": st["uncertain"]})
    spot = list(csv.DictReader(open(LABELS / "human_spot_check.csv", encoding="utf-8")))
    return dump({**tot, "human_spot_check_rows": len(spot), "human_spot_check_filled": sum(bool(r["human_label_0_1_2"].strip()) for r in spot),
                 "table": {"columns": ["Label set", "Items", "Passes 1 and 2 agree", "Settled by pass 3", "Uncertain (excluded)", "Second labeller"],
                           "rows": rows}}, f"{OUT}/labels.json")


def run(ctx):
    print("  week8.final", flush=True)
    final_table()
    labels_summary()
