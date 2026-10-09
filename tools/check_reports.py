"""Fail if any report disagrees with results/*.json or breaks the wording rules.

Checks per .docx (with its .manifest.json written by build_report.py):
  1. every placeholder value still equals what results/*.json gives now, and appears in the document text
  2. no unresolved {{...}} placeholder
  3. required wording: label provenance + limitation, data = semester syllabus booklets, scanned-PDF coverage, OCR as future work
  4. forbidden wording: human gold / human kappa / course-policy / gold set
  5. the template has no hand-typed numbers outside placeholders (whitelist: weeks, roll numbers, years, model names, label scale)
  6. across all reports given: the same placeholder key never has two different values

    python tools/check_reports.py submissions/week5_bundle/*.docx
"""
import json, re, sys
from pathlib import Path

from docx import Document

sys.path.insert(0, str(Path(__file__).parent))
from build_report import PH, ROOT, fmt, lookup  # noqa: E402

REQUIRED = ["LLM-annotated reference labels", "No human annotators were used; results are indicative.",
            "semester syllabus booklets", "OCR"]
FLAG_DISCLOSURE = ("flags (Week 3 onwards) were labelled by two Claude passes with different wording plus a pass 3, with no Phi-3, "
                   "so agreement on flags is not independent")
ONLY_AS = [(r"paraphras", "paraphrase variant not run (time)"), (r"bge-base prerequisite", "bge-base prerequisite re-tuning not run (time)")]
FORBIDDEN = [r"human gold", r"human kappa", r"course[- ]policy", r"\bgold (?:set|labels?|standard)\b", r"Fleiss"]
ALLOWED_DIGITS = re.compile(
    r"95\s?%|^\|\s*\d+\s*\||capabilit(?:y|ies) \d(?:\s?[–-]\s?\d)?|Week\s?\d|Weeks? \d(?:\s?[–-]\s?\d)?|J0\d\d|Groups? \d+(?:\s?/\s?\d+)?|20\d\d(?:[-–/]\d\d)?|\b[KL][1-6]\b|K1[–-]K6|L1[–-]L6|Phi-3|"
    r"v\d+(?:\.\d+)*|MiniLM-L\d+|bge-\w+-en-v\d\.\d|[A-Za-z]+\d+[A-Za-z]*|\b\d{1,2} (?:Sep|Oct|Nov)\b|\b[0-2]\s?=|\b0/1/2\b|"
    r"\bpass(?:es)? \d(?:[–-]\d)?\b|\bPass \d\b|\b\d(?:st|nd|rd|th)\b|\btop[- ]?\d+\b|\b0 / 1 / 2\b|@\d+|\bCapability \d\b|\bcaps? \d(?:[–-]\d)?\b|"
    r"\b[1-7]\.\s|\(\d\)|±1|2-of-3|\b5 members\b|section \d", re.I)


def doc_text(path):
    d = Document(path)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def lint_template(week):
    src = (ROOT / f"reports/templates/week{week}.md").read_text(encoding="utf-8")
    src = re.sub(r"```.*?```", " ", src, flags=re.S)
    errs = []
    for ln, line in enumerate(src.splitlines(), 1):
        if line.strip().startswith("!["):
            continue
        s = PH.sub(" ", line)
        s = re.sub(r"`[^`]*`", " ", s)
        s = ALLOWED_DIGITS.sub(" ", s)
        if re.search(r"\d", s):
            errs.append(f"week{week}.md:{ln}: hand-typed number? {line.strip()[:100]}")
    return errs


def check(paths):
    errs, seen = [], {}
    stats = lookup("dataset/parse_stats", "")
    coverage = f"{fmt(stats['text_pdf'], '')} of {fmt(stats['pdfs_considered'], '')}"
    for p in map(Path, paths):
        man = json.loads(p.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        text = doc_text(p)
        flat = re.sub(r"\s+", " ", text)
        for e in man["values"]:
            k, v = e["key"], e["value"]
            kind, _, rest = k.partition(":")
            if kind in ("table", "json"):
                f, _, path = rest.partition(":")
                now = json.dumps(lookup(f, path), sort_keys=True, ensure_ascii=False)
            else:
                f, path, form = (k.split(":") + ["", ""])[:3]
                now = fmt(lookup(f, path), form)
                if v not in text:
                    errs.append(f"{p.name}: value {v!r} of {k} not found in the document")
            if now != v:
                errs.append(f"{p.name}: {k} is {v!r} in the report but {now!r} in results")
            if seen.setdefault(k, v) != v:
                errs.append(f"{p.name}: {k} differs between reports ({seen[k]!r} vs {v!r})")
        if "{{" in text:
            errs.append(f"{p.name}: unresolved placeholder")
        labels = lookup("week2/labels", "")
        split_sentence = (f"For {fmt(labels['pass2_by']['claude_b'], '')} of {fmt(labels['items'], '')} pairs the second labeller was a Claude prompt "
                          "rather than Phi-3, so agreement on those items is not independent.")
        for k, allowed in ONLY_AS:                  # skipped analyses may only appear as "not run"
            if len(re.findall(k, flat, re.I)) != flat.count(allowed):
                errs.append(f"{p.name}: mentions /{k}/ other than as {allowed!r}")
        for r in REQUIRED + [coverage, split_sentence, FLAG_DISCLOSURE]:
            if r not in flat:
                errs.append(f"{p.name}: missing required wording {r!r}")
        for r in FORBIDDEN:
            if re.search(r, flat, re.I):
                errs.append(f"{p.name}: forbidden wording /{r}/")
        errs += lint_template(man["week"])
    return errs


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    errs = check(sys.argv[1:])
    print("\n".join(errs) or f"OK: {len(sys.argv) - 1} report(s) consistent with results/")
    sys.exit(1 if errs else 0)
