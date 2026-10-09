# Curriculum Intelligence Engine
Group 9 · J048, J049, J050, J052, J053 · AI University NLP Course Project 2026

An NLP engine for curriculum committees at NMIMS. Given courses or whole programmes from our own **semester syllabus booklets**, it
finds **overlapping courses**, **redundant content** inside a programme, **missing prerequisites** (a prerequisite not offered in an
earlier semester, or an assumed topic never taught earlier), compares two curricula (programme vs programme, year vs year),
maps learning outcomes to Bloom's taxonomy, and flags emerging-topic gaps. Every answer is JSON with confidence scores.

## Weekly deliverables (redo, October 2026)
| Week | Deliverable | Git tag | Report |
|---|---|---|---|
| 1 | Proposal v2 | `week-1-redo` | `submissions/week1/` |
| 2 | Dataset & baselines | `week-2-redo` | `submissions/week2/` |
| 3 | Core NLP model | `week-3-redo` | `submissions/week3/` |
| 4 | Improved model | `week-4-redo` | `submissions/week4/` |
| 5 | Evaluation & error analysis | `week-5-redo` | `submissions/week5/` |

The first versions of Weeks 2–4 (MIT OpenCourseWare data, then an early NMIMS parser) are kept for history in `legacy/` and
`legacy_mit/` (tag `pre-redo`). None of their numbers are reused.

## Data
- Source: the NMIMS B.Tech archive `B TECH.zip` (semester syllabus booklets, teaching schemes and past exam papers). Exam papers are skipped.
- Coverage limitation: only PDFs with a text layer are parsed (531 of 2,413 non-exam PDFs); the rest are scanned images and there is
  no OCR in this project yet (future work). See `results/dataset/parse_stats.json`.
- The archive and everything derived from it (`data/`) are **not** in this public repository. The code, ids, labels and metrics are.
  No student data is used; the parsed fields contain no faculty names or e-mail addresses.

## Run
```bash
pip install -r requirements.txt
set NMIMS_ZIP=path\to\B TECH.zip           # Windows (export NMIMS_ZIP=... elsewhere); default: ./B TECH.zip
python -m curriculum_engine.run all         # parse -> dataset stats/split -> weekly results in results/
python -m pytest -q                         # unit tests (no NMIMS data needed)
python tools/build_report.py 1              # report .docx from results/*.json (no hand-typed numbers)
python tools/check_reports.py submissions/week1/*.docx
```
The first parse reads ~2,400 PDFs from the zip (cached afterwards in `data/interim/`). Embeddings use cached sentence-transformer
models; set `HF_HUB_OFFLINE=1` once they are downloaded. The optional LLM (Phi-3) runs locally through Ollama; nothing is sent to
external services.

## Repository layout
```
curriculum_engine/   parser, data (families, programmes, split), embeddings, overlap, labelling, run
tools/               build_report.py, check_reports.py, docx2pdf.ps1
reports/templates/   report text with {{placeholders}} for every number
results/             metrics JSON + figures (dataset/, plan/, week<N>/)
labels/              LLM-annotated reference labels (ids, labels per pass, consensus, one-line rationale)
tests/               pytest (synthetic fixtures)
submissions/week<N>/ report, SUBMIT.md (the code zip is built with git archive, not committed)
```

## Labels and AI assistance
Evaluation uses **LLM-annotated reference labels**: each item is labelled in independent passes with different prompt wordings
(a Claude model in a fresh context, then Phi-3 locally), without model scores; disagreements get a third pass and items without a
2-of-3 majority are excluded. **No human annotators were used; results are indicative.** Agreement is reported as LLM-vs-LLM
agreement. `labels/human_spot_check.csv` lets team members check a sample later. Code and reports were written with AI assistance (Claude).
