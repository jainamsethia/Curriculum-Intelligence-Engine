# Curriculum Intelligence Engine
Group 9 · J048, J049, J050, J052, J053 · AI University NLP Course Project 2026 · **v1.0 (final, October 2026)**

An NLP engine for curriculum committees at NMIMS. Given courses or whole programmes from our own **semester syllabus booklets**, it
finds **overlapping courses**, **redundant content** inside a programme, **missing prerequisites** (a prerequisite not offered in an
earlier semester, or an assumed topic never taught earlier), compares two curricula (programme vs programme, year vs year),
maps learning outcomes to Bloom's taxonomy, flags emerging-topic gaps, and drafts a modernised syllabus from job descriptions.
Every answer is JSON with evidence and a confidence score, meant for a human decision by the Board of Studies.

## Results (v1.0, held-out test split)
Targets were fixed in Proposal v2 before any result existed. Source: `results/week8/final.json` (rebuilt by `run all`).

| Capability | Metric | Result | Target | Met? |
|---|---|---|---|---|
| Overlap | Spearman with reference labels | 0.717 | ≥ 0.6 | yes |
| Redundancy | precision of top flags | 0.800 | ≥ 0.7 | yes |
| Prerequisites (checker v2) | removal recall | 0.711 | ≥ 0.8 | **no** |
| Prerequisites (checker v2) | precision of flags (n = 14) | 0.500 | ≥ 0.6 | **no** |
| Curriculum diff | precision, real year pairs | 0.998 | ≥ 0.85 | yes |
| Bloom levels | accuracy vs faculty tags | 0.776 | ≥ 0.7 | yes |
| Emerging topics | precision of flagged gaps | 0.915 | ≥ 0.7 | yes |
| Topic labels | share accepted | 0.700 | ≥ 0.7 | yes |
| Curriculum modernisation | taught / gap accuracy on JD skills | 0.857 | none set | – |

12 of 14 targets are met (the full table, with every metric, is in the Week 8 report). Both misses are prerequisite flags: checker v2
removed false alarms and raised precision, but recall fell below target.

## Weekly deliverables (redo, October 2026)
| Week | Deliverable | Git tag | Folder |
|---|---|---|---|
| 1 | Proposal v2 | `week-1-redo` | `submissions/week1/` |
| 2 | Dataset & baselines | `week-2-redo` | `submissions/week2/` |
| 3 | Core NLP model | `week-3-redo` | `submissions/week3/` |
| 4 | Improved model | `week-4-redo` | `submissions/week4/` |
| 5 | Evaluation & error analysis | `week-5-redo` | `submissions/week5/` |
| 1–5 | Final bundle (all five reports from one commit) | `week-5-redo` | `submissions/week5_bundle/` |
| 6 | Asset engineering: package + API, prerequisite checker v2 | `week-6-redo-fix` | `submissions/week6/` |
| 7 | Integration & demo: curriculum modernisation from JDs, Group 11 / Group 4 hand-offs, committee UI, responsible-AI analysis | `week-7` | `submissions/week7/` |
| 8 | Final: final testing, final report, presentation, architecture diagram, demo checklist | `v1.0` | `submissions/week8/` |

The first versions of Weeks 2–4 (MIT OpenCourseWare data, then an early NMIMS parser) are kept for history in `legacy/` and
`legacy_mit/` (tag `pre-redo`). None of their numbers are reused.

## Install
```bash
pip install -e .[dev]                       # Python 3.11; or: pip install -r requirements.txt (pinned versions)
python -m spacy download en_core_web_sm     # once, for the job-description skill phrases
```
Data: either put `data/processed/courses.jsonl` from the Teams code zip in place (parsed syllabi), or set `NMIMS_ZIP` to the
original `B TECH.zip` and let `run all` parse it. Sentence-transformer models are downloaded on first use; set `HF_HUB_OFFLINE=1`
afterwards. Nothing is sent to external services (the optional Phi-3 labeller runs locally through Ollama).

## Run
```bash
python -m curriculum_engine.run all         # parse -> dataset stats/split -> results/week2 ... week8
python -m pytest -q                         # unit, API and UI tests (synthetic fixtures, no NMIMS data needed)
python tools/final_test.py                  # clean clone -> tests -> run all -> every committed result reproduced; API latency
python tools/build_report.py 8              # report .docx from results/*.json (no hand-typed numbers)
python tools/check_reports.py submissions/week8/*.docx
```

## API
```bash
uvicorn curriculum_engine.api:app --port 8000      # OpenAPI docs: http://localhost:8000/docs
```
Endpoints: `POST /compare-curricula`, `/map-outcomes`, `/find-overlaps`, `/check-prerequisites`, `/detect-redundancy`,
`/emerging-topics`, `/modernize-curriculum`, `GET /courses`, `GET /health`. Details, confidence definitions and real examples:
`docs/API.md`, `docs/examples/`, `docs/openapi.json`.

## Committee UI and demo
```bash
uvicorn curriculum_engine.api:app --port 8000      # terminal 1
streamlit run app/committee_ui.py                  # terminal 2 (CE_API_URL defaults to http://localhost:8000)
```
The "Modernise curriculum" tab runs the 31 October demo step: five job descriptions (`demo/jds`, synthetic samples) plus the current
NLP syllabi -> overlaps, demanded skills, a traceable draft syllabus (Word / Markdown / JSON), a notice for Group 11 and passages for
Group 4. Demo script and recording checklist: `docs/DEMO_SCRIPT.md`. Integration contracts: `docs/INTEGRATION.md`.

## Architecture, slides
![Architecture](docs/architecture.png)

Mermaid source `docs/architecture.mmd`; details in `docs/ARCHITECTURE.md`. The PNG and the final slides are generated with Node tools:
```bash
cd tools && npm install          # pptxgenjs, mermaid-cli (uses the installed Chrome)
npm run architecture             # docs/architecture.mmd -> docs/architecture.png
npm run slides                   # results/*.json -> submissions/week8/..._Presentation.pptx
```

## Data
- Source: the NMIMS B.Tech archive `B TECH.zip` (semester syllabus booklets, teaching schemes and past exam papers). Exam papers are skipped.
- Coverage: only PDFs with a text layer are parsed (531 of 2,413 non-exam PDFs); the rest are scanned images and there is no OCR in
  this project yet. See `results/dataset/parse_stats.json`.
- The archive and everything derived from it (`data/`, including outputs that quote syllabus text, such as the draft syllabus and the
  Group 4 passages in `data/week7/`) are **not** in this public repository; they ship only in the Teams code zip. The code, ids,
  labels and metrics are public. No student data is used; the parsed fields contain no faculty names or e-mail addresses.

## Labels
Evaluation uses **LLM-annotated reference labels** (`labels/`). **No human annotators were used; results are indicative.**
- 200 stratified course pairs: pass 1 Claude (wording A, fresh context, no model scores); pass 2 Phi-3 (local, Ollama) for PR001-PR106
  and a second Claude prompt (wording B) for PR107-PR200; pass 3 Claude (wording C) on the disagreements; 2-of-3 majority,
  1 item excluded as uncertain. Agreement is reported separately per subset; the Claude-vs-Claude subset is not independent.
- 187 detector flags (Week 3), 45 prerequisite-v2 flags (Week 6) and 70 JD skill-coverage decisions (Week 7): two Claude passes with
  different wording plus a pass 3, no Phi-3, so agreement on these is not independent.
- `labels/human_spot_check.csv`: 50 pairs (10 per team member) for an optional human check; not filled in yet.
- Not run (time), listed as future work: the paraphrase variant of the robustness tests; bge-base prerequisite re-tuning.
- Summary per label set: `results/week8/labels.json`.

## Limitations
- Scanned booklets are not parsed (no OCR), so a course can look removed, or a prerequisite missing, when the evidence sits in an
  unread booklet; programmes with too little parsed data get no prerequisite checks.
- All quality numbers rest on LLM-annotated reference labels, not on human annotation; some samples are small (e.g. n = 14 for
  prerequisite-flag precision), so intervals are wide.
- Missing-prerequisite flags still miss both targets; treat them as prompts for review, not findings.
- The emerging-topic reference list (`curriculum_engine/emerging_topics.json`) is a draft: only entries whose source page was opened are
  marked verified, and the list still awaits approval.
- The demo job descriptions are synthetic samples; the draft syllabus is a starting point for faculty (its hours exceed one semester).
- Group 11's model and Group 4's knowledge base cannot be run from their repositories, so the hand-offs are checked against their
  schemas, with a marked mock (Group 11) and a labelled stand-in (Group 4); Group 10 is not integrated.

## AI assistance
Code, reports and slides were written with AI assistance (Claude, Anthropic). Reference labels are LLM-annotated (Claude, and Phi-3 run
locally). Every number in the reports and slides is generated from `results/*.json`; the reports are checked by
`tools/check_reports.py`.

## Repository layout
```
curriculum_engine/   parser, data (families, programmes, split), embeddings, overlap, prerequisites, modernise, API, run
app/                 committee_ui.py (Streamlit UI over the API)
demo/jds/            synthetic sample job descriptions for the demo
docs/                API, integration, responsible AI, architecture (Mermaid + PNG), demo script
tools/               report builder and checker, submission packer, final test, slides (Node), screenshots
reports/templates/   report text with {{placeholders}} for every number
results/             metrics JSON + figures (dataset/, plan/, week<N>/)
labels/              LLM-annotated reference labels (ids, labels per pass, consensus, one-line rationale)
tests/               pytest (synthetic fixtures)
submissions/week<N>/ report (.docx, .pdf), SUBMIT.md; Week 8 also the slides (code zips are built with git archive, not committed)
```
