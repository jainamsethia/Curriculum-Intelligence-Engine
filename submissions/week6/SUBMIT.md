# Week 6 (Asset Engineering) — what to upload on Teams

| File (in `submissions/week6/`) | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week6_Asset_Engineering.docx` | the Week 6 report (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week6_code.zip` | the repository at tag `week-6-redo-fix` (commit `ee7c1cdf31ee65b973018afccbf33f69968e173f`) **plus** `data/processed/courses.jsonl` (parsed NMIMS syllabi; only in this zip, never on GitHub) |

Repository: https://github.com/jainamsethia/Curriculum-Intelligence-Engine/tree/week-6-redo-fix

## Key numbers
- API: 7 endpoints, all answering on real NMIMS data (`docs/examples/`); engine version 0.6.0.
- Prerequisite checker v2 vs v1 (test): flags 33 → 14; removal recall 0.880 → 0.711; rename false alarms 0.120 → 0.000.
- Flag precision: same labels 0.100 → 0.222; fresh validation sample 0.200 → 0.450 (intervals overlap; target 0.600 still missed).

## Why this should score higher
1. A working reusable asset: installable package + FastAPI service, structured Pydantic input/output, OpenAPI docs, `docs/API.md`, examples.
2. Confidence scores are defined from measured precision on the evaluation, and every response carries version information.
3. The weakest Week 5 result (prerequisite flags) was improved from its error analysis and re-evaluated honestly, including what got worse (removal recall).
4. Code visible: tag, zip, one command, unit and API tests.
5. The grader can run it: the parsed dataset ships in the zip.

## Self-check for the grader
- [ ] `pip install -e .[dev]` then `uvicorn curriculum_engine.api:app` serves `/docs`
- [ ] `python -m pytest -q` passes (API tests need no NMIMS data)
- [ ] Every endpoint has an example in `docs/examples/`
