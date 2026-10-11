# Week 6 (Asset Engineering) — what to upload on Teams

| File (in `submissions/week6/`) | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week6_Asset_Engineering.docx` | the Week 6 report (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week6_code.zip` | the repository at tag `{{git:tag}}` (commit `{{git:commit}}`) **plus** `data/processed/courses.jsonl` (parsed NMIMS syllabi; only in this zip, never on GitHub) |

Repository: {{git:repo}}/tree/{{git:tag}}

## Key numbers
- API: {{week6/api_examples:endpoints}} endpoints, all answering on real NMIMS data (`docs/examples/`); engine version {{week6/api_examples:engine_version}}.
- Prerequisite checker v2 vs v1 (test): flags {{week6/prerequisites_v2:flags_test.v1}} → {{week6/prerequisites_v2:flags_test.v2}}; removal recall {{week6/prerequisites_v2:tests_test.v1.removal_recall.value}} → {{week6/prerequisites_v2:tests_test.v2.removal_recall.value}}; rename false alarms {{week6/prerequisites_v2:tests_test.v1.rename_false_alarm.value}} → {{week6/prerequisites_v2:tests_test.v2.rename_false_alarm.value}}.
- Flag precision: same labels {{week6/prerequisites_v2:same_labels.v1_precision}} → {{week6/prerequisites_v2:same_labels.v2_precision_on_those}}; fresh validation sample {{week6/prerequisites_v2:fresh_sample.val_v1.precision}} → {{week6/prerequisites_v2:fresh_sample.val_v2.precision}} (intervals overlap; target {{plan/targets:prerequisites.flag_precision}} still missed).

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
