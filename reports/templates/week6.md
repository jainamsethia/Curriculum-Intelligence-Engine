# Curriculum Intelligence Engine — Week 6: Asset Engineering (reusable NLP asset)
Group 9  ·  Roll Nos: J048, J049, J050, J052, J053  ·  AI University NLP Course Project 2026  ·  Week 6

## CODE
```
Repository : {{git:repo}}
Tag        : {{git:tag}}
Commit     : {{git:commit}}
Install    : pip install -e .[dev]
Serve      : uvicorn curriculum_engine.api:app --port 8000        (OpenAPI docs at /docs)
Reproduce  : python -m curriculum_engine.run all                  (data → week2 … week6; NMIMS_ZIP or data/processed/courses.jsonl)
Tests      : python -m pytest -q                                   (unit tests + API tests on a synthetic fixture)
New        : curriculum_engine/api.py, api_engine.py, schemas.py, prereq.py (checker v2), experiments/week6.py, docs/API.md
Folders    :
{{git:tree}}
```

## Feedback addressed
| Comment | What we changed | Evidence |
|---|---|---|
| Week 4: "Where is the code? … overlapping content, missing pre-requisites, detect redundancies" | The three core capabilities are now endpoints of an installable package and API with structured input, output and confidence. | `curriculum_engine/api.py`, `docs/API.md` |
| Our Week 5 error analysis: prerequisite flags were mostly wrong (cases F05–F07) | Prerequisite checker v2: phrase cleaning, non-subject filter, alias / umbrella names, own-content filter; re-evaluated on the same split and labels. | `results/week6/prerequisites_v2.json` |
| Week 2: "an asset that will be of value to NMIMS" | Every answer runs on our NMIMS semester syllabus booklets and can also take a new syllabus as raw text. | `docs/examples/` |

## Progress vs previous week
| Item | Week 5 | Week 6 |
|---|---|---|
| Reusable asset | scripts and a Python engine class | installable package `curriculum_engine` version {{week6/api_examples:engine_version}}, FastAPI service with {{week6/api_examples:endpoints}} endpoints, Pydantic schemas, OpenAPI |
| Missing-prerequisite flags on test courses | {{week6/prerequisites_v2:flags_test.v1}} | {{week6/prerequisites_v2:flags_test.v2}} |
| Prerequisite flag precision, fresh validation sample | {{week6/prerequisites_v2:fresh_sample.val_v1.precision}} | {{week6/prerequisites_v2:fresh_sample.val_v2.precision}} |

## 1. Prerequisite checker v2 (before / after, honest)
The Week 5 failure cases showed why most missing-prerequisite flags were wrong: umbrella names ("Engineering Mathematics" when Calculus and Linear Algebra are taught), fragments created by splitting free text ("III", "Even", "Management"), and phrases that are not subjects ("subjects studied till semester V", "geography", topics of the course itself). Checker v2 adds four steps before a flag is raised:
- **Phrase cleaning**: parentheses and course codes are dropped; "and" / "or" only split a phrase when both sides have two or more words; a bare roman numeral is re-attached to the preceding course name.
- **Non-subject filter**: phrases such as "all courses studied till …", generic skills and school subjects are skipped.
- **Alias / umbrella names**: the prerequisite counts as met if every content word of the phrase, or one member subject of a broad subject (mathematics → calculus, algebra, differential equations …), appears in the name of one earlier course.
- **Own-content filter**: a phrase that the course's own units teach is course content, not a prerequisite (threshold {{week6/prerequisites_v2:own_content_threshold_from_val}}, chosen on validation).

| Measure | v1 (Week 3) | v2 (Week 6) |
|---|---|---|
| Removal recall, test cases | {{week6/prerequisites_v2:tests_test.v1.removal_recall.value}} | {{week6/prerequisites_v2:tests_test.v2.removal_recall.value}} |
| False alarms, intact programme | {{week6/prerequisites_v2:tests_test.v1.intact_false_alarm.value}} | {{week6/prerequisites_v2:tests_test.v2.intact_false_alarm.value}} |
| False alarms after renaming the course | {{week6/prerequisites_v2:tests_test.v1.rename_false_alarm.value}} | {{week6/prerequisites_v2:tests_test.v2.rename_false_alarm.value}} |
| Precision on the same {{week6/prerequisites_v2:same_labels.labelled_flags}} labelled test flags | {{week6/prerequisites_v2:same_labels.v1_precision}} | {{week6/prerequisites_v2:same_labels.v2_precision_on_those}} on the {{week6/prerequisites_v2:same_labels.v2_still_flagged}} flags v2 still raises |
| Precision of all v2 test flags (fresh labels where the phrase changed) | – | {{week6/prerequisites_v2:fresh_sample.test_v2.precision}} [{{week6/prerequisites_v2:fresh_sample.test_v2.ci95.0}}, {{week6/prerequisites_v2:fresh_sample.test_v2.ci95.1}}] (n = {{week6/prerequisites_v2:fresh_sample.test_v2.n}}) |
| Precision, fresh validation sample | {{week6/prerequisites_v2:fresh_sample.val_v1.precision}} [{{week6/prerequisites_v2:fresh_sample.val_v1.ci95.0}}, {{week6/prerequisites_v2:fresh_sample.val_v1.ci95.1}}] | {{week6/prerequisites_v2:fresh_sample.val_v2.precision}} [{{week6/prerequisites_v2:fresh_sample.val_v2.ci95.0}}, {{week6/prerequisites_v2:fresh_sample.val_v2.ci95.1}}] |

- On the same labels, v2 drops {{week6/prerequisites_v2:same_labels.wrong_flags_dropped_by_v2}} flags that were judged wrong and {{week6/prerequisites_v2:same_labels.correct_flags_dropped_by_v2}} that were judged correct, and its false alarms in the label-free tests fall to zero.
- The cost is recall: v2 misses more removed prerequisites, because another earlier course with the same subject words still counts (removing "Engineering Mathematics - I" while "Engineering Mathematics - II" is taught is arguably fine, which the removal test cannot see). Removal recall is now below the target of {{plan/targets:prerequisites.removal_recall}}, and precision is still below the target of {{plan/targets:prerequisites.flag_precision}}.
- Caution: the rules were designed from test-split failure cases, so the after-numbers on test labels are optimistic. The fresh validation sample is less biased; its intervals overlap, so the gain is indicative, not proven. A first attempt that pooled words across all earlier course names (removal recall {{week6/prerequisites_v2:v2a_first_attempt.tests_test.removal_recall.value}}) was replaced; its results stay in the results file. The {{week6/prerequisites_v2:fresh_sample_agreement.items}} fresh items were labelled with the Week 3 flag protocol: two Claude passes with different wording agreed on {{week6/prerequisites_v2:fresh_sample_agreement.both_passes_agree}} of them, so pass 3 was needed for {{week6/prerequisites_v2:fresh_sample_agreement.needed_pass3}} items (same model family: this agreement is not independent).

## 2. The reusable asset
- **Package**: `curriculum_engine` (pyproject, `pip install -e .`), entry point `python -m curriculum_engine.run`; the API engine `ApiEngine` can be used directly from Python.
- **API** (FastAPI, Pydantic v2 schemas, OpenAPI at `/docs`, exported to `docs/openapi.json`): `POST /compare-curricula` (two courses, or two programmes / two years of one programme), `POST /map-outcomes`, `POST /find-overlaps`, `POST /check-prerequisites`, `POST /detect-redundancy`, `POST /emerging-topics`, `GET /health`.
- **Structured input**: a course is an id, a course name or raw syllabus text; a programme is a code plus an optional academic year. Invalid requests get `422`, unknown courses or programmes `404`.
- **Confidence scores** are defined from the evaluation, not invented: overlap confidence = P(any overlap) from the Week 4 logistic model (relative, not well calibrated); prerequisite and redundancy flags carry their measured precision ({{week6/prerequisites_v2:fresh_sample.test_v2.precision}} and {{week3/flag_precision:redundancy.core.precision}}); a Bloom level carries the precision of that level against faculty tags (Create {{week6/bloom_confidence:Create}}, Understand {{week6/bloom_confidence:Understand}}, Apply {{week6/bloom_confidence:Apply}}).
- **Versioning**: every response has `meta` (engine version, encoder, number of course versions, split hash).
- **Examples and tests**: `docs/examples/*.json` are real responses regenerated by `run week6`; status codes: compare (courses) {{week6/api_examples:status_codes.compare_courses}}, compare (programmes) {{week6/api_examples:status_codes.compare_programmes}}, map-outcomes {{week6/api_examples:status_codes.map_outcomes}}, find-overlaps {{week6/api_examples:status_codes.find_overlaps}}, check-prerequisites {{week6/api_examples:status_codes.check_prerequisites}}, detect-redundancy {{week6/api_examples:status_codes.detect_redundancy}}, emerging-topics {{week6/api_examples:status_codes.emerging_topics}}, health {{week6/api_examples:status_codes.health}}. API tests run on a synthetic programme, so they need no NMIMS data.
- **Docker**: not provided — Docker is not installed on the development machine.

Example: a real request and response of `POST /compare-curricula` (shortened; full version in `docs/examples/compare_courses.json`):
```
{{json:week6/api_examples:compare_courses_excerpt}}
```

## 3. Limitations
- Data: NMIMS semester syllabus booklets; {{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} non-exam PDFs are parsed, the rest are scanned and need OCR (future work), so the API can report a course as removed or a prerequisite as missing when the evidence sits in an unread booklet.
- Labels and confidence values come from LLM-annotated reference labels. No human annotators were used; results are indicative. For {{week2/labels:pass2_by.claude_b}} of {{week2/labels:items}} pairs the second labeller was a Claude prompt rather than Phi-3, so agreement on those items is not independent. Detector flags (Week 3 onwards) were labelled by two Claude passes with different wording plus a pass 3, with no Phi-3, so agreement on flags is not independent.
- Redundancy pairs include templated units (project or case-study units identical across courses).
- The parsed dataset is in the Teams code zip only, never in the public repository.

## AI assistance
Code and report text were written with AI assistance (Claude). Reference labels are LLM-annotated (Claude, Phi-3). Every number is generated from `results/*.json` by `tools/build_report.py` and verified by `tools/check_reports.py`.
