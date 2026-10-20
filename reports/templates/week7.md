# Curriculum Intelligence Engine — Week 7: Integration, UI and Demo
Group 9  ·  Roll Nos: J048, J049, J050, J052, J053  ·  AI University NLP Course Project 2026  ·  Week 7

## CODE
```
Repository : {{git:repo}}
Tag        : {{git:tag}}
Commit     : {{git:commit}}
Install    : pip install -e .[dev]
Serve      : uvicorn curriculum_engine.api:app --port 8000        (OpenAPI docs at /docs)
UI         : streamlit run app/committee_ui.py                     (calls the API; CE_API_URL, default http://localhost:8000)
Reproduce  : python -m curriculum_engine.run all                  (data → week2 … week7; NMIMS_ZIP or data/processed/courses.jsonl)
Tests      : python -m pytest -q                                   (unit, API and UI tests on a synthetic fixture)
New        : curriculum_engine/modernize.py, integrations.py, experiments/week7.py, app/committee_ui.py, demo/jds/,
             docs/INTEGRATION.md, RESPONSIBLE_AI.md, ARCHITECTURE.md, DEMO_SCRIPT.md
Folders    :
{{git:tree}}
```

## Feedback addressed
| Comment | What we changed | Evidence |
|---|---|---|
| Week 7 brief: demonstrate at least one interaction with another project's component | The draft syllabus goes to Group 11 as a notice in their `/process-notice` format and to Group 4 as passages in their knowledge-base schema; both run from the UI. | `docs/INTEGRATION.md`, `results/week7/group11_handoff.json`, `data/week7/group4_chunks.csv` (Teams zip) |
| Demo plan for 31 October: five JDs on fine-tuning LLMs + current NLP syllabi → overlapping content + a newly structured curriculum | New endpoint `POST /modernize-curriculum` and the UI tab "Modernise curriculum (demo)". | `data/week7/new_curriculum.docx` (Teams zip), `docs/DEMO_SCRIPT.md` |
| Week 7 brief: UI, API integration, documentation, responsible AI, place in the AI University architecture | Streamlit committee UI over the API, architecture diagram, responsible-AI analysis with a per-programme table. | `app/`, `docs/ARCHITECTURE.md`, `docs/RESPONSIBLE_AI.md` |
| Our Week 6 limitation: emerging-topic coverage looked too lenient | Coverage now uses the unit-match threshold instead of the prerequisite-topic threshold, measured on the labelled emerging-topic decisions. | `results/week7/emerging_threshold.json` |

## Progress vs previous week
| Item | Week 6 | Week 7 |
|---|---|---|
| API | {{week6/api_examples:endpoints}} endpoints | {{week7/summary:endpoints}} endpoints (adds `POST /modernize-curriculum`, `GET /courses`) |
| User interface | OpenAPI page only | Streamlit committee UI: demo tab + five analysis tabs |
| Interaction with other groups | planned | Group 11 notice (mode: {{week7/summary:group11_mode}}), Group 4 passages ({{week7/summary:group4_passages}} rows) |
| Emerging-topic coverage accuracy (labelled decisions) | {{week7/emerging_threshold:week3_threshold.accuracy}} | {{week7/emerging_threshold:unit_match_threshold.accuracy}} |
| Responsible AI | limitations list | fairness per programme, hallucination guard, human-in-the-loop, misuse analysis |

## 1. Demo capability: job descriptions + current syllabi → overlaps and a new syllabus
**Input.** {{week7/summary:jds}} job descriptions on fine-tuning LLMs (synthetic samples written for the demo and marked as such; real JDs can be pasted) and {{week7/summary:courses}} current syllabi: Natural Language Processing as taught in AI, COMP / CSE-DS / IT and EXTC, Speech and Natural Language Processing (DS) and Generative AI (AI). **Method** (`modernize.py`, no LLM at run time):
- JDs are split into {{week7/summary:requirement_statements}} requirement statements; spaCy noun chunks give skill phrases (verbs, determiners and generic words removed).
- Phrases are clustered (MiniLM embeddings, cosine, average linkage) into {{week7/summary:skill_themes}} skill themes; {{week7/summary:demanded_themes}} are named by at least two JDs.
- A theme is already taught if a unit topic of the selected syllabi reaches the unit-match threshold ({{week3/thresholds:unit_match:.2f}}, chosen on validation pairs in Week 3 and reused here); the closest unit is shown as evidence.
- Overlap between the syllabi uses the Week 4 fusion score and unit-level MaxSim. Units taught in two or more syllabi form shared topics.
- The draft keeps each shared topic once (from the most recent syllabus), keeps existing units that teach demanded skills, adds new units for demanded gaps, moves tools (Python, cloud GPUs) to the lab environment, reuses source outcomes, takes prerequisites from the sources with the checker-v2 phrase cleaning, and checks the hours. Unit text is copied or templated, never generated, and every unit names its source (traceability check: {{week7/summary:traceable}}).

**Result.** {{week7/summary:identical_pairs}} of the {{week7/summary:pairs}} syllabus pairs are identical: the same NLP syllabus is taught in several programmes. Speech and NLP overlaps it strongly; Generative AI hardly overlaps. Of the {{week7/summary:demanded_themes}} demanded themes, {{week7/summary:demanded_covered}} are taught and {{week7/summary:demanded_gaps}} are gaps:
{{table:week7/summary:theme_table}}
The draft syllabus has {{week7/summary:units}} units ({{week7/summary:units_shared}} shared, {{week7/summary:units_existing_demanded}} existing units demanded by the JDs, {{week7/summary:units_new}} new) and {{week7/summary:outcomes}} outcomes. Its units add up to {{week7/summary:total_hours}} hours, more than one semester's lecture hours, so the draft itself tells the committee to trim it or split the course. It is marked "DRAFT for the Board of Studies".
{{table:week7/summary:unit_table}}

**How good is the "taught / gap" decision?** All {{week7/coverage_precision:labelled}} themes were judged taught or not taught against the syllabus text with LLM-annotated reference labels (two Claude passes with different wording, pass 3 for {{week7/coverage_precision:needed_pass3}}, {{week7/coverage_precision:uncertain}} uncertain; no scores shown). Accuracy {{week7/coverage_precision:all_themes.accuracy}}; gap precision {{week7/coverage_precision:all_themes.gap.precision}} (n = {{week7/coverage_precision:all_themes.gap.n}}), "taught" precision {{week7/coverage_precision:all_themes.covered.precision}} (n = {{week7/coverage_precision:all_themes.covered.n}}). On the demanded themes only: gap precision {{week7/coverage_precision:demanded_themes.gap.precision}} (n = {{week7/coverage_precision:demanded_themes.gap.n}}), "taught" precision {{week7/coverage_precision:demanded_themes.covered.precision}} (n = {{week7/coverage_precision:demanded_themes.covered.n}}); the samples are small and the JDs were also used while building the method, so these numbers are indicative. Weak spots are visible in the table: a generic theme ("learning") and new units titled from one phrase ("Bias").

## 2. Interaction with Group 11 and Group 4
- **Group 11, University Communications Intelligence** (`Ar-Srivas/uni-comms-intelligence`): their FastAPI `POST /process-notice` takes `{doc_id, text}` and returns title, category, audience, entities and per-audience summaries. We turn the draft into a notice and send it; request and response are validated with their Pydantic models (request valid: {{week7/contracts:group11_request_schema}}, response valid: {{week7/contracts:group11_response_schema}}), and the reply becomes one email per audience. Their service cannot be started from the repository because the fine-tuned summariser weights are not committed, so this run used a documented mock (mode: {{week7/group11_handoff:mode}}; every mock reply carries a `_mock` marker). Setting `GROUP11_URL` (or the URL box in the UI) switches to the live call for the joint demo.
- **Group 4, University Knowledge Assistant** (`vedantkokate2104-ux/SNLP_CHATBOT`): no HTTP API; their knowledge base is a passage table (`chunk_id, page, section, part, text`) cut into word windows with overlap. We export the draft as {{week7/contracts:group4_rows}} passages in exactly that schema (columns valid: {{week7/contracts:group4_columns}}; longest passage {{week7/contracts:group4_max_words}} words) for them to add to their knowledge base. To test the hand-off without their data, a stand-in that imitates their offline extractive mode (hybrid BM25 + MiniLM with their keyword weight) answered {{week7/group4_handoff:answered_from_expected_section}} of {{week7/group4_handoff:questions}} student questions (prerequisites, lab work, credits, fine-tuning topics, assessment) from the right section. The miss: for fine-tuning topics it quoted the LLM unit, not "RAG and Fine Tuning". The stand-in is not Group 4's system; their assistant answers in the joint demo.

## 3. User interface and architecture
![Committee UI (Streamlit, calling the API). Left: the five JDs, the selected syllabi and the overlap table. Right: the Group 11 notice turned into emails (mock mode) and the Group 4 hand-off with a stand-in answer and its source.](results/week7/fig_ui_report.png)
The UI has a demo tab (JDs, syllabus selection, overlaps, skills table, draft with `.docx` / `.md` / `.json` download, both hand-offs) and tabs for comparing two courses, two years of a programme, prerequisites, redundancy and emerging topics. A sidebar shows the API health and a fixed warning that outputs are drafts. Group 9 sits between the syllabus data and the student- and faculty-facing services of the AI University application:
![Group 9 in the AI University application (Mermaid source in docs/ARCHITECTURE.md).](results/week7/fig_architecture.png)

## 4. Responsible-AI analysis
| Risk | What we do | Evidence |
|---|---|---|
| Privacy, confidentiality | Raw booklets and parsed data are never in the public repository (Teams zip only); the Week 1 scan found no student or faculty personal data; the JDs are synthetic. | `.gitignore`, `docs/RESPONSIBLE_AI.md` |
| Hallucination | The draft is copied or templated from syllabi and JDs, never generated; every unit names its source; the mock is marked. | traceability check, `_mock` marker |
| Transparency | Every answer shows its evidence (closest unit, source course, JD) and a confidence measured on the evaluation, with the engine version. | `docs/API.md` |
| Human in the loop, misuse | Outputs are drafts for the Board of Studies; the UI warns against cutting or merging courses automatically; the hours check flags overload. | UI sidebar, draft status |
| Fairness across departments | Quality differs by programme (table below); programmes with too little parsed data get no prerequisite checks. | `results/week7/fairness.json` |

{{table:week7/fairness:table}}
Bloom accuracy ranges from {{week7/fairness:bloom_accuracy_min.0}} ({{week7/fairness:bloom_accuracy_min.1}}) to {{week7/fairness:bloom_accuracy_max.0}} ({{week7/fairness:bloom_accuracy_max.1}}); "n/a": not enough labelled items or parsed data. The emerging-topic fix: at the old threshold ({{week7/emerging_threshold:week3_threshold.threshold:.2f}}) almost every topic counted as taught (gap recall {{week7/emerging_threshold:week3_threshold.gap_recall}} on {{week7/emerging_threshold:labelled_decisions}} labelled decisions); at {{week7/emerging_threshold:unit_match_threshold.threshold:.2f}} gap recall is {{week7/emerging_threshold:unit_match_threshold.gap_recall}} and gap precision {{week7/emerging_threshold:unit_match_threshold.gap_precision}}. The threshold was picked after seeing these labels, so these numbers are optimistic.

## 5. Limitations and next steps
- Data: NMIMS semester syllabus booklets; {{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} non-exam PDFs are parsed, the rest are scanned and need OCR (future work), so a topic can look missing when it sits in an unread booklet.
- Labels come from LLM-annotated reference labels. No human annotators were used; results are indicative. For {{week2/labels:pass2_by.claude_b}} of {{week2/labels:items}} pairs the second labeller was a Claude prompt rather than Phi-3, so agreement on those items is not independent. Detector flags (Week 3 onwards) were labelled by two Claude passes with different wording plus a pass 3, with no Phi-3, so agreement on flags is not independent; the same holds for the Week 7 skill-coverage labels.
- The draft syllabus and the Group 4 passages quote NMIMS syllabus text, so they are in the Teams code zip (`data/week7/`) and not in the public repository.
- The JDs are synthetic; the draft is a starting point for faculty, not a finished syllabus (hours over budget, terse new-unit titles).
- Peer services: Group 11's model and Group 4's knowledge base are not runnable from their repositories, so the interactions are checked against their schemas, with a marked mock and a stand-in; the live run is planned for the joint demo on 31 October.
- Week 8: final testing, final report, README, presentation and the live demo with Group 4 and Group 11.

## AI assistance
Code and report text were written with AI assistance (Claude). Reference labels are LLM-annotated (Claude, Phi-3). Every number is generated from `results/*.json` by `tools/build_report.py` and verified by `tools/check_reports.py`.
