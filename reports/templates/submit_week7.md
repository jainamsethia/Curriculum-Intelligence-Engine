# Week 7 (Integration, UI and Demo) — what to upload on Teams

| File (in `submissions/week7/`) | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week7_Integration_Demo.docx` | the Week 7 report (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week7_code.zip` | the repository at tag `{{git:tag}}` (commit `{{git:commit}}`) **plus** `data/processed/courses.jsonl` (parsed NMIMS syllabi) and `data/week7/` (the draft syllabus as `.docx` / `.md`, the Group 4 passage CSV); both only in this zip, never on GitHub |

Repository: {{git:repo}}/tree/{{git:tag}}

## Key numbers
- Demo step (31 October plan): {{week7/summary:jds}} JDs + {{week7/summary:courses}} current syllabi → {{week7/summary:identical_pairs}} of {{week7/summary:pairs}} syllabus pairs identical; {{week7/summary:demanded_themes}} demanded skill themes ({{week7/summary:demanded_covered}} taught, {{week7/summary:demanded_gaps}} gaps); draft syllabus with {{week7/summary:units}} units, every unit traceable to a syllabus or JD.
- Taught / gap decision vs LLM-annotated reference labels: accuracy {{week7/coverage_precision:all_themes.accuracy}} on {{week7/coverage_precision:labelled}} themes.
- Group 11 hand-off: request and response valid against their schema (mode: {{week7/group11_handoff:mode}}); Group 4 hand-off: {{week7/contracts:group4_rows}} passages in their schema; stand-in answered {{week7/group4_handoff:answered_from_expected_section}} of {{week7/group4_handoff:questions}} questions from the right section.
- API: {{week7/summary:endpoints}} endpoints; emerging-topic accuracy {{week7/emerging_threshold:week3_threshold.accuracy}} → {{week7/emerging_threshold:unit_match_threshold.accuracy}}.

## What covers each Week 7 requirement
1. User interface / demo: `app/committee_ui.py` (Streamlit over the API), `docs/DEMO_SCRIPT.md`.
2. API integration: `POST /modernize-curriculum`, `GET /courses`; `curriculum_engine/integrations.py` (Group 11 client, Group 4 export).
3. Documentation: `docs/API.md`, `docs/INTEGRATION.md`, README.
4. Responsible-AI analysis: report section 4, `docs/RESPONSIBLE_AI.md`, `results/week7/fairness.json`.
5. Place in the AI University architecture: `docs/ARCHITECTURE.md`, `results/week7/fig_architecture.png`.
6. Interaction with another project's component: Group 11 notice and Group 4 passage export (report section 2).

## Before the joint demo on 31 October
- Group 11 must run their `/process-notice` service (their summariser weights are not in their repository); then set `GROUP11_URL` or paste the URL in the UI. Without it the UI uses the marked mock.
- Group 4 should load `data/week7/group4_chunks.csv` (in the Teams zip, or downloaded from the UI) into their knowledge base.
- The JDs are synthetic samples; paste real postings if the instructor provides them.

## Self-check for the grader
- [ ] `pip install -e .[dev]`, then `uvicorn curriculum_engine.api:app` and `streamlit run app/committee_ui.py` show the demo tab
- [ ] `python -m pytest -q` passes (no NMIMS data needed)
- [ ] `data/week7/new_curriculum.docx` (in the zip) is the draft syllabus of the last run
