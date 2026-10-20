# Curriculum Intelligence Engine — Week 8: Final Report
Group 9  ·  Roll Nos: J048, J049, J050, J052, J053  ·  AI University NLP Course Project 2026  ·  Week 8 (final, v1.0)

## CODE
```
Repository : {{git:repo}}
Tag        : {{git:tag}}
Commit     : {{git:commit}}
Install    : pip install -e .[dev]  &&  python -m spacy download en_core_web_sm
Reproduce  : python -m curriculum_engine.run all                  (data → week2 … week8; data/processed/courses.jsonl from the Teams zip)
Final test : python tools/final_test.py                            (clean clone → tests → run all → compare results; API latency)
Serve / UI : uvicorn curriculum_engine.api:app --port 8000   |   streamlit run app/committee_ui.py
Slides     : cd tools && npm install && npm run slides             (architecture PNG: npm run architecture)
Folders    :
{{git:tree}}
```

## Feedback addressed
| Comment | What we changed | Evidence |
|---|---|---|
| Week 2: wrong dataset (MIT OpenCourseWare) | NMIMS semester syllabus booklets only, from Proposal v2 on; one dataset story in every report. | `results/dataset/parse_stats.json` |
| Weeks 3–4: "minimal progress", "where is the code?" | Tagged code per week, one command rebuilds every result, tests, and an automated clean-clone reproduction. | `results/week8/final_testing.json` |
| Week 4: focus on overlapping content, missing prerequisites, redundancy | The three core capabilities are measured on a held-out split against targets fixed in advance. | `results/week8/final.json` |
| Weak ground truth | A documented labelling protocol with LLM-annotated reference labels, plus label-free synthetic tests and faculty Bloom tags. | `results/week8/labels.json` |
| Week 8 brief: final testing, report, README, presentation, demonstration, architecture diagram, final repository | Each item below, all generated from the same results files. | `submissions/week8/`, `README.md`, `docs/` |

## Progress vs previous week
| Item | Week 7 | Week 8 |
|---|---|---|
| Testing | unit, API and UI tests; manual clean-clone gate | automated final test: {{week8/final_testing:clean_clone.tests_passed}} tests pass and a clean clone reproduces all {{week8/final_testing:clean_clone.result_files_compared}} result files; latency of every endpoint |
| Results | one report per week | one table of the final v1.0 results against the success criteria |
| Architecture | figure drawn by script | Mermaid source + exported PNG, including Group 10 as a planned consumer |
| Presentation and demo | demo script | slides (.pptx) generated from the results files; recorded-demo checklist |
| README | weekly notes | final: install, run, API, data, limitations, AI use |

## 1. Problem and what v1.0 does
Curriculum committees (Board of Studies, programme chairs) revise syllabi by reading booklets by hand. The final version (v1.0) answers their questions with evidence and a confidence score, as JSON through an API and in a committee UI: which courses overlap, which units are redundant inside a programme, which prerequisites are not taught earlier, what changed between two programmes or two years, which Bloom level each outcome targets, which emerging topics are missing, and, from job descriptions, which demanded skills are already taught and what a modernised syllabus could look like. Every output is a draft for a human decision.

## 2. Data and evaluation protocol
- **Data**: NMIMS semester syllabus booklets. {{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} non-exam PDFs have a text layer and are parsed; {{dataset/parse_stats:scanned_pdf_skipped}} are scanned. Result: {{dataset/stats:course_versions}} course versions in {{dataset/stats:families}} course families and {{dataset/stats:programmes}} programmes.
- **Split**: by course family (versions of one course never cross splits): {{dataset/split:families.train}} / {{dataset/split:families.val}} / {{dataset/split:families.test}} families, hash {{dataset/split:hash}}. Thresholds are tuned on validation; the test split is used once.
- **Reference labels**: {{week8/labels:items}} items labelled by two LLM passes with different wording, without model scores, a third pass for disagreements ({{week8/labels:pass3}} items) and {{week8/labels:uncertain}} item excluded as uncertain:
{{table:week8/labels:table}}
- **Label-free checks** run alongside: injected duplicate units, removed or renamed prerequisites, real year-to-year changes, and the {{dataset/parse_stats:outcomes_with_k_level}} outcomes that carry a faculty Bloom tag.

## 3. System
![Architecture of v1.0 (Mermaid source docs/architecture.mmd, exported with mermaid-cli). Group 10 is a planned consumer, not integrated.](docs/architecture.png)
Booklets are parsed locally into course records and families; MiniLM chunk embeddings and TF-IDF feed the modules (overlap by rank fusion of TF-IDF and chunk MaxSim; unit-level redundancy; prerequisite checker v2 with alias, umbrella-name, non-subject and own-content filters; curriculum diff; Bloom verb lexicon; emerging-topic coverage; curriculum modernisation from job descriptions). A FastAPI service with {{week7/summary:endpoints}} endpoints returns JSON with evidence, confidence and the engine version; a Streamlit UI calls it; the draft syllabus is handed to Group 11 and Group 4.

## 4. Final results against the success criteria
{{table:week8/final:table}}
**{{week8/final:targets_met}} of {{week8/final:targets_total}} targets are met** (Week 5: {{week8/final:week5_targets_met}}). The difference is the prerequisite checker: v2 removed the false alarms and raised flag precision, but removal recall fell below its target, and flag precision still misses its target; that precision comes from a small fresh sample, so its interval is wide. The emerging-topic row now uses the unit-match threshold chosen in Week 7 after seeing those labels, so it is optimistic. Overlap, redundancy, curriculum diff, Bloom levels and topic labels are unchanged since Week 5. The modernisation capability had no target fixed in advance.

## 5. Final testing
- **Clean clone** of commit `{{week8/final_testing:clean_clone.commit}}`: {{week8/final_testing:clean_clone.pytest}}; `run all` finished in {{week8/final_testing:clean_clone.run_all_seconds}} s and reproduced every committed result (files compared: {{week8/final_testing:clean_clone.result_files_compared}}, reproduced: {{week8/final_testing:clean_clone.reproduced}}). Machine: {{week8/final_testing:os}}, Python {{week8/final_testing:python}}, CPU only.
- **API on the real data** (in-process, the requests of `docs/examples`, median of {{week8/final_testing:latency.repeats}} calls after one warm-up call that also loads the models):
{{table:week8/final_testing:latency.table}}
- **Report and slide checks**: every number in the reports is a placeholder resolved from `results/*.json` and re-checked by `tools/check_reports.py`; the slides read the same files; the deck passed an Office XML validation.

## 6. Demonstration and integration
For the joint demo on 31 October our step takes {{week7/summary:jds}} job descriptions on fine-tuning LLMs and {{week7/summary:courses}} current NLP syllabi, finds that {{week7/summary:identical_pairs}} of {{week7/summary:pairs}} syllabus pairs are identical, lists {{week7/summary:demanded_themes}} demanded skills ({{week7/summary:demanded_covered}} taught, {{week7/summary:demanded_gaps}} gaps) and drafts a syllabus of {{week7/summary:units}} units, each naming its source. The draft goes to Group 11 as a notice in their schema (this run: {{week7/group11_handoff:mode}}) and to Group 4 as {{week7/contracts:group4_rows}} passages in their schema; our stand-in for their offline mode answered {{week7/group4_handoff:answered_from_expected_section}} of {{week7/group4_handoff:questions}} student questions from the right section. `docs/DEMO_SCRIPT.md` has the live script and a checklist for a recorded backup.

## 7. Responsible AI
Booklets and parsed text stay local and off GitHub; the draft syllabus is copied or templated, never generated, so nothing is invented; outputs are drafts for the Board of Studies and the UI warns against automatic course cuts; quality differs by programme (Bloom accuracy {{week7/fairness:bloom_accuracy_min.0}} for {{week7/fairness:bloom_accuracy_min.1}} to {{week7/fairness:bloom_accuracy_max.0}} for {{week7/fairness:bloom_accuracy_max.1}}), and programmes with too little parsed data get no prerequisite checks. Details: `docs/RESPONSIBLE_AI.md`.

## 8. Limitations and future work
- Data: NMIMS semester syllabus booklets; {{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} non-exam PDFs are parsed, the rest are scanned and need OCR (future work), so a course can look removed or a prerequisite missing when the evidence sits in an unread booklet.
- Labels come from LLM-annotated reference labels. No human annotators were used; results are indicative. For {{week2/labels:pass2_by.claude_b}} of {{week2/labels:items}} pairs the second labeller was a Claude prompt rather than Phi-3, so agreement on those items is not independent. Detector flags (Week 3 onwards) were labelled by two Claude passes with different wording plus a pass 3, with no Phi-3, so agreement on flags is not independent. The human spot-check sheet ({{week8/labels:human_spot_check_rows}} pairs) is ready but not filled in.
- Not run, and not reported as results: paraphrase variant not run (time); bge-base prerequisite re-tuning not run (time).
- The emerging-topic reference list is a draft awaiting approval; only entries whose source page was opened count as verified.
- The demo job descriptions are synthetic; Group 11's model and Group 4's knowledge base cannot be run from their repositories, so the hand-offs are checked against their schemas with a marked mock and a stand-in; Group 10 is not integrated.
- Future work: OCR for the scanned booklets, the human spot check, a faculty-confirmed prerequisite map, real job postings, and the Group 10 elective hand-off.

## AI assistance
Code, report text and slides were written with AI assistance (Claude). Reference labels are LLM-annotated (Claude, Phi-3). Every number is generated from `results/*.json` by `tools/build_report.py` and verified by `tools/check_reports.py`.
