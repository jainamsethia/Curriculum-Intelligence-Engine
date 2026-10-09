# Group 9 — Curriculum Intelligence Engine: Weeks 1–5 (redo) — index

All five reports in this folder were rebuilt from one commit, so every number agrees across them and with `results/*.json`.

- Repository: {{git:repo}}
- Bundle built from tag `{{git:tag}}`, commit `{{git:commit}}`
- Data: NMIMS semester syllabus booklets ({{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} non-exam PDFs parsed; no OCR yet)
- Labels: LLM-annotated reference labels. No human annotators were used; results are indicative. Pass 2 of the course pairs: Phi-3 for {{week2/labels:pass2_by.phi3}} pairs, a second Claude prompt for {{week2/labels:pass2_by.claude_b}} pairs (agreement on those is not independent). Detector flags: two Claude passes plus a pass 3, no Phi-3 (agreement not independent).
- Not run (time): paraphrase variant of the robustness tests; bge-base prerequisite re-tuning.

| Week | Report | Week tag | Key metric |
|---|---|---|---|
| 1 | `Group9_Curriculum_Intelligence_Engine_Week1_Proposal_v2.docx` | `week-1-redo` | targets fixed per capability |
| 2 | `Group9_Curriculum_Intelligence_Engine_Week2_Dataset_Baseline.docx` | `week-2-redo` | {{dataset/stats:course_versions}} course versions; TF-IDF overlap Spearman {{week2/overlap:methods.tfidf_cosine.test.spearman.value}} |
| 3 | `Group9_Curriculum_Intelligence_Engine_Week3_Core_NLP_Model.docx` | `week-3-redo` | MaxSim overlap Spearman {{week3/overlap:methods.maxsim_soft.test.spearman.value}}; rename false alarms {{week3/prerequisites:tests.rename_false_alarm.value}} |
| 4 | `Group9_Curriculum_Intelligence_Engine_Week4_Improved_Model.docx` | `week-4-redo` | selected overlap method: {{week4/overlap:selected_on_validation}} |
| 5 | `Group9_Curriculum_Intelligence_Engine_Week5_Evaluation_Error_Analysis.docx` | `week-5-redo` | {{week5/final:targets_met}} of {{week5/final:targets_total}} targets met; {{week5/failures:n}} failure cases |

Code: `Group9_Curriculum_Intelligence_Engine_Weeks1-5_code.zip` = `git archive` of `{{git:tag}}`. Run `python -m curriculum_engine.run all` (needs the NMIMS zip; see README) and `python -m pytest -q`.
