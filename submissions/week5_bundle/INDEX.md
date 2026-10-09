# Group 9 — Curriculum Intelligence Engine: Weeks 1–5 (redo) — index

All five reports in this folder were rebuilt from one commit, so every number agrees across them and with `results/*.json`.

- Repository: https://github.com/jainamsethia/Curriculum-Intelligence-Engine
- Bundle built from tag `week-5-redo`, commit `85f3791d2804d1fd1a55c371cd156e2121d902b7`
- Data: NMIMS semester syllabus booklets (531 of 2,413 non-exam PDFs parsed; no OCR yet)
- Labels: LLM-annotated reference labels. No human annotators were used; results are indicative. Pass 2 of the course pairs: Phi-3 for 106 pairs, a second Claude prompt for 94 pairs (agreement on those is not independent). Detector flags: two Claude passes plus a pass 3, no Phi-3 (agreement not independent).
- Not run (time): paraphrase variant of the robustness tests; bge-base prerequisite re-tuning.

| Week | Report | Week tag | Key metric |
|---|---|---|---|
| 1 | `Group9_Curriculum_Intelligence_Engine_Week1_Proposal_v2.docx` | `week-1-redo` | targets fixed per capability |
| 2 | `Group9_Curriculum_Intelligence_Engine_Week2_Dataset_Baseline.docx` | `week-2-redo` | 2,366 course versions; TF-IDF overlap Spearman 0.708 |
| 3 | `Group9_Curriculum_Intelligence_Engine_Week3_Core_NLP_Model.docx` | `week-3-redo` | MaxSim overlap Spearman 0.667; rename false alarms 0.120 |
| 4 | `Group9_Curriculum_Intelligence_Engine_Week4_Improved_Model.docx` | `week-4-redo` | selected overlap method: fusion_rank_tfidf_maxsim |
| 5 | `Group9_Curriculum_Intelligence_Engine_Week5_Evaluation_Error_Analysis.docx` | `week-5-redo` | 13 of 14 targets met; 12 failure cases |

Code: `Group9_Curriculum_Intelligence_Engine_Weeks1-5_code.zip` = `git archive` of `week-5-redo`. Run `python -m curriculum_engine.run all` (needs the NMIMS zip; see README) and `python -m pytest -q`.
