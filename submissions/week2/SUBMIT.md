# Week 2 (Dataset & Baseline, redo) — what to upload on Teams

Upload these files from `submissions/week2/`:

| File | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week2_Dataset_Baseline.docx` | the redone Week 2 report (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week2_code.zip` | the repository at tag `week-2-redo` (commit `c3a1e1b2358da725cfa81f4b28838a2c1c3172a4`) |

Repository: https://github.com/jainamsethia/Curriculum-Intelligence-Engine/tree/week-2-redo

## Key numbers
- NMIMS dataset: 2,366 course versions, 645 course families, 13 programmes; 531 of 2,413 PDFs parsed (rest scanned).
- Split (families): 387 / 129 / 129; LLM-annotated reference labels: 200 pairs, LLM-vs-LLM agreement Claude vs Phi-3 74% (106 pairs), Claude vs Claude 99% (94 pairs, not independent).
- Overlap baseline (TF-IDF): Spearman 0.708, ROC-AUC 0.959.
- Prerequisite baseline: removal recall 1.000, but false alarms 47% (intact) and 100% (renamed course).
- Bloom lexicon vs faculty tags: accuracy 0.776.

## Why this should score higher
1. Uses our own college's data, as the feedback asked: NMIMS semester syllabus booklets only.
2. Real data engineering: parser bug fixed (19 records), course families, programme structure, family-level split.
3. Baselines for overlap, redundancy and missing prerequisites (the core objective), not side tasks, each with a test-split metric.
4. Honest ground truth: LLM-annotated reference labels with a documented multi-pass protocol and agreement, plus label-free tests.
5. Code is visible and reproducible: CODE block, tag, zip, one command, unit tests.

## Self-check for the grader
- [ ] CODE block shows tag `week-2-redo`, commit `c3a1e1b2358da725cfa81f4b28838a2c1c3172a4`, run command
- [ ] Dataset section names NMIMS semester syllabus booklets and states the scanned-PDF coverage limit
- [ ] Split is by course family (no course on both sides)
- [ ] Baseline table covers overlap, redundancy, prerequisites, diff, Bloom
- [ ] `python -m pytest -q` passes in the unzipped code
