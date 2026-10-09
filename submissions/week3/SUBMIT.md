# Week 3 (Core NLP Model, redo) — what to upload on Teams

| File (in `submissions/week3/`) | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week3_Core_NLP_Model.docx` | the redone Week 3 report (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week3_code.zip` | the repository at tag `week-3-redo` (commit `2d41ffa154bd37c4217c20468f7c8d07d4f7463b`) |

Repository: https://github.com/jainamsethia/Curriculum-Intelligence-Engine/tree/week-3-redo

## Key numbers (test split; Week 2 baseline → Week 3 core)
- Overlap Spearman: 0.708 → 0.667
- Redundancy, injected near-duplicates in top 20: 0.600 → 1.000
- Prerequisites, false alarms after renaming: 1.000 → 0.120
- Diff precision with renamed courses: 0.793 → 0.993

## Why this should score higher
1. Clear progress over Week 2 on the same split and metrics, shown in one table.
2. The three core outcomes (overlap, redundancy, missing prerequisites) each get a semantic model, not side tasks.
3. Prerequisites are checked against each NMIMS programme's semester structure.
4. Example JSON on real NMIMS courses; flags evaluated with LLM-annotated reference labels.
5. Code visible: tag, commit, zip, one command.

## Self-check for the grader
- [ ] Progress table compares Week 2 and Week 3 on every capability
- [ ] Example output shows common and unique topics of two real courses
- [ ] `python -m pytest -q` passes
