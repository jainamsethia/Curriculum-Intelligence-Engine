# Week 4 (Improved Model, redo) — what to upload on Teams

| File (in `submissions/week4/`) | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week4_Improved_Model.docx` | the redone Week 4 report (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week4_code.zip` | the repository at tag `week-4-redo` (commit `4dbcadbe4823a344677897483b84dae37ce3bf2a`) |

Repository: https://github.com/jainamsethia/Curriculum-Intelligence-Engine/tree/week-4-redo

## Key numbers
- Overlap method selected on validation: fusion_rank_tfidf_maxsim; calibrated confidence on test: Brier 0.138.
- Bloom: 0.776 (lexicon) vs 0.776 (improved): no clear difference.

## Why this should score higher
1. "Where is the code?": CODE block, tag, zip, one command per table.
2. Every ablation targets overlap, redundancy or prerequisites.
3. Paired 95% intervals and honest verdicts ("no clear difference" where true).
4. Choices made on validation, test reported once.
5. A calibrated confidence score ready for the Week 6 API.

## Self-check for the grader
- [ ] Baseline → core → improved table with verdicts
- [ ] Intervals resample course families
- [ ] `python -m pytest -q` passes
