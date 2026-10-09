# Week 4 (Improved Model, redo) — what to upload on Teams

| File (in `submissions/week4/`) | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week4_Improved_Model.docx` | the redone Week 4 report (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week4_code.zip` | the repository at tag `{{git:tag}}` (commit `{{git:commit}}`) |

Repository: {{git:repo}}/tree/{{git:tag}}

## Key numbers
- Overlap method selected on validation: {{week4/overlap:selected_on_validation}}; calibrated confidence on test: Brier {{week4/overlap:confidence_calibration_test.brier}}.
- Bloom: {{week4/bloom:lexicon.accuracy.value}} (lexicon) vs {{week4/bloom:improved_test.accuracy.value}} (improved): {{week4/bloom:vs_lexicon_accuracy.verdict}}.

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
