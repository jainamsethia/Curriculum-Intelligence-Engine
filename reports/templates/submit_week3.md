# Week 3 (Core NLP Model, redo) — what to upload on Teams

| File (in `submissions/week3/`) | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week3_Core_NLP_Model.docx` | the redone Week 3 report (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week3_code.zip` | the repository at tag `{{git:tag}}` (commit `{{git:commit}}`) |

Repository: {{git:repo}}/tree/{{git:tag}}

## Key numbers (test split; Week 2 baseline → Week 3 core)
- Overlap Spearman: {{week3/overlap:methods.tfidf_cosine.test.spearman.value}} → {{week3/overlap:methods.maxsim_soft.test.spearman.value}}
- Redundancy, injected near-duplicates in top 20: {{week2/redundancy:synthetic_injection.recall_at_20}} → {{week3/redundancy:synthetic_injection.recall_at_20}}
- Prerequisites, false alarms after renaming: {{week2/prerequisites:tests.rename_false_alarm.value}} → {{week3/prerequisites:tests.rename_false_alarm.value}}
- Diff precision with renamed courses: {{week2/diff:rename_test.precision}} → {{week3/diff:rename_test.precision}}

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
