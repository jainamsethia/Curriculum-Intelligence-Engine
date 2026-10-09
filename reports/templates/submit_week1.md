# Week 1 (Proposal v2) — what to upload on Teams

Upload these files from `submissions/week1/`:

| File | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week1_Proposal_v2.docx` | the revised proposal (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week1_code.zip` | the repository at tag `{{git:tag}}` (commit `{{git:commit}}`) |

Repository: {{git:repo}}/tree/{{git:tag}}

## Why this should score higher
1. Dataset is now our own NMIMS semester syllabus booklets ({{dataset/parse_stats:courses}} course versions), as the Week 2 feedback asked; MIT data is archived.
2. Every capability has a numeric target fixed before any result (`results/plan/targets.json`), so progress is measurable week on week.
3. Overlap, redundancy and missing prerequisites are the first three capabilities and the headline metrics, as the Week 4 feedback asked.
4. Code is visible: CODE block with tag and commit on page 1, plus the code zip.
5. Labels are described honestly: LLM-annotated reference labels with a multi-pass protocol; the scanned-PDF coverage limit is stated.

## Self-check for the grader
- [ ] CODE block on page 1 shows tag `{{git:tag}}` and commit `{{git:commit}}`
- [ ] Feedback table maps each Week 2-4 comment to a change and an evidence file
- [ ] Success criteria table gives a target for each capability
- [ ] `python -m pytest -q` passes in the unzipped code
