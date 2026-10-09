# Week 2 (Dataset & Baseline, redo) — what to upload on Teams

Upload these files from `submissions/week2/`:

| File | What it is |
|---|---|
| `Group9_Curriculum_Intelligence_Engine_Week2_Dataset_Baseline.docx` | the redone Week 2 report (a `.pdf` copy is in the same folder) |
| `Group9_Curriculum_Intelligence_Engine_Week2_code.zip` | the repository at tag `{{git:tag}}` (commit `{{git:commit}}`) |

Repository: {{git:repo}}/tree/{{git:tag}}

## Key numbers
- NMIMS dataset: {{dataset/stats:course_versions}} course versions, {{dataset/stats:families}} course families, {{dataset/stats:programmes}} programmes; {{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} PDFs parsed (rest scanned).
- Split (families): {{dataset/split:families.train}} / {{dataset/split:families.val}} / {{dataset/split:families.test}}; LLM-annotated reference labels: {{week2/labels:items}} pairs, LLM-vs-LLM agreement Claude vs Phi-3 {{week2/labels:agreement_claude_vs_phi3.percent_agreement:.0%}} ({{week2/labels:pass2_by.phi3}} pairs), Claude vs Claude {{week2/labels:agreement_claude_vs_claude.percent_agreement:.0%}} ({{week2/labels:pass2_by.claude_b}} pairs, not independent).
- Overlap baseline (TF-IDF): Spearman {{week2/overlap:methods.tfidf_cosine.test.spearman.value}}, ROC-AUC {{week2/overlap:methods.tfidf_cosine.test.roc_auc.value}}.
- Prerequisite baseline: removal recall {{week2/prerequisites:tests.removal_recall.value}}, but false alarms {{week2/prerequisites:tests.intact_false_alarm.value:.0%}} (intact) and {{week2/prerequisites:tests.rename_false_alarm.value:.0%}} (renamed course).
- Bloom lexicon vs faculty tags: accuracy {{week2/bloom:lexicon.accuracy.value}}.

## Why this should score higher
1. Uses our own college's data, as the feedback asked: NMIMS semester syllabus booklets only.
2. Real data engineering: parser bug fixed ({{dataset/parse_stats:prerequisite_fix_changed}} records), course families, programme structure, family-level split.
3. Baselines for overlap, redundancy and missing prerequisites (the core objective), not side tasks, each with a test-split metric.
4. Honest ground truth: LLM-annotated reference labels with a documented multi-pass protocol and agreement, plus label-free tests.
5. Code is visible and reproducible: CODE block, tag, zip, one command, unit tests.

## Self-check for the grader
- [ ] CODE block shows tag `{{git:tag}}`, commit `{{git:commit}}`, run command
- [ ] Dataset section names NMIMS semester syllabus booklets and states the scanned-PDF coverage limit
- [ ] Split is by course family (no course on both sides)
- [ ] Baseline table covers overlap, redundancy, prerequisites, diff, Bloom
- [ ] `python -m pytest -q` passes in the unzipped code
