# Curriculum Intelligence Engine — Week 2: Dataset & Baseline (redo)
Group 9  ·  Roll Nos: J048, J049, J050, J052, J053  ·  AI University NLP Course Project 2026  ·  redone 9 Oct 2026

## CODE
```
Repository : {{git:repo}}
Tag        : {{git:tag}}
Commit     : {{git:commit}}
Run        : python -m curriculum_engine.run data && python -m curriculum_engine.run week2     (NMIMS_ZIP=<path to "B TECH.zip">)
Tests      : python -m pytest -q
Folders    :
{{git:tree}}
```

## Feedback addressed
| Instructor comment | What we changed | Evidence |
|---|---|---|
| "Please use the course catalog from your own college … not for MIT" | The whole dataset is now NMIMS B.Tech semester syllabus booklets; MIT data is archived and unused. | `curriculum_engine/parse.py`, `results/dataset/` |
| (Week 4) "missing pre-requisites" | Parser bug fixed: an empty prerequisite field swallowed the course objectives in {{dataset/parse_stats:prerequisite_fix_changed}} records. Programme structure built so prerequisites can be checked against earlier semesters. | `tests/test_data.py`, `curriculum_engine/data.py` |
| (Week 4) "Where is the code?" | CODE block above, tagged commit, code zip, one command per step. | `curriculum_engine/run.py` |

## Progress vs previous week
| Item | Week 1 (proposal) | Week 2 (this report) |
|---|---|---|
| Parsed NMIMS course versions | planned | {{dataset/stats:course_versions}} in {{dataset/stats:families}} course families, {{dataset/stats:programmes}} programmes |
| Train / validation / test families | planned | {{dataset/split:families.train}} / {{dataset/split:families.val}} / {{dataset/split:families.test}} |
| Reference-labelled course pairs | protocol only | {{week2/labels:items}} pairs, two LLM labellers + a third on disagreements |
| Capabilities with a measured baseline | none | overlap, redundancy, prerequisites, curriculum diff, Bloom |

## 1. Dataset: NMIMS semester syllabus booklets
- **Collection.** Our college's B.Tech archive (all years and branches). {{dataset/parse_stats:pdfs_considered}} non-exam PDFs are considered (past exam papers are not syllabi and are skipped). Coverage limitation: {{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} PDFs have a text layer; {{dataset/parse_stats:scanned_pdf_skipped}} are scanned images and need OCR, which is future work.
- **Cleaning.** Repeated page headers and footers are detected per booklet and removed; {{dataset/parse_stats:exact_duplicates}} exact duplicate course blocks (the same syllabus printed in several programme booklets) are merged, keeping every source booklet; {{dataset/parse_stats:dropped_legacy_or_unknown_year}} blocks before 2020-21 or without a year are dropped. Grading, attendance and assessment tables are cut by the parser and are not modelled; no student data and no faculty names are present.
- **Schema (one record per course version).** Name, code, programme, semester, academic year, L/P/T hours, credits, prerequisite text, objectives, outcomes with the faculty's K/L Bloom tag, numbered units (title, text, hours), books, source booklets. Result: {{dataset/parse_stats:courses}} course versions; {{dataset/parse_stats:outcomes_total}} outcomes ({{dataset/parse_stats:outcomes_with_k_level}} with a faculty tag); {{dataset/parse_stats:units_total}} units.
- **Versions and families.** All versions of one course (same code or same normalised name) form a family: {{dataset/stats:families}} families, {{dataset/stats:families_with_2plus_versions}} with two or more versions. Comparing versions is a separate version-change task, never mixed with cross-course overlap.
- **Programme structure.** Each version is placed in its programmes (from the booklet folders) and semester; {{dataset/stats:common_courses}} versions are common first-year or all-programme courses and count for every programme. Prerequisite checks skip programmes with too little parsed data.

{{table:week2/eda:per_programme}}

![Left: parsed course versions per programme and academic year (gaps are mostly scanned booklets). Right: words per course.](results/week2/fig_eda.png)

Median course length {{week2/eda:words_per_course.median}} words; {{week2/eda:with_prerequisite_phrase_share:.0%}} of versions name at least one prerequisite.

## 2. Split and LLM-annotated reference labels
- **Split by family** (seed {{dataset/split:seed}}, hash `{{dataset/split:hash}}`): {{dataset/split:families.train}} / {{dataset/split:families.val}} / {{dataset/split:families.test}} families = {{dataset/split:versions.train}} / {{dataset/split:versions.val}} / {{dataset/split:versions.test}} versions for train / validation / test. TF-IDF and the Bloom fallback are fitted on train; every threshold is chosen on validation; the test split is used once.
- **LLM-annotated reference labels.** {{week2/labels:items}} cross-course pairs (both courses in the same split; versions of one course excluded), stratified by programme relation and by score band of both TF-IDF and embeddings, plus near-name, disagreement and random pairs. Labels: 0 = no meaningful overlap, 1 = partial (at least one unit-level topic shared), 2 = substantial / redundant. Pass 1: a Claude model in a fresh context; pass 2: Phi-3 running locally for {{week2/labels:pass2_by.phi3}} pairs and a second Claude prompt (wording B) for the other {{week2/labels:pass2_by.claude_b}}; no labeller saw any model score. Pass 3 settled disagreements; items without a 2-of-3 majority are excluded.
- LLM-vs-LLM agreement, reported separately: Claude vs Phi-3 on {{week2/labels:pass2_by.phi3}} pairs {{week2/labels:agreement_claude_vs_phi3.percent_agreement:.0%}} (weighted kappa {{week2/labels:agreement_claude_vs_phi3.cohen_kappa}}); Claude vs Claude (wording B) on {{week2/labels:pass2_by.claude_b}} pairs {{week2/labels:agreement_claude_vs_claude.percent_agreement:.0%}} (weighted kappa {{week2/labels:agreement_claude_vs_claude.cohen_kappa}}). {{week2/labels:resolved_by_pass3}} items were settled by pass 3, {{week2/labels:uncertain_excluded}} excluded as uncertain. Usable labels — validation: {{week2/labels:by_split.val.0}} / {{week2/labels:by_split.val.1}} / {{week2/labels:by_split.val.2}}, test: {{week2/labels:by_split.test.0}} / {{week2/labels:by_split.test.1}} / {{week2/labels:by_split.test.2}} (labels 0 / 1 / 2). No human annotators were used; results are indicative. For {{week2/labels:pass2_by.claude_b}} of {{week2/labels:items}} pairs the second labeller was a Claude prompt rather than Phi-3, so agreement on those items is not independent. Detector flags (Week 3 onwards) were labelled by two Claude passes with different wording plus a pass 3, with no Phi-3, so agreement on flags is not independent. `labels/human_spot_check.csv` lets each member check ten pairs later.
- **Label-free tests** (no annotation needed): faculty Bloom tags; near-duplicate units injected into an unrelated course of the same programme; removal or renaming of a prerequisite course; known year-over-year course changes, also with renamed courses.

## 3. Baselines on the test split
{{table:week2/summary:table}}

- **Overlap.** On the reference labels, TF-IDF has Spearman {{week2/overlap:methods.tfidf_cosine.test.spearman.value}} and ROC-AUC {{week2/overlap:methods.tfidf_cosine.test.roc_auc.value}} for "any overlap"; at the threshold chosen on validation ({{week2/overlap:methods.tfidf_cosine.at_val_threshold.threshold}}) precision is {{week2/overlap:methods.tfidf_cosine.at_val_threshold.precision}} and recall {{week2/overlap:methods.tfidf_cosine.at_val_threshold.recall}}. Versions of one course (reported separately, {{week2/overlap:version_pairs.n}} pairs) have median TF-IDF {{week2/overlap:version_pairs.tfidf_median}}, against {{week2/overlap:version_pairs.cross_course_median_by_label.2}} for substantially overlapping different courses.
- **Redundancy.** Course pairs in a programme above the validation threshold ({{week2/redundancy:threshold_from_val_substantial_overlap}}): {{week2/redundancy:flags_test}} flags among test courses. When one lightly edited unit of an unrelated course is copied into another course of the same programme, the course-level score ranks that pair in the top 20 in {{week2/redundancy:synthetic_injection.recall_at_20:.0%}} of {{week2/redundancy:synthetic_injection.cases}} cases: one shared unit barely moves a whole-course score.
- **Missing prerequisites.** {{week2/prerequisites:phrases_checked}} prerequisite phrases of {{week2/prerequisites:course_programme_pairs_checked}} test course-programme pairs were checked; {{week2/prerequisites:flags}} were flagged ({{week2/prerequisites:flags_by_type.unresolved}} because no course has exactly that name). On {{week2/prerequisites:tests.cases}} test cases the baseline finds every removed prerequisite course (recall {{week2/prerequisites:tests.removal_recall.value}}) but also flags {{week2/prerequisites:tests.intact_false_alarm.value:.0%}} of intact programmes and {{week2/prerequisites:tests.rename_false_alarm.value:.0%}} after the prerequisite course is renamed: exact names are too brittle.
- **Curriculum diff.** Across {{week2/diff:year_pairs}} programme year-pairs, name matching gives precision {{week2/diff:precision}} and recall {{week2/diff:recall}} for added and removed courses, but precision falls to {{week2/diff:rename_test.precision}} when courses are renamed. Many "changes" are booklets missing in one year (scanned), not real revisions.
- **Bloom.** The verb lexicon matches the faculty tag for {{week2/bloom:lexicon.accuracy.value:.0%}} of {{week2/bloom:test_outcomes}} test outcomes (majority class: {{week2/bloom:majority_class.accuracy.value:.0%}}).

## 4. Observations and next steps
- Lexical baselines are strong where syllabi re-use text, and weak exactly where the committee needs help: a single shared unit inside different courses, renamed courses, and prerequisites written in free text.
- Week 3 adds unit-level embeddings (MaxSim) for overlap and redundancy, embedding-based prerequisite resolution with a topic check over earlier semesters, content matching for the diff, and topic labels, evaluated on this same split and these same metrics.
- Limitations: labels come from LLMs; scanned booklets leave gaps in the programme structure; the test split has {{week2/bloom:test_outcomes}} faculty-tagged outcomes, so Bloom intervals are wide.

## AI assistance
Code and report text were written with AI assistance (Claude). Reference labels are LLM-annotated (Claude, Phi-3). Every number is generated from `results/*.json` by `tools/build_report.py` and verified by `tools/check_reports.py`.
