# Curriculum Intelligence Engine — Week 4: Improved Model (redo)
Group 9  ·  Roll Nos: J048, J049, J050, J052, J053  ·  AI University NLP Course Project 2026  ·  redone 9 Oct 2026

## CODE
```
Repository : {{git:repo}}
Tag        : {{git:tag}}
Commit     : {{git:commit}}
Run        : python -m curriculum_engine.run week4      (after: run data, week2, week3; NMIMS_ZIP=<path to "B TECH.zip">)
Code       : curriculum_engine/experiments/week4.py (ablations), curriculum_engine/evaluate.py (paired intervals)
Folders    :
{{git:tree}}
```

## Feedback addressed
| Instructor comment | What we changed | Evidence |
|---|---|---|
| Week 4: "Where is the code?" | CODE block, tagged commit and code zip; every table below is produced by one command. | `curriculum_engine/experiments/week4.py` |
| Week 4: "overlapping content, missing pre-requisites, detect redundancies … minimal progress" | Every ablation targets overlap, redundancy or missing prerequisites; side tasks (course search, summaries) from the first Week 4 are dropped. | `results/week4/summary.json` |
| Week 3: "Progress … is minimal" | Baseline → core → improved on the same test items, with a 95% interval for each difference and an honest verdict. | `results/week4/overlap.json` |

## Progress vs previous week
{{table:week4/summary:table}}

## 1. Overlap: encoder, aggregation and fusion
All methods are scored on the same {{week4/overlap:n_test}} reference-labelled test pairs; differences to the Week 3 core use paired intervals that resample course families. Choices are made on the {{week4/overlap:n_val}} validation pairs.

{{table:week4/overlap:table}}

- Selected on validation: **{{week4/overlap:selected_on_validation}}** (test Spearman {{week4/overlap:methods.fusion_rank_tfidf_maxsim.test.spearman.value}}, ROC-AUC {{week4/overlap:methods.fusion_rank_tfidf_maxsim.test.roc_auc.value}}). It beats the Week 3 core, but it is only marginally above TF-IDF alone ({{week4/overlap:methods.tfidf_cosine.test.spearman.value}}): for ranking whole courses, lexical overlap carries most of the signal. Verdicts: "improved" only if the interval of the difference is above zero, "worse" if below, otherwise "no clear difference".
- The logistic fusion turns the scores into a probability-like confidence for the API. On test: Brier score {{week4/overlap:confidence_calibration_test.brier}}, expected calibration error {{week4/overlap:confidence_calibration_test.ece}}; the calibration error shows it is not yet well calibrated (only {{week4/overlap:n_val}} validation pairs to fit it), so the API will report it as a relative confidence.

## 2. Redundancy: encoder ablation
- Redundancy, injected near-duplicate units found in a programme's top 20: MiniLM {{week4/redundancy:minilm.synthetic_injection.recall_at_20}}, bge-base {{week4/redundancy:bge_base.synthetic_injection.recall_at_20}} (course-level TF-IDF in Week 2: {{week2/redundancy:synthetic_injection.recall_at_20}}).
- Prerequisites keep the Week 3 model (MiniLM, thresholds from validation cases).

## 3. Bloom level: lexicon vs lexicon + learned verbs + classifier
- Faculty K/L tags are the ground truth. The improved model looks the outcome's verb up in the lexicon and in a verb map learned from training outcomes, and falls back to a logistic regression on the sentence embedding (order chosen on validation: {{week4/bloom:improved}}).
- Test accuracy {{week4/bloom:improved_test.accuracy.value}} vs {{week4/bloom:lexicon.accuracy.value}} for the lexicon; difference {{week4/bloom:vs_lexicon_accuracy.difference}} [{{week4/bloom:vs_lexicon_accuracy.ci95.0}}, {{week4/bloom:vs_lexicon_accuracy.ci95.1}}]: {{week4/bloom:vs_lexicon_accuracy.verdict}}.

## 4. Findings and limitations
- Improvements are kept only where the interval supports them; otherwise the simpler model stays.
- Labels for overlap and flags are LLM-annotated reference labels (two passes, majority with a third). No human annotators were used; results are indicative. For {{week2/labels:pass2_by.claude_b}} of {{week2/labels:items}} pairs the second labeller was a Claude prompt rather than Phi-3, so agreement on those items is not independent. Detector flags (Week 3 onwards) were labelled by two Claude passes with different wording plus a pass 3, with no Phi-3, so agreement on flags is not independent.
- Future work: bge-base prerequisite re-tuning not run (time).
- Data: NMIMS semester syllabus booklets; only {{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} PDFs have a text layer (no OCR yet), so programme structures have gaps.
- Not repeated from the first Week 4: course search, LLM summaries and LLM prerequisite extraction, which do not serve the core objective directly.

## AI assistance
Code and report text were written with AI assistance (Claude). Reference labels are LLM-annotated (Claude, Phi-3). Every number is generated from `results/*.json` by `tools/build_report.py` and verified by `tools/check_reports.py`.
