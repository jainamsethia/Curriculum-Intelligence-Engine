# Curriculum Intelligence Engine — Week 5: Evaluation & Error Analysis
Group 9  ·  Roll Nos: J048, J049, J050, J052, J053  ·  AI University NLP Course Project 2026  ·  9 Oct 2026

## CODE
```
Repository : {{git:repo}}
Tag        : {{git:tag}}
Commit     : {{git:commit}}
Run        : python -m curriculum_engine.run all        (data → week2 → week3 → week4 → week5; NMIMS_ZIP=<path to "B TECH.zip">)
This week  : curriculum_engine/experiments/week5.py, reports/failure_notes.json
Folders    :
{{git:tree}}
```

## Feedback addressed
| Instructor comment | What we changed | Evidence |
|---|---|---|
| Week 4: "overlapping content, missing pre-requisites, detect redundancies" | The evaluation is organised around these three capabilities first, each against a target fixed in the proposal. | `results/week5/final.json` |
| Week 4: "Where is the code?" | CODE block, tag, code zip; one command reruns every number in this report. | `curriculum_engine/run.py` |
| Week 2: "value to NMIMS" | Breakdown by NMIMS programme, failure cases on real NMIMS courses. | `results/week5/per_programme.json` |

## Progress vs previous week
| Item | Week 4 | Week 5 (this report) |
|---|---|---|
| Overlap method | ablations, selected on validation | final evaluation of **{{week5/final:overlap.method}}** on the held-out test split |
| Evidence of robustness | none | perturbations of the input text |
| Error analysis | none | {{week5/failures:n}} failure cases with root cause and fix |
| Targets checked | none | {{week5/final:targets_met}} of {{week5/final:targets_total}} met, {{week5/final:targets_missed}} missed |

## 1. Test results against the proposal targets
Held-out test split: course families never seen in training or tuning; thresholds come from validation. Overlap and flag precision use LLM-annotated reference labels (pass 1 Claude, pass 2 Phi-3 or a second Claude prompt, pass 3 on disagreements). No human annotators were used; results are indicative. For {{week2/labels:pass2_by.claude_b}} of {{week2/labels:items}} pairs the second labeller was a Claude prompt rather than Phi-3, so agreement on those items is not independent. Detector flags (Week 3 onwards) were labelled by two Claude passes with different wording plus a pass 3, with no Phi-3, so agreement on flags is not independent.

{{table:week5/final:table}}

The one missed target is the precision of missing-prerequisite flags: most flags concern prerequisites that are taught earlier under another name, umbrella names such as "Engineering Mathematics", or phrases that are not subjects at all (cases F05–F07 below). Removal recall and rename false alarms meet their targets, so the checker finds real gaps but raises too many warnings to be used without review.

The overlap method has Spearman {{week5/final:overlap.test.spearman.value}} [{{week5/final:overlap.test.spearman.ci95.0}}, {{week5/final:overlap.test.spearman.ci95.1}}] and ROC-AUC {{week5/final:overlap.test.roc_auc.value}} [{{week5/final:overlap.test.roc_auc.ci95.0}}, {{week5/final:overlap.test.roc_auc.ci95.1}}] on {{week5/final:overlap.test.n}} test pairs; at the validation threshold, precision {{week5/final:overlap.at_val_threshold.precision}} and recall {{week5/final:overlap.at_val_threshold.recall}}.

## 2. Breakdown by programme
{{table:week5/per_programme:table}}

Programmes with few labelled pairs get no AUC (shown as n/a). Bloom accuracy uses the verb lexicon against faculty tags; removal recall is the share of removed prerequisite courses that raise a flag.

## 3. Threshold sensitivity
![Left: overlap precision, recall and F1 on test as the threshold moves (dashed: the threshold chosen on validation). Right: prerequisite removal recall and false alarms as the resolution threshold moves.](results/week5/fig_sensitivity.png)

At the validation threshold the test F1 is {{week5/sensitivity:test_f1_at_chosen}}; the best F1 any threshold could reach on test is {{week5/sensitivity:best_test_f1_any_threshold}}, a clear gap: a threshold tuned on {{week4/overlap:n_val}} validation pairs is noisy, which is why the ranking metrics (Spearman, ROC-AUC) are the more reliable summary.

## 4. Robustness
Course B of every labelled test pair is perturbed and the pair is re-scored (left of the arrow: clean text).

{{table:week5/robustness:table}}

Typos and shuffled units change nothing and dropping words very little; keeping only half of the units costs the most. Without outcomes and objectives TF-IDF loses more than the fusion, because MaxSim still matches the units. A perturbed copy of a course stays very close to the original (median self-similarity after typos {{week5/robustness:self_similarity_median.typos}}, after truncation {{week5/robustness:self_similarity_median.truncated_half}}), so versions of one course are always separated from different courses: versions are a separate version-change task.

## 5. Failure cases
Selected by fixed rules, not hand-picked: {{week5/failures:selection_rules}}

{{table:week5/failures:table}}

## 6. Limitations
- Data: NMIMS semester syllabus booklets; {{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} non-exam PDFs have a text layer and are parsed, the rest are scanned (OCR is future work), so programme structures and year-over-year comparisons have gaps.
- Labels: LLM-annotated reference labels; LLM-vs-LLM agreement, reported separately: Claude vs Phi-3 on {{week2/labels:pass2_by.phi3}} pairs {{week2/labels:agreement_claude_vs_phi3.percent_agreement:.0%}} (weighted kappa {{week2/labels:agreement_claude_vs_phi3.cohen_kappa}}); Claude vs Claude (wording B) on {{week2/labels:pass2_by.claude_b}} pairs {{week2/labels:agreement_claude_vs_claude.percent_agreement:.0%}} (weighted kappa {{week2/labels:agreement_claude_vs_claude.cohen_kappa}}); the same model family wrote the code and pass 1, which is why label-free tests (injection, removal, renaming, faculty Bloom tags) run alongside.
- Synthetic tests use simple edits (word drop, typos, renamed titles); real revisions can be subtler. Robustness: paraphrase variant not run (time).

## AI assistance
Code and report text were written with AI assistance (Claude). Reference labels are LLM-annotated (Claude, Phi-3). Every number is generated from `results/*.json` by `tools/build_report.py` and verified by `tools/check_reports.py`.
