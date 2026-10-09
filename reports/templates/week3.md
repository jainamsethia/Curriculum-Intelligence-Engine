# Curriculum Intelligence Engine — Week 3: Core NLP Model (redo)
Group 9  ·  Roll Nos: J048, J049, J050, J052, J053  ·  AI University NLP Course Project 2026  ·  redone 9 Oct 2026

## CODE
```
Repository : {{git:repo}}
Tag        : {{git:tag}}
Commit     : {{git:commit}}
Run        : python -m curriculum_engine.run week3      (after: run data, run week2; NMIMS_ZIP=<path to "B TECH.zip">)
Core code  : curriculum_engine/overlap.py, redundancy.py, prereq.py, diff.py, topics.py, emerging.py, engine.py
Folders    :
{{git:tree}}
```

## Feedback addressed
| Instructor comment | What we changed | Evidence |
|---|---|---|
| Week 3: "Progress relative to previous week is minimal" | Every capability of Week 2 gets a semantic core model, evaluated on the same split, labels and synthetic cases; the progress table below compares them metric by metric. | `results/week3/summary.json` |
| Week 4: "overlapping content, missing pre-requisites, detect redundancies" | Overlap, redundancy and missing prerequisites are the three core models of this week, with flag precision on reference labels. | `curriculum_engine/overlap.py`, `redundancy.py`, `prereq.py` |
| Week 2: "value to NMIMS" | Example output below is on real NMIMS courses; prerequisite checks follow each programme's semester structure. | `results/week3/examples.json` |

## Progress vs previous week (same test split, same metrics)
{{table:week3/summary:progress}}

## 1. Core models
- **Overlap (capability 1).** Each course is split into chunks (its units; outcomes when it has no units) and embedded with all-MiniLM-L6-v2. Every chunk of A is matched to its most similar chunk of B and vice versa (MaxSim). The soft score is the mean best-match similarity; matched units (cosine ≥ {{week3/thresholds:unit_match}}, chosen on validation) are the common topics, unmatched units are unique to each course, and the share of matched units gives the overlap percentage.
- **Redundancy (capability 2).** Inside one programme, every pair of different courses is reduced to its most similar unit pair; pairs above {{week3/thresholds:unit_duplicate:.3f}} (tuned on validation) are flagged with the two unit titles as evidence.
- **Missing prerequisites (capability 3).** Each prerequisite phrase is resolved by embedding similarity to course names, preferring courses taught earlier in the same programme (threshold {{week3/thresholds:prereq_resolve}}). If no course matches, the phrase is checked as a topic against every unit of earlier-semester courses (threshold {{week3/thresholds:prereq_topic}}). Flags: "not offered earlier" and "assumed topic not taught". Both thresholds were tuned on validation removal and rename cases, without labels.
- **Curriculum diff (capability 4).** Courses are matched by name first, then by content (MaxSim ≥ the validation threshold for substantial overlap), so a renamed course counts as retained.
- **Topic model (capability 7) and emerging topics (capability 6).** Unit embeddings of training courses are clustered into {{week3/topics:k}} topics named by class-based TF-IDF; common topics in a report carry these labels. Emerging-topic gaps compare each programme's units with {{week3/emerging:verified_topics}} verified topics from cited sources (World Economic Forum Top 10 Emerging Technologies 2025 and 2026, Stanford AI Index 2025 and 2026, AICTE emerging areas); {{week3/emerging:gaps}} of {{week3/emerging:pairs_checked}} topic–programme checks are flagged as gaps. The list is a draft pending the group's approval; unverified entries are not used.

## 2. Results on the test split
{{table:week3/overlap:table}}

- **Overlap.** MaxSim reaches Spearman {{week3/overlap:methods.maxsim_soft.test.spearman.value}} against {{week3/overlap:methods.tfidf_cosine.test.spearman.value}} for TF-IDF, and ROC-AUC {{week3/overlap:methods.maxsim_soft.test.roc_auc.value}} against {{week3/overlap:methods.tfidf_cosine.test.roc_auc.value}}: no gain for ranking whole courses. NMIMS syllabi re-use wording across programmes, so lexical similarity is already strong; what MaxSim adds is the unit-level explanation (which topics are common and which are unique). Versions of one course (separate task, {{week3/overlap:version_pairs.n}} pairs) score a median {{week3/overlap:version_pairs.maxsim_median}}, above substantially overlapping different courses ({{week3/overlap:version_pairs.cross_course_median_by_label.2}}).
- **Redundancy.** Injected near-duplicate units are ranked in a programme's top 20 in {{week3/redundancy:synthetic_injection.recall_at_20:.0%}} of cases (course-level TF-IDF: {{week2/redundancy:synthetic_injection.recall_at_20:.0%}}). On the test courses {{week3/redundancy:flags_test}} unit pairs are flagged.
- **Missing prerequisites.** Removal recall {{week3/prerequisites:tests.removal_recall.value}} (baseline {{week2/prerequisites:tests.removal_recall.value}}), false alarms on intact programmes {{week3/prerequisites:tests.intact_false_alarm.value}} (baseline {{week2/prerequisites:tests.intact_false_alarm.value}}) and after renaming the prerequisite course {{week3/prerequisites:tests.rename_false_alarm.value}} (baseline {{week2/prerequisites:tests.rename_false_alarm.value}}). Flags on the test courses: {{week3/prerequisites:flags}} (baseline {{week2/prerequisites:flags}}).
- **Curriculum diff.** With renamed courses, precision rises from {{week2/diff:rename_test.precision}} (names only) to {{week3/diff:rename_test.precision}}; on the real year pairs it is {{week3/diff:precision}}.
- **Flag precision (LLM-annotated reference labels).** Top {{week3/flag_precision:redundancy.core.flags_labelled}} redundancy flags: precision {{week3/flag_precision:redundancy.baseline.precision}} for the course-level baseline and {{week3/flag_precision:redundancy.core.precision}} for unit-level MaxSim. Missing-prerequisite flags (a sample of {{week3/flag_precision:prerequisite.core.flags_labelled}} per method): precision {{week3/flag_precision:prerequisite.baseline.precision}} (baseline) and {{week3/flag_precision:prerequisite.core.precision}} (core); most flagged prerequisites are in fact taught earlier under another name or are not subjects at all (analysed in Week 5). Emerging topics: the {{week3/flag_precision:emerging_gap.core.flags_labelled}} flagged gaps are all judged correct (precision {{week3/flag_precision:emerging_gap.core.precision}}), but only {{week3/flag_precision:emerging_gap_recall:.0%}} of the {{week3/flag_precision:emerging_real_gaps}} gaps seen by the labels are flagged, so the threshold is too lenient. Topic labels accepted: {{week3/flag_precision:topic_label.core.precision:.0%}}. Detector flags (Week 3 onwards) were labelled by two Claude passes with different wording plus a pass 3, with no Phi-3, so agreement on flags is not independent. Claude-vs-Claude agreement on flags: {{week3/flag_precision:agreement_pass1_vs_pass2.percent_agreement:.0%}} (same model family, not independent). Pair labels: LLM-vs-LLM agreement, reported separately: Claude vs Phi-3 on {{week2/labels:pass2_by.phi3}} pairs {{week2/labels:agreement_claude_vs_phi3.percent_agreement:.0%}} (weighted kappa {{week2/labels:agreement_claude_vs_phi3.cohen_kappa}}); Claude vs Claude (wording B) on {{week2/labels:pass2_by.claude_b}} pairs {{week2/labels:agreement_claude_vs_claude.percent_agreement:.0%}} (weighted kappa {{week2/labels:agreement_claude_vs_claude.cohen_kappa}}). No human annotators were used; results are indicative. For {{week2/labels:pass2_by.claude_b}} of {{week2/labels:items}} pairs the second labeller was a Claude prompt rather than Phi-3, so agreement on those items is not independent.

## 3. Example output (real NMIMS courses, test split)
```
{{json:week3/examples:excerpt}}
```

## 4. Findings and limitations
- Unit-level semantics help where the committee needs help: one repeated unit inside otherwise different courses (redundancy), renamed courses (diff, prerequisite false alarms). They do not beat TF-IDF at ranking whole courses, and missing-prerequisite flags are still mostly wrong, which Week 5 analyses.
- Prerequisite checks depend on the programme structure, which has gaps where booklets are scanned ({{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} non-exam PDFs parsed); programmes with little data are skipped. OCR is future work.
- Labels are LLM-annotated reference labels; the label-free tests (injection, removal, renaming) do not depend on them.
- Week 4: encoder and aggregation ablations, score fusion and calibrated confidence, Bloom classifier, with paired confidence intervals.

## AI assistance
Code and report text were written with AI assistance (Claude). Reference labels are LLM-annotated (Claude, Phi-3). Every number is generated from `results/*.json` by `tools/build_report.py` and verified by `tools/check_reports.py`.
