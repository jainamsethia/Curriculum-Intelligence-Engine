# Curriculum Intelligence Engine
Group 9 · J048, J049, J050, J052, J053 · AI University NLP Course Project 2026

Compares course syllabi and returns a structured JSON report: overlap score with confidence, common and unique topics,
redundancy flag, prerequisite check, and Bloom's-taxonomy level of each learning outcome.

| Week | Deliverable | Where |
|---|---|---|
| 1 | Project proposal | – |
| 2 | Dataset (MIT OpenCourseWare) + TF-IDF baseline | `src/collect.py`, `preprocess.py`, `baseline.py` |
| 3 | NMIMS syllabus dataset + pipeline v1 (embeddings, information extraction, Bloom, topic map) | `src/parse_nmims.py`, `pipeline.py`, `evaluate_v1.py`, … |
| 4 | **Improved approaches vs baselines (hybrid search, transformer / LLM classification and extraction, LLM summaries)** | `src/improved.py`, `src/w4_*.py` |

## Week 4 — improved approaches vs baselines

Four components were re-implemented with an improved approach and compared with their baseline on the same test data
(details, tables and figure in the Week 4 report and `results/w4_*.json`):

| Task | Baseline | Improved approach | Metric | Baseline | Improved | Verdict (95% interval of the difference) |
|---|---|---|---|---|---|---|
| Course search (which course teaches this outcome?) | TF-IDF cosine | Hybrid: BM25 + bge-base, RRF | MRR@10 | 0.582 | 0.617 | improved, +0.035 [0.011, 0.057] |
| | | Hybrid: BM25 + domain-adapted MiniLM | MRR@10 | 0.582 | 0.630 | improved, +0.048 [0.023, 0.073] |
| | | Hybrid + cross-encoder rerank | MRR@10 | 0.582 | 0.607 | no clear difference |
| Bloom level (326 faculty-tagged outcomes) | Rule-based verb lexicon | Fine-tuned transformer + rule hint | accuracy | 0.727 | 0.733 | no clear difference |
| | | Phi-3 zero-shot | accuracy | 0.727 | 0.718 | no clear difference |
| | | Phi-3 few-shot | accuracy | 0.727 | 0.607 | worse |
| Prerequisite extraction (60 strings) | Regex splitter | spaCy noun chunks | F1 | 0.784 | 0.606 | worse |
| | | Phi-3 few-shot | F1 | 0.784 | 0.852 | improved, +0.068 [0.007, 0.130] |
| Course summary (40 courses, vs faculty objective) | Extractive, first two outcomes | Phi-3 (unit titles + outcomes) | ROUGE-L | 0.217 | 0.257 | improved, but 32% of its content words are not in the input |
| | | Phi-3 (full unit text + outcomes) | ROUGE-L | 0.217 | 0.267 | improved, but 31% of its content words are not in the input |

Honest summary: hybrid retrieval and LLM prerequisite extraction are real gains; nothing beat the verb lexicon for Bloom's taxonomy (Phi-3
zero-shot matches it with no training data); LLM summaries score higher but add unsupported wording, and all three LLM comparison summaries
tried in `w4_examples.py` failed the faithfulness guard and fell back to a template.

### Run (Week 4)
Needs `pip install -r requirements.txt` and, for the LLM steps, a local [Ollama](https://ollama.com) with `phi3` and `nomic-embed-text`
pulled. Once the encoders are cached, set `HF_HUB_OFFLINE=1` (the sentence-transformers loader otherwise contacts huggingface.co even for cached models).
```bash
python src/w4_finetune_encoder.py          # domain-adapted MiniLM (contrastive, ~25 min on CPU) -> data/interim/
python src/w4_retrieval.py --models minilm,bge-small,nomic,bge-base,ft      # task 1 -> results/w4_retrieval.json
python src/w4_bloom.py                      # task 2 (fine-tuning + Phi-3, ~70 min on CPU; resumes from data/interim/ checkpoints)
python src/w4_bloom_vote.py                 # exploratory vote between Bloom classifiers
python src/w4_prereq_extraction.py          # task 3 (gold strings: data/annotation/prereq_extraction_gold.json)
python src/w4_summarize.py                  # task 4
python src/w4_examples.py                   # search and comparison-summary examples
python src/w4_figures.py                    # results/fig_w4_comparison.png

python src/pipeline.py search "transformers and large language models" -k 5 --mode hybrid   # hybrid | dense | keyword | course-vector (Week 3)
python src/pipeline.py compare --a nm59a8f694 --b nm7677637d --summary                      # adds an LLM summary with a faithfulness guard
```
Labels: the prerequisite test strings (`prereq_extraction_gold.json`) were labelled by a single LLM annotator, not by humans. Retrieval,
Bloom and summary ground truths come from the syllabi themselves (source course, faculty K/L tag, faculty objective).

## Week 3 — pipeline v1

```
NMIMS syllabus PDF ──parse_nmims.py──> structured course (units, outcomes, prerequisites, ...)
                                         │
                       pipeline.py: sentence-transformer embeddings (all-MiniLM-L6-v2)
                         ├─ unit-level alignment  -> overlap score, common / unique topics
                         ├─ + TF-IDF, logistic calibration on gold pairs -> confidence, redundancy flag
                         ├─ prerequisite extraction -> match to catalogue course -> missing / sequencing warnings
                         └─ outcome -> Bloom level (verb look-ups + embedding fallback)
```

### Run
```bash
pip install -r requirements.txt

python src/parse_nmims.py "B TECH.zip"        # 1. syllabus PDFs (read straight from the zip) -> data/processed/nmims_courses.jsonl
python src/evaluate_v1.py                     # 2. metrics, figures, data/processed/calibration.json  (uses the committed gold labels)
python src/examples_v1.py                     # 3. example reports  -> results/example_reports_v1.json
python src/topic_trends.py                    # 4. topic map        -> results/topic_trends.*

python src/pipeline.py compare --a nm59a8f694 --b nm7677637d          # two courses from the dataset (ids are in nmims_courses.jsonl)
python src/pipeline.py compare --a "booklet.pdf::Discrete Mathematics" --b other_syllabus.docx
python src/pipeline.py outcomes --course nm7677637d               # Bloom level of every outcome
python src/pipeline.py search "transformers and large language models" -k 5
```
`parse_nmims.py` caches page text in `data/interim/` (git-ignored), so re-running after a parser change takes about 30 seconds.
The first `pipeline.py` call embeds the course catalogue (a few minutes on CPU, cached in `data/interim/` afterwards).
`make_gold_sample.py` regenerates the annotation samples; it is not needed to reproduce the results.

### Data
The source archive (`B TECH.zip`, 9,359 PDFs, 16 GB) is **not** in this repository: it is too large, and about 6,900 of the files
are past exam papers. Only text-layer PDFs can be read (531 of them, mostly semester syllabus booklets, academic years 2020-21 to 2026-27); scanned PDFs
(all exam papers, most pre-2020 syllabi) need OCR, which is not set up.

| | |
|---|---|
| Course versions (`data/processed/nmims_courses.jsonl`) | 2,366 (655 distinct course names, 472 with two or more versions) |
| Units / learning outcomes | 14,747 / 8,650 (546 outcomes carry a faculty Bloom tag K1–K6 or L1–L6) |
| Schema | `id, course_name, code, course_key, program, semester, academic_year, credits, prerequisites, objectives, outcomes[{text,k_levels}], units[{title,text,hours}], text_books, reference_books, lab_work, source, n_source_files` |

Known parser limits: unit titles are best effort (first line of a unit), `hours` is the first duration cell of a unit, and
multi-row units occasionally merge. See `results/extraction_audit.json`.

### Evaluation (details, tables and figures in `results/metrics_v1.json` and the Week 3 report)
Split by course name: 458 train / 197 test courses. TF-IDF, the unit threshold and the Bloom classifier are fitted on train courses only.

| Task | Result |
|---|---|
| Extraction audit, 45 fresh records | fully correct: course name and code 100%, semester and prerequisites 98%, outcomes 96%, objectives 91% (+9% cosmetic), books 89% (+9% cosmetic), units 58% (+31% cosmetic, 11% wrong) |
| Same-course retrieval, 222 test queries | TF-IDF (Week 2 baseline) MRR 0.927, embeddings 0.929: **no meaningful difference** |
| Unit alignment between versions of a course | On 199 revised units embeddings 98.0% vs TF-IDF 95.0% top-1 (embeddings right / TF-IDF wrong in 6 units, the reverse in 0; exact McNemar p = 0.031) |
| Overlap on 113 gold pairs | Every method AUC ≈ 1.0: the set is "same course vs unrelated" and cannot separate methods |
| Bloom level, 98 held-out outcomes | 69.4% (95% CI 60–79) vs 70.4% for the hand-written verb list alone, 40.8% majority class |
| Prerequisite matching, audited | 87% precise at similarity ≥ 0.75 (n = 54; threshold chosen on this audit, so optimistic) |

Embeddings did **not** beat TF-IDF at ranking or detecting overlapping courses on this corpus. The embedding pipeline adds topic-level
alignment, prerequisite and Bloom analysis, and calibrated confidence rather than a better ranking score.

### Provenance of labels (please read)
* **Gold overlap labels** (`data/annotation/nmims_gold_pairs.csv`, `nmims_gold_annotators.csv`): 113 course pairs labelled 0/1/2 by a
  **panel of three independent LLM annotators** (median label, 98% unanimous), blind to all model scores. They were not labelled by humans,
  and only 2 pairs are "partial", so partial overlap is untested.
* **Prerequisite audit** (`prereq_audit.csv`): 88 matches graded by two LLM graders. **Extraction audit**: LLM auditors compared raw PDF text with parsed records.
* **Bloom ground truth** is the faculty K/L tag printed in the syllabi, not an LLM label.
* Week 2 gold pairs (`gold_pairs.csv`, MIT OCW) were labelled by a single LLM annotator.

## Week 2 — dataset and baseline (MIT OpenCourseWare)
```bash
python src/collect.py      # MIT Learn API + OCW syllabus pages -> data/raw/ocw
python src/preprocess.py   # clean, spaCy lemmatise, silver labels, grouped split -> data/processed/courses.jsonl
python src/baseline.py     # TF-IDF cosine, TF-IDF + logistic regression, lexicon extraction -> results/metrics.json
```
87 courses, split 60/27 by course number; TF-IDF cosine reaches ROC-AUC 0.826 against silver pair labels and Spearman 0.636 against the
41 gold pairs. Own course files can be dropped into `data/raw/local/` (`.txt`, `.docx`, `.pdf`).

## Layout
```
src/           improved.py  w4_retrieval.py  w4_finetune_encoder.py  w4_bloom.py  w4_bloom_vote.py  w4_prereq_extraction.py
               w4_summarize.py  w4_examples.py  w4_figures.py                                                     (Week 4)
src/           parse_nmims.py  pipeline.py  evaluate_v1.py  examples_v1.py  topic_trends.py  make_gold_sample.py   (Week 3)
               collect.py  preprocess.py  baseline.py                                                          (Week 2)
data/raw/ocw/  Week 2 raw MIT OpenCourseWare pages (CC BY-NC-SA 4.0)
data/processed nmims_courses.jsonl  calibration.json  courses.jsonl/.csv (Week 2)
data/annotation  gold / audit labels (see provenance above)
results/       w4_*.json  w4_bloom_predictions.csv  fig_w4_comparison.png                          (Week 4)
               metrics_v1.json  extraction_audit.json  example_reports_v1.json  topic_trends.*  figures  (Week 2: metrics.json, ...)
report/        Week 2 and Week 3 reports (.docx)
```
