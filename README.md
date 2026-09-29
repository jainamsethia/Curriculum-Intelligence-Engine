# Curriculum Intelligence Engine
Group 9 · J048, J049, J050, J052, J053 · AI University NLP Course Project 2026

Compares course syllabi and returns a structured JSON report: overlap score with confidence, common and unique topics,
redundancy flag, prerequisite check, and Bloom's-taxonomy level of each learning outcome.

| Week | Deliverable | Where |
|---|---|---|
| 1 | Project proposal | – |
| 2 | Dataset (MIT OpenCourseWare) + TF-IDF baseline | `src/collect.py`, `preprocess.py`, `baseline.py` |
| 3 | **NMIMS syllabus dataset + pipeline v1 (embeddings, information extraction, Bloom, topic map)** | `src/parse_nmims.py`, `pipeline.py`, `evaluate_v1.py`, … |

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
`make_gold_sample.py` regenerates the annotation samples; it is not needed to reproduce the results.

### Data
The source archive (`B TECH.zip`, 9,359 PDFs, 16 GB) is **not** in this repository: it is too large, and about 6,900 of the files
are past exam papers. Only text-layer PDFs can be read (531 syllabus booklets, academic years 2020-21 to 2026-27); scanned PDFs
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
src/           parse_nmims.py  pipeline.py  evaluate_v1.py  examples_v1.py  topic_trends.py  make_gold_sample.py   (Week 3)
               collect.py  preprocess.py  baseline.py                                                          (Week 2)
data/raw/ocw/  Week 2 raw MIT OpenCourseWare pages (CC BY-NC-SA 4.0)
data/processed nmims_courses.jsonl  calibration.json  courses.jsonl/.csv (Week 2)
data/annotation  gold / audit labels (see provenance above)
results/       metrics_v1.json  extraction_audit.json  example_reports_v1.json  topic_trends.*  figures  (Week 2: metrics.json, ...)
report/        Week 2 and Week 3 reports (.docx)
```
