# Group 9 — Curriculum Intelligence Engine: INTERIM bundle (Weeks 1–2 redone)

Interim upload so that a submission exists before the deadline; Weeks 3–5 follow in the final bundle.

- Repository: {{git:repo}}
- Bundle built from tag `{{git:tag}}`, commit `{{git:commit}}`
- Data: NMIMS semester syllabus booklets ({{dataset/parse_stats:text_pdf}} of {{dataset/parse_stats:pdfs_considered}} non-exam PDFs parsed; no OCR yet)
- Labels: LLM-annotated reference labels for {{week2/labels:items}} course pairs. No human annotators were used; results are indicative. Pass 2 was Phi-3 for {{week2/labels:pass2_by.phi3}} pairs and a second Claude prompt for {{week2/labels:pass2_by.claude_b}} pairs (agreement on those is not independent).

| Week | Report | Week tag | Key metric |
|---|---|---|---|
| 1 | `Group9_Curriculum_Intelligence_Engine_Week1_Proposal_v2.docx` | `week-1-redo` | targets fixed per capability |
| 2 | `Group9_Curriculum_Intelligence_Engine_Week2_Dataset_Baseline.docx` | `week-2-redo` | {{dataset/stats:course_versions}} course versions; TF-IDF overlap Spearman {{week2/overlap:methods.tfidf_cosine.test.spearman.value}} |
