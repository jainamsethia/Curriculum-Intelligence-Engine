# Demo script — 31 October (Group 9, Curriculum Intelligence Engine)

Setup (before the session): `pip install -e .[dev]`; `data/processed/courses.jsonl` from the Teams zip in place;
terminal 1 `uvicorn curriculum_engine.api:app --port 8000`; terminal 2 `streamlit run app/committee_ui.py`.
If Group 11's service is running, set its URL in the UI (or `GROUP11_URL`); otherwise the documented mock is used.

1. **Input** (tab "Modernise curriculum"): show the five job descriptions on LLM fine-tuning (synthetic samples, or paste real
   ones) and the current NLP syllabi selected from several programmes (AI, COMP/CSE-DS/IT, EXTC, DS, plus Generative AI).
2. **Overlap**: click *Analyse*. Point out that the Natural Language Processing syllabus is identical across programmes and that
   Speech and NLP overlaps it heavily; Generative AI hardly overlaps.
3. **Demanded skills**: the table of skills asked for by two or more JDs, which are already taught and which are gaps, each with
   the closest syllabus topic as evidence.
4. **New structured syllabus**: the draft (shared topics kept once, industry-demanded units kept, new units from the JDs), every
   unit with its source; note the hours check and the DRAFT status. Download the Word file.
5. **Group 11 hand-off**: send the notice; show the email per audience (live reply from Group 11, or the marked mock).
6. **Group 4 hand-off**: download the passage CSV for Group 4; ask "What are the prerequisites for the new course?",
   "Does the course have lab work?", "How many credits does the course carry?" (stand-in answers with sources), then Group 4
   answers the same questions in their assistant after loading the CSV.
7. Close with the limitations: drafts for the Board of Studies, LLM-annotated reference labels, scanned booklets not parsed.

Backup if the network or a peer service fails: everything above runs offline; `results/week7/` holds the outputs of the
last full run (draft syllabus, hand-off payloads, answers).
