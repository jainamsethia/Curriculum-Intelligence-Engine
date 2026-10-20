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

Backup if the network or a peer service fails: everything above runs offline; the outputs of the last full run are in
`data/week7/` (draft syllabus .docx / .md, Group 4 passages; Teams zip only) and `results/week7/` (hand-off payloads, answers).

## Recorded-demo checklist (backup video, about 5 minutes)
Record before 31 October so the demo survives a network or laptop failure.

Before recording
- [ ] Fresh terminal: `python -m pytest -q` passes; `python tools/final_test.py` reproduced every result (see `results/week8/final_testing.json`).
- [ ] `data/processed/courses.jsonl` in place; API and UI started as above; the sidebar shows "API ok" and the engine version.
- [ ] Browser zoom 100 %, window at least 1500 px wide, notifications and other apps closed; no personal tabs or bookmarks visible.
- [ ] If Group 11's service is running, its URL pasted in the UI; otherwise say on camera that the reply is the marked mock.
- [ ] Group 4 has the passage CSV (`data/week7/group4_chunks.csv` or the UI download) loaded in their assistant.

Recording (screen + voice; one take per step is fine, cut later)
1. 0:00–0:30 Title slide: the problem and who uses the tool (slides 1–2).
2. 0:30–1:15 Input: the five JDs (say they are synthetic samples) and the selected NLP syllabi; click *Analyse*.
3. 1:15–2:00 Overlap table: the identical NLP syllabi across programmes; Speech and NLP overlap; Generative AI distinct.
4. 2:00–2:45 Skills table: demanded skills, taught or gap, with the closest unit as evidence.
5. 2:45–3:30 Draft syllabus: units with sources, the hours warning, the DRAFT status; download the Word file and open it.
6. 3:30–4:15 Group 11 notice → emails per audience; Group 4 CSV download and the three student questions with sources.
7. 4:15–5:00 Limitations (slide 12) and where the code is (slide 13).

After recording
- [ ] Watch it once: every number on screen matches `results/week7/summary.json` and the slides; nothing private is visible.
- [ ] Export as MP4 (1080p); keep it next to the slides for the demo day; do not upload syllabus screens to a public site.
