# Integration with other AI University components (Week 7)

The instructor's demo plan (31 October) chains three groups:
**Group 9** (us) turns job descriptions and the current NLP syllabi into a draft unified syllabus →
**Group 11** writes structured emails to faculty and students about it →
**Group 4** answers students' questions about the new course (prerequisites, labs and exams, credits).

Code: `curriculum_engine/modernize.py` (hand-off payloads), `curriculum_engine/integrations.py` (clients and contracts),
`POST /modernize-curriculum` (returns both payloads), `app/committee_ui.py` (buttons for both hand-offs).

## Group 11 — University Communications Intelligence
Source: github.com/Ar-Srivas/uni-comms-intelligence, `submission/week_6/src/api.py` (read on 20 Oct 2026).

| | Contract (their FastAPI service) |
|---|---|
| Endpoint | `POST {GROUP11_URL}/process-notice` |
| Request | `{"doc_id": str, "text": str}` (`NoticeRequest`) |
| Response | `{"doc_id", "title", "category", "audience": [str], "entities": {dates_and_deadlines, organizations, contact_emails, urls, action_roles}, "summaries": {"student": str, "faculty": str, "admin": str}}` (`NoticeResponse`) |

- What we send: `modernize.group11_request(doc)` — a notice announcing the draft unified syllabus (courses merged, programmes,
  new units, prerequisites, who must act). Schema-checked with a copy of their Pydantic models (`integrations.NoticeRequest`).
- Live call: set `GROUP11_URL` (or type the URL in the UI) while their service runs; `integrations.send_to_group11` posts the
  request and validates the reply against `NoticeResponse`.
- Fallback: their service cannot start from the repository as published (it needs `models/summarizer_final/`, their fine-tuned
  BART summariser, which is not committed). Until it runs, `group11_mock` returns a reply in their exact schema, built from our
  syllabus fields, and every mock reply carries `"_mock"` so it can never be mistaken for their model's output.
- `integrations.email_from_group11(response)` turns their reply into one email per audience (subject, body, deadlines, contact).

## Group 4 — University Knowledge Assistant
Source: github.com/vedantkokate2104-ux/SNLP_CHATBOT, `Week_6/Group4_Week6.ipynb` and Week 6 report (read on 20 Oct 2026).

| | Contract |
|---|---|
| Knowledge base | a passage table: `chunk_id, page, section, part, text`, 100-word windows with 25 words of overlap (`make_chunks`) |
| Question answering | `RAGChatbot.ask(question)` → `answer, source (part / section / page), evidence_quote, confidence` |
| HTTP API | none published (the Week 6 package is not in the repository; their knowledge base also needs the SRB PDF) |

- What we hand over: `modernize.group4_chunks(doc)` → CSV in their schema (`data/week7/group4_chunks.csv`, git-ignored because it quotes syllabus text; in the Teams zip, or the UI download).
  Sections: overview, credits and teaching scheme, prerequisites, one section per unit, lab work, assessment, outcomes.
- Demo stand-in: `modernize.group4_standin(rows, question, emb)` answers from those passages the way their offline mode does
  (hybrid BM25 + MiniLM, best sentence quoted verbatim with its source). It is labelled as our stand-in in every reply;
  on demo day Group 4 loads the CSV into their own assistant.

## Contract tests
`tests/test_modernize.py` checks: the Group 11 request and the mock reply validate against their models, an unreachable
Group 11 URL falls back to the mock and says so, Group 4 rows have exactly their columns and at most 100 words, and the
stand-in answers the credits question from the credits section. `results/week7/contracts.json` records the same checks on the
real demo output.
