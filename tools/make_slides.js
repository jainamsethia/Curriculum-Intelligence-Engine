// Final presentation (Week 8), built from results/*.json so every number matches the reports.
//   cd tools && npm install && node make_slides.js [out.pptx]
// Default output: submissions/week8/Group9_Curriculum_Intelligence_Engine_Week8_Presentation.pptx
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const OUT = process.argv[2] || path.join(ROOT, "submissions/week8/Group9_Curriculum_Intelligence_Engine_Week8_Presentation.pptx");
const cache = {};
const get = (file, p = "") => p.split(".").filter(Boolean).reduce((d, k) => d[k],
  cache[file] ??= JSON.parse(fs.readFileSync(path.join(ROOT, "results", `${file}.json`), "utf8")));
const fmt = (v, d = 3) => (typeof v !== "number" ? String(v) : Number.isInteger(v) ? v.toLocaleString("en-US") : v.toFixed(d));
const R = (file, p, d) => fmt(get(file, p), d);
const D3 = (file, p) => get(file, p).toFixed(3);
const T = (p) => String(get("plan/targets", p));                   // success criteria fixed in Proposal v2
const testing = fs.existsSync(path.join(ROOT, "results/week8/final_testing.json")) ? get("week8/final_testing") : null;

// palette: deep slate for dark slides, teal dominant, amber accent
const INK = "1B2B34", TEAL = "0E7C7B", TINT = "E3F1F0", AMBER = "D98E04", GREY = "5B6770", LINE = "D5DCE0", WHITE = "FFFFFF", MISS = "B4473C";
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";                                   // 10 x 5.625 in
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };
pres.author = "Group 9 (J048, J049, J050, J052, J053)";
pres.title = "Curriculum Intelligence Engine - final presentation";

const footer = { text: { text: "Group 9 · Curriculum Intelligence Engine v1.0", options: { x: 0.5, y: 5.22, w: 6, h: 0.25, fontSize: 9, color: GREY, fontFace: BODY, margin: 0 } } };
pres.defineSlideMaster({
  title: "DARK", background: { color: INK },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.35, w: 8.8, h: 1.3, fontSize: 40, bold: true, color: WHITE, fontFace: HEAD, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.85, w: 8.8, h: 1.9, fontSize: 18, color: "CFE3E2", fontFace: BODY, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: WHITE },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.25, w: 9, h: 0.75, fontSize: 32, bold: true, color: INK, fontFace: HEAD, valign: "middle", margin: 0 }, text: "" } },
    footer,
  ],
  slideNumber: { x: 9.0, y: 5.22, w: 0.5, h: 0.25, fontSize: 9, color: GREY, fontFace: BODY, align: "right" },
});

let section = "";
const slide = (master, title, notes) => {
  const s = pres.addSlide({ masterName: master, sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  s.addNotes(notes);
  return s;
};
const sec = (t) => { section = t; pres.addSection({ title: t }); };
const text = (s, t, o) => s.addText(t, { isTextBox: true, fontFace: BODY, fontSize: 14, color: INK, margin: 0, valign: "top", ...o });
const card = (s, x, y, w, h, name, fill = TINT) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08, objectName: name });
const badge = (s, x, y, label, fill = TEAL) => {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.42, h: 0.42, fill: { color: fill }, line: { color: fill }, objectName: `badge ${label}` });
  text(s, String(label), { x, y, w: 0.42, h: 0.42, fontSize: 14, bold: true, color: WHITE, align: "center", valign: "middle" });
};
const stat = (s, x, y, w, value, label, color = TEAL) => {
  text(s, value, { x, y, w, h: 0.75, fontSize: 40, bold: true, color, fontFace: HEAD, valign: "bottom" });
  text(s, label, { x, y: y + 0.8, w, h: 0.55, fontSize: 12, color: GREY });
};
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  catAxisLabelColor: GREY, valAxisLabelColor: GREY, dataLabelColor: INK, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontSize: 11,
  valGridLine: { color: "E6EAEC", size: 0.5 }, catGridLine: { style: "none" } };

// 1 ---------------------------------------------------------------- title
sec("Opening");
let s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
s.addText("Curriculum Intelligence Engine", { placeholder: "title" });
s.addText([
  { text: "Overlap, redundancy, missing prerequisites and industry gaps in NMIMS B.Tech syllabi", options: { breakLine: true } },
  { text: " ", options: { breakLine: true, fontSize: 10 } },
  { text: "Group 9 · J048, J049, J050, J052, J053 · AI University NLP Course Project 2026", options: { fontSize: 14, color: "9FC3C1" } },
], { placeholder: "body" });
s.addNotes("Introduce the group and the one-line purpose: a tool for curriculum committees that reads NMIMS syllabus booklets and answers which courses overlap, which prerequisites are missing, and what industry asks for that we do not teach.");

// 2 ---------------------------------------------------------------- problem
sec("Problem and data");
s = slide("CONTENT", "The committee's three questions", "Who uses it: the Board of Studies and programme chairs, who revise syllabi every year by reading booklets by hand. The engine answers three questions with evidence, and never decides on its own.");
const qs = [["Which courses teach the same thing?", "Overlap and redundancy between courses, programmes and years"],
  ["Which prerequisites are missing?", "A declared prerequisite that no earlier semester teaches"],
  ["What does industry ask that we do not teach?", "Skills in job descriptions compared with what the syllabi cover"]];
qs.forEach(([q, d], i) => {
  const x = 0.5 + i * 3.05;
  card(s, x, 1.35, 2.85, 2.9, `question ${i + 1}`);
  badge(s, x + 0.25, 1.6, i + 1);
  text(s, q, { x: x + 0.25, y: 2.2, w: 2.35, h: 0.95, fontSize: 17, bold: true, fontFace: HEAD });
  text(s, d, { x: x + 0.25, y: 3.2, w: 2.35, h: 0.9, fontSize: 13, color: GREY });
});
text(s, "Output: structured JSON with evidence and a confidence score, for a human decision", { x: 0.5, y: 4.5, w: 9, h: 0.4, fontSize: 14, italic: true, color: TEAL });

// 3 ---------------------------------------------------------------- data
s = slide("CONTENT", "Data: NMIMS semester syllabus booklets",
  `Only text-layer PDFs are parsed: ${R("dataset/parse_stats", "text_pdf")} of ${R("dataset/parse_stats", "pdfs_considered")} non-exam PDFs. The rest are scanned and need OCR, which is future work. Exam papers are skipped. The parsed data never goes to the public repository.`);
[[R("dataset/parse_stats", "pdfs_considered"), "non-exam PDFs in the archive"], [R("dataset/parse_stats", "text_pdf"), "with a text layer, parsed (no OCR)"],
 [R("dataset/stats", "course_versions"), "course versions after deduplication"], [R("dataset/stats", "families"), `course families, ${R("dataset/stats", "programmes")} programmes`]]
  .forEach(([v, l], i) => stat(s, 0.5 + (i % 2) * 2.2, 1.2 + Math.floor(i / 2) * 1.6, 2.0, v, l, i === 1 ? AMBER : TEAL));
const progs = Object.entries(get("dataset/stats", "versions_per_programme")).sort((a, b) => a[1] - b[1]);
s.addChart(pres.charts.BAR, [{ name: "Course versions", labels: progs.map((p) => p[0]), values: progs.map((p) => p[1]) }], {
  x: 5.0, y: 1.1, w: 4.6, h: 3.95, barDir: "bar", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd", showLegend: false,
  showTitle: true, title: "Course versions per programme", titleFontFace: "+mn-lt", titleFontSize: 12, titleColor: INK, valAxisHidden: true, ...chartText,
  catAxisLabelFontSize: 10, dataLabelFontSize: 10, valGridLine: { style: "none" } });
text(s, `Academic years ${get("week2/eda", "academic_years")[0]} to ${get("week2/eda", "academic_years").at(-1)} · split by course family ${["train", "val", "test"].map((k) => Math.round(get("dataset/split", `fractions.${k}`) * 100)).join("/")} (hash ${get("dataset/split", "hash")})`,
  { x: 0.5, y: 4.55, w: 4.3, h: 0.5, fontSize: 11, color: GREY });

// 4 ---------------------------------------------------------------- architecture
sec("System");
s = slide("CONTENT", "Architecture", "Left to right: booklets are parsed locally into course records and families; embeddings and TF-IDF feed the analysis modules; the FastAPI service returns JSON with evidence and confidence to the committee UI and to the other groups. Group 10 is a planned consumer, not integrated yet.");
s.addImage({ path: path.join(ROOT, "docs/architecture.png"), x: 0.5, y: 1.15, w: 9, h: 9 * 1248 / 3968 });
[["Runs offline", "no cloud services; data never on GitHub"], ["Evidence with every answer", "matched units, sources, confidence"], [`${get("week7/summary", "endpoints")} API endpoints`, "one JSON for the UI and peers"]]
  .forEach(([h, d], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 4.2, 2.85, 0.85, `architecture note ${i + 1}`);
    text(s, h, { x: x + 0.2, y: 4.28, w: 2.5, h: 0.35, fontSize: 14, bold: true });
    text(s, d, { x: x + 0.2, y: 4.62, w: 2.5, h: 0.35, fontSize: 12, color: GREY });
  });

// 5 ---------------------------------------------------------------- labels
s = slide("CONTENT", "How we measured quality",
  "The test split is held out by course family, so versions of one course never sit in two splits. Reference labels come from LLM passes with different wordings; disagreements went to a third pass. Say clearly: no human annotators were used, so results are indicative. A human spot-check sheet is ready but not filled in.");
stat(s, 0.5, 1.15, 2.6, R("week8/labels", "items"), "LLM-annotated reference labels");
stat(s, 0.5, 2.6, 2.6, R("week8/labels", "pass3"), `settled by a third pass, ${R("week8/labels", "uncertain")} left uncertain`, AMBER);
const lt = get("week8/labels", "table");
s.addTable([lt.columns.slice(0, 4).map((c) => ({ text: c, options: { bold: true, color: WHITE, fill: { color: TEAL } } })),
  ...lt.rows.map((r) => r.slice(0, 4).map((c) => ({ text: fmt(c) })))],
  { x: 3.4, y: 1.2, w: 6.1, colW: [2.5, 0.9, 1.4, 1.3], fontFace: BODY, fontSize: 12, color: INK, border: { type: "solid", pt: 0.5, color: LINE }, valign: "middle", rowH: 0.42 });
card(s, 3.4, 3.65, 6.1, 1.2, "label caveat", "FBF1DE");
text(s, [{ text: "No human annotators were used; results are indicative.", options: { bold: true, breakLine: true } },
  { text: "Second labeller: Phi-3 for part of the pairs, otherwise a second Claude prompt, so agreement there is not independent.", options: { fontSize: 12, color: GREY } }],
  { x: 3.6, y: 3.78, w: 5.7, h: 0.95, fontSize: 14 });

// 6 ---------------------------------------------------------------- results
sec("Results");
s = slide("CONTENT", "Results against our success criteria",
  `Targets were fixed in Proposal v2 before any result existed and are measured on the held-out test split. ${R("week8/final", "targets_met")} of ${R("week8/final", "targets_total")} are met. The two misses are both about prerequisite flags: checker v2 traded recall for precision.`);
stat(s, 0.5, 1.2, 2.4, `${get("week8/final", "targets_met")} / ${get("week8/final", "targets_total")}`, "success targets met on the test split");
stat(s, 0.5, 2.75, 2.4, String(get("week8/final", "targets_missed")), "missed: both on prerequisite flags", MISS);
const ft = get("week8/final", "table");
let cap = "";
const headline = ft.rows.map((r) => { cap = r[0] || cap; return [cap.replace(/^\d+ /, ""), ...r.slice(1)]; })
  .filter((r, i, a) => i === 0 || r[0] !== a[i - 1][0] || r[5] === "no");
s.addTable([["Capability", "Metric", "Final", "Target", "Met?"].map((c) => ({ text: c, options: { bold: true, color: WHITE, fill: { color: TEAL } } })),
  ...headline.map((r) => [r[0], r[1], r[3], r[4].replace(">=", "≥").replace("<=", "≤"), r[5]].map((c, j) => ({ text: c, options: j === 4 && c === "no" ? { color: MISS, bold: true } : {} })))],
  { x: 3.2, y: 1.15, w: 6.3, colW: [1.55, 2.6, 0.65, 0.95, 0.55], fontFace: BODY, fontSize: 11, color: INK, border: { type: "solid", pt: 0.5, color: LINE }, valign: "middle", rowH: 0.36 });

// 7 ---------------------------------------------------------------- overlap
s = slide("CONTENT", "Overlap and redundancy work",
  "Overlap fuses TF-IDF with MiniLM chunk MaxSim. Spearman against the reference labels and ROC-AUC for any overlap are on the test split. Redundancy flags near-duplicate units inside a programme. The demo example: the same NLP syllabus is taught in several programmes.");
[[R("week5/final", "table.rows.0.2"), `Spearman with reference labels (target ${T("overlap.spearman")})`], [R("week5/final", "table.rows.1.2"), `ROC-AUC, any overlap (target ${T("overlap.roc_auc")})`],
 [R("week5/final", "table.rows.3.2"), `precision of top redundancy flags (target ${T("redundancy.precision_at_20")})`]].forEach(([v, l], i) => stat(s, 0.5 + i * 3.05, 1.2, 2.8, v, l));
card(s, 0.5, 3.05, 9, 1.3, "overlap example");
badge(s, 0.75, 3.3, "e.g.", AMBER);
text(s, [{ text: `${get("week7/summary", "identical_pairs")} of ${get("week7/summary", "pairs")} NLP-course pairs are identical`, options: { bold: true, fontSize: 17, fontFace: HEAD, breakLine: true } },
  { text: "Natural Language Processing is taught with the same units in AI, COMP / CSE-DS / IT and EXTC; Speech and NLP overlaps it strongly; Generative AI hardly overlaps. A committee can merge these into one course.", options: { fontSize: 13, color: GREY } }],
  { x: 1.35, y: 3.28, w: 7.9, h: 1.0 });

// 8 ---------------------------------------------------------------- prerequisites
s = slide("CONTENT", "Prerequisites: an honest trade-off",
  "Our Week 5 error analysis showed most missing-prerequisite flags were wrong: umbrella names, fragments, non-subjects. Checker v2 removed false alarms and lifted precision, but removal recall fell below the target. Precision samples are small, so the intervals are wide.");
const pv = "week6/prerequisites_v2";
s.addChart(pres.charts.BAR, [
  { name: "checker v1 (Week 3)", labels: ["removal recall", "precision of flags", "false alarms after rename"],
    values: [get(pv, "tests_test.v1.removal_recall.value"), get(pv, "same_labels.v1_precision"), get(pv, "tests_test.v1.rename_false_alarm.value")] },
  { name: "checker v2 (Week 6)", labels: ["removal recall", "precision of flags", "false alarms after rename"],
    values: [get(pv, "tests_test.v2.removal_recall.value"), get(pv, "fresh_sample.test_v2.precision"), get(pv, "tests_test.v2.rename_false_alarm.value")] }],
  { x: 0.5, y: 1.1, w: 5.6, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: ["9AA7AE", TEAL], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "b", legendFontSize: 11, ...chartText });
[["Fewer false alarms", `rename false alarms ${D3(pv, "tests_test.v1.rename_false_alarm.value")} → ${D3(pv, "tests_test.v2.rename_false_alarm.value")}`, TEAL],
 ["Higher precision", `${R(pv, "same_labels.v1_precision")} → ${R(pv, "fresh_sample.test_v2.precision")} (n = ${get(pv, "fresh_sample.test_v2.n")}); target ${T("prerequisites.flag_precision")} still missed`, AMBER],
 ["Lower recall", `${R(pv, "tests_test.v1.removal_recall.value")} → ${R(pv, "tests_test.v2.removal_recall.value")}; below the ${T("prerequisites.removal_recall")} target`, MISS]]
  .forEach(([h, d, c], i) => {
    text(s, h, { x: 6.4, y: 1.25 + i * 1.2, w: 3.1, h: 0.4, fontSize: 16, bold: true, color: c, fontFace: HEAD });
    text(s, d, { x: 6.4, y: 1.65 + i * 1.2, w: 3.1, h: 0.65, fontSize: 12, color: GREY });
  });

// 9 ---------------------------------------------------------------- demo flow
sec("Demo and integration");
s = slide("CONTENT", "Demo: from JDs to a new syllabus",
  "This is our step in the 31 October demo chain. Input: five job descriptions on fine-tuning LLMs (synthetic samples) and the current NLP syllabi of several programmes. Output: overlaps, demanded skills and a draft syllabus where every unit names its source. Then the draft goes to Group 11 as a notice and to Group 4 as passages.");
const steps = [["Input", `${get("week7/summary", "jds")} JDs + ${get("week7/summary", "courses")} current NLP syllabi`],
  ["Overlap", `${get("week7/summary", "identical_pairs")} of ${get("week7/summary", "pairs")} syllabus pairs identical`],
  ["Skills", `${get("week7/summary", "demanded_themes")} demanded: ${get("week7/summary", "demanded_covered")} taught, ${get("week7/summary", "demanded_gaps")} gaps`],
  ["Draft", `${get("week7/summary", "units")} units: ${get("week7/summary", "units_shared")} shared, ${get("week7/summary", "units_existing_demanded")} kept, ${get("week7/summary", "units_new")} new`]];
steps.forEach(([h, d], i) => {
  const x = 0.5 + i * 2.3;
  card(s, x, 1.2, 2.1, 1.65, `step ${i + 1}`);
  badge(s, x + 0.2, 1.38, i + 1);
  text(s, h, { x: x + 0.75, y: 1.4, w: 1.25, h: 0.4, fontSize: 16, bold: true, fontFace: HEAD, valign: "middle" });
  text(s, d, { x: x + 0.2, y: 1.95, w: 1.75, h: 0.8, fontSize: 13 });
  if (i < 3) s.addShape(pres.shapes.CHEVRON, { x: x + 2.12, y: 1.9, w: 0.16, h: 0.26, fill: { color: LINE }, line: { color: LINE }, objectName: `arrow ${i + 1}` });
});
const g4 = "week7/group4_handoff";
[["Group 11 · Communications", `Notice in their /process-notice schema → one email per audience (this run: ${get("week7/group11_handoff", "mode")}; live once their service runs)`],
 ["Group 4 · Knowledge Assistant", `${get("week7/contracts", "group4_rows")} passages in their schema; our stand-in answered ${get(g4, "answered_from_expected_section")} of ${get(g4, "questions")} student questions`],
 ["Group 10 · Admin Copilot", "Can read the same draft syllabus JSON for elective advice (not integrated yet)"]]
  .forEach(([h, d], i) => {
    const x = 0.5 + i * 3.05;
    card(s, x, 3.15, 2.85, 1.7, `handoff ${i + 1}`, "F2F5F6");
    text(s, h, { x: x + 0.2, y: 3.28, w: 2.5, h: 0.4, fontSize: 14, bold: true, color: TEAL });
    text(s, d, { x: x + 0.2, y: 3.7, w: 2.5, h: 1.05, fontSize: 12, color: GREY });
  });

// 10 --------------------------------------------------------------- UI
s = slide("CONTENT", "Committee UI",
  "Streamlit over the API. Demo tab: paste JDs, pick syllabi, see overlaps and skills, download the draft as Word, Markdown or JSON, send the notice to Group 11, export passages for Group 4. Other tabs compare two courses, two years, prerequisites, redundancy and emerging topics. The sidebar always says outputs are drafts.");
s.addImage({ path: path.join(ROOT, "results/week7/fig_ui_report.png"), x: 0.9, y: 1.1, w: 8.2, h: 8.2 * 1050 / 2310 });
text(s, "Left: JDs, selected syllabi, overlaps. Right: Group 11 emails (marked mock), a Group 4 answer with its source.",
  { x: 0.9, y: 4.88, w: 8.2, h: 0.28, fontSize: 11, color: GREY, align: "center" });

// 11 --------------------------------------------------------------- responsible AI
sec("Responsible AI and close");
s = slide("CONTENT", "Responsible AI",
  "Four risks and what the system does about each. Fairness: quality is not equal across programmes; programmes with too little parsed data get no prerequisite checks at all, instead of unreliable ones.");
[["Privacy", "Booklets and parsed text stay local; only ids, metrics and labels are public"],
 ["No invented content", "Draft syllabi are copied or templated, never generated; every unit names its source"],
 ["Human in the loop", "Outputs are drafts for the Board of Studies; the UI warns against automatic course cuts"],
 ["Fairness", `Bloom accuracy ranges ${R("week7/fairness", "bloom_accuracy_min.0")} (${get("week7/fairness", "bloom_accuracy_min.1")}) to ${R("week7/fairness", "bloom_accuracy_max.0")} (${get("week7/fairness", "bloom_accuracy_max.1")}); small programmes excluded from prerequisite checks`]]
  .forEach(([h, d], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.2 + Math.floor(i / 2) * 1.85;
    card(s, x, y, 4.4, 1.65, `risk ${i + 1}`);
    badge(s, x + 0.25, y + 0.25, i + 1);
    text(s, h, { x: x + 0.85, y: y + 0.25, w: 3.3, h: 0.42, fontSize: 17, bold: true, fontFace: HEAD, valign: "middle" });
    text(s, d, { x: x + 0.85, y: y + 0.75, w: 3.35, h: 0.8, fontSize: 13, color: GREY });
  });

// 12 --------------------------------------------------------------- limitations
s = slide("CONTENT", "Limitations and next steps",
  "Be direct about the limits: coverage of scanned booklets, LLM labels, prerequisite precision, synthetic job descriptions. Each one has a concrete next step.");
const lim = [["Scanned booklets", `${R("dataset/parse_stats", "text_pdf")} of ${R("dataset/parse_stats", "pdfs_considered")} PDFs parsed`, "Add OCR, then re-run coverage"],
  ["Labels", "LLM-annotated, no human annotators", `Fill the ${get("week8/labels", "human_spot_check_rows")}-pair human spot check`],
  ["Prerequisite flags", `precision ${R(pv, "fresh_sample.test_v2.precision")}, removal recall ${R(pv, "tests_test.v2.removal_recall.value")}`, "Faculty-confirmed prerequisite map"],
  ["Job descriptions", "synthetic samples in the demo", "Real postings; Group 10 integration"]];
s.addTable([["Limitation", "Today", "Next step"].map((c) => ({ text: c, options: { bold: true, color: WHITE, fill: { color: TEAL } } })),
  ...lim.map((r) => r.map((c, j) => ({ text: c, options: j === 0 ? { bold: true } : {} })))],
  { x: 0.5, y: 1.2, w: 9, colW: [2.2, 3.4, 3.4], fontFace: BODY, fontSize: 14, color: INK, border: { type: "solid", pt: 0.5, color: LINE }, valign: "middle", rowH: 0.62 });
text(s, "Every number in the reports and these slides is generated from results/*.json; skipped analyses are reported as not run.",
  { x: 0.5, y: 4.55, w: 9, h: 0.4, fontSize: 13, italic: true, color: TEAL });

// 13 --------------------------------------------------------------- close
s = pres.addSlide({ masterName: "DARK", sectionTitle: section });
s.addText("Thank you · Questions?", { placeholder: "title" });
const tested = testing ? `${testing.clean_clone.tests_passed} tests pass; a clean clone reproduces all ${testing.clean_clone.result_files_compared} result files` : "Final testing: see results/week8/final_testing.json";
s.addText([
  { text: "github.com/jainamsethia/Curriculum-Intelligence-Engine (tag v1.0)", options: { breakLine: true } },
  { text: tested, options: { breakLine: true, fontSize: 14, color: "9FC3C1" } },
  { text: "Code and text written with AI assistance (Claude); reference labels are LLM-annotated.", options: { fontSize: 12, color: "9FC3C1" } },
], { placeholder: "body" });
s.addNotes("Close with where to find the code and how it was tested. Invite questions; the live demo follows.");

pres.writeFile({ fileName: OUT }).then((f) => console.log("written", f));
