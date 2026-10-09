"""Build a weekly report (.docx) from reports/templates/week<N>.md and results/**/*.json. No number is typed by hand.

Template syntax (a small markdown subset):
  # Title / ## Heading / ### Subheading, paragraphs, '- ' bullets, **bold**, `code`, | tables |, ![caption](path.png), ``` code ```
  {{file:path:fmt}}   value from results/<file>.json at dotted <path> (list index allowed), formatted with <fmt> (e.g. .3f, ,d, .0%)
  {{table:file:path}} a table stored as {"columns": [...], "rows": [[...], ...]}
  {{json:file:path}}  a JSON excerpt as a code block
  {{git:commit}} {{git:tag}} {{git:tree}} {{git:repo}}
Every resolved value is written to <out>.manifest.json; tools/check_reports.py re-checks them against the results.

    python tools/build_report.py <N> [--tag week-N-redo] [--out submissions/weekN/<name>.docx]
"""
import argparse, json, re, subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPO = "https://github.com/jainamsethia/Curriculum-Intelligence-Engine"
TITLES = {1: "Proposal_v2", 2: "Dataset_Baseline", 3: "Core_NLP_Model", 4: "Improved_Model", 5: "Evaluation_Error_Analysis",
          6: "Asset_Engineering", 7: "Integration_Demo", 8: "Final"}
PH = re.compile(r"\{\{([^{}]+)\}\}")
ACCENT = RGBColor(0x1F, 0x4E, 0x79)


def doc_name(n):
    return f"Group9_Curriculum_Intelligence_Engine_Week{n}_{TITLES[n]}"


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def lookup(file, path):
    d = json.loads((RESULTS / f"{file}.json").read_text(encoding="utf-8"))
    for k in path.split(".") if path else []:
        d = d[int(k)] if isinstance(d, list) else d[k]
    return d


def fmt(v, f):
    if v is None:
        return "n/a"
    if f:
        return format(v, f)
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v)


class Resolver:
    def __init__(self, tag):
        self.tag = tag
        self.commit = git("rev-list", "-n", "1", tag) if tag and git("tag", "-l", tag) else git("rev-parse", "HEAD")
        self.manifest = []

    def value(self, key):
        kind, _, rest = key.partition(":")
        if kind == "git":
            return {"commit": self.commit, "tag": self.tag or "(untagged)", "repo": REPO,
                    "tree": tree(self.commit)}[rest]
        if kind in ("table", "json"):
            f, _, p = rest.partition(":")
            v = lookup(f, p)
            self.manifest.append({"key": key, "value": json.dumps(v, sort_keys=True, ensure_ascii=False)})
            return v
        f, p, form = (key.split(":") + ["", ""])[:3]
        s = fmt(lookup(f, p), form)
        self.manifest.append({"key": key, "value": s})
        return s

    def text(self, s):
        return PH.sub(lambda m: self.value(m.group(1).strip()), s)


def tree(commit):
    """Top-level folders of the commit with what each holds (one line each)."""
    desc = {"curriculum_engine": "the NLP engine (parser, data, overlap, redundancy, prerequisites, diff, Bloom, evaluation)",
            "tools": "report builder, report checker, submission packer", "tests": "unit tests (pytest)",
            "labels": "LLM-annotated reference labels (ids + labels per pass + consensus)", "results": "metrics JSON + figures per week",
            "reports": "report templates (numbers are placeholders)", "submissions": "per-week submission folders",
            "legacy": "old Week 3-4 scripts and reports (history only)", "legacy_mit": "old MIT OCW scripts (not used)", "docs": "documentation"}
    names = [x for x in git("ls-tree", "--name-only", commit).splitlines() if x]
    return "\n".join(f"{n + '/' if n in desc else n:<20} {desc.get(n, '')}".rstrip() for n in names)


# ------------------------------------------------------------------ docx helpers
def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hex_fill)
    tcPr.append(sh)


def runs(par, text, size=None, bold=False):
    """**bold** and `code` inline."""
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        if part.startswith("**"):
            r = par.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1]); r.font.name = "Consolas"
        else:
            r = par.add_run(part); r.bold = bold
        if size:
            r.font.size = Pt(size)


def add_table(doc, header, rows, R):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""
        runs(c.paragraphs[0], R.text(str(h)), size=8, bold=True); shade(c, "D9E2F3")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row[:len(header)]):
            cells[i].text = ""
            s = R.text(v) if isinstance(v, str) else fmt(v, "")
            runs(cells[i].paragraphs[0], s, size=8)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def code_block(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text); r.font.name = "Consolas"; r.font.size = Pt(7)
    pPr = p._p.get_or_add_pPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "F2F2F2"); pPr.append(sh)


def build(n, tag, out):
    R = Resolver(tag)
    src = (ROOT / f"reports/templates/week{n}.md").read_text(encoding="utf-8")
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    for m in ("left_margin", "right_margin"):
        setattr(sec, m, Cm(1.8))
    sec.top_margin = sec.bottom_margin = Cm(1.5)
    st = doc.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(9.5)
    st.paragraph_format.space_after = Pt(3)
    for h, size in (("Heading 1", 12), ("Heading 2", 10.5), ("Title", 16)):
        doc.styles[h].font.size = Pt(size); doc.styles[h].font.color.rgb = ACCENT

    lines, i = src.splitlines(), 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            code_block(doc, R.text("\n".join(lines[i + 1:j])))
            i = j + 1
            continue
        s = l.strip()
        m = re.fullmatch(r"\{\{(table|json):([^{}]+)\}\}", s)
        if m:
            v = R.value(f"{m.group(1)}:{m.group(2)}")
            if m.group(1) == "table":
                add_table(doc, v["columns"], v["rows"], R)
            else:
                code_block(doc, json.dumps(v, indent=1, ensure_ascii=False))
        elif s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append([c.strip() for c in lines[i].strip().strip("|").split("|")]); i += 1
            add_table(doc, block[0], [r for r in block[1:] if not re.fullmatch(r"[-: ]+", "".join(r))], R)
            continue
        elif s.startswith("!["):
            cap, path = re.fullmatch(r"!\[(.*)\]\((.*)\)", s).groups()
            doc.add_picture(str(ROOT / path), width=Cm(16.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = doc.add_paragraph(); runs(p, R.text(cap), size=8); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif s.startswith("# "):
            doc.add_paragraph(R.text(s[2:]), style="Title")
        elif s.startswith("## "):
            doc.add_heading(R.text(s[3:]), level=1)
        elif s.startswith("### "):
            doc.add_heading(R.text(s[4:]), level=2)
        elif s.startswith("- "):
            runs(doc.add_paragraph(style="List Bullet"), R.text(s[2:]))
        elif s:
            runs(doc.add_paragraph(), R.text(s))
        i += 1
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    out.with_suffix(".manifest.json").write_text(json.dumps({"week": n, "tag": tag, "commit": R.commit, "values": R.manifest}, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("week", type=int)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / f"submissions/week{a.week}/{doc_name(a.week)}.docx"
    print(build(a.week, a.tag or f"week-{a.week}-redo", out))
