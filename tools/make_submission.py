"""Pack one week's submission from its git tag:  python tools/make_submission.py <N>

submissions/week<N>/: report .docx (+ .pdf, page count checked), code zip (git archive of tag week-<N>-redo), SUBMIT.md
(rendered from reports/templates/submit_week<N>.md with the same placeholders as the report). Fails if the report check fails.
"""
import subprocess, sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).parent))
from build_report import ROOT, Resolver, build, doc_name, git  # noqa: E402
from check_reports import check  # noqa: E402

PS = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
MAX_PAGES = 5


def to_pdf(docx):
    pdf = docx.with_suffix(".pdf")
    subprocess.run([PS, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "tools/docx2pdf.ps1"), str(docx), str(pdf)], check=True)
    return pdf, len(fitz.open(pdf))


def pack(n, out_dir=None, tag=None):
    tag = tag or f"week-{n}-redo"
    assert git("tag", "-l", tag), f"tag {tag} does not exist yet"
    out_dir = Path(out_dir or ROOT / f"submissions/week{n}")
    docx = build(n, tag, out_dir / f"{doc_name(n)}.docx")
    errs = check([docx])
    if errs:
        sys.exit("report check failed:\n" + "\n".join(errs))
    pdf, pages = to_pdf(docx)
    if pages > MAX_PAGES:
        print(f"WARNING: {pdf.name} has {pages} pages (limit about {MAX_PAGES})")
    zip_path = out_dir / f"Group9_Curriculum_Intelligence_Engine_Week{n}_code.zip"
    subprocess.run(["git", "archive", "--format=zip", "--prefix=Curriculum-Intelligence-Engine/", "-o", str(zip_path), tag], cwd=ROOT, check=True)
    tpl = ROOT / f"reports/templates/submit_week{n}.md"
    if tpl.exists():
        (out_dir / "SUBMIT.md").write_text(Resolver(tag).text(tpl.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"{docx.name}: {pages} pages\n{zip_path.name}: {zip_path.stat().st_size // 1024} KB\nSUBMIT.md: {'written' if tpl.exists() else 'MISSING'}")
    return docx, pdf, zip_path


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pack(int(sys.argv[1]))
