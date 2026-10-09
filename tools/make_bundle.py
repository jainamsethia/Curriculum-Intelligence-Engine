"""Build the Week 5 submission bundle: every finished week's report rebuilt from ONE commit (so all numbers agree),
one code zip, INDEX.md and SUBMIT.md, after a cross-report consistency check.

    python tools/make_bundle.py --tag week-5-redo --weeks 1 2 3 4 5 [--out submissions/week5_bundle]
"""
import argparse, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_report import ROOT, Resolver, build, doc_name, git  # noqa: E402
from check_reports import check  # noqa: E402
from make_submission import MAX_PAGES, to_pdf  # noqa: E402

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weeks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", default=str(ROOT / "submissions/week5_bundle"))
    ap.add_argument("--interim", action="store_true", help="use the interim INDEX/SUBMIT templates")
    a = ap.parse_args()
    assert git("tag", "-l", a.tag), f"tag {a.tag} does not exist"
    out = Path(a.out).resolve()                 # Word needs an absolute path
    docs = [build(n, a.tag, out / f"{doc_name(n)}.docx") for n in a.weeks]
    errs = check(docs)
    if errs:
        sys.exit("consistency check failed:\n" + "\n".join(errs))
    pages = {d.name: to_pdf(d)[1] for d in docs}
    zip_path = out / f"Group9_Curriculum_Intelligence_Engine_Weeks{a.weeks[0]}-{a.weeks[-1]}_code.zip"
    subprocess.run(["git", "archive", "--format=zip", "--prefix=Curriculum-Intelligence-Engine/", "-o", str(zip_path), a.tag], cwd=ROOT, check=True)
    R = Resolver(a.tag)
    for name in ("INDEX", "SUBMIT"):
        tpl = ROOT / f"reports/templates/bundle_{'interim_' if a.interim else ''}{name.lower()}.md"
        (out / f"{name}.md").write_text(R.text(tpl.read_text(encoding="utf-8")), encoding="utf-8")
    for d, p in pages.items():
        print(f"{d}: {p} pages" + ("  (over the page limit)" if p > MAX_PAGES else ""))
    print(f"{zip_path.name}: {zip_path.stat().st_size // 1024} KB\nOK: {len(docs)} reports consistent with results/ at {a.tag}")
