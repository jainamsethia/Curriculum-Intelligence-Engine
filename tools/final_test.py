"""Final testing (Week 8): clean clone of HEAD -> pytest -> full pipeline -> every committed result must come out identical;
then the latency of every API endpoint on the real NMIMS data. Writes results/week8/final_testing.json.

    python tools/final_test.py            (about 5 minutes; needs data/processed/courses.jsonl, i.e. the Teams zip or NMIMS_ZIP)

Not part of `run all`: timings differ from machine to machine, so this file records one measured run.
"""
import json, os, platform, re, shutil, stat, statistics, subprocess, sys, tempfile, time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/week8/final_testing.json"


def sh(cmd, cwd, env=None):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")


def clean_clone():
    tmp = Path(tempfile.mkdtemp(prefix="ce-final-"))
    try:
        sh(["git", "clone", "-q", str(ROOT), str(tmp / "repo")], ROOT)
        repo = tmp / "repo"
        commit = sh(["git", "rev-parse", "HEAD"], repo).stdout.strip()
        t = sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], repo)
        line = (t.stdout.strip().splitlines() or [""])[-1]
        counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|errors?|skipped)", line)}
        env = {**os.environ, "CE_DATA_DIR": str(ROOT / "data"), "HF_HUB_OFFLINE": "1"}
        t0 = time.time()
        r = sh([sys.executable, "-m", "curriculum_engine.run", "all"], repo, env)
        secs = round(time.time() - t0)
        changed = [ln[3:] for ln in sh(["git", "status", "--porcelain"], repo).stdout.splitlines()]
        compared = [f for f in sh(["git", "ls-files", "results", "docs/examples", "docs/openapi.json"], repo).stdout.splitlines()
                    if not f.endswith("final_testing.json")]
        return {"commit": commit, "pytest": line, "tests_passed": counts.get("passed", 0),
                "tests_failed": counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0),
                "run_all_exit_code": r.returncode, "run_all_seconds": secs, "result_files_compared": len(compared),
                "changed_files": changed, "reproduced": r.returncode == 0 and not changed}
    finally:
        shutil.rmtree(tmp, onerror=lambda f, p, _: (os.chmod(p, stat.S_IWRITE), f(p)))   # .git objects are read-only on Windows


def latency(repeats=3):
    """Median wall time per endpoint (in-process, after one warm-up call), with the requests of docs/examples."""
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient
    from curriculum_engine import api
    from curriculum_engine.experiments import week7 as W7
    c = TestClient(api.app)
    t0 = time.time()
    assert c.get("/health").status_code == 200
    start = round(time.time() - t0, 1)
    calls = {}
    for f in sorted((ROOT / "docs/examples").glob("*.json")):
        req = json.loads(f.read_text(encoding="utf-8"))["request"]
        calls[f"{req['method']} {req['path']} ({f.stem})"] = (req["method"], req["path"], req.get("body"))
    calls["POST /modernize-curriculum (31 October demo)"] = ("POST", "/modernize-curriculum",
                                                             {"jds": [{"id": k, "text": v} for k, v in W7.jds().items()], "courses": [{"id": i} for i in W7.DEMO_COURSES]})
    calls["GET /courses?q=language"] = ("GET", "/courses?q=language", None)
    rows = []
    for name, (method, path, body) in calls.items():
        times = []
        for i in range(repeats + 1):
            t = time.time()
            r = c.request(method, path, json=body)
            assert r.status_code == 200, (name, r.status_code, r.text[:200])
            times.append(time.time() - t)
        rows.append([name, r.status_code, round(statistics.median(times[1:]) * 1000)])
    return {"engine_start_seconds": start, "repeats": repeats,
            "table": {"columns": ["Endpoint (example request)", "Status", "Median latency (ms)"], "rows": rows},
            "slowest_ms": max(r[2] for r in rows), "all_ok": all(r[1] == 200 for r in rows)}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    res = {"date": date.today().isoformat(), "python": platform.python_version(), "os": platform.platform(terse=True),
           "cpu_count": os.cpu_count(), "clean_clone": clean_clone()}
    print(json.dumps(res["clean_clone"], indent=1))
    res["latency"] = latency()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res["latency"], indent=1))
    sys.exit(0 if res["clean_clone"]["reproduced"] and res["clean_clone"]["tests_failed"] == 0 else 1)
