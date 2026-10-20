"""Screenshot of the running committee UI (for the reports), via headless Chrome and the DevTools protocol.

    uvicorn curriculum_engine.api:app --port 8765
    CE_API_URL=http://127.0.0.1:8765 CE_UI_AUTORUN=1 streamlit run app/committee_ui.py --server.address 127.0.0.1 --server.port 8766
    python tools/ui_screenshot.py http://127.0.0.1:8766 data/week7/fig_ui.png "retrieval confidence" demo=0:1680 handoffs=3500:5200

Waits until the page text contains the given marker (the UI has finished the demo step), then saves a full-page PNG,
the named crops (<out>_<name>.png) and the side-by-side report figure results/week7/fig_ui_report.png. The full page shows
syllabus text, so <out> belongs in data/ (git-ignored); the report figure shows course names and our own notice only.
"""
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request

import websocket

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def main(url, out, marker, crops=(), width=1500, height=5200, timeout=600):
    prof = tempfile.mkdtemp(prefix="ui-shot-")
    p = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--remote-debugging-port=9333",
                          "--remote-allow-origins=*", f"--user-data-dir={prof}", f"--window-size={width},{height}", "about:blank"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(30):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json"))
                break
            except OSError:
                time.sleep(1)
        ws = websocket.create_connection(next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page"), timeout=timeout)
        n = [0]

        def cmd(method, **params):
            n[0] += 1
            ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == n[0]:
                    return msg.get("result", {})

        cmd("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)
        cmd("Page.navigate", url=url)
        t0 = time.time()
        while time.time() - t0 < timeout:
            text = cmd("Runtime.evaluate", expression="document.body.innerText", returnByValue=True).get("result", {}).get("value", "")
            if marker in text and "Running..." not in text:
                break
            time.sleep(3)
        else:
            sys.exit(f"marker {marker!r} not found within {timeout} s")
        time.sleep(5)                                            # let the last widgets paint
        png = cmd("Page.captureScreenshot", format="png")["data"]
        open(out, "wb").write(base64.b64decode(png))
        print(out, "saved after", round(time.time() - t0), "s")
        from PIL import Image
        for c in crops:                                          # name=y0:y1 -> <out>_<name>.png
            name, rng = c.split("=")
            y0, y1 = map(int, rng.split(":"))
            Image.open(out).crop((0, y0, width, y1)).save(out.replace(".png", f"_{name}.png"))
    finally:
        p.terminate()


def report_figure(out, dest="results/week7/fig_ui_report.png"):
    """Week 7 report figure: inputs + overlap table | Group 11 and Group 4 hand-offs, side by side, sidebar cut off.
    ponytail: pixel boxes fit the current UI layout at width 1500; re-pick them if the UI changes."""
    from PIL import Image
    a = Image.open(out.replace(".png", "_demo.png")).crop((300, 240, 1440, 1290))
    b = Image.open(out.replace(".png", "_handoffs.png")).crop((300, 870, 1440, 1700))
    fig = Image.new("RGB", (a.width + b.width + 30, max(a.height, b.height)), "white")
    fig.paste(a, (0, 0)); fig.paste(b, (a.width + 30, 0))
    fig.save(dest)


if __name__ == "__main__":
    if sys.argv[1] == "--report-figure":                         # python tools/ui_screenshot.py --report-figure data/week7/fig_ui.png
        report_figure(sys.argv[2])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "retrieval confidence", sys.argv[4:])
        report_figure(sys.argv[2])
