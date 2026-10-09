"""Optional local LLM (Phi-3 via Ollama). Nothing is sent to external services; every caller has a non-LLM fallback."""
import json, time, urllib.request

OLLAMA = "http://localhost:11434"
MODEL = "phi3"


def _post(path, body, timeout=900):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def available(model=MODEL):
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5) as r:
            return any(m["name"].split(":")[0] == model for m in json.loads(r.read())["models"])
    except Exception:
        return False


def generate(prompt, model=MODEL, num_predict=120, temperature=0.0, fmt=None, system=None, num_ctx=4096, retries=2):
    """One deterministic completion. Returns the text; raises after `retries` failures."""
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "num_predict": num_predict, "seed": 7, "num_ctx": num_ctx}}
    if fmt:
        body["format"] = fmt
    if system:
        body["system"] = system
    for attempt in range(retries + 1):
        try:
            return _post("/api/generate", body)["response"].strip()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(3)
