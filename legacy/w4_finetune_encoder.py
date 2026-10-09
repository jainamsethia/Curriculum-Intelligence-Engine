"""Week 4 - domain adaptation of the sentence encoder (all-MiniLM-L6-v2) with contrastive learning.

Training pairs come from the FIT courses only (the benchmark's validation and test courses are never touched):
  anchor   = a learning-outcome statement of a course
  positive = one of the two unit descriptions of the same course that the base encoder finds closest to that outcome
  negatives = the other pairs in the batch (every pair in a batch is from a different course, so there are no false negatives)
Loss: symmetric InfoNCE (scale 20). The epoch with the best MRR on the validation courses is kept.

    python src/w4_finetune_encoder.py
"""
import json, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).parent))
import w4_retrieval as R
import pipeline as P

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/interim/w4_minilm_ft"
EPOCHS, BATCH, LR, SEED = 5, 32, 2e-5, 42
STEPS_LIMIT = None                                            # set by a smoke test only


def main():
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    D = R.build()
    fams = D["fams"]
    base = P.Embedder("sentence-transformers/all-MiniLM-L6-v2")
    Ec = base(D["chunks"])
    fit_idx = [i for i, f in enumerate(fams) if f in D["fit"]]
    anchors = {}                                                   # course index -> [(outcome, [two closest chunk indices])]
    for i in fit_idx:
        outs = [o for o in D["outs"][fams[i]].values() if len(o.split()) >= R.MIN_Q_WORDS]
        if not outs:
            continue
        lo = D["starts"][i]; hi = D["starts"][i + 1] if i + 1 < len(fams) else len(D["chunks"])
        Eo = base(outs)
        top = np.argsort(-(Eo @ Ec[lo:hi].T), axis=1)[:, :2] + lo
        anchors[i] = [(o, list(t)) for o, t in zip(outs, top)]
    n_anchor = sum(len(v) for v in anchors.values())
    print(f"fit courses with outcomes: {len(anchors)}, training anchors: {n_anchor}", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    model.max_seq_length = 128
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps_per_epoch = n_anchor // BATCH if STEPS_LIMIT is None else STEPS_LIMIT
    total = EPOCHS * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * total)) * max(0.0, (total - s) / total))
    enc = lambda texts: torch.nn.functional.normalize(model(model.tokenize(texts))["sentence_embedding"], dim=-1)

    def val_mrr():
        model.eval()
        with torch.no_grad():
            Ecv = model.encode(D["chunks"], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            m = D["qsplit"] == "val"
            Eq = model.encode([q for q, v in zip(D["queries"], m) if v], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        S = np.full((len(D["queries"]), len(fams)), -1.0, np.float32)
        S[np.flatnonzero(m)] = R.best_unit(Eq @ Ecv.T, D["starts"])
        r = R.first_rank(S[m], {**D, "qrel": [q for q, v in zip(D["qrel"], m) if v]})
        model.train()
        return float(np.mean(np.where(r <= 10, 1 / r, 0)))

    log = {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "anchors": n_anchor, "fit_courses": len(anchors), "val_MRR@10": {}}
    log["val_MRR@10"]["epoch 0 (base encoder)"] = round(val_mrr(), 4)
    print("val MRR@10, base encoder:", log["val_MRR@10"]["epoch 0 (base encoder)"], flush=True)
    best, t0 = -1.0, time.time()
    model.train()
    for ep in range(1, EPOCHS + 1):
        pools = {i: random.sample(v, len(v)) for i, v in anchors.items()}
        ce = torch.nn.CrossEntropyLoss()
        losses = []
        for step in range(steps_per_epoch):
            live = [i for i, v in pools.items() if v]
            if len(live) < BATCH:
                break
            chosen = random.sample(live, BATCH)
            pairs = [(pools[i].pop(), i) for i in chosen]
            qs = [a[0] for a, _ in pairs]
            ds = [D["chunks"][random.choice(a[1])] for a, _ in pairs]
            logits = enc(qs) @ enc(ds).T * 20.0
            y = torch.arange(len(qs))
            loss = (ce(logits, y) + ce(logits.T, y)) / 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(float(loss))
            if step % 40 == 0:
                print(f"  epoch {ep} step {step}/{steps_per_epoch} loss {np.mean(losses[-40:]):.3f} ({time.time() - t0:.0f}s)", flush=True)
        v = val_mrr()
        log["val_MRR@10"][f"epoch {ep}"] = round(v, 4)
        print(f"epoch {ep}: mean loss {np.mean(losses):.3f}, val MRR@10 {v:.4f}", flush=True)
        if v > best:
            best = v
            OUT.mkdir(parents=True, exist_ok=True)
            model.save(str(OUT))
            log["best_epoch"] = ep
    log["seconds"] = round(time.time() - t0)
    (ROOT / "results/w4_finetune_log.json").write_text(json.dumps(log, indent=1))
    print("done; best epoch", log["best_epoch"], "val MRR@10", best, flush=True)


if __name__ == "__main__":
    main()
