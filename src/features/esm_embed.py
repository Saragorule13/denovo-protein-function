"""Phase 3a: frozen ESM-2 35M per-residue embeddings (from repo root).

  python src/features/esm_embed.py [--n N] [--batch-tokens 4000]

In : data/processed/clean.csv
Out : data/processed/embeddings/{accession}.pt (fp16 [L,480]) + embeddings_manifest.csv
Resume-safe: skips accessions with an existing .pt file.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
EMB = PROC / "embeddings"
EMB.mkdir(parents=True, exist_ok=True)
CKPT = "facebook/esm2_t12_35M_UR50D"


def main(n, batch_tokens):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    df = pd.read_csv(PROC / "clean.csv")
    if n:
        df = df.head(n)
    done = {p.stem for p in EMB.glob("*.pt")}
    todo = df[~df["accession"].isin(done)].sort_values(
        "length", ascending=False).reset_index(drop=True)
    print(f"total={len(df)} done={len(done)} todo={len(todo)}", flush=True)
    if len(todo) == 0:
        return
    tok = EsmTokenizer.from_pretrained(CKPT)
    model = EsmModel.from_pretrained(CKPT).to(device).eval()
    rows, t0 = [], time.time()
    i = 0
    while i < len(todo):
        # length-bucketed batch within token budget
        j, budget = i, 0
        while j < len(todo) and budget + todo.loc[j, "length"] <= batch_tokens:
            budget += todo.loc[j, "length"]
            j += 1
        j = max(j, i + 1)
        chunk = todo.iloc[i:j]
        enc = tok(list(chunk["sequence"]), return_tensors="pt",
                  padding=True, truncation=True, max_length=1024)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state  # [B, Lpad+2, 480]
        for k, row in enumerate(chunk.itertuples()):
            L = min(int(row.length), h.size(1) - 2)
            emb = h[k, 1:1 + L, :].detach().cpu().to(torch.float16)
            torch.save(emb, EMB / f"{row.accession}.pt")
            rows.append({"accession": row.accession, "L": L})
        i = j
        if len(rows) % 200 < (j - i + len(chunk)) % 200 or True:
            pass
        if (len(rows) // 50) != ((len(rows) - len(chunk)) // 50):
            dt = time.time() - t0
            print(f"  ...{len(rows)}/{len(todo)} ({dt:.0f}s, "
                  f"{len(rows) / max(dt, 1):.1f} prot/s)", flush=True)
    man = PROC / "embeddings_manifest.csv"
    old = pd.read_csv(man) if man.exists() else pd.DataFrame(columns=["accession", "L"])
    man_df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("accession")
    man_df.to_csv(man, index=False)
    print(f"done {len(rows)} new, total {len(man_df)} -> {man}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch-tokens", type=int, default=4000)
    a = ap.parse_args()
    main(a.n, a.batch_tokens)