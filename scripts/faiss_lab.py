"""FAISS lab: exact vs approximate nearest-neighbour search on the real index.

    python -m scripts.faiss_lab              # 13.1 exact top 10 + 13.2 index sweep

13.1: load the bge-small chunk vectors from Chroma, embed every question in
the golden sets (golden, conversations, policy; deduplicated), build an exact
IndexFlatIP and save its top 10: the reference every other index is measured
against. Checks that Flat's top 1 matches Chroma's own search.

13.2: build HNSW, IVF and IVF-PQ at several settings and record, per setting,
ANN recall@10 against Flat, p50/p95 latency of one query at a time (a user
sends one query, not a batch), build time, and memory (serialized size).

Vectors are L2-normalised, so inner product = cosine, and every index is
built with METRIC_INNER_PRODUCT (HNSW and IVF default to L2, which gives the
same ranking on unit vectors, but only on unit vectors).

FAISS prints training warnings (too few points per cluster) to stderr; they
are left visible on purpose.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "faiss"
K = 10


def ann_recall(found: np.ndarray, truth: np.ndarray, k: int = K) -> float:
    """Mean share of the true top k that the index found (FAISS pads misses with -1)."""
    hits = [len(set(f[:k]) & set(t[:k]) - {-1}) / k for f, t in zip(found, truth)]
    return float(np.mean(hits))


def normalise(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype="float32")
    faiss.normalize_L2(x)
    return x


def load_vectors(variant: str = "plain") -> tuple[list[str], np.ndarray]:
    from helpdesk.embedder import DEFAULT_MODEL
    from helpdesk.index import DEFAULT_PERSIST_DIR, get_collection

    payload = get_collection(DEFAULT_PERSIST_DIR, DEFAULT_MODEL, variant).get(include=["embeddings"])
    return list(payload["ids"]), normalise(np.asarray(payload["embeddings"]))


def golden_questions() -> list[str]:
    data = ROOT / "data"
    texts = [q["question"] for q in json.loads((data / "golden.json").read_text(encoding="utf-8"))]
    for convo in json.loads((data / "golden_conversations.json").read_text(encoding="utf-8")):
        texts += [turn["question"] for turn in convo["turns"]]
    for case in json.loads((data / "golden_policy.json").read_text(encoding="utf-8")):
        texts += [turn["user"] for turn in case["turns"]]
    return list(dict.fromkeys(texts))


def run(index, name: str, params: dict, xb: np.ndarray, xq: np.ndarray, truth: np.ndarray,
        k: int = K, train: bool = False) -> dict:
    """Build (train + add), then time each query on its own; one row per setting."""
    started = time.perf_counter()
    if train:
        index.train(xb)
    index.add(xb)
    build_s = time.perf_counter() - started
    found = np.empty((len(xq), k), dtype="int64")
    latencies = []
    for i in range(len(xq)):
        t0 = time.perf_counter()
        _, ids = index.search(xq[i : i + 1], k)
        latencies.append((time.perf_counter() - t0) * 1000)
        found[i] = ids[0]
    return {
        "index": name, "params": params, "recall@10": round(ann_recall(found, truth, k), 4),
        "p50_ms": round(float(np.percentile(latencies, 50)), 4),
        "p95_ms": round(float(np.percentile(latencies, 95)), 4),
        "build_s": round(build_s, 4),
        "memory_mb": round(faiss.serialize_index(index).nbytes / 1e6, 4),
    }


def configure(index, **search):
    if "efSearch" in search:
        index.hnsw.efSearch = search["efSearch"]
    if "nprobe" in search:
        index.nprobe = search["nprobe"]
    return index


def sweep(xb: np.ndarray, xq: np.ndarray, truth: np.ndarray) -> list[dict]:
    n, d = xb.shape
    rows = [run(faiss.IndexFlatIP(d), "flat", {}, xb, xq, truth)]
    for m in (16, 32):
        for ef in (16, 32, 64, 128, 256):
            index = configure(faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT), efSearch=ef)
            rows.append(run(index, "hnsw", {"M": m, "efSearch": ef}, xb, xq, truth))
    nlist = max(1, round(math.sqrt(n)))
    for nprobe in sorted({p for p in (1, 2, 4, 8, 16, 32) if p <= nlist} | {nlist}):
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = nprobe
        row = run(index, "ivf", {"nlist": nlist, "nprobe": nprobe}, xb, xq, truth, train=True)
        row["_quantizer"] = quantizer  # keep alive until the row is printed
        rows.append(row)
    for nbits in (8, 4):
        for nprobe in (1, 8, 32):
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(quantizer, d, nlist, 48, nbits, faiss.METRIC_INNER_PRODUCT)
            index.nprobe = min(nprobe, nlist)
            params = {"nlist": nlist, "m": 48, "nbits": nbits, "nprobe": nprobe}
            try:
                rows.append(run(index, "ivf-pq", params, xb, xq, truth, train=True))
            except RuntimeError as error:  # PQ needs >= 2**nbits training points per codebook
                rows.append({"index": "ivf-pq", "params": params,
                             "error": str(error).strip().splitlines()[-1][:160]})
    for row in rows:
        row.pop("_quantizer", None)
    return rows


def main() -> None:
    from transformers.utils import logging as hf_logging

    from helpdesk.embedder import DEFAULT_MODEL, Embedder
    from helpdesk.index import DEFAULT_PERSIST_DIR, search

    hf_logging.set_verbosity_error()
    ids, xb = load_vectors()
    questions = golden_questions()
    embedder = Embedder(DEFAULT_MODEL)
    xq = normalise(np.stack([embedder.embed_query(q) for q in questions]))
    n, d = xb.shape
    print(f"{n} chunk vectors x {d} dims; {len(questions)} questions")

    # 13.1: exact reference
    flat = faiss.IndexFlatIP(d)
    flat.add(xb)
    _, truth = flat.search(xq, K)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "flat_top10_idx.npy", truth)
    (OUT / "flat_top10.json").write_text(json.dumps(
        {q: [ids[i] for i in row] for q, row in zip(questions, truth)}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    chroma_top1 = search(questions[0], embedder, DEFAULT_PERSIST_DIR, k=1)[0].chunk_id
    agree = sum(search(q, embedder, DEFAULT_PERSIST_DIR, k=1)[0].chunk_id == ids[row[0]]
                for q, row in zip(questions, truth))
    print(f"Flat top 1 for q1 = Chroma top 1: {ids[truth[0][0]] == chroma_top1}; "
          f"all questions: {agree}/{len(questions)} agree")

    # 13.2: sweep
    rows = sweep(xb, xq, truth)
    (OUT / f"sweep_{n}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n{'index':8}{'params':40}{'recall@10':>10}{'p50 ms':>9}{'p95 ms':>9}{'build s':>9}{'memory MB':>11}")
    for row in rows:
        params = " ".join(f"{k}={v}" for k, v in row["params"].items()) or "-"
        if "error" in row:
            print(f"{row['index']:8}{params:40}  cannot train: {row['error']}")
            continue
        print(f"{row['index']:8}{params:40}{row['recall@10']:>10.3f}{row['p50_ms']:>9.3f}"
              f"{row['p95_ms']:>9.3f}{row['build_s']:>9.3f}{row['memory_mb']:>11.3f}")
    print(f"\nsaved results/faiss/sweep_{n}.json, flat_top10.json, flat_top10_idx.npy")


if __name__ == "__main__":
    main()
