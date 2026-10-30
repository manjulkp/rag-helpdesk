"""FAISS lab, 13.3-13.5, on saved vectors (NumPy + FAISS only, no PyTorch).

    python -m scripts.faiss_queries                       # once: embed questions, save vectors
    python -m scripts.faiss_scale scale --n 250000        # 13.3 scale-up, sigma 0.05 and 0.02
    python -m scripts.faiss_scale metrics                 # 13.4 distance metrics
    python -m scripts.faiss_scale quant                   # 13.5 quantization on the real vectors

13.3: the real corpus is 97 vectors, so the scale-up adds synthetic ones:
each is a random real chunk vector plus Gaussian noise (per-dimension sigma),
re-normalised. sigma sets how close a synthetic vector stays to its source
(reported as the mean cosine), which turns out to decide which index wins.
Builds use all cores; each query is timed alone on one thread. Memory is the
index's size on disk (faiss.write_index), so no second in-RAM copy is made.

13.4 and 13.5 also report golden-set file recall@8 and answer-chunk recall@20
for the 15 golden questions (the first 15 saved questions are golden.json's).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import faiss
import numpy as np

from scripts.faiss_lab import OUT, ann_recall

K = 10
CORES = os.cpu_count() or 1


def load() -> tuple[np.ndarray, np.ndarray, list[str], list[dict]]:
    xb = np.load(OUT / "vectors.npy").astype("float32")
    faiss.normalize_L2(xb)
    xq = np.load(OUT / "queries.npy").astype("float32")
    faiss.normalize_L2(xq)
    questions = json.loads((OUT / "questions.json").read_text(encoding="utf-8"))
    chunks = json.loads((OUT / "chunks.json").read_text(encoding="utf-8"))
    return xb, xq, questions, chunks


def disk_mb(index) -> float:
    handle, path = tempfile.mkstemp(suffix=".faiss")
    os.close(handle)
    try:
        if isinstance(index, faiss.IndexBinary):
            faiss.write_index_binary(index, path)
        else:
            faiss.write_index(index, path)
        return round(Path(path).stat().st_size / 1e6, 2)
    finally:
        os.unlink(path)


def timed_search(index, xq: np.ndarray, k: int = K) -> tuple[np.ndarray, list[float]]:
    faiss.omp_set_num_threads(1)  # one user's query runs alone
    found = np.empty((len(xq), k), dtype="int64")
    latencies = []
    for i in range(len(xq)):
        t0 = time.perf_counter()
        _, ids = index.search(xq[i : i + 1], k)
        latencies.append((time.perf_counter() - t0) * 1000)
        found[i] = ids[0]
    faiss.omp_set_num_threads(CORES)
    return found, latencies


def row(name: str, params: dict, index, build_s: float, xq, truth) -> dict:
    found, lat = timed_search(index, xq)
    return {"index": name, "params": params, "recall@10": round(ann_recall(found, truth), 4),
            "p50_ms": round(float(np.percentile(lat, 50)), 3),
            "p95_ms": round(float(np.percentile(lat, 95)), 3),
            "build_s": round(build_s, 2), "memory_mb": disk_mb(index)}


def synthetic(real: np.ndarray, n: int, sigma: float, seed: int = 0) -> tuple[np.ndarray, float]:
    """n vectors: a random real vector + N(0, sigma^2) per dimension, normalised; also the mean
    cosine to the source vector."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, real.shape[1]), dtype="float32")
    cos = []
    for start in range(0, n, 50_000):
        stop = min(n, start + 50_000)
        src = rng.integers(0, len(real), stop - start)
        block = real[src] + rng.normal(0, sigma, (stop - start, real.shape[1])).astype("float32")
        faiss.normalize_L2(block)
        out[start:stop] = block
        cos.append(np.einsum("ij,ij->i", block, real[src]))
    return out, float(np.mean(np.concatenate(cos)))


def scale(n: int, sigmas: list[float]) -> None:
    real, xq, _, _ = load()
    faiss.omp_set_num_threads(CORES)
    for sigma in sigmas:
        synth, mean_cos = synthetic(real, n - len(real), sigma)
        xb = np.concatenate([real, synth])
        del synth
        print(f"\nN={len(xb):,}  sigma={sigma}  mean cos(synthetic, source)={mean_cos:.3f}", flush=True)
        rows = []
        t0 = time.perf_counter()
        flat = faiss.IndexFlatIP(xb.shape[1])
        flat.add(xb)
        build = time.perf_counter() - t0
        _, truth = flat.search(xq, K)
        real_in_truth = float(np.mean(truth < len(real)))
        rows.append(row("flat", {}, flat, build, xq, truth))
        del flat
        for m, efs in ((16, (16, 64, 256)), (32, (64, 256))):
            t0 = time.perf_counter()
            hnsw = faiss.IndexHNSWFlat(xb.shape[1], m, faiss.METRIC_INNER_PRODUCT)
            hnsw.hnsw.efConstruction = 40
            hnsw.add(xb)
            build = time.perf_counter() - t0
            for ef in efs:
                hnsw.hnsw.efSearch = ef
                rows.append(row("hnsw", {"M": m, "efSearch": ef}, hnsw, build, xq, truth))
                print(f"  {rows[-1]}", flush=True)
            del hnsw
        nlist = round(math.sqrt(len(xb)))
        quantizer = faiss.IndexFlatIP(xb.shape[1])
        t0 = time.perf_counter()
        ivf = faiss.IndexIVFFlat(quantizer, xb.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(xb)
        ivf.add(xb)
        build = time.perf_counter() - t0
        for nprobe in (1, 8, 32, 128):
            ivf.nprobe = nprobe
            rows.append(row("ivf", {"nlist": nlist, "nprobe": nprobe}, ivf, build, xq, truth))
            print(f"  {rows[-1]}", flush=True)
        del ivf, quantizer
        quantizer = faiss.IndexFlatIP(xb.shape[1])
        t0 = time.perf_counter()
        pq = faiss.IndexIVFPQ(quantizer, xb.shape[1], nlist, 48, 8, faiss.METRIC_INNER_PRODUCT)
        pq.train(xb)
        pq.add(xb)
        build = time.perf_counter() - t0
        for nprobe in (8, 32, 128):
            pq.nprobe = nprobe
            rows.append(row("ivf-pq", {"nlist": nlist, "m": 48, "nbits": 8, "nprobe": nprobe},
                            pq, build, xq, truth))
            print(f"  {rows[-1]}", flush=True)
        # exact re-score of the PQ top 100 against the stored float vectors
        refine = faiss.IndexRefineFlat(pq, faiss.swig_ptr(xb))
        refine.k_factor = 10
        pq.nprobe = 32
        rows.append(row("ivf-pq+refine", {"nprobe": 32, "rescore_top": 100}, refine, build, xq, truth))
        print(f"  {rows[-1]}", flush=True)
        del refine, pq, quantizer, xb
        result = {"n": n, "sigma": sigma, "mean_cos_to_source": round(mean_cos, 3),
                  "share_of_true_top10_that_is_real": round(real_in_truth, 3), "rows": rows}
        path = OUT / f"sweep_{n}_sigma{sigma}.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print_table(rows)
        print(f"  true top 10 that are real chunks: {real_in_truth:.0%}; saved {path.name}")


def print_table(rows: list[dict]) -> None:
    print(f"  {'index':14}{'params':34}{'recall@10':>10}{'p50 ms':>9}{'p95 ms':>9}{'build s':>9}{'MB':>9}")
    for r in rows:
        params = " ".join(f"{k}={v}" for k, v in r["params"].items()) or "-"
        print(f"  {r['index']:14}{params:34}{r['recall@10']:>10.3f}{r['p50_ms']:>9.3f}"
              f"{r['p95_ms']:>9.3f}{r['build_s']:>9.2f}{r['memory_mb']:>9.2f}")


def golden_metrics(found: np.ndarray, questions: list[str], chunks: list[dict]) -> dict:
    """File recall@8 and answer-chunk recall@20 on the golden questions, from FAISS ids."""
    from helpdesk.evaluate import answer_rank, first_relevant_rank, load_golden

    golden = {q.question: q for q in load_golden(Path(__file__).resolve().parents[1] / "data" / "golden.json")}
    file_hits, answer_hits, labelled = [], [], []
    for qi, text in enumerate(questions):
        q = golden.get(text)
        if q is None or q.type == "negative":
            continue
        hits = [chunks[i] for i in found[qi] if i >= 0]
        rank = first_relevant_rank(hits, q.expected)
        file_hits.append(rank is not None and rank <= 8)
        if q.expected_chunks:
            ar = answer_rank(hits, q.expected_chunks)
            answer_hits.append(ar is not None and ar <= 20)
    return {"R@8": round(float(np.mean(file_hits)), 3),
            "ansR@20": round(float(np.mean(answer_hits)), 3) if answer_hits else None,
            "n": len(file_hits)}


def search_all(index, xq: np.ndarray, k: int) -> np.ndarray:
    return index.search(xq, k)[1]


def metrics() -> None:
    xb, xq, questions, chunks = load()
    k = min(20, len(xb))
    ip = faiss.IndexFlatIP(xb.shape[1]); ip.add(xb)
    l2 = faiss.IndexFlatL2(xb.shape[1]); l2.add(xb)
    top_ip, top_l2 = search_all(ip, xq, K), search_all(l2, xq, K)
    same_sets = sum(set(a) == set(b) for a, b in zip(top_ip, top_l2))
    same_order = sum(np.array_equal(a, b) for a, b in zip(top_ip, top_l2))
    # the identity behind it: |a - b|^2 = 2 - 2 cos(a, b) for unit vectors
    d_ip, i_ip = ip.search(xq, K)
    d_l2, _ = l2.search(xq, K)
    identity_err = float(np.max(np.abs(np.sort(d_l2, axis=1) - np.sort(2 - 2 * d_ip, axis=1))))
    print(f"unit vectors: IP and L2 give the same top-10 set for {same_sets}/{len(xq)} questions, "
          f"the same order for {same_order}/{len(xq)}; max | |a-b|^2 - (2 - 2cos) | = {identity_err:.1e}")
    if same_order < len(xq):
        ties = 0
        for a, b, d in zip(top_ip, top_l2, d_ip):
            if not np.array_equal(a, b):
                ties += len(set(np.round(d, 5))) < len(d)
        print(f"  order differences: {len(xq) - same_order}, of which {ties} have tied scores in the top 10")

    # rescale 20% of the vectors by random factors in [0.5, 2]: no longer unit vectors
    rng = np.random.default_rng(0)
    scaled = xb.copy()
    pick = rng.choice(len(xb), size=max(1, len(xb) // 5), replace=False)
    factors = rng.uniform(0.5, 2.0, len(pick)).astype("float32")
    scaled[pick] *= factors[:, None]
    result = {"unit": {"same_set": same_sets, "same_order": same_order, "identity_max_err": identity_err}}
    for name, index in (("cosine", None), ("inner product", faiss.IndexFlatIP(xb.shape[1])),
                        ("L2", faiss.IndexFlatL2(xb.shape[1]))):
        if index is None:  # cosine = inner product on re-normalised vectors
            index = faiss.IndexFlatIP(xb.shape[1])
            normed = scaled.copy(); faiss.normalize_L2(normed); index.add(normed)
        else:
            index.add(scaled)
        found = search_all(index, xq, k)
        inflated = float(np.mean(np.isin(found[:, :K], pick[factors > 1])))
        shrunk = float(np.mean(np.isin(found[:, :K], pick[factors < 1])))
        g = golden_metrics(found, questions, chunks)
        result[name] = {**g, "share_top10_inflated": round(inflated, 3), "share_top10_shrunk": round(shrunk, 3)}
        print(f"scaled 20%: {name:13} golden R@8 {g['R@8']:.3f} ansR@20 {g['ansR@20']:.3f}  "
              f"top-10 share: inflated {inflated:.0%}, shrunk {shrunk:.0%} "
              f"(they are {np.mean(factors > 1) * 20:.0f}% / {np.mean(factors < 1) * 20:.0f}% of the corpus)")
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def quant() -> None:
    xb, xq, questions, chunks = load()
    n, d = xb.shape
    k = min(20, n)
    flat = faiss.IndexFlatIP(d); flat.add(xb)
    truth = search_all(flat, xq, K)
    rows = []

    def add(name: str, found: np.ndarray, code_bytes: float) -> None:
        g = golden_metrics(found, questions, chunks)
        rows.append({"index": name, "bytes_per_vector": code_bytes, "recall@10": round(ann_recall(found, truth), 4), **g})

    add("float32 (flat)", search_all(flat, xq, k), d * 4)
    for name, qtype, nbytes in (("fp16", faiss.ScalarQuantizer.QT_fp16, d * 2),
                                ("int8 (SQ8)", faiss.ScalarQuantizer.QT_8bit, d)):
        sq = faiss.IndexScalarQuantizer(d, qtype, faiss.METRIC_INNER_PRODUCT)
        sq.train(xb); sq.add(xb)
        add(name, search_all(sq, xq, k), nbytes)
    # binary: one sign bit per dimension, Hamming distance
    bits_b = np.packbits(xb > 0, axis=1)
    bits_q = np.packbits(xq > 0, axis=1)
    binary = faiss.IndexBinaryFlat(d)
    binary.add(bits_b)
    add("binary (sign bits)", binary.search(bits_q, k)[1], d / 8)
    rescore = max(k, n // 3)  # top 100 would be the whole 97-vector corpus, i.e. exact
    cand = binary.search(bits_q, rescore)[1]
    rescored = np.array([c[np.argsort(-(xb[c] @ q))][:k] for c, q in zip(cand, xq)])
    add(f"binary + float re-score of top {rescore}", rescored, d / 8)
    (OUT / "quantization.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"N={n} real vectors, {len(xq)} questions (golden metrics on {rows[0]['n']} answerable)")
    print(f"{'index':34}{'bytes/vec':>10}{'recall@10':>10}{'R@8':>7}{'ansR@20':>9}")
    for r in rows:
        print(f"{r['index']:34}{r['bytes_per_vector']:>10.0f}{r['recall@10']:>10.3f}{r['R@8']:>7.3f}{r['ansR@20']:>9.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scale")
    s.add_argument("--n", type=int, default=500_000)
    s.add_argument("--sigma", type=float, nargs="+", default=[0.05, 0.02])
    sub.add_parser("metrics")
    sub.add_parser("quant")
    args = parser.parse_args()
    if args.cmd == "scale":
        scale(args.n, args.sigma)
    elif args.cmd == "metrics":
        metrics()
    else:
        quant()


if __name__ == "__main__":
    main()
