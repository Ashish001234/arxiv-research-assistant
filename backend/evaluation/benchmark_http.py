"""
HTTP-based latency benchmark.

Hits a running FastAPI server (no in-process model loading), so the laptop's
RAM only holds a single copy of the indices. Measures end-to-end wall-clock
latency exactly as a real user would experience.

Usage (two terminals):
    # Terminal 1 — start the server
    py -3.11 -m uvicorn backend.api:app

    # Terminal 2 — run this benchmark
    py -3.11 -m backend.evaluation.benchmark_http
"""
import csv
import json
import statistics
import sys
import time
import urllib.parse
from http.client import HTTPConnection
from pathlib import Path

# Force IPv4 + persistent connection to eliminate any IPv6 fallback delay.
HOST = "127.0.0.1"
PORT = 8000
TEST_QUERIES_PATH = Path(__file__).parent / "test_queries.json"
RESULTS_JSON = Path(__file__).parent / "latency_results.json"
RESULTS_CSV = Path(__file__).parent / "latency_raw.csv"

METHODS = ["bm25", "dense", "hybrid", "hybrid_rerank"]

# Single persistent HTTP connection reused across all requests.
_conn = HTTPConnection(HOST, PORT, timeout=120)


def now_ms() -> float:
    return time.perf_counter_ns() / 1_000_000.0


def http_get_ms(path: str) -> float:
    t0 = now_ms()
    _conn.request("GET", path)
    resp = _conn.getresponse()
    _ = resp.read()
    return now_ms() - t0


def http_post_ms(path: str, body: dict) -> float:
    data = json.dumps(body).encode("utf-8")
    t0 = now_ms()
    _conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
    resp = _conn.getresponse()
    _ = resp.read()
    return now_ms() - t0


def check_server() -> bool:
    try:
        ms = http_get_ms("/api/stats")
        # Also fetch + parse to verify it's actually our server
        _conn.request("GET", "/api/stats")
        resp = _conn.getresponse()
        data = json.loads(resp.read())
        print(f"  Server is up at {HOST}:{PORT}. "
              f"Total papers: {data.get('total_papers'):,}. "
              f"First request: {ms:.2f}ms")
        return True
    except Exception as e:
        print(f"\nERROR: cannot reach http://{HOST}:{PORT} — is the server running?")
        print(f"  Start it in another terminal:  py -3.11 -m uvicorn backend.api:app")
        print(f"  Underlying error: {e}")
        return False


def stats(samples: list[float], trim_pct: float = 0.10) -> dict:
    if not samples:
        return {"n": 0}
    samples_sorted = sorted(samples)
    if len(samples) >= 5 and trim_pct > 0:
        k = max(1, int(len(samples) * trim_pct))
        trimmed = samples_sorted[k:-k] if len(samples) - 2 * k > 0 else samples_sorted
    else:
        trimmed = samples_sorted
    return {
        "n_total": len(samples),
        "n_trimmed": len(trimmed),
        "mean_ms": round(statistics.mean(trimmed), 2),
        "median_ms": round(statistics.median(trimmed), 2),
        "stdev_ms": round(statistics.stdev(trimmed), 2) if len(trimmed) > 1 else 0.0,
        "p95_ms": round(samples_sorted[int(0.95 * (len(samples_sorted) - 1))], 2),
        "min_ms": round(min(trimmed), 2),
        "max_ms": round(max(trimmed), 2),
    }


def benchmark_search(queries: list[str]) -> list[dict]:
    """For each (query, method) pair: first call = miss, second call = hit."""
    raw = []
    print(f"\nBenchmarking /api/search ({len(queries)} queries x {len(METHODS)} methods x 2 calls)...")

    # MISS pass: each (query, method) is fresh in the server cache
    print("\n  --- MISS pass (full pipeline) ---")
    for q in queries:
        for method in METHODS:
            path = f"/api/search?q={urllib.parse.quote(q)}&method={method}&top_k=10"
            try:
                ms = http_get_ms(path)
                raw.append({"query": q, "stage": f"search_{method}_miss", "ms": round(ms, 2)})
                print(f"    {method:<14} | {q[:42]:<42} | {ms:>9.2f} ms")
            except Exception as e:
                print(f"    {method:<14} | {q[:42]:<42} | ERROR: {e}")

    # HIT pass: same URL again — should hit the server-side cache
    print("\n  --- HIT pass (cache hit) ---")
    for q in queries:
        for method in METHODS:
            path = f"/api/search?q={urllib.parse.quote(q)}&method={method}&top_k=10"
            try:
                ms = http_get_ms(path)
                raw.append({"query": q, "stage": f"search_{method}_hit", "ms": round(ms, 2)})
                print(f"    {method:<14} | {q[:42]:<42} | {ms:>9.2f} ms")
            except Exception as e:
                print(f"    {method:<14} | {q[:42]:<42} | ERROR: {e}")

    return raw


def benchmark_ask(queries: list[str]) -> list[dict]:
    raw = []
    print(f"\nBenchmarking /api/ask ({len(queries)} queries x 2 calls)...")

    print("\n  --- MISS pass (full pipeline + LLM) ---")
    for q in queries:
        try:
            ms = http_post_ms("/api/ask", {"query": q})
            raw.append({"query": q, "stage": "ask_miss", "ms": round(ms, 2)})
            print(f"    {q[:60]:<60} | {ms:>9.2f} ms")
        except Exception as e:
            print(f"    {q[:60]:<60} | ERROR: {e}")

    print("\n  --- HIT pass (cache hit) ---")
    for q in queries:
        try:
            ms = http_post_ms("/api/ask", {"query": q})
            raw.append({"query": q, "stage": "ask_hit", "ms": round(ms, 2)})
            print(f"    {q[:60]:<60} | {ms:>9.2f} ms")
        except Exception as e:
            print(f"    {q[:60]:<60} | ERROR: {e}")

    return raw


def main():
    print("=" * 86)
    print("ArXiv RAG — HTTP Latency Benchmark")
    print("=" * 86)
    if not check_server():
        sys.exit(1)

    with open(TEST_QUERIES_PATH) as f:
        test_queries = json.load(f)
    queries = [item["query"] for item in test_queries]
    print(f"  Loaded {len(queries)} queries.")

    # Warm-up: hit the server with one throwaway request per method.
    print("\nWarm-up (one request per method)...")
    try:
        for m in METHODS:
            ms = http_get_ms(f"/api/search?q=warmup&method={m}&top_k=5")
            print(f"    {m:<14} {ms:>9.2f} ms")
        print("  Warm-up done.")
    except Exception as e:
        print(f"  Warm-up failed: {e}")

    raw_search = benchmark_search(queries)
    raw_ask = benchmark_ask(queries)
    raw_all = raw_search + raw_ask

    # Save raw CSV
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "stage", "ms"])
        writer.writeheader()
        writer.writerows(raw_all)
    print(f"\nWrote raw timings: {RESULTS_CSV}")

    # Aggregate
    by_stage: dict[str, list[float]] = {}
    for row in raw_all:
        by_stage.setdefault(row["stage"], []).append(row["ms"])
    aggregate = {stage: stats(samples) for stage, samples in by_stage.items()}

    summary = {
        "benchmark_type": "http_end_to_end",
        "n_queries": len(queries),
        "stage_stats": aggregate,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote aggregate stats: {RESULTS_JSON}")

    # Pretty table
    print("\n" + "=" * 92)
    print("LATENCY SUMMARY (end-to-end via HTTP, 10% trimmed; raw timings in latency_raw.csv)")
    print("=" * 92)
    print(f"{'Stage':<26} {'n':>4}  {'median':>10}  {'mean':>10}  {'p95':>10}  {'stdev':>10}")
    print("-" * 92)
    order = [
        "search_bm25_miss", "search_dense_miss", "search_hybrid_miss", "search_hybrid_rerank_miss",
        "ask_miss",
        "search_bm25_hit", "search_dense_hit", "search_hybrid_hit", "search_hybrid_rerank_hit",
        "ask_hit",
    ]
    for stage in order:
        s = aggregate.get(stage)
        if not s:
            continue
        n = s["n_trimmed"]
        print(f"{stage:<26} {n:>4}  {s['median_ms']:>8.2f}ms  "
              f"{s['mean_ms']:>8.2f}ms  {s['p95_ms']:>8.2f}ms  "
              f"{s['stdev_ms']:>8.2f}ms")
    print("=" * 92)

    # Derived stage estimates (subtraction)
    def med(stage):
        s = aggregate.get(stage)
        return s["median_ms"] if s else None

    print("\nDerived stage estimates (median, by subtraction):")
    bm25 = med("search_bm25_miss")
    dense = med("search_dense_miss")
    hybrid = med("search_hybrid_miss")
    hyb_rr = med("search_hybrid_rerank_miss")
    ask = med("ask_miss")

    if bm25 and dense and hybrid:
        rrf_est = max(0.0, hybrid - max(bm25, dense))
        print(f"  RRF overhead ~  hybrid - max(bm25, dense)            = {rrf_est:.2f} ms")
    if hybrid and hyb_rr:
        print(f"  Re-rank only ~  hybrid_rerank - hybrid                = {hyb_rr - hybrid:.2f} ms")
    if hyb_rr and ask:
        print(f"  LLM only     ~  ask - hybrid_rerank                   = {ask - hyb_rr:.2f} ms")


if __name__ == "__main__":
    main()
