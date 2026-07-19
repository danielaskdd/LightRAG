#!/usr/bin/env python3
"""Manager-proxy RPC micro-benchmarks for the shared_storage optimizations.

Manual, threshold-gated, NOT part of CI. Produces the numbers that gate two
decisions from the optimization plan:

  3a  queue-stats aggregation structure (A vs B vs C) for a /health sweep.
  3b  log-path write forms (A/B/C/D) — LAYER 1 ONLY (pure Manager IPC).

IMPORTANT — what this does and does NOT decide:

  * Layer 1 here measures the *pure IPC* cost of each write form under a real
    Manager + real keyed pipeline_status lock. It explains the mechanism and
    bounds the per-op difference. It does NOT, by itself, justify PR-C.
  * The PR-C go/no-go is a LAYER 2 question — replay a real pipeline trace with
    each log event tagged (1) already-lockfree, (2) lockable pure log, or
    (3) shares a critical section with coordination state and must keep the
    lock. Only events of type (2) may drop the lock in the B/D replay. Layer 2
    needs a captured trace and lives in bench_log_replay.py (see its docstring).

Preset thresholds (fixed BEFORE running, per the plan; edit before a run only
with review, never after seeing results):

  3a  a candidate structure must, versus the current A:
        - /health sweep median wall-clock improvement >= 20%
        - marshaled-bytes per sweep <= 2x current
      (Manager CPU is not directly observable from the client; wall-clock and
      bytes are its proxies here.)
  3b  layer-1 is reported raw. The plan's >=5% end-to-end gate is a layer-2
      judgement and is intentionally NOT evaluated here.

Usage:
    python scripts/benchmarks/bench_manager_rpc.py \
        --workers 4 --queues 6 --ops 3000 --warmup 300 --repeats 10

Record in the PR/issue: the exact command, `python --version`, machine specs,
raw output, and the summary tables.
"""

from __future__ import annotations

import argparse
import asyncio
import pickle
import statistics
import sys
import time
from dataclasses import dataclass, field

from lightrag.kg.shared_storage import (
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_pipeline_status,
    initialize_share_data,
)

KEY_SEP = "\x1f"
_WS = "bench"


# ---------------------------------------------------------------------------
# stats helpers
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    label: str
    per_op_us: list[float] = field(default_factory=list)
    sweep_ms: list[float] = field(default_factory=list)
    bytes_per_unit: int = 0

    def op_stats(self) -> dict[str, float]:
        xs = sorted(self.per_op_us)
        return {
            "median_us": statistics.median(xs),
            "p95_us": xs[min(len(xs) - 1, int(len(xs) * 0.95))],
            "p99_us": xs[min(len(xs) - 1, int(len(xs) * 0.99))],
            "stdev_us": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        }

    def sweep_stats(self) -> dict[str, float]:
        xs = sorted(self.sweep_ms)
        return {
            "median_ms": statistics.median(xs),
            "p95_ms": xs[min(len(xs) - 1, int(len(xs) * 0.95))],
            "p99_ms": xs[min(len(xs) - 1, int(len(xs) * 0.99))],
            "stdev_ms": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
        }


def _pct_delta(candidate: float, baseline: float) -> float:
    """Percent improvement of candidate over baseline (positive = faster)."""
    if baseline == 0:
        return 0.0
    return (baseline - candidate) / baseline * 100.0


# ---------------------------------------------------------------------------
# 3b layer 1: log-path write forms
# ---------------------------------------------------------------------------


async def _reset_history(ps) -> None:
    # In-place clear preserves the ListProxy identity (the cached-handle forms
    # depend on it), matching the production del [:] pattern.
    ps["history_messages"][:] = []


async def bench_log_path(ops: int, warmup: int, repeats: int) -> dict[str, Sample]:
    ps = await get_namespace_data("pipeline_status", workspace=_WS)
    lock = get_namespace_lock("pipeline_status", workspace=_WS)
    msg = "x" * 80  # representative status line

    async def form_A():  # baseline: lock + setitem + getitem + append
        async with lock:
            ps["latest_message"] = msg
            ps["history_messages"].append(msg)

    async def form_B():  # #3432: no lock, setitem + get + extend
        ps["latest_message"] = msg
        h = ps.get("history_messages")
        h.extend((msg,))

    handle = ps["history_messages"]  # cached once, as a writer would

    async def form_C():  # writer: lock + setitem + cached-handle extend
        async with lock:
            ps["latest_message"] = msg
            handle.extend((msg,))

    async def form_D():  # combined: no lock + cached-handle extend
        ps["latest_message"] = msg
        handle.extend((msg,))

    forms = {
        "A_baseline": form_A,
        "B_lockfree": form_B,
        "C_writer": form_C,
        "D_both": form_D,
    }
    samples = {name: Sample(name) for name in forms}

    for name, fn in forms.items():
        for _ in range(warmup):
            await fn()
        await _reset_history(ps)
        for _ in range(repeats):
            for _ in range(ops):
                t0 = time.perf_counter()
                await fn()
                samples[name].per_op_us.append((time.perf_counter() - t0) * 1e6)
            await _reset_history(ps)
    return samples


# ---------------------------------------------------------------------------
# 3a: queue-stats aggregation structures, whole /health sweep
# ---------------------------------------------------------------------------


def _snapshot(pid: int, q: str) -> dict:
    return {
        "pid": pid,
        "queue": q,
        "pending": 3,
        "active": 1,
        "updated_at": time.time(),
    }


async def bench_queue_stats(
    queues: int, workers: int, warmup: int, repeats: int
) -> dict[str, Sample]:
    ns = await get_namespace_data("bench_queue_stats", workspace=_WS)
    queue_names = [f"q{i}" for i in range(queues)]
    for qi, q in enumerate(queue_names):
        for w in range(workers):
            ns[f"{q}{KEY_SEP}{1000 + qi * 100 + w}"] = _snapshot(1000 + qi * 100 + w, q)

    # One "sweep" == aggregate every queue once (what /health does per request).
    def sweep_A():  # current: keys() once, then get() each matching key, per queue
        for q in queue_names:
            prefix = f"{q}{KEY_SEP}"
            for k in [k for k in ns.keys() if k.startswith(prefix)]:
                _ = dict(ns.get(k))

    def sweep_B():  # per-queue full copy(), then local filter
        for q in queue_names:
            prefix = f"{q}{KEY_SEP}"
            snap = ns.copy()
            _ = {k: v for k, v in snap.items() if k.startswith(prefix)}

    def sweep_C():  # ONE copy() for the whole sweep, local group-by
        snap = ns.copy()
        buckets: dict[str, dict] = {q: {} for q in queue_names}
        for k, v in snap.items():
            q = k.split(KEY_SEP, 1)[0]
            if q in buckets:
                buckets[q][k] = v

    sweeps = {
        "A_keys_get": sweep_A,
        "B_copy_per_queue": sweep_B,
        "C_copy_once": sweep_C,
    }
    samples = {name: Sample(name) for name in sweeps}

    # Marshaled-bytes estimate per sweep (payload the Manager pickles back).
    full = {k: ns.get(k) for k in ns.keys()}
    one_value_bytes = len(pickle.dumps(next(iter(full.values()))))
    full_bytes = len(pickle.dumps(full))
    samples["A_keys_get"].bytes_per_unit = one_value_bytes * queues * workers
    samples["B_copy_per_queue"].bytes_per_unit = full_bytes * queues
    samples["C_copy_once"].bytes_per_unit = full_bytes

    for name, fn in sweeps.items():
        for _ in range(warmup):
            fn()
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            samples[name].sweep_ms.append((time.perf_counter() - t0) * 1e3)
    return samples


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def report_log_path(samples: dict[str, Sample]) -> None:
    base = samples["A_baseline"].op_stats()["median_us"]
    print("\n=== 3b LAYER 1: log-path per-op latency (pure Manager IPC) ===")
    print(
        f"{'form':<14}{'median_us':>12}{'p95_us':>10}{'p99_us':>10}{'stdev':>10}{'vs A':>10}"
    )
    for name in ("A_baseline", "B_lockfree", "C_writer", "D_both"):
        s = samples[name].op_stats()
        print(
            f"{name:<14}{s['median_us']:>12.2f}{s['p95_us']:>10.2f}"
            f"{s['p99_us']:>10.2f}{s['stdev_us']:>10.2f}"
            f"{_pct_delta(s['median_us'], base):>9.1f}%"
        )
    print(
        "NOTE: layer-1 only. PR-C go/no-go is a LAYER-2 trace-replay decision "
        "(>=5% end-to-end), not decided here."
    )


def report_queue_stats(samples: dict[str, Sample]) -> None:
    base = samples["A_keys_get"].sweep_stats()["median_ms"]
    base_bytes = samples["A_keys_get"].bytes_per_unit
    print("\n=== 3a: queue-stats /health sweep (aggregate all queues once) ===")
    print(
        f"{'structure':<18}{'median_ms':>12}{'p95_ms':>10}{'p99_ms':>10}{'vs A':>9}{'bytes':>12}{'bytes vs A':>12}{'verdict':>9}"
    )
    for name in ("A_keys_get", "B_copy_per_queue", "C_copy_once"):
        s = samples[name].sweep_stats()
        b = samples[name].bytes_per_unit
        impr = _pct_delta(s["median_ms"], base)
        bytes_ratio = (b / base_bytes) if base_bytes else 1.0
        if name == "A_keys_get":
            verdict = "baseline"
        else:
            verdict = "PASS" if (impr >= 20.0 and bytes_ratio <= 2.0) else "FAIL"
        print(
            f"{name:<18}{s['median_ms']:>12.4f}{s['p95_ms']:>10.4f}{s['p99_ms']:>10.4f}"
            f"{impr:>8.1f}%{b:>12}{bytes_ratio:>11.2f}x{verdict:>9}"
        )
    print(
        "THRESHOLD: candidate PASSES only if median improvement >=20% AND bytes <=2x A."
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4, help="workers per queue (W)")
    ap.add_argument("--queues", type=int, default=6, help="number of queues (Q)")
    ap.add_argument("--ops", type=int, default=3000, help="log ops per repeat (3b)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    print(
        f"python {sys.version.split()[0]} | workers={args.workers} queues={args.queues} "
        f"ops={args.ops} warmup={args.warmup} repeats={args.repeats}"
    )

    finalize_share_data()
    initialize_share_data(2)  # real Manager
    try:
        await initialize_pipeline_status(workspace=_WS)
        log_samples = await bench_log_path(args.ops, args.warmup, args.repeats)
        report_log_path(log_samples)

        qs_samples = await bench_queue_stats(
            args.queues, args.workers, args.warmup, args.repeats
        )
        report_queue_stats(qs_samples)
    finally:
        finalize_share_data()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
