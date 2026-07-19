#!/usr/bin/env python3
"""Layer-2 log-path replay — the trace-driven half of plan stage 3b.

Layer 1 (bench_manager_rpc.py) measures the pure per-op IPC cost of the four
log-write forms in a tight loop. That is NOT enough to decide PR-C / reopening
#3432, because in a real pipeline log writes are interleaved with LLM/embedding/
graph work that dwarfs a per-write µs/ms delta. This script answers the real
question: replay a representative trace and measure the END-TO-END effect.

Two things make the replay faithful (both were review requirements):

1. EVENT CLASSIFICATION. Every log event carries a ``lock_class``:
     "lockfree"     — already written outside pipeline_status_lock today.
     "lockable"     — pure status log currently under the lock; MAY drop it.
     "coordinated"  — shares a critical section with coordination state
                      (busy / cur_batch / owner tokens / trim); MUST keep the
                      lock in every form.
   The B/D (lock-free) forms drop the lock ONLY for "lockable" events. A/C keep
   the lock exactly where it is today. Collapsing all events to lock-free would
   overstate the win and would not correspond to a safe migration.

2. INTER-EVENT GAPS. Each event carries ``gap_ms`` — the wall time between it
   and the previous log write in the real run (LLM latency, IO, CPU work). The
   replay sleeps that gap, so a per-write saving is measured as a fraction of
   realistic end-to-end time, not of a synthetic tight loop.

TRACE CAPTURE (how to produce a real trace):
  Instrument the status-log call sites (or post-process a completed run's
  history_messages plus timestamps) to emit newline-delimited JSON, one object
  per log write:
      {"gap_ms": 12.4, "lock_class": "lockable", "msg_len": 80, "n_msgs": 1}
  gap_ms is time since the previous write; lock_class per the taxonomy above;
  n_msgs>1 for grouped writes (extend of several lines). Save as trace.jsonl and
  pass --trace trace.jsonl.

SYNTHETIC MODE (--synthetic) generates a trace from explicit parameters so the
harness is runnable/validatable WITHOUT a capture. Synthetic numbers must NEVER
be used for the actual decision — they only exercise the machinery. The real
PR-C / #3432 verdict requires --trace from a captured run.

Preset gate (fixed before running): a form PASSES only if its end-to-end median
improves >= 5% over A. Reported, not auto-applied — the decision is a review.

Usage:
    python scripts/benchmarks/bench_log_replay.py --trace trace.jsonl --repeats 5
    python scripts/benchmarks/bench_log_replay.py --synthetic --events 4000 --repeats 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass

from lightrag.kg.shared_storage import (
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_pipeline_status,
    initialize_share_data,
)

_WS = "bench"
_LOCK_CLASSES = ("lockfree", "lockable", "coordinated")


@dataclass
class Event:
    gap_ms: float
    lock_class: str
    msg_len: int = 80
    n_msgs: int = 1


def load_trace(path: str) -> list[Event]:
    events: list[Event] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            lc = d.get("lock_class", "lockable")
            if lc not in _LOCK_CLASSES:
                raise ValueError(f"bad lock_class {lc!r} (expected {_LOCK_CLASSES})")
            events.append(
                Event(
                    gap_ms=float(d.get("gap_ms", 0.0)),
                    lock_class=lc,
                    msg_len=int(d.get("msg_len", 80)),
                    n_msgs=int(d.get("n_msgs", 1)),
                )
            )
    if not events:
        raise ValueError(f"trace {path} is empty")
    return events


def synthetic_trace(events: int, seed_ratio=(0.15, 0.70, 0.15)) -> list[Event]:
    """A DELIBERATELY SYNTHETIC trace. Not representative — validation only.

    Distributes events across the three lock classes by ``seed_ratio`` and
    assigns a fixed gap. A real capture will differ in count, gaps, grouping and
    class mix, which is exactly why the real decision needs --trace.
    """
    lockfree, lockable, coordinated = seed_ratio
    out: list[Event] = []
    for i in range(events):
        r = (i % 100) / 100.0
        if r < lockfree:
            lc = "lockfree"
        elif r < lockfree + lockable:
            lc = "lockable"
        else:
            lc = "coordinated"
        # Fixed synthetic gap standing in for real inter-write work.
        out.append(Event(gap_ms=5.0, lock_class=lc))
    return out


async def replay(events: list[Event], form: str, honor_gaps: bool) -> float:
    """Replay the trace once under one write form; return end-to-end seconds.

    form in {A, B, C, D}:
      A: current — lock every lockable+coordinated write; 3 RPC (rematerialize).
      B: #3432   — drop lock for 'lockable'; 3 RPC.
      C: writer  — lock every lockable+coordinated write; 2 RPC (cached handle).
      D: both    — drop lock for 'lockable'; 2 RPC (cached handle).
    'coordinated' events ALWAYS take the lock; 'lockfree' events NEVER do.
    """
    ps = await get_namespace_data("pipeline_status", workspace=_WS)
    lock = get_namespace_lock("pipeline_status", workspace=_WS)
    handle = ps["history_messages"]
    cached = form in ("C", "D")
    drop_lockable = form in ("B", "D")

    def do_write(ev: Event) -> None:
        msg = "x" * ev.msg_len
        msgs = tuple(msg for _ in range(ev.n_msgs))
        ps["latest_message"] = msg
        if cached:
            handle.extend(msgs)
        else:
            ps["history_messages"].extend(msgs)

    t0 = time.perf_counter()
    for ev in events:
        if honor_gaps and ev.gap_ms > 0:
            await asyncio.sleep(ev.gap_ms / 1000.0)
        takes_lock = ev.lock_class == "coordinated" or (
            ev.lock_class == "lockable" and not drop_lockable
        )
        if takes_lock:
            async with lock:
                do_write(ev)
        else:
            do_write(ev)
        ps["history_messages"][:] = []  # keep the list bounded, identity intact
    return time.perf_counter() - t0


def _class_mix(events: list[Event]) -> str:
    n = len(events)
    counts = {c: 0 for c in _LOCK_CLASSES}
    for e in events:
        counts[e.lock_class] += 1
    return ", ".join(
        f"{c}={counts[c]}({counts[c] / n * 100:.0f}%)" for c in _LOCK_CLASSES
    )


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="path to a captured trace.jsonl")
    src.add_argument("--synthetic", action="store_true", help="generate a fake trace")
    ap.add_argument("--events", type=int, default=4000, help="synthetic event count")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--no-gaps",
        action="store_true",
        help="ignore inter-event gaps (measures the tight-loop upper bound only)",
    )
    args = ap.parse_args()

    if args.synthetic:
        events = synthetic_trace(args.events)
        print("!!! SYNTHETIC trace — validation only, NOT a basis for any decision")
    else:
        events = load_trace(args.trace)

    honor_gaps = not args.no_gaps
    print(
        f"python {sys.version.split()[0]} | events={len(events)} repeats={args.repeats} "
        f"honor_gaps={honor_gaps}"
    )
    print(f"class mix: {_class_mix(events)}")
    total_gap_ms = sum(e.gap_ms for e in events)
    print(f"trace models ~{total_gap_ms / 1000.0:.2f}s of inter-write work per replay")

    finalize_share_data()
    initialize_share_data(2)
    try:
        await initialize_pipeline_status(workspace=_WS)
        medians: dict[str, float] = {}
        for form in ("A", "B", "C", "D"):
            runs = [await replay(events, form, honor_gaps) for _ in range(args.repeats)]
            medians[form] = statistics.median(runs)
    finally:
        finalize_share_data()

    base = medians["A"]
    print("\n=== 3b LAYER 2: end-to-end replay ===")
    print(f"{'form':<10}{'median_s':>12}{'vs A':>10}{'gate(>=5%)':>12}")
    for form in ("A", "B", "C", "D"):
        impr = (base - medians[form]) / base * 100.0 if base else 0.0
        gate = "-" if form == "A" else ("PASS" if impr >= 5.0 else "FAIL")
        print(f"{form:<10}{medians[form]:>12.4f}{impr:>9.1f}%{gate:>12}")
    print(
        "\nDecision (per plan, review-gated): B/D pass -> consider reopening #3432 "
        "(migrate 'lockable' events only); only C passes -> implement PR-C; "
        "none pass -> close the log-path direction."
    )
    if args.synthetic:
        print(
            "Reminder: SYNTHETIC run — re-run with --trace from a real capture "
            "before deciding anything."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
