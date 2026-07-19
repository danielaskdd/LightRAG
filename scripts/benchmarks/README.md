# shared_storage Manager-RPC benchmarks

Manual, threshold-gated benchmarks that gate two decisions from the Manager-proxy
RPC optimization plan. **Not part of CI** — they spin up a real
`multiprocessing.Manager` and are run by hand, with results recorded in the
driving issue/PR.

| Script | Plan stage | Decides |
|---|---|---|
| `rpc_profiler.py` | attribution | WHERE the remaining Manager RPCs concentrate in a real run — run this FIRST to pick the next target with data |
| `bench_manager_rpc.py` | 3a + 3b layer 1 | queue-stats aggregation structure; pure per-op IPC cost of the four log-write forms |
| `bench_log_replay.py` | 3b layer 2 | end-to-end effect of the log-write forms on a **captured** pipeline trace — the actual PR-C / reopen-#3432 gate |

## Picking the next target: profile first

After #3433/#3434 landed a real ~10% end-to-end win (60s → 54s ingest) purely by
cutting RPC counts, the next target should be chosen from a real-run attribution,
not guessed. `rpc_profiler.py` monkeypatches `BaseProxy._callmethod`/`_incref` in
each worker and dumps a per-pid count of Manager round trips by method and by
nearest `lightrag/` call site. Enable it (see the module docstring — gunicorn
`--preload`, `usercustomize`, or an in-process driver), reproduce the ingest,
signal `kill -USR1 <pid>` after the batch (or let it dump at exit), then:

```bash
python scripts/benchmarks/rpc_profiler.py report /tmp/rpcprof
```

A synthetic mini-run already shows the keyed-lock machinery (acquire/release plus
its registry `get`/`__setitem__`/`pop`, ~18–20 round trips per lock cycle) as the
dominant category — the real profile confirms whether that, or the hot-path log
writes, is the bigger lever before committing to either.

## Preset thresholds (fixed before running)

- **3a**: a candidate structure passes only if its `/health` sweep median
  wall-clock improves **≥ 20%** over the current form **and** marshals
  **≤ 2×** the bytes.
- **3b**: a form passes only if its end-to-end replay median improves **≥ 5%**
  over the baseline. Layer 1 is reported raw (mechanism only); the gate is a
  **layer-2** judgement on a real trace and is review-applied, never automatic.

## Why two layers for 3b

Layer 1 is a tight loop of pure writes — it bounds the per-op IPC delta but
cannot represent end-to-end value, because real log writes are interleaved with
LLM/embedding/graph work that dwarfs a per-write µs/ms delta. #3432 was closed as
"no significant positive impact" precisely because that end-to-end reality
swamped the per-write saving; its measurement also came from too small a run
(854 nodes / 1225 edges → only 218 history lines). Layer 2 replays a real trace
with (a) each event classified as `lockfree` / `lockable` / `coordinated` (the
lock-free forms drop the lock **only** for `lockable` events) and (b) real
inter-write gaps honored, so the saving is measured against realistic wall time.

**A real `--trace` capture is required for the actual decision.** `--synthetic`
only exercises the machinery and must never drive a decision (the fixed 5 ms
synthetic gap is far smaller than real LLM latency, so it overstates the win).

## Reproduce

```bash
# 3a + 3b layer 1
python scripts/benchmarks/bench_manager_rpc.py \
    --workers 4 --queues 6 --ops 2000 --warmup 200 --repeats 8

# 3b layer 2 — real decision needs a captured trace (see bench_log_replay.py docstring)
python scripts/benchmarks/bench_log_replay.py --trace trace.jsonl --repeats 5
# machinery check only (NOT a decision):
python scripts/benchmarks/bench_log_replay.py --synthetic --events 4000 --repeats 5
```

## Sample run (mechanism reference, not a verdict)

Machine: Intel Core i9-9980HK @ 2.40GHz, 16 logical cores, macOS; CPython 3.12.11.
Command: `bench_manager_rpc.py --workers 4 --queues 6 --ops 2000 --warmup 200 --repeats 8`.

**3a — queue-stats `/health` sweep**

| structure | median (ms) | vs A | bytes vs A | verdict |
|---|---|---|---|---|
| A `keys()`+`get()` per queue | 1.088 | — | 1.00× | baseline |
| B `copy()` per queue | 0.444 | +59% | 5.90× | **FAIL** (bytes) |
| C one `copy()` per sweep | 0.073 | **+93%** | 0.98× | **PASS** |

→ Structure **C** (one snapshot for the whole sweep, local group-by) is the
clear candidate: it beats the 20% wall-clock bar by a wide margin without
inflating marshaled bytes. B's per-queue full copy blows the byte budget as Q
grows. Implementing C means having `/health` aggregate all queues from a single
snapshot instead of calling `aggregate_queue_stats` per queue.

**3b layer 1 — log-write per-op latency (pure IPC, tight loop)**

| form | median (µs) | vs A | notes |
|---|---|---|---|
| A baseline (lock + setitem + getitem + append) | 1893 | — | re-materializes the ListProxy each write |
| B #3432 (no lock, 3 RPC) | 807 | +57% | removing the lock is the big lever |
| C writer (lock, cached handle, 2 RPC) | 986 | +48% | cached handle avoids proxy re-materialization |
| D both (no lock, cached handle) | 56 | +97% | — |

→ The keyed-lock cycle (~1 ms) dominates a single log write, and fetching
`history_messages` fresh each time is costly because the returned ListProxy is
re-materialized (an incref round-trip) per write. **This is layer 1 only** — the
tight-loop upper bound. Whether any of it matters end-to-end is decided by layer
2 on a real trace; do not read a PR-C verdict from this table.
