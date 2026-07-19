#!/usr/bin/env python3
"""Manager-RPC attribution profiler for a real (workers>1) LightRAG run.

Counts every ``multiprocessing.managers.BaseProxy`` round trip — method calls
(``_callmethod``) plus proxy-creation churn (``_incref``, the hidden cost of a
call returning a fresh ListProxy/DictProxy, e.g. a per-write
``pipeline_status["history_messages"]`` re-materialization) — and attributes each
to the nearest ``lightrag/`` call site. Use it to see WHERE the remaining Manager
RPCs concentrate after PRs #3433/#3434, so the next optimization is data-chosen.

WHY the raw proxy layer: it is the single chokepoint every proxy op funnels
through, so attribution is exact and does not depend on knowing the hot paths in
advance.

--------------------------------------------------------------------------------
ENABLE (must be installed in the process that HOLDS the proxies — i.e. workers>1).
lightrag-gunicorn already sets preload_app=True and creates the Manager in the
master before forking, so patching the master's BaseProxy class makes every
worker inherit it via fork.

Option 1 — sitecustomize hook (RECOMMENDED; zero code edit, works with
lightrag-gunicorn). sitecustomize runs at interpreter startup, before the master
creates the Manager:
    mkdir -p /tmp/rpcprof_hook
    cp scripts/benchmarks/rpc_profiler.py /tmp/rpcprof_hook/
    printf 'import rpc_profiler\n' > /tmp/rpcprof_hook/sitecustomize.py
    rm -rf /tmp/rpcprof && mkdir -p /tmp/rpcprof
    LIGHTRAG_RPC_PROFILE=1 LIGHTRAG_RPC_PROFILE_DIR=/tmp/rpcprof \
        LIGHTRAG_RPC_PROFILE_EVERY=200000 PYTHONPATH=/tmp/rpcprof_hook \
        lightrag-gunicorn --workers 4
    (Use sitecustomize, NOT usercustomize: virtualenvs commonly disable user-site
    so usercustomize never fires. If the env already ships a sitecustomize, this
    one shadows it for the run — fine for a temporary profile.)

Option 2 — explicit env-gated import in lightrag/api/gunicorn_config.py (add near
the top; runs in the master before initialize_share_data):
    if os.environ.get("LIGHTRAG_RPC_PROFILE"):
        import rpc_profiler  # noqa: F401  (needs scripts/benchmarks on PYTHONPATH)
    then launch with the same LIGHTRAG_RPC_PROFILE* env + PYTHONPATH as Option 1.

Option 3 — in-process reproduction (a driver script using initialize_share_data(N)):
    import rpc_profiler; rpc_profiler.install()  # then run the workload

--------------------------------------------------------------------------------
DUMP: each process (re)writes ``<DIR>/rpc_profile_<pid>.jsonl`` atomically on
  - normal exit (atexit — runs on gunicorn's graceful worker shutdown), and
  - every LIGHTRAG_RPC_PROFILE_EVERY records (default 1,000,000) while running.

  Signals are deliberately NOT used: gunicorn workers reinstall their own signal
  handlers after fork, which would clobber ours. The periodic flush makes the
  latest counts available on disk without stopping the server — after an ingest
  batch, either read the already-flushed file or stop the server (atexit writes
  the final tail). Set EVERY smaller (e.g. 100000) to flush more often.

REPORT (merge all per-pid files into one ranked table):
    python scripts/benchmarks/rpc_profiler.py report /tmp/rpcprof

TUNING: LIGHTRAG_RPC_PROFILE_SAMPLE=N samples the (expensive) call-site walk on
1/N calls while still counting EVERY call per method — set e.g. 20 on a run that
makes millions of RPCs to keep overhead down. Totals stay exact; per-site numbers
are scaled by N in the report.
"""

from __future__ import annotations

import atexit
import itertools
import json
import os
import sys
import threading
from collections import Counter

_method_counts: Counter = Counter()  # methodname -> exact total across all calls
_site_counts: Counter = Counter()  # (methodname, "file:line") -> sampled count
_lock = threading.Lock()
_dump_lock = threading.Lock()
_counter = itertools.count()
_SAMPLE = max(1, int(os.environ.get("LIGHTRAG_RPC_PROFILE_SAMPLE", "1")))
_EVERY = max(0, int(os.environ.get("LIGHTRAG_RPC_PROFILE_EVERY", "1000000")))
_installed = False


def _callsite() -> str:
    """Nearest lightrag frame that caused this RPC.

    Prefers a caller OUTSIDE shared_storage.py (the real originator such as
    operate.py / pipeline.py); falls back to the shared_storage function when the
    whole visible stack is internal plumbing.
    """
    try:
        f = sys._getframe(3)  # _callsite -> wrapper -> proxy method -> caller
    except ValueError:
        return "?"
    fallback = None  # a lightrag/shared_storage.py frame (internal plumbing)
    outer = None  # first non-multiprocessing, non-profiler frame (last resort)
    depth = 0
    while f is not None and depth < 40:
        fn = f.f_code.co_filename
        base = os.path.basename(fn)
        if "lightrag" in fn:
            tag = f"{base}:{f.f_lineno}"
            if base != "shared_storage.py":
                return tag
            if fallback is None:
                fallback = tag
        elif (
            outer is None and "multiprocessing" not in fn and base != "rpc_profiler.py"
        ):
            outer = f"{base}:{f.f_lineno}"
        f = f.f_back
        depth += 1
    return fallback or outer or "?"


def _record(methodname: str) -> None:
    n = next(_counter)
    sampled = n % _SAMPLE == 0
    site = _callsite() if sampled else None
    with _lock:
        _method_counts[methodname] += 1
        if sampled:
            _site_counts[(methodname, site)] += 1
    if _EVERY and n and n % _EVERY == 0:
        dump()


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True

    from multiprocessing.managers import BaseProxy

    orig_call = BaseProxy._callmethod
    orig_incref = BaseProxy._incref

    def call(self, methodname, args=(), kwds={}):
        _record(methodname)
        return orig_call(self, methodname, args, kwds)

    def incref(self):
        _record("#incref")
        return orig_incref(self)

    # NOTE: _decref is a 6-arg staticmethod registered as a Finalize callback, not
    # an instance method — wrapping it is fragile and it only reports cleanup, so
    # we count creation (#incref) as the proxy-churn signal and leave _decref be.
    BaseProxy._callmethod = call
    BaseProxy._incref = incref

    atexit.register(dump)


def dump() -> None:
    """Atomically (re)write this process's per-pid dump. Safe to call repeatedly
    (periodic flush) and from any thread; the last writer wins."""
    out_dir = os.environ.get("LIGHTRAG_RPC_PROFILE_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"rpc_profile_{os.getpid()}.jsonl")
    with _lock:
        methods = dict(_method_counts)
        sites = {f"{m}\t{s}": c for (m, s), c in _site_counts.items()}
    payload = (
        json.dumps(
            {
                "pid": os.getpid(),
                "sample": _SAMPLE,
                "type": "methods",
                "counts": methods,
            }
        )
        + "\n"
        + json.dumps(
            {"pid": os.getpid(), "sample": _SAMPLE, "type": "sites", "counts": sites}
        )
        + "\n"
    )
    with _dump_lock:  # serialize concurrent flushes; write-then-rename for atomicity
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            f.write(payload)
        os.replace(tmp, path)


# ---------------------------------------------------------------------------
# report: merge per-pid dumps into a ranked table
# ---------------------------------------------------------------------------


def report(directory: str, top: int = 30) -> int:
    method_total: Counter = Counter()
    site_total: Counter = Counter()
    pids = 0
    for name in sorted(os.listdir(directory)):
        if not (name.startswith("rpc_profile_") and name.endswith(".jsonl")):
            continue
        pids += 1
        with open(os.path.join(directory, name)) as f:
            for line in f:
                rec = json.loads(line)
                sample = rec.get("sample", 1)
                if rec["type"] == "methods":
                    for m, c in rec["counts"].items():
                        method_total[m] += c
                else:  # sites are sampled; scale back up
                    for key, c in rec["counts"].items():
                        site_total[key] += c * sample

    grand = sum(method_total.values())
    print(
        f"merged {pids} process dump(s) | total Manager round-trips (exact): {grand:,}\n"
    )

    print("=== by method (exact totals) ===")
    print(f"{'method':<22}{'count':>14}{'%':>8}")
    for m, c in method_total.most_common():
        print(f"{m:<22}{c:>14,}{c / grand * 100:>7.1f}%")

    print("\n=== by call site (sampled x scaled; method | file:line) ===")
    print(f"{'count~':>14}{'%~':>8}  method | site")
    for key, c in site_total.most_common(top):
        m, _, s = key.partition("\t")
        print(f"{c:>14,}{c / grand * 100:>7.1f}%  {m} | {s}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "report":
        raise SystemExit(report(sys.argv[2]))
    print(__doc__)
    raise SystemExit(2)

if os.environ.get("LIGHTRAG_RPC_PROFILE"):
    install()
