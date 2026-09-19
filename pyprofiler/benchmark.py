#!/usr/bin/env python3
"""
benchmark - a common tester for llama.cpp submissions.

You should provide your own `llama-server` and LLM to run. Point this harness at
the binary and the model; it starts the server, runs the suite, stops the
server, and prints two numbers:

    score       a single 0-100 quality mark
    throughput  decode tok/s

Pass --report for the full breakdown (per-section scores, load time, peak RSS,
time to first token (TTFT)).

Quality suite (50 items by default):
    20 x HellaSwag   ranked, scored by rectified Brier skill
    20 x ARC-Easy    ranked, scored by rectified Brier skill
    10 x IFEval-ish  generated, scored by fraction of constraints satisfied

Each item is worth at most 1 point; the total is rescaled to 100(%).
Brier skill: 1.0 = confident and correct, 0.0 = no better than guessing.

Usage
-----
    python benchmark.py --llama-server ./llama-server --model model.gguf

    # run with a detailed report
    python benchmark.py --llama-server ./llama-server --model model.gguf \
        --threads 1 --team t1 --report --out results-t1.json

The task data ships alongside this script in data/<split>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from bench.backend import ServerBackend
from bench.data import DEFAULT_DATA_DIR, load_ifeval, load_ranking
from bench.ifeval import IFEvalTask
from bench.ranking import ARCEasyTask, HellaSwagTask
from bench.report import build_report, print_report, print_summary
from bench.results import ItemResult
from bench.server import (ManagedServer, build_server_cmd, detect_mode,
                          ensure_no_server_running)
from bench.telemetry import Telemetry

HOST = "127.0.0.1"


# ===========================================================================
# Runner
# ===========================================================================

def run_suite(backend: ServerBackend, plan: list[tuple[Any, list[dict]]],
              item_timeout: float, verbose: bool = False) -> list[ItemResult]:
    results: list[ItemResult] = []
    for task, items in plan:
        if not items:
            continue
        print(f"\n[{task.name}] {len(items)} items", file=sys.stderr)
        for i, item in enumerate(items, 1):
            item_id = str(item.get("id", i))
            try:
                r = task.run(item, backend, timeout=item_timeout)
            except Exception as e:
                # A hung or broken submission costs one item, not the run.
                r = ItemResult(task=task.name, item_id=item_id, score=0.0,
                               raw_score=0.0, correct=False,
                               detail={"error": repr(e)[:200]}, error=repr(e)[:200])
            results.append(r)
            flag = "!" if r.error else " "
            print(f"  {i:>3}/{len(items)} {flag} score={r.score:5.3f} "
                  f"({r.wall_s:5.2f}s)", file=sys.stderr)
            if verbose:
                print(f"        {json.dumps(r.detail, default=str)[:300]}",
                      file=sys.stderr)
    return results


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    srv = ap.add_argument_group("server (started by this script)")
    srv.add_argument("--llama-server", required=True,
                     help="path to the team's llama-server binary")
    srv.add_argument("--model", required=True, help="path to the .gguf model")
    srv.add_argument("--threads", type=int, default=None,
                     help="max threads, passed to both -t and -tb "
                          "(default max supported on the system)")
    srv.add_argument("--port", type=int, default=8080)
    srv.add_argument("--server-args", default=None,
                     help="extra arguments appended to the llama-server command. "
                          "Use the = form so argparse does not eat the leading "
                          "dashes, e.g. --server-args=\"-c 4096 --mlock\"")
    srv.add_argument("--boot-timeout", type=float, default=300.0,
                     help="seconds to wait for the server to answer (default 300)")

    suite = ap.add_argument_group("suite")
    suite.add_argument("--team", default="unnamed")
    suite.add_argument("--split", default="public", help="data/<split>/ subdirectory")
    suite.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    suite.add_argument("--n-hellaswag", type=int, default=20)
    suite.add_argument("--n-arc", type=int, default=20)
    suite.add_argument("--n-ifeval", type=int, default=10)
    suite.add_argument("--max-tokens", type=int, default=256)
    suite.add_argument("--item-timeout", type=float, default=60.0)
    suite.add_argument("--n-probs", type=int, default=40)
    suite.add_argument("--seed", type=int, default=42,
                       help="seed for item selection; identical for every team")

    out = ap.add_argument_group("output")
    out.add_argument("--report", action="store_true",
                     help="print the full quality and performance breakdown")
    out.add_argument("--out", default=None,
                     help="write the machine-readable JSON report to this path")
    out.add_argument("--verbose", action="store_true",
                     help="per-item detail, and live llama-server output")
    return ap.parse_args(argv)


def build_plan(args: argparse.Namespace) -> list[tuple[Any, list[dict]]]:
    """Load every task file up front, so bad data fails before the server starts."""
    data_dir = Path(args.data_dir).expanduser()
    return [
        (HellaSwagTask(),
         load_ranking("hellaswag", args.n_hellaswag, args.split, data_dir, args.seed)),
        (ARCEasyTask(),
         load_ranking("arc_easy", args.n_arc, args.split, data_dir, args.seed)),
        (IFEvalTask(max_tokens=args.max_tokens),
         load_ifeval(args.n_ifeval, args.split, data_dir, args.seed)),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    binary = Path(args.llama_server).expanduser()
    model = Path(args.model).expanduser()
    if not binary.is_file():
        print(f"\nERROR: no llama-server binary at '{binary}'.\n", file=sys.stderr)
        return 2
    if not os.access(binary, os.X_OK):
        print(f"\nERROR: '{binary}' is not executable. Try:\n"
              f"    chmod +x {binary}\n", file=sys.stderr)
        return 2
    if not model.is_file():
        print(f"\nERROR: no model file at '{model}'.\n", file=sys.stderr)
        return 2

    threads = args.threads if args.threads and args.threads > 0 else (os.cpu_count() or 1)

    plan = build_plan(args)            # fail fast on missing data
    ensure_no_server_running(binary, HOST, args.port)

    cmd = build_server_cmd(binary, model, threads, HOST, args.port, args.server_args)
    tel = Telemetry()
    backend = ServerBackend(f"http://{HOST}:{args.port}", chat=False,
                            timeout=args.item_timeout, n_probs=args.n_probs,
                            telemetry=tel)

    try:
        with ManagedServer(cmd, HOST, args.port, tel,
                           boot_timeout=args.boot_timeout,
                           show_output=args.verbose):
            try:
                info = backend.probe()
            except Exception as e:
                info = {}
                print(f"  (probe failed: {e})", file=sys.stderr)

            chat, warnings = detect_mode(info, model)
            backend.chat = chat
            for w in warnings:
                print(f"\nWARNING: {w}\n", file=sys.stderr)
            print(f"  mode: {'chat (/v1/chat/completions)' if chat else 'completion (/completion)'}"
                  f"  threads: {threads}", file=sys.stderr)

            results = run_suite(backend, plan, args.item_timeout, args.verbose)
    except KeyboardInterrupt:
        # ManagedServer.__exit__ has already stopped the server on the way out.
        print("aborted.", file=sys.stderr)
        return 130

    rep = build_report(results, tel, args.team, args.split)
    if args.report:
        print_report(rep)
    else:
        print_summary(rep)

    if args.out:
        out = Path(args.out).expanduser()
        out.write_text(json.dumps(rep, indent=2, default=str))
        print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
