#!/usr/bin/env python3
"""Calibrate ICARM curve #302 and run a dual-GPU Mestre-Nagao specialization sieve."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction

from rank31_mn_common import (
    ICARM_URL,
    MESTRE_NAGAO_REFERENCE_URL,
    RECORD_T,
    RESULTS_DIR,
    calibration,
    commit_and_push,
    compile_helper,
    environment,
    generate_candidates,
    git,
    percentile,
    score_candidates,
    score_json,
    validate_gpu_counts,
)


def parse_fraction(text: str) -> Fraction:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"invalid rational value: {text}") from exc


def stage(number: int, title: str) -> float:
    print(f"\n=== Stage {number}: {title} ===", flush=True)
    return time.perf_counter()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--candidates", type=int, default=50_000,
                        help="new broad-scan candidates, excluding record control")
    parser.add_argument("--broad-prime-bound", type=int, default=1000)
    parser.add_argument("--refine-top", type=int, default=512)
    parser.add_argument("--refine-prime-bound", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=302)
    parser.add_argument("--denominator-min", type=int, default=1_000)
    parser.add_argument("--denominator-max", type=int, default=1_000_000)
    parser.add_argument("--t-span", type=parse_fraction, default=Fraction(1, 5))
    parser.add_argument("--batch-candidates", type=int, default=12_500)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    devices = [int(item) for item in args.devices.split(",") if item.strip()]
    if not devices or len(set(devices)) != len(devices):
        parser.error("--devices must contain distinct comma-separated integers")
    if args.candidates < 1 or args.refine_top < 1 or args.refine_top > args.candidates:
        parser.error("invalid candidate/refinement count")
    if args.broad_prime_bound < 3 or args.refine_prime_bound < args.broad_prime_bound:
        parser.error("invalid prime bounds")
    if args.denominator_min < 1 or args.denominator_max < args.denominator_min:
        parser.error("invalid denominator range")
    if args.batch_candidates < 1:
        parser.error("--batch-candidates must be positive")

    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    started = stage(1, "exact coordinate calibration against ICARM curve #302")
    cal = calibration()
    timings["calibration"] = time.perf_counter() - started
    print(f"Derived exact u = {cal['change_of_variables']['u']}")
    print("31/31 published witnesses transported and verified exactly.")
    print(f"Minimum family-model x projective height: {cal['minimum_family_x_projective_height']}")
    print(f"Median family-model x projective height:  {cal['median_family_x_projective_height']}")
    print(f"Maximum family-model x projective height: {cal['maximum_family_x_projective_height']}")
    print(f"Reachable at old H=2,000,000:  {cal['reachable_at_h_2m']}/31")
    print(f"Reachable at old H=20,000,000: {cal['reachable_at_h_20m']}/31")
    if args.calibration_only:
        return

    started = stage(2, "compile and validate CUDA finite-field scorer")
    helper, helper_meta = compile_helper(args.force_rebuild)
    candidates = generate_candidates(
        args.candidates,
        args.seed,
        args.denominator_min,
        args.denominator_max,
        args.t_span,
    )
    print(f"Generated {len(candidates)-1:,} new T values plus record control T={RECORD_T}.")
    validation = validate_gpu_counts(helper, devices[0], candidates, args.broad_prime_bound)
    print(
        f"GPU/CPU exact point-count validation passed for "
        f"{validation['candidate_prime_pairs_checked']} candidate-prime pairs."
    )
    timings["build_and_validation"] = time.perf_counter() - started

    started = stage(3, "broad dual-GPU Mestre-Nagao sieve")
    broad_rows, broad_stats = score_candidates(
        helper,
        candidates,
        devices,
        args.broad_prime_bound,
        "broad",
        args.batch_candidates,
    )
    timings["broad_gpu"] = time.perf_counter() - started
    broad_control = percentile(broad_rows)
    print(
        f"Record control: score={broad_control['score']:.9f}, "
        f"rank={broad_control['rank_higher_score_is_better']}/{broad_control['population']}, "
        f"percentile={broad_control['percentile']:.3f}%"
    )

    broad_new = [row for row in broad_rows if row.index != 0]
    broad_new.sort(key=lambda row: (row.score, row.good_primes), reverse=True)
    selected = broad_new[:args.refine_top]
    refined_candidates = [RECORD_T] + [row.t for row in selected]

    started = stage(4, "refine top candidates to the larger prime bound")
    refine_batch = max(
        1,
        min(
            args.batch_candidates,
            (len(refined_candidates) + len(devices) - 1) // len(devices),
        ),
    )
    refined_rows, refined_stats = score_candidates(
        helper,
        refined_candidates,
        devices,
        args.refine_prime_bound,
        "refine",
        refine_batch,
    )
    timings["refined_gpu"] = time.perf_counter() - started
    refined_control = percentile(refined_rows)
    print(
        f"Record refined control: score={refined_control['score']:.9f}, "
        f"rank={refined_control['rank_higher_score_is_better']}/{refined_control['population']}, "
        f"percentile={refined_control['percentile']:.3f}%"
    )

    refined_new = [row for row in refined_rows if row.index != 0]
    refined_new.sort(key=lambda row: (row.score, row.good_primes), reverse=True)
    print("\nTop new leads (Mestre-Nagao heuristic, not rank proofs):")
    for rank, row in enumerate(refined_new[:25], 1):
        print(
            f"  {rank:2d}. T={row.t.numerator}/{row.t.denominator} "
            f"score={row.score:.9f} good_primes={row.good_primes}"
        )

    stage(5, "write reproducible report")
    generated = datetime.now(timezone.utc)
    stamp = generated.strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamped = RESULTS_DIR / f"rank31-mestre-nagao-{stamp}.json"
    latest = RESULTS_DIR / "rank31-mestre-nagao-latest.json"
    timings["total_before_publication"] = time.perf_counter() - total_started
    report = {
        "generated_at_utc": generated.isoformat(),
        "git_head_before_result_commit": git(["rev-parse", "HEAD"]).stdout.strip(),
        "sources": {
            "icarm_curve_302": ICARM_URL,
            "mestre_nagao_reference": MESTRE_NAGAO_REFERENCE_URL,
        },
        "research_note": (
            "Mestre-Nagao scores are heuristic candidate rankings, not rank proofs. "
            "Promising specializations require exact independence/descent certification."
        ),
        "environment": environment(),
        "parameters": {
            "devices": devices,
            "new_candidate_count": args.candidates,
            "seed": args.seed,
            "denominator_min": args.denominator_min,
            "denominator_max": args.denominator_max,
            "t_span": str(args.t_span),
            "broad_prime_bound": args.broad_prime_bound,
            "refine_top": args.refine_top,
            "refine_prime_bound": args.refine_prime_bound,
            "batch_candidates": args.batch_candidates,
        },
        "calibration": cal,
        "cuda_helper": helper_meta,
        "gpu_cpu_validation": validation,
        "record_control": {
            "t": str(RECORD_T),
            "broad": broad_control,
            "refined_population": refined_control,
        },
        "broad_results": [score_json(row) for row in broad_rows],
        "refined_results": [score_json(row) for row in refined_rows],
        "top_new_leads": [score_json(row) for row in refined_new[:100]],
        "gpu_work": {"broad": broad_stats, "refined": refined_stats},
        "timings_seconds": timings,
    }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    timestamped.write_text(text)
    latest.write_text(text)
    print(f"Wrote {timestamped}")
    print(f"Wrote {latest}")

    if args.no_push:
        print("--no-push selected; result files were not committed.")
        return

    stage(6, "commit, push, and verify GitHub results")
    publication = commit_and_push(
        [timestamped, latest],
        f"Record rank-31 Mestre-Nagao GPU sieve {stamp}",
    )
    print(json.dumps(publication, indent=2))
    if publication.get("pushed"):
        print("Result publication verified: local HEAD == origin/master")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ArithmeticError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            if exc.stdout:
                print(exc.stdout, file=sys.stderr)
            if exc.stderr:
                print(exc.stderr, file=sys.stderr)
        print(f"rank31_mestre_nagao_gpu failed: {exc}", file=sys.stderr)
        sys.exit(1)
