#!/usr/bin/env python3
"""Run a sharded, multi-seed Mestre-Nagao specialization campaign."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from rank31_mn_common import (
    ICARM_URL, MESTRE_NAGAO_REFERENCE_URL, RECORD_T, RESULTS_DIR,
    build_validation_oracle, calibration, commit_and_push, compile_helper,
    environment, generate_candidates, git, percentile, score_candidates,
    score_json, validate_gpu_counts,
)

MAX_GITHUB_FILE_BYTES = 100_000_000
ARTIFACT_SPLIT_BYTES = 90_000_000
ARTIFACT_SPLIT_ROWS = 1_100_000


@dataclass(frozen=True)
class Lead:
    t: Fraction
    score: float
    good_primes: int
    shard: int
    seed: int
    stage_a_rank: int
    stage_a_index: int


def parse_fraction(value: str) -> Fraction:
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"invalid rational value: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("--t-span must be positive")
    return result


def parse_seeds(text: str) -> list[int]:
    try:
        seeds = [int(value.strip()) for value in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("--seeds must contain distinct integers")
    return seeds


def ranked_new_leads(rows, shard: int, seed: int, limit: int) -> list[Lead]:
    new_rows = [row for row in rows if row.index != 0 and row.t != RECORD_T]
    new_rows.sort(key=lambda row: (-row.score, -row.good_primes, row.index))
    return [Lead(row.t, row.score, row.good_primes, shard, seed, rank, row.index)
            for rank, row in enumerate(new_rows[:limit], 1)]


def merge_finalists(shard_leads: list[list[Lead]], limit: int) -> list[Lead]:
    """A shard's top K contains every possible global top-K member from it."""
    ranked = sorted((lead for leads in shard_leads for lead in leads),
                    key=lambda lead: (-lead.score, -lead.good_primes,
                                      lead.shard, lead.stage_a_index))
    seen = {RECORD_T}
    result = []
    for lead in ranked:
        if lead.t not in seen:
            seen.add(lead.t)
            result.append(lead)
            if len(result) == limit:
                break
    if len(result) < limit:
        raise RuntimeError(f"only {len(result)} distinct new finalists found; requested {limit}")
    return result


def lead_json(lead: Lead) -> dict:
    return {"t": f"{lead.t.numerator}/{lead.t.denominator}",
            "stage_a_score": lead.score, "stage_a_good_primes": lead.good_primes,
            "source_shard": lead.shard, "source_seed": lead.seed,
            "stage_a_rank_in_shard": lead.stage_a_rank,
            "stage_a_index_in_shard": lead.stage_a_index}


def summarize_gpu_work(stats) -> list[dict]:
    totals = {}
    for batch in stats:
        row = totals.setdefault(batch["device"], {"device": batch["device"],
                                                   "candidate_count": 0, "seconds": 0.0})
        row["candidate_count"] += batch["candidate_count"]
        row["seconds"] += batch["seconds"]
    return [dict(row, candidates_per_second=row["candidate_count"] / row["seconds"])
            for _, row in sorted(totals.items())]


def artifact_info(path: Path) -> dict:
    size = path.stat().st_size
    if size >= MAX_GITHUB_FILE_BYTES:
        raise RuntimeError(f"artifact exceeds GitHub's 100 MB limit: {path} ({size} bytes)")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.relative_to(RESULTS_DIR)), "bytes": size,
            "sha256": digest.hexdigest()}


def write_json(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_info(path)


def write_stage_a_parts(directory: Path, prefix: str, rows) -> list[dict]:
    """Write deterministic gzip JSONL parts, splitting before the GitHub limit."""
    parts = []
    position = 0
    while position < len(rows):
        path = directory / f"{prefix}.part-{len(parts)+1:03d}.jsonl.gz"
        count = 0
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                               compresslevel=6, mtime=0) as output:
                while position < len(rows) and count < ARTIFACT_SPLIT_ROWS:
                    output.write((json.dumps(score_json(rows[position]),
                                             sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
                    position += 1
                    count += 1
                    if count % 1000 == 0:
                        output.flush()
                        if raw.tell() >= ARTIFACT_SPLIT_BYTES:
                            break
        info = artifact_info(path)
        info["row_count"] = count
        parts.append(info)
    if sum(part["row_count"] for part in parts) != len(rows):
        raise AssertionError("incomplete Stage A artifact")
    return parts


def write_gzip_json(path: Path, value: dict) -> dict:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return artifact_info(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_seeds,
                        help="comma-separated seeds (default: 304,305,306,307,308)")
    parser.add_argument("--seed-start", type=int, default=304)
    parser.add_argument("--shards", type=int, default=5)
    parser.add_argument("--candidates-per-shard", type=int, default=1_000_000)
    parser.add_argument("--first-prime-bound", type=int, default=5000)
    parser.add_argument("--global-finalists", type=int, default=50_000)
    parser.add_argument("--second-prime-bound", type=int, default=100_000)
    parser.add_argument("--denominator-min", type=int, default=1000)
    parser.add_argument("--denominator-max", type=int, default=5_000_000)
    parser.add_argument("--t-span", type=parse_fraction, default=Fraction(1, 2))
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--batch-candidates", type=int, default=50_000)
    parser.add_argument("--validation-primes", type=int, default=256)
    parser.add_argument("--validation-exhaustive", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()
    if args.seeds is None:
        if args.shards < 1:
            parser.error("--shards must be positive")
        args.seeds = list(range(args.seed_start, args.seed_start + args.shards))
    elif args.shards != 5 or args.seed_start != 304:
        parser.error("use either --seeds or --seed-start/--shards")
    try:
        args.devices = [int(item.strip()) for item in args.devices.split(",")]
    except ValueError:
        parser.error("--devices must be comma-separated device indices")
    if not args.devices or any(device < 0 for device in args.devices) or len(set(args.devices)) != len(args.devices):
        parser.error("--devices must contain distinct nonnegative device indices")
    if args.candidates_per_shard < 1 or args.global_finalists < 1:
        parser.error("candidate and finalist counts must be positive")
    if args.global_finalists > len(args.seeds) * args.candidates_per_shard:
        parser.error("--global-finalists exceeds the number of generated candidates")
    if args.first_prime_bound < 3 or args.second_prime_bound < args.first_prime_bound:
        parser.error("invalid prime bounds")
    if args.denominator_min < 1 or args.denominator_max < args.denominator_min:
        parser.error("invalid denominator range")
    if args.batch_candidates < 1 or args.validation_primes < 1:
        parser.error("batch and validation-prime counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.no_push:
        branch = git(["branch", "--show-current"]).stdout.strip()
        if branch != "master":
            raise RuntimeError(f"automatic campaign publication requires master, found {branch!r}")
        if git(["diff", "--cached", "--quiet"], check=False).returncode != 0:
            raise RuntimeError("automatic campaign publication requires an empty Git staging area")
    started = time.perf_counter()
    source_head = git(["rev-parse", "HEAD"]).stdout.strip()
    generated = datetime.now(timezone.utc)
    stamp = generated.strftime("%Y%m%dT%H%M%SZ")
    campaign_dir = RESULTS_DIR / f"rank31-mestre-nagao-campaign-{stamp}"
    campaign_dir.mkdir(parents=True, exist_ok=False)
    artifacts: list[Path] = []

    print("=== Exact calibration and shared CPU/GPU validation ===", flush=True)
    cal = calibration()
    helper, helper_meta = compile_helper()
    first_candidates = generate_candidates(args.candidates_per_shard, args.seeds[0],
                                           args.denominator_min, args.denominator_max,
                                           args.t_span)
    oracle = build_validation_oracle(first_candidates, args.second_prime_bound,
                                     args.validation_primes, args.validation_exhaustive)
    gpu_checks = [validate_gpu_counts(helper, device, oracle) for device in args.devices]
    validation = {"oracle": oracle.report(), "gpu_checks": gpu_checks}
    print(f"Validated {len(oracle.expected)} exact CPU pairs on each of {len(args.devices)} GPU(s); "
          f"mode={oracle.mode}, CPU={oracle.cpu_seconds:.2f}s", flush=True)

    shard_summaries = []
    shard_leads = []
    stage_a_higher = 0
    stage_a_at_or_below = 1  # The control is counted once globally.
    stage_a_new_count = 0
    control_score = None
    stage_a_started = time.perf_counter()
    for shard, seed in enumerate(args.seeds, 1):
        shard_started = time.perf_counter()
        print(f"\n=== Stage A shard {shard}/{len(args.seeds)}, seed={seed} ===", flush=True)
        candidates = first_candidates if shard == 1 else generate_candidates(
            args.candidates_per_shard, seed, args.denominator_min,
            args.denominator_max, args.t_span)
        rows, gpu_work = score_candidates(helper, candidates, args.devices,
                                          args.first_prime_bound, f"shard{shard:02d}",
                                          args.batch_candidates)
        control = percentile(rows)
        if control_score is None:
            control_score = control["score"]
        elif control["score"] != control_score:
            raise RuntimeError("record control Stage A score differs across shards")
        stage_a_higher += control["rank_higher_score_is_better"] - 1
        stage_a_at_or_below += control["population"] - control["rank_higher_score_is_better"]
        stage_a_new_count += len(rows) - 1
        leads = ranked_new_leads(rows, shard, seed, args.global_finalists)
        shard_leads.append(leads)
        prefix = f"shard-{shard:02d}-seed-{seed}-stage-a"
        parts = write_stage_a_parts(campaign_dir, prefix, rows)
        artifacts.extend(campaign_dir / Path(part["path"]).name for part in parts)
        shard_summary = {
            "shard": shard, "seed": seed,
            "source_commit": source_head,
            "parameters": {"new_candidate_count": args.candidates_per_shard,
                           "prime_bound": args.first_prime_bound,
                           "denominator_min": args.denominator_min,
                           "denominator_max": args.denominator_max,
                           "t_span": str(args.t_span)},
            "record_control": control,
            "stage_a_artifacts": parts,
            "top_new_leads": [lead_json(lead) for lead in leads[:100]],
            "gpu_work": gpu_work,
            "gpu_totals": summarize_gpu_work(gpu_work),
            "elapsed_seconds": time.perf_counter() - shard_started,
        }
        summary_path = campaign_dir / f"shard-{shard:02d}-seed-{seed}.summary.json"
        summary_info = write_json(summary_path, shard_summary)
        artifacts.append(summary_path)
        shard_summaries.append({"shard": shard, "seed": seed,
                                "record_control": control,
                                "stage_a_artifacts": parts,
                                "summary_artifact": summary_info,
                                "elapsed_seconds": shard_summary["elapsed_seconds"],
                                "gpu_totals": shard_summary["gpu_totals"]})
        print(f"Shard {shard}: control rank {control['rank_higher_score_is_better']}/{control['population']}; "
              f"wrote {len(parts)} compressed part(s)", flush=True)
        if shard == 1:
            first_candidates = None
        del candidates, rows
    stage_a_seconds = time.perf_counter() - stage_a_started

    print("\n=== Merge global finalists and score Stage B ===", flush=True)
    finalists = merge_finalists(shard_leads, args.global_finalists)
    if len({lead.t for lead in finalists}) != len(finalists) or any(lead.t == RECORD_T for lead in finalists):
        raise AssertionError("global finalists contain duplicates or the record control")
    stage_b_started = time.perf_counter()
    stage_b_candidates = [RECORD_T] + [lead.t for lead in finalists]
    stage_b_rows, stage_b_gpu_work = score_candidates(
        helper, stage_b_candidates, args.devices, args.second_prime_bound,
        "global-refine", args.batch_candidates)
    stage_b_seconds = time.perf_counter() - stage_b_started
    stage_b_control = percentile(stage_b_rows)
    stage_b_new = []
    for row in stage_b_rows[1:]:
        item = score_json(row)
        item.update(lead_json(finalists[row.index - 1]))
        stage_b_new.append(item)
    stage_b_new.sort(key=lambda row: (-row["score"], -row["good_primes"], row["index"]))
    gap = stage_b_control["score"] - stage_b_new[0]["score"]
    stage_a_global = {
        "score": control_score,
        "rank_higher_score_is_better": stage_a_higher + 1,
        "population_including_control_once": stage_a_new_count + 1,
        "percentile": 100 * stage_a_at_or_below / (stage_a_new_count + 1),
        "population_note": "Counts scored rows across shards; repeated T values in different shards are counted separately.",
    }
    stage_b_report = {
        "source_commit": source_head,
        "record_control": stage_b_control,
        "record_t": str(RECORD_T),
        "record_score_gap_over_best_new": gap,
        "finalists": [dict(score_json(stage_b_rows[0]), control=True)] + stage_b_new,
        "gpu_work": stage_b_gpu_work,
        "gpu_totals": summarize_gpu_work(stage_b_gpu_work),
        "elapsed_seconds": stage_b_seconds,
    }
    stage_b_path = campaign_dir / "global-stage-b.json.gz"
    stage_b_info = write_gzip_json(stage_b_path, stage_b_report)
    artifacts.append(stage_b_path)
    print(f"Stage B control rank {stage_b_control['rank_higher_score_is_better']}/{stage_b_control['population']}; "
          f"best-new gap {gap:+.9f}", flush=True)
    for rank, lead in enumerate(stage_b_new[:100], 1):
        print(f"{rank:3d}. T={lead['t']} score={lead['score']:.9f} "
              f"seed={lead['source_seed']} shard={lead['source_shard']}")

    summary = {
        "generated_at_utc": generated.isoformat(),
        "source_commit": source_head,
        "research_note": "Mestre-Nagao scores are heuristic rankings, not rank proofs.",
        "sources": {"icarm_curve_302": ICARM_URL,
                    "mestre_nagao_reference": MESTRE_NAGAO_REFERENCE_URL},
        "parameters": {"seeds": args.seeds,
                       "candidates_per_shard": args.candidates_per_shard,
                       "first_prime_bound": args.first_prime_bound,
                       "global_finalists": args.global_finalists,
                       "second_prime_bound": args.second_prime_bound,
                       "denominator_min": args.denominator_min,
                       "denominator_max": args.denominator_max,
                       "t_span": str(args.t_span),
                       "devices": args.devices,
                       "batch_candidates": args.batch_candidates,
                       "validation_primes": args.validation_primes,
                       "validation_exhaustive": args.validation_exhaustive},
        "environment": environment(),
        "cuda_helper": helper_meta,
        "calibration": {"source": cal["source"],
                        "witness_count": len(cal["witnesses"]),
                        "change_of_variables_u": cal["change_of_variables"]["u"]},
        "validation": validation,
        "shards": shard_summaries,
        "global_finalist_count": len(finalists),
        "record_control": {"stage_a_global": stage_a_global,
                           "stage_b_global": stage_b_control},
        "record_score_gap_over_best_new": gap,
        "top_new_leads": stage_b_new[:100],
        "stage_b_artifact": stage_b_info,
        "stage_b_gpu_totals": stage_b_report["gpu_totals"],
        "timings_seconds": {"stage_a": stage_a_seconds,
                            "stage_b": stage_b_seconds,
                            "total_before_publication": time.perf_counter() - started},
    }
    summary_path = campaign_dir / "campaign.summary.json"
    summary_info = write_json(summary_path, summary)
    artifacts.append(summary_path)
    latest = RESULTS_DIR / "rank31-mestre-nagao-campaign-latest.json"
    latest_data = {"campaign_summary_artifact": summary_info,
                   "generated_at_utc": generated.isoformat(),
                   "source_commit": source_head,
                   "record_control": summary["record_control"],
                   "top_new_leads": stage_b_new[:100],
                   "research_note": summary["research_note"]}
    write_json(latest, latest_data)
    artifacts.append(latest)
    print(f"Wrote {len(artifacts)} campaign artifacts under {campaign_dir}", flush=True)
    if args.no_push:
        print("--no-push selected; campaign artifacts were not committed.")
        return
    publication = commit_and_push(artifacts, f"Record rank-31 Mestre-Nagao campaign {stamp}")
    print(json.dumps(publication, indent=2))
    if publication.get("pushed"):
        print("Campaign publication verified: local HEAD == origin/master")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ArithmeticError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"rank31_mestre_nagao_campaign failed: {exc}", file=sys.stderr)
        sys.exit(1)
