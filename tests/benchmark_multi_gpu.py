#!/usr/bin/env python3
"""Compare identical bounded searches on individual GPUs and their combination."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

RECORD_CURVE = (
    "247747600 -985905640 567207969 2396040466 "
    "52485681 -470135160 82342800"
)


def positive(text):
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_devices(text):
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", text):
        raise argparse.ArgumentTypeError("use comma-separated GPU IDs, e.g. 0,1")
    ids = [int(item) for item in text.split(",")]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("select at least two distinct GPU IDs")
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default="./ratpoints_gpu")
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("0,1"))
    parser.add_argument("--height", type=positive, default=1_000_000)
    parser.add_argument("--denominator-min", type=positive, default=1)
    parser.add_argument("--denominator-max", type=positive)
    parser.add_argument("--coefficients", default=RECORD_CURVE)
    parser.add_argument("--repeats", type=positive, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--batch-size", type=positive, default=65_536)
    parser.add_argument("--timeout", type=positive, default=3600)
    parser.add_argument("--json", type=Path, help="write complete results after all checks pass")
    args = parser.parse_args()
    upper = args.denominator_max if args.denominator_max is not None else args.height
    if (args.warmups < 0 or args.batch_size > 65_536
            or not args.denominator_min <= upper <= min(args.height, 2_147_483_647)):
        parser.error("invalid warmup count, batch size, or denominator bounds")
    binary = str(Path(args.binary).resolve(strict=True))
    env = os.environ.copy()
    env["RATPOINTS_GPU_BENCHMARK"] = "1"
    listing = subprocess.run([binary, "--list-devices"], check=True,
                             capture_output=True, text=True, timeout=args.timeout,
                             env=env).stdout
    visible = {int(item) for item in re.findall(r"^device=(\d+)", listing, re.M)}
    if not set(args.devices) <= visible:
        parser.error(f"selected devices are not visible; available IDs: {sorted(visible)}")
    configurations = [str(device) for device in args.devices]
    configurations.append(",".join(map(str, args.devices)))
    samples = {config: [] for config in configurations}
    expected_output = None
    expected_survivors = None
    sites = (2 * args.height + 1) * (upper - args.denominator_min + 1)

    def run(config, measured):
        nonlocal expected_output, expected_survivors
        command = [binary, args.coefficients, str(args.height), "-dl",
                   str(args.denominator_min), "-du", str(upper),
                   "--devices", config, "--batch-size", str(args.batch_size), "-v"]
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, check=True,
                                env=env, timeout=args.timeout)
        elapsed = time.perf_counter() - started
        if expected_output is None:
            expected_output = result.stdout
        elif result.stdout != expected_output:
            raise RuntimeError(f"ordered point output differs for GPU selection {config}; no valid benchmark")
        diagnostic = result.stderr.decode("utf-8", errors="replace")
        summary = re.search(r"^wall_ms=.*$", diagnostic, re.M)
        if summary is None:
            raise RuntimeError("missing search metrics; rebuild the multi-GPU executable")
        values = dict(re.findall(r"([a-z_]+)=([0-9.]+)", summary.group()))
        if int(values["denominators"]) != upper - args.denominator_min + 1:
            raise RuntimeError("benchmark did not search the requested denominator range")
        survivors = (int(values["survivors"]), int(values["exact_survivors"]))
        if expected_survivors is None:
            expected_survivors = survivors
        elif survivors != expected_survivors:
            raise RuntimeError(f"survivor counts differ for GPU selection {config}")
        if measured:
            samples[config].append({"elapsed_seconds": elapsed,
                                    "metrics": values, "diagnostics": diagnostic})
        print(f"{'run' if measured else 'warmup'} GPUs={config} elapsed={elapsed:.6f}s output=OK",
              file=sys.stderr, flush=True)

    print(listing, end="")
    print(f"Height={args.height}, denominators={args.denominator_min}..{upper}, "
          f"sites={sites}, repeats={args.repeats}, warmups={args.warmups}", flush=True)
    for _ in range(args.warmups):
        for config in configurations:
            run(config, False)
    # Rotate order to reduce systematic warm-cache and thermal ordering bias.
    for repeat in range(args.repeats):
        offset = repeat % len(configurations)
        for config in configurations[offset:] + configurations[:offset]:
            run(config, True)
    medians = {config: statistics.median(item["elapsed_seconds"] for item in runs)
               for config, runs in samples.items()}
    fastest_single = min(medians[str(device)] for device in args.devices)
    results = []
    print("\nGPU(s)           median_s      min_s       Gsites/s   speedup_vs_fastest_single")
    for config in configurations:
        median = medians[config]
        minimum = min(item["elapsed_seconds"] for item in samples[config])
        speedup = fastest_single / median
        throughput = sites / median / 1e9
        print(f"{config:<16} {median:>10.6f} {minimum:>10.6f} {throughput:>13.3f} {speedup:>10.3f}x")
        results.append({"devices": config, "median_seconds": median,
                        "minimum_seconds": minimum, "gsites_per_second": throughput,
                        "speedup_vs_fastest_single": speedup, "runs": samples[config]})
    digest = hashlib.sha256(expected_output).hexdigest()
    print(f"\nAll ordered outputs and survivor counts match. Output SHA-256: {digest}")
    print("Elapsed times include process/CUDA startup, GMP verification, and output. "
          "Gsites/s counts the bounding rectangle, not individually tested points.")
    if args.json:
        report = {"binary": binary,
                  "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "device_listing": listing, "coefficients": args.coefficients,
                  "height": args.height, "denominator_min": args.denominator_min,
                  "denominator_max": upper, "sites": sites, "batch_size": args.batch_size,
                  "repeats": args.repeats, "warmups": args.warmups,
                  "output_sha256": digest, "output_bytes": len(expected_output),
                  "modular_survivors": expected_survivors[0],
                  "exact_survivors": expected_survivors[1], "results": results}
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.json}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            detail = error.stderr or b""
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            print(detail, file=sys.stderr)
        print(f"Benchmark failed: {error}", file=sys.stderr)
        sys.exit(1)
