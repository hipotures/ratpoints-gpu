#!/usr/bin/env python3
"""Build isolated CUDA variants and compare exact height-10M searches."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time

from benchmark_multi_gpu import RECORD_CURVE


def command_output(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def summary(samples):
    values = [sample["seconds"] for sample in samples]
    return {"median_seconds": statistics.median(values),
            "minimum_seconds": min(values), "mean_seconds": statistics.mean(values),
            "stddev_seconds": statistics.stdev(values) if len(values) > 1 else 0,
            "coefficient_of_variation":
                statistics.stdev(values) / statistics.mean(values)
                if len(values) > 1 else 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--height", type=int, default=10_000_000)
    parser.add_argument("--denominator-max", type=int)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--num-primes", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--primes", type=int, nargs="+", default=[14, 12, 10, 8])
    parser.add_argument("--masks", nargs="+", choices=["shared", "global"],
                        default=["shared", "global"])
    args = parser.parse_args()
    if (args.height < 1 or args.warmups < 0 or args.repeats < 1
            or args.device < 0 or args.num_primes < 1 or args.num_primes > 93
            or args.block < 32 or args.block > 1024 or args.block % 32
            or args.batch_size < 1 or args.batch_size > 65_536
            or any(n < 1 or n > args.num_primes for n in args.primes)):
        parser.error("invalid height, repeats, device, or tuning parameter")
    if args.block >= 512:
        parser.error("BLOCK >= 512 is unsupported: selected primes must exceed "
                     "BLOCK, but the sieve uses primes below 512")
    upper = args.height if args.denominator_max is None else args.denominator_max
    if upper < 1 or upper > args.height:
        parser.error("denominator max must be in 1..height")

    root = Path(__file__).resolve().parent.parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = command_output(["git", "ls-files"], root).splitlines()
    report = {"git_commit": command_output(["git", "rev-parse", "HEAD"], root),
              "git_dirty": bool(command_output(["git", "status", "--porcelain"], root)),
              "nvcc_version": command_output(["nvcc", "--version"]).splitlines()[-2],
              "nvidia_smi": command_output([
                  "nvidia-smi", "--query-gpu=name,pci.bus_id,compute_cap,driver_version",
                  "--format=csv,noheader"]),
              "height": args.height, "denominator_min": 1,
              "denominator_max": upper, "device": args.device,
              "warmups": args.warmups, "repeats": args.repeats,
              "arch": args.arch, "block": args.block,
              "num_primes": args.num_primes, "batch_size": args.batch_size,
              "coefficients": RECORD_CURVE,
              "variants": [], "baseline": None}
    report_path = output / "results.json"
    env = os.environ.copy()
    env["RATPOINTS_GPU_BENCHMARK"] = "1"

    for initial in args.primes:
        for mask in args.masks:
            name = (f"block{args.block}_primes{args.num_primes}_"
                    f"initial{initial}_{mask}_batch{args.batch_size}")
            directory = output / name
            directory.mkdir(exist_ok=True)
            for relative in files:
                source = root / relative
                if source.is_file():
                    target = directory / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            flags = (f"-DBLOCK={args.block} -DNUM_PRIMES={args.num_primes} "
                     f"-DINITIAL_PRIMES={initial}")
            if mask == "global":
                flags += " -DGLOBAL_MASK_ROWS"
            build = ["make", "-j", str(min(os.cpu_count() or 1, 8)),
                     f"CPPFLAGS={flags}", f"ARCH_FLAGS=-arch={args.arch}"]
            print(f"Building {name}", file=sys.stderr, flush=True)
            with (directory / "build.log").open("w") as build_log:
                subprocess.run(build, cwd=directory, check=True,
                               stdout=build_log, stderr=subprocess.STDOUT)
            binary = directory / "ratpoints_gpu"
            variant = {"name": name, "block": args.block,
                       "num_primes": args.num_primes, "batch_size": args.batch_size,
                       "initial_primes": initial, "mask": mask,
                       "build_command": build,
                       "binary_sha256": digest(binary.read_bytes()), "samples": []}
            search = [str(binary), RECORD_CURVE, str(args.height), "-dl", "1",
                      "-du", str(upper), "--devices", str(args.device),
                      "--batch-size", str(args.batch_size), "-v"]
            for index in range(args.warmups + args.repeats):
                started = time.perf_counter()
                result = subprocess.run(search, env=env, capture_output=True, check=True)
                seconds = time.perf_counter() - started
                line = re.search(r"^wall_ms=.*$", result.stderr.decode(), re.M)
                if not line:
                    raise RuntimeError(f"missing metrics for {name}")
                metrics = dict(re.findall(r"([a-z_]+)=([0-9.]+)", line.group()))
                identity = {"output_sha256": digest(result.stdout),
                            "output_bytes": len(result.stdout),
                            "modular_survivors": int(metrics["survivors"]),
                            "exact_survivors": int(metrics["exact_survivors"])}
                if report["baseline"] is None:
                    report["baseline"] = identity
                    (output / "baseline.stdout").write_bytes(result.stdout)
                if (identity != report["baseline"]
                        or result.stdout != (output / "baseline.stdout").read_bytes()):
                    raise RuntimeError(f"exact output or survivor mismatch: {name}")
                if index >= args.warmups:
                    variant["samples"].append({"seconds": seconds, "metrics": metrics})
                print(f"{name} {'warmup' if index < args.warmups else 'run'} "
                      f"{index + 1}: {seconds:.6f}s output=OK", file=sys.stderr,
                      flush=True)
            variant.update(summary(variant["samples"]))
            report["variants"].append(variant)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
    baseline = report["variants"][0]["median_seconds"]
    for variant in report["variants"]:
        variant["speedup_vs_first"] = baseline / variant["median_seconds"]
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    for variant in sorted(report["variants"], key=lambda item: item["median_seconds"]):
        print(f"{variant['name']}: {variant['median_seconds']:.6f}s "
              f"CV={variant['coefficient_of_variation']:.2%} "
              f"speedup={variant['speedup_vs_first']:.3f}x")
    print(f"Exact output SHA-256: {report['baseline']['output_sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print(f"Autotuning failed: {error}", file=sys.stderr)
        sys.exit(1)
