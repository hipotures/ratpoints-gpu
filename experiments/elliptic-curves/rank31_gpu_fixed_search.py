#!/usr/bin/env python3
"""Bounded two-GPU search for rational points on the T=-47/80 minimal model.

For x = center + n/d, the discriminant (2y+x)^2 = 4x^3+x^2+4a4*x+4a6
is a square exactly when a point exists. The existing CUDA sieve tests this
integer cubic at n/d modulo primes coprime to d, so it cannot reject a point.
Every emitted result is checked again with exact Python integers.
"""

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
A4 = -1322791863836678632756897906456499538
A6 = 648069501075857258662767528794908106200954623218194692
KNOWN_CENTERS = {
    "P0": -1076035859747408082,
    "PD": 525828187343488308,
    "PE": 688507480409292636,
    "S": 866040525691648982,
}


def root_ceiling():
    """First integer to the right of the unique real branch endpoint."""
    lo, hi = -2*10**18, -10**18
    def cubic(x):
        return 4*x**3+x*x+4*A4*x+4*A6
    if not cubic(lo) < 0 < cubic(hi):
        raise ArithmeticError("real-root bracket changed")
    while hi-lo > 1:
        middle = (lo+hi)//2
        if cubic(middle) < 0:
            lo = middle
        else:
            hi = middle
    return hi


KNOWN_CENTERS["R"] = root_ceiling()


def coefficients(center):
    c = center
    return (4*c**3+c*c+4*A4*c+4*A6,
            12*c*c+2*c+4*A4, 12*c+1, 4)


def run_command(command, timeout):
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True,
                               start_new_session=True)
    samples = []
    stop = threading.Event()

    def sample():
        while not stop.wait(0.25):
            sample_process = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False)
            if sample_process.returncode == 0:
                samples.append(sample_process.stdout.strip())

    observer = threading.Thread(target=sample, daemon=True)
    observer.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        stop.set()
        observer.join(timeout=6)
    return {"command": command, "elapsed_seconds": time.monotonic()-started,
            "timed_out": timed_out, "returncode": process.returncode,
            "stdout": stdout, "stderr": stderr, "utilization_samples": samples}


def parse_points(stdout, center, coeffs):
    points = []
    for line in stdout.splitlines():
        n_text, root_text, d_text = line.split()
        n, root, d = int(n_text), int(root_text), int(d_text)
        if d <= 0 or math.gcd(n, d) != 1:
            raise ArithmeticError("invalid reduced GPU numerator/denominator")
        value = coeffs[3]*n**3*d + coeffs[2]*n*n*d*d + coeffs[1]*n*d**3 + coeffs[0]*d**4
        if value != root*root:
            raise ArithmeticError("GPU result failed exact square identity")
        x = Fraction(center*d+n, d)
        y = (Fraction(root, d*d)-x)/2
        if y*y+x*y != x**3+A4*x+A6:
            raise ArithmeticError("GPU result failed minimal curve equation")
        points.append({"n": n, "root": str(root), "d": d,
                       "x": str(x), "y": str(y)})
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", choices=KNOWN_CENTERS, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--denominators", type=int, required=True)
    parser.add_argument("--devices", choices=("0", "1", "0,1"), required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    if args.height < 1 or not 1 <= args.denominators <= min(args.height, 2**31-1):
        parser.error("invalid explicit search bounds")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        parser.error("invalid tag")
    center = KNOWN_CENTERS[args.center]
    coeffs = coefficients(center)
    # f(center+X) is squarefree because the elliptic curve is nonsingular.
    command = [str(ROOT / "ratpoints_gpu"), " ".join(map(str, coeffs)),
               str(args.height), "-dl", "1", "-du", str(args.denominators),
               "--devices", args.devices, "-i", "-v", "-f", "%x %y %z\\n"]
    run = run_command(command, args.timeout)
    points = parse_points(run["stdout"], center, coeffs) if run["returncode"] == 0 else []
    metrics_line = next((line for line in run["stderr"].splitlines()
                         if line.startswith("wall_ms=")), None)
    metrics = dict(re.findall(r"([a-z_]+)=([0-9.]+)", metrics_line)) if metrics_line else {}
    utilization = {"0": [], "1": []}
    for sample in run["utilization_samples"]:
        for line in sample.splitlines():
            index, value = [x.strip() for x in line.split(",")]
            if index in utilization:
                utilization[index].append(int(value))
    sites = (2*args.height+1)*args.denominators
    report = {"model": [1, 0, 0, str(A4), str(A6)],
              "center_label": args.center, "center": str(center),
              "discriminant_cubic_coefficients_ascending": list(map(str, coeffs)),
              "height": args.height, "denominators": [1, args.denominators],
              "devices": args.devices, "sites": sites,
              "sites_per_second": sites/run["elapsed_seconds"],
              "elapsed_seconds": run["elapsed_seconds"],
              "timed_out": run["timed_out"], "returncode": run["returncode"],
              "command": command, "metrics": metrics,
              "gpu_utilization_percent": {
                  k: {"samples": len(v), "mean": sum(v)/len(v) if v else None,
                      "max": max(v) if v else None} for k, v in utilization.items()},
              "points": points, "stderr_tail": run["stderr"][-3000:],
              "sieve_proof": "At selected primes p coprime to d, every square value of d^4*f(n/d) reduces to a quadratic residue. Every GPU survivor is checked by exact square and curve identities."}
    target = HERE / "results" / f"rank31-gpu-fixed-{args.tag}.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(f"{target}: {len(points)} points, {run['elapsed_seconds']:.2f}s, "
          f"{sites/run['elapsed_seconds']:.3g} sites/s")
    if run["returncode"] != 0:
        raise SystemExit(124 if run["timed_out"] else 1)


if __name__ == "__main__":
    main()
