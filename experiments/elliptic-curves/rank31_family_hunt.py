#!/usr/bin/env python3
"""GPU hunt for promising specializations of the public rank-31 elliptic-curve family.

This is a research search, not a replay of a known answer.  For many rational
specializations T, it constructs the cubic model

    Y^2 = (L x + B)^2 + 4 x (x + D) (x + E)

of the family published with ICARM curve #302, searches bounded rational x with
ratpoints_gpu on all requested GPUs, ranks specializations by the number of
exact rational points found, then deepens the most promising candidates.

The final JSON report is committed and pushed automatically by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "experiments" / "elliptic-curves" / "results"
RECORD_T = Fraction(164518, 924945)
SOURCE_URL = "https://elliptic-rank.icarm.cloud/curve/302"


def lcm(a: int, b: int) -> int:
    return abs(a // math.gcd(a, b) * b)


def frac_poly(t: Fraction) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction]:
    L = 446667 * t**2 + 471466 * t + 239031
    p = 318552 * t**2 + 368554 * t - 72570
    q = 733413 * t**2 - 45082 * t - 14960
    D = 5 * (7174492962 * t**4 - 7114589515 * t**3 - 22069002960 * t**2
             + 3909144679 * t - 205134150)
    E = (882769396002 * t**4 + 811447034567 * t**3 - 1174040743 * t**2
         - 32493137198 * t - 2386325360)
    B = p * q * (L + p + q) - p * E - q * D
    return L, B, D, E, p * q


def integer_cubic(t: Fraction) -> tuple[tuple[int, int, int, int], dict[str, str]]:
    L, B, D, E, pq = frac_poly(t)
    scale = 1
    for value in (L, B, D, E):
        scale = lcm(scale, value.denominator)
    s = scale
    coeffs_f = (
        B * B,
        2 * L * B + 4 * D * E,
        L * L + 4 * (D + E),
        Fraction(4),
    )
    coeffs = tuple(int(value * s * s) for value in coeffs_f)
    if any(Fraction(c) != value * s * s for c, value in zip(coeffs, coeffs_f)):
        raise ArithmeticError("failed to clear family denominators exactly")
    g = 0
    for c in coeffs:
        g = math.gcd(g, abs(c))
    root = math.isqrt(g)
    if root > 1 and root * root == g:
        coeffs = tuple(c // g for c in coeffs)
        s //= root if s % root == 0 else 1
    meta = {
        "L": str(L), "B": str(B), "D": str(D), "E": str(E),
        "pq": str(pq), "scale": str(scale),
    }
    return coeffs, meta


def metric_line(stderr: str) -> dict[str, str]:
    match = re.search(r"^wall_ms=.*$", stderr, re.M)
    if match is None:
        raise RuntimeError("ratpoints_gpu did not emit benchmark metrics")
    return dict(re.findall(r"([a-z_]+)=([0-9.]+)", match.group()))


def visible_devices(binary: Path) -> tuple[str, list[int]]:
    result = subprocess.run([str(binary), "--list-devices"], check=True,
                            capture_output=True, text=True)
    ids = [int(v) for v in re.findall(r"^device=(\d+)", result.stdout, re.M)]
    return result.stdout, ids


def candidate_parameters(count: int) -> list[Fraction]:
    """Deterministic arithmetic neighborhood of the known record specialization."""
    values: list[Fraction] = [RECORD_T]
    seen = {RECORD_T}
    # Vary both denominator scale and the nearest numerator.  Different
    # denominator arithmetic is more useful than only perturbing one numerator.
    q_min, q_max = 20_000, 1_000_000
    golden = 0.6180339887498949
    k = 0
    while len(values) < count:
        frac = (k * golden) % 1.0
        den = q_min + int(frac * (q_max - q_min))
        center = int(round(float(RECORD_T) * den))
        offset_cycle = (0, 1, -1, 2, -2, 3, -3, 5, -5, 8, -8, 13, -13)
        num = center + offset_cycle[k % len(offset_cycle)]
        if den > 0 and math.gcd(num, den) == 1:
            t = Fraction(num, den)
            if t not in seen:
                seen.add(t)
                values.append(t)
        k += 1
    return values


@dataclass
class RunResult:
    t: Fraction
    height: int
    exact_survivors: int
    modular_survivors: int
    seconds: float
    wall_ms: float
    stdout: str
    coeffs: tuple[int, int, int, int]
    family_meta: dict[str, str]


def run_search(binary: Path, devices: str, t: Fraction, height: int,
               dl: int = 1, du: int | None = None, keep_output: bool = False) -> RunResult:
    coeffs, meta = integer_cubic(t)
    upper = height if du is None else du
    command = [str(binary), " ".join(str(c) for c in coeffs), str(height),
               "-dl", str(dl), "-du", str(upper), "--devices", devices,
               "--batch-size", "65536", "-i", "-v"]
    if not keep_output:
        command.append("-z")
    env = os.environ.copy()
    env["RATPOINTS_GPU_BENCHMARK"] = "1"
    started = time.perf_counter()
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    seconds = time.perf_counter() - started
    metrics = metric_line(result.stderr)
    return RunResult(
        t=t, height=height,
        exact_survivors=int(metrics["exact_survivors"]),
        modular_survivors=int(metrics["survivors"]),
        seconds=seconds, wall_ms=float(metrics["wall_ms"]),
        stdout=result.stdout if keep_output else "",
        coeffs=coeffs, family_meta=meta,
    )


def deepen(binary: Path, devices: str, t: Fraction, height: int,
           chunk_denominators: int) -> dict:
    coeffs, meta = integer_cubic(t)
    exact = modular = 0
    outputs: set[str] = set()
    chunks = (height + chunk_denominators - 1) // chunk_denominators
    started = time.perf_counter()
    for index, lo in enumerate(range(1, height + 1, chunk_denominators), 1):
        hi = min(height, lo + chunk_denominators - 1)
        result = run_search(binary, devices, t, height, lo, hi, keep_output=True)
        exact += result.exact_survivors
        modular += result.modular_survivors
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                outputs.add(line)
        elapsed = time.perf_counter() - started
        rate = index / elapsed if elapsed else 0.0
        eta = (chunks - index) / rate if rate else 0.0
        print(f"  deep {index:3d}/{chunks:3d}  den={lo:,}..{hi:,}  "
              f"exact={exact:,}  unique-lines={len(outputs):,}  "
              f"elapsed={elapsed:7.1f}s  ETA={eta:7.1f}s", flush=True)
    return {
        "t": f"{t.numerator}/{t.denominator}",
        "height": height,
        "exact_survivors_sum": exact,
        "modular_survivors_sum": modular,
        "unique_output_lines": len(outputs),
        "points": sorted(outputs),
        "coefficients": list(coeffs),
        "family": meta,
        "seconds": time.perf_counter() - started,
    }


def git(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *cmd], cwd=ROOT, check=check,
                          capture_output=True, text=True)


def commit_and_push(paths: list[Path], message: str) -> dict:
    # Never scoop up unrelated local work.  Only explicit result files are added.
    for path in paths:
        git(["add", "--", str(path.relative_to(ROOT))])
    staged = git(["diff", "--cached", "--quiet"], check=False)
    if staged.returncode == 0:
        head = git(["rev-parse", "HEAD"]).stdout.strip()
        return {"committed": False, "head": head, "pushed": False,
                "reason": "result files unchanged"}
    git(["commit", "-m", message])
    head = git(["rev-parse", "HEAD"]).stdout.strip()
    branch = git(["branch", "--show-current"]).stdout.strip()
    if branch != "master":
        raise RuntimeError(f"refusing automatic push from branch {branch!r}; expected master")
    git(["push", "origin", "master"])
    remote = git(["ls-remote", "origin", "refs/heads/master"]).stdout.split()[0]
    if remote != head:
        raise RuntimeError(f"push verification failed: HEAD={head}, origin/master={remote}")
    return {"committed": True, "head": head, "pushed": True, "origin_master": remote}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "ratpoints_gpu")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--candidates", type=int, default=384,
                        help="family specializations in broad GPU scan (default: 384)")
    parser.add_argument("--broad-height", type=int, default=2_000_000)
    parser.add_argument("--top", type=int, default=10,
                        help="number of best specializations to deepen (default: 10)")
    parser.add_argument("--deep-height", type=int, default=20_000_000)
    parser.add_argument("--deep-chunk", type=int, default=1_000_000)
    parser.add_argument("--no-push", action="store_true",
                        help="write results but do not commit/push them")
    args = parser.parse_args()
    if args.candidates < 1 or args.top < 1 or args.top > args.candidates:
        parser.error("invalid --candidates/--top")
    if min(args.broad_height, args.deep_height, args.deep_chunk) < 1:
        parser.error("heights and chunk size must be positive")
    binary = args.binary.resolve(strict=True)

    print("=== Rank-31 family GPU hunt ===")
    print("Research target: new promising specializations of the public family behind ICARM #302.")
    print("A high point count is a candidate signal, not a rank proof.\n")

    print("=== Stage 1: GPU and family validation ===", flush=True)
    listing, ids = visible_devices(binary)
    print(listing, end="")
    requested = [int(x) for x in args.devices.split(",")]
    if not set(requested).issubset(ids):
        raise RuntimeError(f"requested GPUs {requested} are not all visible: {ids}")
    record_coeffs, _ = integer_cubic(RECORD_T)
    print(f"Record specialization T={RECORD_T.numerator}/{RECORD_T.denominator}")
    print("Derived cubic coefficient bit lengths:",
          [abs(c).bit_length() for c in record_coeffs], flush=True)

    print("\n=== Stage 2: broad specialization scan on GPUs ===", flush=True)
    params = candidate_parameters(args.candidates)
    broad: list[RunResult] = []
    started = time.perf_counter()
    failures: list[dict[str, str]] = []
    for index, t in enumerate(params, 1):
        try:
            item = run_search(binary, args.devices, t, args.broad_height)
            broad.append(item)
            best = max(x.exact_survivors for x in broad)
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(params) - index) / rate if rate else 0.0
            marker = " RECORD" if t == RECORD_T else ""
            print(f"[{index:4d}/{len(params):4d}] T={t.numerator}/{t.denominator}{marker}  "
                  f"exact={item.exact_survivors:4d}  best={best:4d}  "
                  f"wall={item.seconds:6.2f}s  ETA={eta/60:6.1f}m", flush=True)
        except subprocess.CalledProcessError as exc:
            failures.append({"t": str(t), "stderr": exc.stderr[-2000:] if exc.stderr else ""})
            print(f"[{index:4d}/{len(params):4d}] T={t} FAILED", flush=True)
    if not broad:
        raise RuntimeError("all broad searches failed")

    broad.sort(key=lambda x: (x.exact_survivors, -x.seconds), reverse=True)
    selected = broad[:args.top]
    print("\nTop broad-scan candidates:")
    for rank, item in enumerate(selected, 1):
        print(f"  {rank:2d}. T={item.t} exact={item.exact_survivors} "
              f"modular={item.modular_survivors:,} time={item.seconds:.2f}s")

    print("\n=== Stage 3: deep exact GPU searches of top candidates ===", flush=True)
    deep: list[dict] = []
    for index, item in enumerate(selected, 1):
        print(f"\nCandidate {index}/{len(selected)}: T={item.t} "
              f"broad exact={item.exact_survivors}", flush=True)
        deep.append(deepen(binary, args.devices, item.t, args.deep_height,
                           args.deep_chunk))

    print("\n=== Stage 4: rank candidate summary ===", flush=True)
    deep.sort(key=lambda x: (x["unique_output_lines"], x["exact_survivors_sum"]), reverse=True)
    for rank, item in enumerate(deep, 1):
        print(f"  {rank:2d}. T={item['t']}  unique-output={item['unique_output_lines']}  "
              f"exact-sum={item['exact_survivors_sum']}  time={item['seconds']:.1f}s")
    print("\nInterpretation: candidates with unusually many bounded rational points deserve "
          "independence testing (e.g. exact 2-descent). This scan alone does not prove rank.")

    print("\n=== Stage 5: write reproducible report ===", flush=True)
    generated = datetime.now(timezone.utc)
    stamp = generated.strftime("%Y%m%dT%H%M%SZ")
    DEFAULT_RESULTS.mkdir(parents=True, exist_ok=True)
    timestamped = DEFAULT_RESULTS / f"rank31-family-hunt-{stamp}.json"
    latest = DEFAULT_RESULTS / "rank31-family-hunt-latest.json"
    report = {
        "generated_at_utc": generated.isoformat(),
        "source_family": SOURCE_URL,
        "record_specialization": str(RECORD_T),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "binary": str(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "device_listing": listing,
        "parameters": {
            "devices": args.devices, "candidates": args.candidates,
            "broad_height": args.broad_height, "top": args.top,
            "deep_height": args.deep_height, "deep_chunk": args.deep_chunk,
        },
        "broad_seconds": time.perf_counter() - started,
        "broad_results": [
            {"t": str(x.t), "exact_survivors": x.exact_survivors,
             "modular_survivors": x.modular_survivors, "seconds": x.seconds,
             "wall_ms": x.wall_ms, "coefficients": list(x.coeffs)}
            for x in broad
        ],
        "failures": failures,
        "deep_results": deep,
        "research_note": (
            "This is an original bounded specialization scan of a published family. "
            "Point abundance is a heuristic candidate signal; rank requires independent certification."
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    timestamped.write_text(payload, encoding="utf-8")
    latest.write_text(payload, encoding="utf-8")
    print(f"Wrote {timestamped}")
    print(f"Wrote {latest}")

    if args.no_push:
        print("\n=== Stage 6: automatic GitHub publication disabled (--no-push) ===")
        return

    print("\n=== Stage 6: commit, push, and verify GitHub results ===", flush=True)
    publication = commit_and_push(
        [timestamped, latest],
        f"Record rank-31 family GPU hunt {stamp}",
    )
    print(json.dumps(publication, indent=2))
    if publication.get("pushed"):
        print("Result publication verified: local HEAD == origin/master")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, ArithmeticError, RuntimeError,
            subprocess.SubprocessError) as error:
        print(f"\nrank31 family hunt FAILED: {error}", file=sys.stderr)
        sys.exit(1)
