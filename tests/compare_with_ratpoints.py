#!/usr/bin/env python3
"""Compare GPU output with ratpoints 2.1.3 or newer on fixed generic models."""

from pathlib import Path
import math
import re
import shutil
import subprocess
import sys


RECORD_CURVE = (
    "247747600 -985905640 567207969 2396040466 "
    "52485681 -470135160 82342800"
)

CASES = (
    ("1 0 1 0 1", 500, 40),
    ("-3 5 -7 11 2", 500, 40),
    ("1 1 0 1", 500, 40),
    ("+1 0 +1", 100, 20),
    ("1 " + "0 " * 99 + "1", 30, 10),
    (RECORD_CURVE, 1_000, 200),
)

OPTION_CASES = (
    ("1 0 1 0 1", 500, ["-du", "40", "-q", "-i"]),
    ("1 0 1 0 1", 500, ["-dl", "7", "-du", "40", "-q"]),
    ("1 0 1 0 1", 500, ["-du", "40", "-l", "-1", "-u", "2", "-q"]),
)


def dense_sieve_polynomial():
    """Return a linear polynomial passing every default sieve prime."""
    primes = []
    for candidate in range(2, 512):
        divisors = range(2, math.isqrt(candidate) + 1)
        if all(candidate % divisor for divisor in divisors):
            primes.append(candidate)
    return f"1 {math.prod(primes[-32:])}"


def points(command):
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return sorted(
        line for line in result.stdout.splitlines() if line.startswith("(")
    )


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: compare_with_ratpoints.py GPU_EXECUTABLE")
    gpu = str(Path(sys.argv[1]).resolve())
    cpu = shutil.which("ratpoints")
    if cpu is None:
        raise SystemExit("ratpoints 2.1.3 or newer is required for comparison")
    for coefficients, height, denominator in CASES:
        options = ["-du", str(denominator), "-q"]
        expected = points([cpu, coefficients, str(height), *options])
        actual = points([gpu, coefficients, str(height), *options])
        if actual != expected:
            raise AssertionError(
                f"backend mismatch for {coefficients}\n"
                f"CPU-only: {sorted(set(expected) - set(actual))}\n"
                f"GPU-only: {sorted(set(actual) - set(expected))}"
            )
    for coefficients, height, options in OPTION_CASES:
        expected = points([cpu, coefficients, str(height), *options])
        actual = points([gpu, coefficients, str(height), *options])
        if actual != expected:
            raise AssertionError(f"option mismatch for {' '.join(options)}")
    rejected = subprocess.run(
        [gpu, "1 -2 1", "10", "-q"], text=True, capture_output=True
    )
    if rejected.returncode == 0 or "not squarefree" not in rejected.stderr:
        raise AssertionError("non-squarefree polynomial was not rejected")

    dense = subprocess.run(
        [gpu, dense_sieve_polynomial(), "5000", "-du", "2", "-z", "-v"],
        check=True,
        text=True,
        capture_output=True,
    )
    match = re.search(r"survivors=(\d+) exact_survivors=(\d+)", dense.stderr)
    if match is None or tuple(map(int, match.groups())) != (20_002, 1):
        raise AssertionError(
            "multi-flush sieve mismatch; expected 20002 modular and 1 exact "
            f"survivor in {dense.stderr!r}"
        )

    comparisons = len(CASES) + len(OPTION_CASES)
    print(
        f"CPU/GPU agreement on {comparisons} point-search cases; "
        "validation and multi-flush checks passed"
    )


if __name__ == "__main__":
    main()
