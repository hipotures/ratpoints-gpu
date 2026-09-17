#!/usr/bin/env python3
"""Exact group-law demonstration on y^2 = x^3 - 2."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.elliptic import EllipticCurve, INFINITY, Point  # noqa: E402
from common.progress import Progress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--associativity-limit", type=int, default=4,
                        help="test triples of nP for -N <= n <= N (default: 4)")
    args = parser.parse_args()
    if args.associativity_limit < 1:
        parser.error("--associativity-limit must be positive")

    curve = EllipticCurve(Fraction(0), Fraction(-2))
    p = Point(Fraction(3), Fraction(5))
    curve.require_point(p)

    print("Curve: y^2 = x^3 - 2")
    print(f"Discriminant: {curve.discriminant} (nonzero, so the curve is nonsingular)")
    print(f"P = {p}")
    print(f"-P = {curve.negate(p)}")
    print(f"P + O = {curve.add(p, INFINITY)}")
    print(f"P + (-P) = {curve.add(p, curve.negate(p))}")
    print(f"2P = {curve.multiply(2, p)}")
    print(f"3P = {curve.multiply(3, p)}")
    print(f"7P = {curve.multiply(7, p)}")

    values = list(range(-args.associativity_limit, args.associativity_limit + 1))
    points = {n: curve.multiply(n, p) for n in values}
    total = len(values) ** 3
    progress = Progress("associativity-sample", total=total)
    checked = 0
    for i in values:
        for j in values:
            for k in values:
                left = curve.add(curve.add(points[i], points[j]), points[k])
                right = curve.add(points[i], curve.add(points[j], points[k]))
                if left != right:
                    raise AssertionError(f"associativity failed for {i}, {j}, {k}")
                checked += 1
                progress.update(checked)
    progress.done(checked)
    print(f"Finite associativity check: {checked} triples passed.")
    print("This finite check is evidence for the implementation, not a proof of the group law.")


if __name__ == "__main__":
    main()
