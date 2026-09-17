#!/usr/bin/env python3
"""Observe exact coordinate growth and naive heights of nP on y^2=x^3-2."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.elliptic import EllipticCurve, Point, naive_x_height  # noqa: E402
from common.progress import Progress  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-multiple", type=int, default=40,
                        help="compute P through nP (default: 40)")
    args = parser.parse_args()
    if args.max_multiple < 1:
        parser.error("--max-multiple must be positive")

    curve = EllipticCurve(Fraction(0), Fraction(-2))
    p = Point(Fraction(3), Fraction(5))
    progress = Progress("heights", total=args.max_multiple)

    print("n  x-num-bits  x-den-bits  h_x(nP)      h_x(nP)/n^2")
    print("-  ----------  ----------  -----------  -------------")
    for n in range(1, args.max_multiple + 1):
        q = curve.multiply(n, p)
        assert q.x is not None
        height = naive_x_height(q)
        ratio = height / (n * n)
        print(f"{n:2d} {q.x.numerator.bit_length():11d} {q.x.denominator.bit_length():11d} "
              f"{height:11.6f} {ratio:13.8f}")
        progress.update(n, num_bits=q.x.numerator.bit_length(),
                        den_bits=q.x.denominator.bit_length())
    progress.done(args.max_multiple)
    print()
    print("h_x(Q) = log(max(|numerator(x(Q))|, denominator(x(Q)))).")
    print("The sequence h_x(nP)/n^2 is an empirical route toward the canonical-height idea.")
    print("These floating-point height values are presentation only; point arithmetic is exact.")


if __name__ == "__main__":
    main()
