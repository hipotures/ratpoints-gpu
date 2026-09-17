#!/usr/bin/env python3
"""Bounded exact rational-point search on y^2 = x^3 - 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.progress import Progress  # noqa: E402
from common.search import search_integral_curve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerator-bound", type=int, default=2000)
    parser.add_argument("--denominator-bound", type=int, default=1000)
    parser.add_argument("--max-print", type=int, default=30,
                        help="maximum number of points to print (default: 30)")
    args = parser.parse_args()
    if args.numerator_bound < 1 or args.denominator_bound < 1 or args.max_print < 0:
        parser.error("bounds must be positive and --max-print must be nonnegative")

    total = (2 * args.numerator_bound + 1) * args.denominator_bound
    progress = Progress("rational-points", total=total)

    def report(completed: int, reduced: int, points: int) -> None:
        progress.update(completed, reduced=reduced, points=points)

    result = search_integral_curve(
        0, -2, args.numerator_bound, args.denominator_bound, report
    )
    progress.done(result.raw_pairs, reduced=result.reduced_x_values,
                  points=len(result.points))

    print("Curve: y^2 = x^3 - 2")
    print(f"Raw (a,b) pairs visited: {result.raw_pairs:,}")
    print(f"Reduced rational x values tested: {result.reduced_x_values:,}")
    print(f"Rational affine points found in the box: {len(result.points):,}")
    for point in result.points[:args.max_print]:
        print(point)
    if len(result.points) > args.max_print:
        print(f"... {len(result.points) - args.max_print} more points not printed")
    print("The search is exhaustive only inside the requested numerator/denominator box.")


if __name__ == "__main__":
    main()
