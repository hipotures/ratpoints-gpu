#!/usr/bin/env python3
"""Check the published record-curve points inside projective height 10^6.

Point data: https://www.mathe2.uni-bayreuth.de/stoll/recordcurve.html
Reference: J. Steffen Mueller and Michael Stoll, "Canonical Heights on Genus
Two Jacobians," Algebra & Number Theory 10 (2016), 2153-2234.
https://doi.org/10.2140/ant.2016.10.2153
"""

from fractions import Fraction
from pathlib import Path
import re
import subprocess
import sys


COEFFICIENTS = (
    "247747600 -985905640 567207969 2396040466 "
    "52485681 -470135160 82342800"
)
HEIGHT = 1_000_000
POINT = re.compile(r"^\((-?\d+) : (\d+)\)$")


def published_points():
    path = Path(__file__).with_name("data") / "record_curve_x.txt"
    return {Fraction(line) for line in path.read_text().splitlines()}


def searched_points(executable):
    process = subprocess.run(
        [str(executable), COEFFICIENTS, str(HEIGHT), "-q", "-y", "-i"],
        check=True,
        text=True,
        capture_output=True,
    )
    output = set()
    for line in process.stdout.splitlines():
        match = POINT.match(line)
        if not match:
            raise AssertionError(f"malformed point output: {line!r}")
        output.add(Fraction(int(match[1]), int(match[2])))
    return output


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_record_curve.py GPU_EXECUTABLE")
    expected = {
        point for point in published_points()
        if abs(point.numerator) <= HEIGHT and point.denominator <= HEIGHT
    }
    actual = searched_points(Path(sys.argv[1]).resolve())
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise AssertionError(
            f"point mismatch: {len(missing)} missing, {len(unexpected)} unexpected\n"
            f"missing: {missing}\nunexpected: {unexpected}"
        )
    print(
        f"found all {len(expected)} published x-coordinates with height <= {HEIGHT}"
    )


if __name__ == "__main__":
    main()
