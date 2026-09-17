"""Exact bounded rational-point search for integral short Weierstrass curves."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isqrt
from typing import Callable

from .elliptic import Point


@dataclass(frozen=True)
class SearchResult:
    points: tuple[Point, ...]
    raw_pairs: int
    reduced_x_values: int


def _square_root_fraction(numerator: int, denominator: int) -> Fraction | None:
    if numerator < 0:
        return None
    common = gcd(numerator, denominator)
    numerator //= common
    denominator //= common
    nroot = isqrt(numerator)
    droot = isqrt(denominator)
    if nroot * nroot != numerator or droot * droot != denominator:
        return None
    return Fraction(nroot, droot)


def search_integral_curve(
    curve_a: int,
    curve_b: int,
    numerator_bound: int,
    denominator_bound: int,
    progress: Callable[[int, int, int], None] | None = None,
) -> SearchResult:
    """Search reduced x=a/b for y^2=x^3+A*x+B using exact integer tests."""
    if numerator_bound < 1 or denominator_bound < 1:
        raise ValueError("bounds must be positive")
    points: set[Point] = set()
    reduced = 0
    raw = 0
    width = 2 * numerator_bound + 1
    for denominator in range(1, denominator_bound + 1):
        b2 = denominator * denominator
        b3 = b2 * denominator
        for numerator in range(-numerator_bound, numerator_bound + 1):
            raw += 1
            if gcd(numerator, denominator) != 1:
                continue
            reduced += 1
            value_num = numerator**3 + curve_a * numerator * b2 + curve_b * b3
            y = _square_root_fraction(value_num, b3)
            if y is None:
                continue
            x = Fraction(numerator, denominator)
            points.add(Point(x, y))
            if y:
                points.add(Point(x, -y))
        if progress is not None:
            progress(denominator * width, reduced, len(points))
    ordered = tuple(sorted(points, key=lambda p: (p.x, p.y)))
    return SearchResult(ordered, raw, reduced)
