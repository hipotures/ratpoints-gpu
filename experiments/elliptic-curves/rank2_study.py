#!/usr/bin/env python3
"""End-to-end exact rank-2 Mordell-Weil computation on LMFDB curve 389.a1."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common.progress import Progress  # noqa: E402

LMFDB_URL = "https://www.lmfdb.org/EllipticCurve/Q/389/a/1"
REFERENCE_HEIGHT_P = 0.32700077365160495184325924541
REFERENCE_HEIGHT_Q = 0.47671165934373953737948605888
REFERENCE_REGULATOR = 0.15246017794314375162432475705

EXPECTED_INTEGRAL_POINTS = {
    (-2, 0), (-2, -1), (-1, 1), (-1, -2), (0, 0), (0, -1),
    (1, 0), (1, -1), (3, 5), (3, -6), (4, 8), (4, -9),
    (6, 15), (6, -16), (39, 246), (39, -247),
    (133, 1539), (133, -1540), (188, 2584), (188, -2585),
}


@dataclass(frozen=True)
class Point:
    x: Fraction | None = None
    y: Fraction | None = None

    @property
    def is_infinity(self) -> bool:
        return self.x is None and self.y is None

    def __str__(self) -> str:
        return "O" if self.is_infinity else f"({self.x}, {self.y})"


O = Point()


@dataclass(frozen=True)
class GeneralWeierstrassCurve:
    a1: Fraction
    a2: Fraction
    a3: Fraction
    a4: Fraction
    a6: Fraction

    @property
    def discriminant(self) -> Fraction:
        b2 = self.a1 * self.a1 + 4 * self.a2
        b4 = 2 * self.a4 + self.a1 * self.a3
        b6 = self.a3 * self.a3 + 4 * self.a6
        b8 = (self.a1 * self.a1 * self.a6 + 4 * self.a2 * self.a6
              - self.a1 * self.a3 * self.a4 + self.a2 * self.a3 * self.a3
              - self.a4 * self.a4)
        return -b2 * b2 * b8 - 8 * b4**3 - 27 * b6**2 + 9 * b2 * b4 * b6

    def is_on_curve(self, point: Point) -> bool:
        if point.is_infinity:
            return True
        assert point.x is not None and point.y is not None
        x, y = point.x, point.y
        return (y * y + self.a1 * x * y + self.a3 * y
                == x**3 + self.a2 * x * x + self.a4 * x + self.a6)

    def require_point(self, point: Point) -> None:
        if not self.is_on_curve(point):
            raise ArithmeticError(f"point is not on curve: {point}")

    def negate(self, point: Point) -> Point:
        if point.is_infinity:
            return point
        assert point.x is not None and point.y is not None
        result = Point(point.x, -point.y - self.a1 * point.x - self.a3)
        self.require_point(result)
        return result

    def add(self, left: Point, right: Point) -> Point:
        if left.is_infinity:
            return right
        if right.is_infinity:
            return left
        assert left.x is not None and left.y is not None
        assert right.x is not None and right.y is not None
        x1, y1, x2, y2 = left.x, left.y, right.x, right.y
        if x1 == x2 and y1 + y2 + self.a1 * x1 + self.a3 == 0:
            return O
        if left == right:
            denominator = 2 * y1 + self.a1 * x1 + self.a3
            if denominator == 0:
                return O
            slope = (3 * x1 * x1 + 2 * self.a2 * x1 + self.a4 - self.a1 * y1) / denominator
            intercept = (-x1**3 + self.a4 * x1 + 2 * self.a6 - self.a3 * y1) / denominator
        else:
            denominator = x2 - x1
            if denominator == 0:
                return O
            slope = (y2 - y1) / denominator
            intercept = (y1 * x2 - y2 * x1) / denominator
        x3 = slope * slope + self.a1 * slope - self.a2 - x1 - x2
        y3 = -(slope + self.a1) * x3 - intercept - self.a3
        result = Point(x3, y3)
        self.require_point(result)
        return result

    def multiply(self, scalar: int, point: Point) -> Point:
        if scalar < 0:
            return self.multiply(-scalar, self.negate(point))
        result = O
        addend = point
        n = scalar
        while n:
            if n & 1:
                result = self.add(result, addend)
            addend = self.add(addend, addend)
            n >>= 1
        return result


def stage(number: int, title: str) -> float:
    print(f"\n=== Stage {number}: {title} ===", flush=True)
    return time.perf_counter()


def elapsed(started: float) -> float:
    return time.perf_counter() - started


def naive_x_height(point: Point) -> float:
    if point.is_infinity:
        return 0.0
    assert point.x is not None
    return math.log(max(abs(point.x.numerator), point.x.denominator))


def height_estimate(curve: GeneralWeierstrassCurve, point: Point, scale: int) -> float:
    multiple = curve.multiply(scale, point)
    return naive_x_height(multiple) / (scale * scale)


def powers_to(limit: int) -> list[int]:
    values: list[int] = []
    value = 4
    while value <= limit:
        values.append(value)
        value *= 2
    if not values or values[-1] != limit:
        values.append(limit)
    return sorted(set(values))


def git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent.parent,
            check=True, capture_output=True, text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def point_bits(point: Point) -> tuple[int, int, int, int]:
    if point.is_infinity:
        return (0, 0, 0, 0)
    assert point.x is not None and point.y is not None
    return (
        abs(point.x.numerator).bit_length(), point.x.denominator.bit_length(),
        abs(point.y.numerator).bit_length(), point.y.denominator.bit_length(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, default=80,
                        help="enumerate -R <= m,n <= R (default: 80)")
    parser.add_argument("--height-scale", type=int, default=128,
                        help="largest multiplier used for height convergence (default: 128)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "rank2-study-latest.json")
    parser.add_argument("--no-json", action="store_true",
                        help="do not write the JSON report")
    args = parser.parse_args()
    if args.radius < 1:
        parser.error("--radius must be positive")
    if args.height_scale < 4:
        parser.error("--height-scale must be at least 4")

    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    curve = GeneralWeierstrassCurve(*(Fraction(v) for v in (0, 1, 1, -2, 0)))
    p = Point(Fraction(0), Fraction(0))
    q = Point(Fraction(1), Fraction(0))

    started = stage(1, "exact curve and generator validation")
    curve.require_point(p)
    curve.require_point(q)
    if curve.discriminant != 389:
        raise AssertionError(f"unexpected discriminant: {curve.discriminant}")
    p_plus_q = curve.add(p, q)
    two_p = curve.multiply(2, p)
    two_q = curve.multiply(2, q)
    print("Curve: y^2 + y = x^3 + x^2 - 2x")
    print(f"Exact discriminant: {curve.discriminant}")
    print(f"P = {p}")
    print(f"Q = {q}")
    print(f"P + Q = {p_plus_q}")
    print(f"2P = {two_p}")
    print(f"2Q = {two_q}")
    timings["stage1_validation"] = elapsed(started)

    started = stage(2, "canonical-height convergence and regulator estimate")
    height_rows: list[dict[str, float | int]] = []
    print("scale        h(P) estimate        h(Q) estimate      regulator estimate")
    print("-----  -------------------  -------------------  -------------------")
    final_hp = final_hq = final_cross = final_regulator = 0.0
    for scale in powers_to(args.height_scale):
        hp = height_estimate(curve, p, scale)
        hq = height_estimate(curve, q, scale)
        hpq = height_estimate(curve, p_plus_q, scale)
        cross = (hpq - hp - hq) / 2.0
        regulator = hp * hq - cross * cross
        if regulator <= 0:
            raise ArithmeticError("estimated height pairing is not positive definite")
        print(f"{scale:5d}  {hp:19.15f}  {hq:19.15f}  {regulator:19.15f}")
        height_rows.append({
            "scale": scale, "height_p": hp, "height_q": hq,
            "height_p_plus_q": hpq, "pairing_pq": cross,
            "regulator": regulator,
        })
        final_hp, final_hq, final_cross, final_regulator = hp, hq, cross, regulator
    print("Reference (LMFDB):")
    print(f"  h(P)      = {REFERENCE_HEIGHT_P:.15f}")
    print(f"  h(Q)      = {REFERENCE_HEIGHT_Q:.15f}")
    print(f"  regulator = {REFERENCE_REGULATOR:.15f}")
    print(f"Absolute regulator error at scale {args.height_scale}: "
          f"{abs(final_regulator - REFERENCE_REGULATOR):.3e}")
    timings["stage2_heights"] = elapsed(started)

    started = stage(3, "precompute exact generator multiples")
    pre_progress = Progress("rank2-precompute", total=2 * args.radius)
    p_multiples = {0: O}
    q_multiples = {0: O}
    current_p = O
    current_q = O
    completed = 0
    for n in range(1, args.radius + 1):
        current_p = curve.add(current_p, p)
        p_multiples[n] = current_p
        p_multiples[-n] = curve.negate(current_p)
        completed += 1
        pre_progress.update(completed, p_done=n, q_done=0)
    for n in range(1, args.radius + 1):
        current_q = curve.add(current_q, q)
        q_multiples[n] = current_q
        q_multiples[-n] = curve.negate(current_q)
        completed += 1
        pre_progress.update(completed, p_done=args.radius, q_done=n)
    pre_progress.done(completed, p_done=args.radius, q_done=args.radius)
    timings["stage3_precompute"] = elapsed(started)

    started = stage(4, "exact rank-2 lattice enumeration")
    total_pairs = (2 * args.radius + 1) ** 2 - 1
    print(f"Enumerating {total_pairs:,} nonzero coefficient pairs (m,n) with "
          f"-R <= m,n <= R, R={args.radius}.", flush=True)
    lattice_progress = Progress("rank2-lattice", total=total_pairs)
    completed = 0
    relations: list[tuple[int, int]] = []
    integral_points: set[tuple[int, int]] = set()
    max_bits = 0
    max_bits_pair = (0, 0)
    max_bits_components = (0, 0, 0, 0)
    for m in range(-args.radius, args.radius + 1):
        mp = p_multiples[m]
        for n in range(-args.radius, args.radius + 1):
            if m == 0 and n == 0:
                continue
            point = curve.add(mp, q_multiples[n])
            completed += 1
            if point.is_infinity:
                relations.append((m, n))
            else:
                assert point.x is not None and point.y is not None
                if point.x.denominator == 1 and point.y.denominator == 1:
                    integral_points.add((int(point.x), int(point.y)))
                bits = point_bits(point)
                largest = max(bits)
                if largest > max_bits:
                    max_bits = largest
                    max_bits_pair = (m, n)
                    max_bits_components = bits
            if completed % 64 == 0:
                lattice_progress.update(
                    completed, integral=len(integral_points),
                    relations=len(relations), max_bits=max_bits,
                )
    lattice_progress.done(
        completed, integral=len(integral_points),
        relations=len(relations), max_bits=max_bits,
    )
    if completed != total_pairs:
        raise AssertionError(f"enumeration count mismatch: {completed} != {total_pairs}")
    if relations:
        raise AssertionError(f"nonzero relation(s) found in coefficient box: {relations[:5]}")

    expected_recovered = EXPECTED_INTEGRAL_POINTS.intersection(integral_points)
    missing_integral = EXPECTED_INTEGRAL_POINTS - integral_points
    extra_integral = integral_points - EXPECTED_INTEGRAL_POINTS
    if args.radius >= 4 and missing_integral:
        raise AssertionError(f"known integral points missing from lattice: {sorted(missing_integral)}")
    print(f"Nonzero relations to O in box: {len(relations)}")
    print(f"Integral points observed: {len(integral_points)}")
    print(f"Known LMFDB integral points recovered: {len(expected_recovered)}/20")
    if extra_integral:
        print(f"Additional integral points observed beyond reference list: {len(extra_integral)}")
    print(f"Largest coordinate component observed: {max_bits:,} bits at (m,n)={max_bits_pair}")
    print("Bit lengths there (x numerator, x denominator, y numerator, y denominator): "
          f"{max_bits_components}")
    timings["stage4_lattice"] = elapsed(started)

    started = stage(5, "compare lattice directions with the height quadratic form")
    half = max(1, args.radius // 2)
    sample_pairs = [
        (args.radius, 0), (0, args.radius),
        (args.radius, args.radius), (args.radius, -args.radius),
        (half, args.radius), (args.radius, half),
    ]
    direction_rows: list[dict[str, float | int]] = []
    print("       (m,n)       observed h_x      predicted q(m,n)      difference")
    print("------------  -----------------  -------------------  --------------")
    for m, n in sample_pairs:
        point = curve.add(p_multiples[m], q_multiples[n])
        observed = naive_x_height(point)
        predicted = (m * m * final_hp + 2 * m * n * final_cross
                     + n * n * final_hq)
        difference = observed - predicted
        print(f"({m:4d},{n:4d})  {observed:17.6f}  {predicted:19.6f}  {difference:14.6f}")
        direction_rows.append({
            "m": m, "n": n, "observed_naive_x_height": observed,
            "predicted_height_quadratic": predicted, "difference": difference,
        })
    timings["stage5_directions"] = elapsed(started)

    started = stage(6, "summary and report")
    print("Computed in this run:")
    print(f"  exact lattice points evaluated: {completed:,}")
    print(f"  nonzero relations found in the box: {len(relations)}")
    print(f"  integral points observed: {len(integral_points)}")
    print(f"  final regulator estimate: {final_regulator:.15f}")
    print(f"  largest observed coordinate component: {max_bits:,} bits")
    print("External reference facts used for comparison (LMFDB 389.a1):")
    print("  Mordell-Weil rank = 2, torsion = trivial, P and Q are generators.")
    print(f"  regulator = {REFERENCE_REGULATOR:.15f}")
    print("The bounded lattice computation is not, by itself, a proof of rank 2 or of completeness of integral points.")

    timings["stage6_summary"] = elapsed(started)
    timings["total"] = elapsed(total_started)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "environment": {
            "python": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "curve": {
            "lmfdb_label": "389.a1",
            "equation": "y^2 + y = x^3 + x^2 - 2x",
            "a_invariants": [0, 1, 1, -2, 0],
            "discriminant": 389,
            "generator_p": [0, 0],
            "generator_q": [1, 0],
        },
        "reference": {
            "source": LMFDB_URL,
            "rank": 2,
            "torsion": "trivial",
            "height_p": REFERENCE_HEIGHT_P,
            "height_q": REFERENCE_HEIGHT_Q,
            "regulator": REFERENCE_REGULATOR,
            "integral_points_count": 20,
        },
        "parameters": {"radius": args.radius, "height_scale": args.height_scale},
        "height_convergence": height_rows,
        "height_pairing_final": {
            "matrix": [[final_hp, final_cross], [final_cross, final_hq]],
            "regulator": final_regulator,
            "regulator_absolute_error": abs(final_regulator - REFERENCE_REGULATOR),
        },
        "lattice": {
            "nonzero_pairs": completed,
            "relations_to_infinity": relations,
            "integral_points_found": [list(item) for item in sorted(integral_points)],
            "known_integral_points_recovered": len(expected_recovered),
            "known_integral_points_missing": [list(item) for item in sorted(missing_integral)],
            "extra_integral_points": [list(item) for item in sorted(extra_integral)],
            "maximum_component_bits": max_bits,
            "maximum_component_pair": list(max_bits_pair),
            "maximum_component_bit_lengths": list(max_bits_components),
        },
        "direction_samples": direction_rows,
        "timings_seconds": timings,
    }
    if not args.no_json:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"JSON report: {args.output}")
    print(f"Total wall time: {timings['total']:.3f} s")


if __name__ == "__main__":
    main()
