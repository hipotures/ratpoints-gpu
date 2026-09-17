from __future__ import annotations

import io
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.elliptic import EllipticCurve, INFINITY, Point, naive_x_height
from common.progress import Progress
from common.search import search_integral_curve


class EllipticMathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.curve = EllipticCurve(Fraction(0), Fraction(-2))
        self.p = Point(Fraction(3), Fraction(5))

    def test_known_point_and_group_identities(self) -> None:
        self.assertTrue(self.curve.is_on_curve(self.p))
        self.assertEqual(self.curve.add(self.p, INFINITY), self.p)
        self.assertEqual(self.curve.add(self.p, self.curve.negate(self.p)), INFINITY)
        self.assertEqual(
            self.curve.multiply(2, self.p),
            Point(Fraction(129, 100), Fraction(-383, 1000)),
        )

    def test_scalar_multiplication_matches_repeated_addition(self) -> None:
        running = INFINITY
        for n in range(1, 12):
            running = self.curve.add(running, self.p)
            self.assertEqual(self.curve.multiply(n, self.p), running)

    def test_associativity_sample(self) -> None:
        points = [self.curve.multiply(n, self.p) for n in range(-3, 4)]
        for p in points:
            for q in points:
                for r in points:
                    self.assertEqual(
                        self.curve.add(self.curve.add(p, q), r),
                        self.curve.add(p, self.curve.add(q, r)),
                    )

    def test_bounded_rational_search_contains_known_points(self) -> None:
        result = search_integral_curve(0, -2, 10, 10)
        self.assertIn(self.p, result.points)
        self.assertIn(self.curve.negate(self.p), result.points)
        for point in result.points:
            self.assertTrue(self.curve.is_on_curve(point))

    def test_height_is_positive_for_nontrivial_point(self) -> None:
        self.assertGreater(naive_x_height(self.curve.multiply(4, self.p)), 0.0)

    def test_progress_non_tty_emits_lines(self) -> None:
        stream = io.StringIO()
        progress = Progress("test-stage", total=10, stream=stream, interval=0.0)
        progress.update(5, points=2)
        progress.done(10, points=3)
        text = stream.getvalue()
        self.assertIn("test-stage", text)
        self.assertIn("50.0%", text)
        self.assertIn("points=3", text)
        self.assertGreaterEqual(text.count("\n"), 3)


if __name__ == "__main__":
    unittest.main()
