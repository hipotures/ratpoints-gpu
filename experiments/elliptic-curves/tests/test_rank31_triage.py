from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank31_mn_common import discriminant, on_curve
from rank31_t_minus_47_80 import T, add, exact_report, negate


class ExactTriageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = exact_report()

    def test_exact_specialization_and_integral_change(self) -> None:
        report = self.report
        self.assertEqual(T, Fraction(-47, 80))
        self.assertEqual(report["family_values"]["L"], "743773643/6400")
        family = tuple(map(Fraction, report["models"]["family"]["ainvariants"]))
        integral = tuple(map(Fraction, report["models"]["integral"]["ainvariants"]))
        self.assertTrue(all(value.denominator == 1 for value in integral))
        u = Fraction(report["family_to_integral_change"]["u"])
        self.assertEqual(discriminant(family)/u**12, discriminant(integral))
        self.assertEqual(report["models"]["family"]["j_invariant"],
                         report["models"]["integral"]["j_invariant"])

    def test_points_relation_and_rigorous_lower_bound(self) -> None:
        report = self.report
        family = tuple(map(Fraction, report["models"]["family"]["ainvariants"]))
        integral = tuple(map(Fraction, report["models"]["integral"]["ainvariants"]))
        points = report["visible_points"]
        self.assertEqual(len(points), 6)
        for row in points:
            self.assertTrue(on_curve(tuple(map(Fraction, row["family_xy"])), family))
            self.assertTrue(on_curve(tuple(map(Fraction, row["integral_xy"])), integral))
        representatives = [tuple(map(Fraction, points[i]["family_xy"])) for i in (0, 2, 4)]
        self.assertIsNone(add(add(representatives[0], representatives[1], family),
                              representatives[2], family))
        self.assertIsNone(add(representatives[0], negate(representatives[0], family), family))
        self.assertEqual(report["rank"]["rigorous_lower_bound"], 1)
        self.assertIsNone(report["rank"]["global_upper_bound"])

    def test_torsion_is_excluded_by_two_good_reductions(self) -> None:
        counts = [(row["prime"], row["point_count"])
                  for row in self.report["torsion"]["good_prime_reductions"]]
        self.assertEqual(counts, [(7, 13), (13, 21)])
        self.assertEqual(self.report["torsion"]["status"], "proven trivial")


if __name__ == "__main__":
    unittest.main()
