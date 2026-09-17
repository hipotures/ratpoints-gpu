from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank31_mn_common import (
    build_validation_oracle,
    odd_primes_up_to,
    select_validation_primes,
)


class ValidationSelectionTests(unittest.TestCase):
    def test_bounded_sample_spans_large_bound_deterministically(self) -> None:
        primes = odd_primes_up_to(100_000)
        selected = select_validation_primes(primes)
        self.assertEqual(selected, select_validation_primes(primes))
        self.assertEqual(len(selected), 256)
        self.assertTrue(set(p for p in primes if p <= 1000).issubset(selected))
        self.assertTrue(set(primes[-4:]).issubset(selected))
        self.assertEqual(selected[-1], primes[-1])

    def test_exhaustive_and_invalid_budget(self) -> None:
        primes = odd_primes_up_to(5000)
        self.assertEqual(select_validation_primes(primes, exhaustive=True), tuple(primes))
        with self.assertRaises(ValueError):
            select_validation_primes(primes, max_sampled_primes=100)

    def test_cpu_oracle_is_built_once_for_device_independent_pairs(self) -> None:
        candidates = [Fraction(1, 2), Fraction(2, 3), Fraction(3, 4), Fraction(4, 5)]
        with patch("rank31_mn_common.cpu_np", return_value=7) as cpu:
            oracle = build_validation_oracle(candidates, 29)
        self.assertEqual(cpu.call_count, len(oracle.candidates) * len(oracle.sampled_primes))
        self.assertEqual(len(oracle.expected), cpu.call_count)
        self.assertEqual(oracle.report()["candidate_indices"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
