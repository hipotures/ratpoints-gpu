from __future__ import annotations

import gzip
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank31_mestre_nagao_campaign import (
    merge_finalists, parse_args, ranked_new_leads, write_stage_a_parts,
)
from rank31_mn_common import RECORD_T, RESULTS_DIR, ScoreRow, generate_candidates


class CampaignTests(unittest.TestCase):
    def test_defaults_and_seed_configuration(self) -> None:
        with patch.object(sys, "argv", ["campaign"]):
            args = parse_args()
        self.assertEqual(args.seeds, [304, 305, 306, 307, 308])
        self.assertEqual((args.candidates_per_shard, args.global_finalists),
                         (1_000_000, 50_000))
        self.assertEqual((args.first_prime_bound, args.second_prime_bound),
                         (5000, 100_000))
        with patch.object(sys, "argv", ["campaign", "--seeds", "9,11"]):
            self.assertEqual(parse_args().seeds, [9, 11])

    def test_global_merge_deduplicates_and_resolves_ties(self) -> None:
        shared = Fraction(3, 7)
        first = ranked_new_leads([
            ScoreRow(0, RECORD_T, 20.0, 7),
            ScoreRow(1, shared, 10.0, 5),
            ScoreRow(2, Fraction(4, 9), 9.0, 5),
        ], 1, 304, 2)
        second = ranked_new_leads([
            ScoreRow(0, RECORD_T, 20.0, 7),
            ScoreRow(1, shared, 10.0, 5),
            ScoreRow(2, Fraction(5, 11), 9.0, 6),
        ], 2, 305, 2)
        merged = merge_finalists([first, second], 3)
        self.assertEqual([lead.t for lead in merged],
                         [shared, Fraction(5, 11), Fraction(4, 9)])
        self.assertEqual([lead.seed for lead in merged], [304, 305, 304])
        self.assertTrue(all(lead.t != RECORD_T for lead in merged))

    def test_seed_shards_are_repeatable_and_distinct(self) -> None:
        first = generate_candidates(12, 304, 1000, 5000, Fraction(1, 2))
        self.assertEqual(first, generate_candidates(12, 304, 1000, 5000, Fraction(1, 2)))
        self.assertNotEqual(first[1:], generate_candidates(12, 305, 1000, 5000, Fraction(1, 2))[1:])
        self.assertEqual(first[0], RECORD_T)

    def test_stage_a_parts_are_complete_and_deterministic(self) -> None:
        rows = [ScoreRow(i, Fraction(i + 1, i + 2), float(i), i)
                for i in range(5)]
        with tempfile.TemporaryDirectory(dir=RESULTS_DIR) as first, tempfile.TemporaryDirectory(dir=RESULTS_DIR) as second:
            with patch("rank31_mestre_nagao_campaign.ARTIFACT_SPLIT_ROWS", 2):
                a = write_stage_a_parts(Path(first), "sample", rows)
                b = write_stage_a_parts(Path(second), "sample", rows)
            self.assertEqual([part["row_count"] for part in a], [2, 2, 1])
            self.assertEqual([part["sha256"] for part in a],
                             [part["sha256"] for part in b])
            lines = []
            for path in sorted(Path(first).glob("*.gz")):
                with gzip.open(path, "rt") as source:
                    lines.extend(source.readlines())
            self.assertEqual(len(lines), len(rows))


if __name__ == "__main__":
    unittest.main()
