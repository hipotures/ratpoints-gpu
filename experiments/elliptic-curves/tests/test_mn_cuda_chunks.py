from __future__ import annotations

import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank31_mn_common import compile_helper, cpu_np, odd_primes_up_to


@unittest.skipUnless(shutil.which("nvcc"), "CUDA compiler is unavailable")
class CUDAChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binary, _ = compile_helper()

    def test_large_cumulative_offsets_and_deterministic_chunks(self) -> None:
        cmd = [str(self.binary), "--plan-only", "--prime-bound", "500000",
               "--chi-chunk-bytes", str(256 * 1024 * 1024)]
        first = subprocess.check_output(cmd, text=True)
        self.assertEqual(first, subprocess.check_output(cmd, text=True))
        fields = dict(item.split("=", 1) for item in first.split())
        self.assertEqual(int(fields["total_chi_entries"]), 9_914_236_193)
        self.assertGreater(int(fields["total_chi_entries"]), 2**31 - 1)
        self.assertGreater(int(fields["chunks"]), 1)
        self.assertLessEqual(int(fields["max_chi_bytes"]), 256 * 1024 * 1024)

    def test_unsupported_chunk_budget_fails_cleanly(self) -> None:
        done = subprocess.run([str(self.binary), "--plan-only", "--prime-bound", "97",
                               "--chi-chunk-bytes", "50"], capture_output=True, text=True)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("smaller than prime", done.stderr)

    @unittest.skipUnless(shutil.which("nvidia-smi"), "CUDA device is unavailable")
    def test_multi_chunk_scores_and_counts_match_one_chunk_and_cpu(self) -> None:
        values = [Fraction(164518, 924945), Fraction(1, 2), Fraction(3, 7)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidates.tsv"
            source.write_text("".join(f"{i}\t{t.numerator}\t{t.denominator}\n"
                                      for i, t in enumerate(values)))
            results = []
            for budget in (1_000_000, 200):
                output = root / f"scores-{budget}.tsv"
                counts = root / f"counts-{budget}.tsv"
                done = subprocess.run([str(self.binary), "--device", "0",
                                       "--input", str(source), "--output", str(output),
                                       "--prime-bound", "97", "--counts", str(counts),
                                       "--chi-chunk-bytes", str(budget)],
                                      capture_output=True, text=True, check=True)
                self.assertIn("prime_chunks=1" if budget == 1_000_000 else "prime_chunks=6",
                              done.stderr)
                results.append((output.read_text().splitlines(), counts.read_text().splitlines()))
            self.assertEqual(results[0][1], results[1][1])
            for original, chunked in zip(results[0][0], results[1][0]):
                oi, oscore, ogood = original.split()
                ci, cscore, cgood = chunked.split()
                self.assertEqual((oi, ogood), (ci, cgood))
                self.assertTrue(math.isclose(float(oscore), float(cscore),
                                             rel_tol=1e-10, abs_tol=1e-9))
            self.assertEqual(len(results[1][1]), len(values) * len(odd_primes_up_to(97)))
            for line in results[1][1]:
                index, prime, count = map(int, line.split())
                expected = cpu_np(values[index], prime)
                self.assertEqual(count, -1 if expected is None else expected)


if __name__ == "__main__":
    unittest.main()
