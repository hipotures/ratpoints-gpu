#!/usr/bin/env python3
"""Test ordered multi-GPU equivalence; --mock adds CPU-only fault injection."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from benchmark_multi_gpu import parse_devices


def oracle(coefficients, height, first, last):
    """Independent exhaustive Python integer oracle for small test boxes."""
    coeff = list(map(int, coefficients.split()))
    degree = len(coeff) - 1
    output = []

    def append(a, value, b):
        if value < 0:
            return
        root = math.isqrt(value)
        if root * root != value:
            return
        output.append(f"({a} : {root} : {b})\n")
        if root:
            output.append(f"({a} : {-root} : {b})\n")

    append(1, 0 if degree % 2 else coeff[-1], 0)
    for b in range(first, last + 1):
        for a in range(-height, height + 1):
            if math.gcd(a, b) != 1:
                continue
            value = sum(c * a ** i * b ** (degree - i) for i, c in enumerate(coeff))
            append(a, value * (b if degree % 2 else 1), b)
    return "".join(output).encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary")
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("0,1"))
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    binary = str(Path(args.binary).resolve(strict=True))
    checks = 0
    env = os.environ.copy()
    env.pop("RATPOINTS_GPU_BENCHMARK", None)
    if args.mock:
        env["RATPOINTS_TEST_DEVICE_COUNT"] = "3"
        env.pop("CUDA_VISIBLE_DEVICES", None)

    def run(arguments, extra_env=None, success=True):
        nonlocal checks
        current = dict(env)
        current.update(extra_env or {})
        result = subprocess.run([binary, *arguments], env=current,
                                capture_output=True, timeout=60)
        if (result.returncode == 0) != success:
            raise AssertionError(f"unexpected exit {result.returncode}: {arguments}\n{result.stderr!r}")
        checks += 1
        return result

    listing = run(["--list-devices"]).stdout.decode()
    visible = {int(x) for x in re.findall(r"^device=(\d+)", listing, re.M)}
    if not set(args.devices) <= visible:
        raise AssertionError(f"requested GPUs not available: {listing}")
    if args.mock != ("Mock GPU" in listing):
        raise AssertionError("mock/real backend mismatch: do not mistake CPU tests for GPU validation")
    selections = [str(args.devices[0]), ",".join(map(str, args.devices)),
                  ",".join(map(str, reversed(args.devices)))]
    run(["--help"], {"CUDA_VISIBLE_DEVICES": ""})
    cases = [
        ("1 0 1", 35, 1, 17),
        ("1 1 0 1", 35, 3, 15),
        ("1 0 1 0 1", 35, 1, 20),
        ("-3 5 -7 11 2", 35, 4, 19),
        ("+1 0 +1", 5, 1, 1),
        ("1 " + "0 " * 99 + "1", 7, 1, 3),
    ]
    for coefficients, height, first, last in cases:
        expected = oracle(coefficients, height, first, last)
        common = [coefficients, str(height), "-dl", str(first), "-du", str(last)]
        for selection in selections:
            result = run([*common, "--devices", selection, "--batch-size", "3", "-v"])
            if result.stdout != expected:
                raise AssertionError(f"oracle mismatch: {common}, GPUs={selection}")
            summary = re.search(rb"denominators=(\d+)", result.stderr)
            if not summary or int(summary[1]) != last - first + 1:
                raise AssertionError("incorrect completed-denominator count")
    options = [[], ["-i"], ["-y"], ["-z"], ["-1"], ["-i", "-1"],
               ["-l", "-1", "-u", "0", "-l", "0.5", "-u", "2"],
               ["-i", "-1", "-l", "1", "-u", "2"],
               ["-f", "%x,%y,%z", "-fs", "[", "-fm", ";", "-fe", "]"],
               ["-f", "--devices"], ["-i", "-I", "-y", "-Y"]]
    for option in options:
        command = ["1 0 1", "40", "-du", "23", "--batch-size", "4", *option]
        expected = run([*command, "--devices", selections[0]]).stdout
        for selection in selections[1:]:
            if run([*command, "--devices", selection]).stdout != expected:
                raise AssertionError(f"ordered output/first-point mismatch: {option}")
        stop_requested = option[:1] == ["-1"] or option[:2] == ["-i", "-1"]
        if stop_requested and len(expected.splitlines()) != 1:
            raise AssertionError("-1 did not emit exactly one point")
    default = run(["1 0 1", "15"]).stdout
    if default != run(["1 0 1", "15", "--devices", "all"]).stdout:
        raise AssertionError("default GPU selection differs from all")
    for value in ("", ",", "0,", ",0", "0,,1", "0,0", "00,0", "-1", "+0",
                  "1.0", "0 1", "all,0", "99999999999999999999999", str(max(visible) + 1)):
        run(["1 0 1", "5", "--devices", value], success=False)
    run(["1 0 1", "5", "--devices"], success=False)
    for size in ("0", "-1", "65537", "abc"):
        run(["1 0 1", "5", "--batch-size", size], success=False)
    run(["1 0 1", "5", "--batch-size"], success=False)
    run(["1 -2 1", "5"], success=False)
    run(["--list-devices"], {"CUDA_VISIBLE_DEVICES": ""}, success=False)

    # A dense polynomial forces multiple survivor flushes in the real CUDA sieve.
    primes = [p for p in range(2, 512)
              if all(p % d for d in range(2, math.isqrt(p) + 1))]
    dense = f"1 {math.prod(primes[-32:])}"
    expected = oracle(dense, 5000, 1, 2)
    for selection in selections[:2]:
        actual = run([dense, "5000", "-du", "2", "--devices", selection, "-v"])
        if actual.stdout != expected or b"survivors=20002 exact_survivors=1" not in actual.stderr:
            raise AssertionError("streaming/dense-survivor regression")

    if args.mock:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "workers.log"
            base = ["1 0 1", "20", "-i", "--devices", "0,2", "--batch-size", "3"]
            run(base, {"RATPOINTS_TEST_LOG": str(log),
                       "RATPOINTS_TEST_REQUIRE_PARALLEL": "1"})
            lines = [line.split() for line in log.read_text().splitlines()]
            starts = [tuple(map(int, line[1:])) for line in lines if line[0] == "start"]
            ends = [tuple(map(int, line[1:])) for line in lines if line[0] == "end"]
            covered = sorted(b for _, lo, hi in starts for b in range(lo, hi + 1))
            if covered != list(range(1, 21)) or sorted(starts) != sorted(ends):
                raise AssertionError("gap, overlap, or unjoined worker")
            if {device for device, _, _ in starts} != {0, 2}:
                raise AssertionError("wrong GPU selected")
            for name in ("RATPOINTS_TEST_FAIL_DEVICE", "RATPOINTS_TEST_ACTIVATE_ERROR"):
                result = run(base, {name: "2"}, success=False)
                if b"GPU 2" not in result.stderr or result.stdout:
                    raise AssertionError("worker failure was lost or failing wave emitted output")
            run(["--list-devices"], {"RATPOINTS_TEST_DRIVER_ERROR": "1"}, success=False)
            run(["--list-devices"], {"RATPOINTS_TEST_DEVICE_COUNT": "0"}, success=False)
            result = run(["1 0 1", "5", "--devices", "all", "-v"],
                         {"RATPOINTS_TEST_DEVICE_COUNT": "1"})
            if b"devices=1" not in result.stderr:
                raise AssertionError("single visible GPU failed")
            # Exercise the full 65,536 boundary and the signed-int endpoint
            # without simulating billions of numerators on a CPU.
            for lo, hi in ((1, 131075), (2147483645, 2147483647)):
                log.write_text("")
                run(["1 0 1", str(hi), "-dl", str(lo), "-du", str(hi),
                     "--devices", "0,1", "-z"],
                    {"RATPOINTS_TEST_BOUNDARIES_ONLY": "1", "RATPOINTS_TEST_LOG": str(log)})
                starts = [tuple(map(int, line.split()[1:])) for line in log.read_text().splitlines()
                          if line.startswith("start ")]
                covered = sorted(b for _, first, last in starts for b in range(first, last + 1))
                if covered != list(range(lo, hi + 1)):
                    raise AssertionError("boundary coverage regression")
            report = Path(directory) / "benchmark.json"
            command = [sys.executable, str(Path(__file__).with_name("benchmark_multi_gpu.py")),
                       "--binary", binary, "--devices", "0,1", "--height", "12",
                       "--repeats", "1", "--warmups", "0", "--json", str(report)]
            subprocess.run(command, env=env, check=True, capture_output=True, timeout=60)
            data = json.loads(report.read_text())
            if len(data["results"]) != 3 or not data["output_sha256"]:
                raise AssertionError("invalid benchmark report")
            bad_env = dict(env, RATPOINTS_TEST_DROP_DEVICE="1")
            mismatch = subprocess.run(command, env=bad_env, capture_output=True, timeout=60)
            if mismatch.returncode == 0 or b"output differs" not in mismatch.stderr:
                raise AssertionError("benchmark accepted inconsistent outputs")
            checks += 2
    print(f"Passed {checks} checks ({'mock CUDA; no GPU execution' if args.mock else 'real CUDA GPUs'}).")


if __name__ == "__main__":
    main()
