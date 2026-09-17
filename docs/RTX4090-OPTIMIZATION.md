# RTX 4090 CUDA profiling and tuning

## Reproduce the search

The workload is Michael Stoll's record curve, numerator height 10,000,000,
and denominators 1 through 10,000,000. Run on an otherwise idle RTX 4090.

```bash
python3 tests/tune_cuda.py --output /tmp/ratpoints-gpu-issue1-sweep \
  --num-primes 28
```

The tuner makes a separate source tree and binary for each combination of
`INITIAL_PRIMES={14,12,10,8}` and shared or global mask rows. It uses two
discarded warmups and five measured runs per variant, checks byte-for-byte
ordered output and both survivor counts against the first run, and writes
`results.json` after each completed variant. It does not replace the repository
binary. The output directory should be outside the repository because it
contains eight full builds.

For a shorter screening run, pass `--height` and `--denominator-max` explicitly.
Those results cannot substitute for the full-box timing decision.

## Profilers

Build with line information and resource reporting only when profiling:

```bash
make clean
make NVCCFLAGS='-O3 -std=c++14 -lineinfo -Xptxas=-v' ARCH_FLAGS='-arch=sm_89'
bash tests/profile_cuda.sh nsys ./ratpoints_gpu /tmp/ratpoints-nsys
bash tests/profile_cuda.sh ncu ./ratpoints_gpu /tmp/ratpoints-ncu
```

The Nsight Systems command profiles the complete search's CUDA API timeline,
transfers, launches, and GPU idle gaps. The executable's phase timers measure
host plan construction and GMP verification. The Nsight Compute
command profiles one `sieve_kernel` launch for a 65,536-denominator batch at
height 10M and records occupancy, memory, scheduler, and instruction metrics.
The Nsight report files and full stdout belong outside the repository.

## Measurements

The initial Nsight Compute profile found `sieve_kernel<unsigned int,0>` at
about 36.8% achieved occupancy, with 26.32 KiB dynamic shared memory per
256-thread block limiting theoretical occupancy to 50%. DRAM throughput was
about 1.5% and compute/L1 throughput about 62%. This motivated the mask-row
and initial-prime sweep. `basis_kernel` launches were only 4–6 microseconds;
they were not optimized.

The original issue reported about 27.605 seconds on the faster single GPU and
14.240 seconds on two GPUs (height 10M, older software run). New measurements
and software provenance are recorded below.

## Full-box sweep, 17 September 2026

Source commit: `14e3c4aaaa34d434ce0bd364645eba77ab14cdc6` on `master`.
Hardware: two NVIDIA GeForce RTX 4090 cards, compute capability 8.9, PCI IDs
`0000:01:00.0` and `0000:02:00.0`. Driver: 615.71.09. CUDA compiler:
13.4.92. Builds used `-O3 -std=c++14 -arch=sm_89`, `BLOCK=256`,
`NUM_PRIMES=28`, and the indicated `INITIAL_PRIMES` and mask mode. GPU 0 was
used for all variant comparisons. Every variant had two discarded warmups and
five measured runs.

| Initial primes | Mask rows | Median (s) | Minimum (s) | Mean (s) | Std. dev. (s) | CV | Median sieve event sum (s) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 | shared | 27.823 | 27.791 | 27.824 | 0.027 | 0.098% | 24.608 |
| 14 | global | 43.055 | 42.981 | 43.040 | 0.038 | 0.088% | 39.790 |
| 12 | shared | 30.031 | 29.992 | 30.023 | 0.023 | 0.076% | 26.811 |
| 12 | global | 44.501 | 44.447 | 44.496 | 0.029 | 0.065% | 41.239 |
| 10 | shared | 41.930 | 41.860 | 41.921 | 0.046 | 0.111% | 38.695 |
| 10 | global | 51.285 | 51.229 | 51.265 | 0.033 | 0.064% | 48.057 |
| 8 | shared | 54.878 | 54.839 | 54.881 | 0.042 | 0.077% | 51.601 |
| 8 | global | 64.328 | 64.234 | 64.323 | 0.058 | 0.090% | 61.043 |

All 56 full-box runs produced byte-identical ordered output, SHA-256
`4d30d69412568b0459b9ca0f25a7519f662948dd0b7269a5bb07502d9268eef9`,
with 71,497,766 modular survivors and 313 exact points. The output was 19,025
bytes. The executable SHA-256 for each build and all individual measurements
are retained in the local tuner JSON; the compact summary is in
`docs/RTX4090-SWEEP.json`.

The acceptance threshold is `max(2%, 3 × CV)`, or 2% for the baseline. The
nearest alternative, 12/shared, is 7.94% slower. Thus no kernel configuration
from this sweep is accepted, and the portable defaults remain 14/shared. The
kernel event time increases with each rejected variant while basis time stays
about 0.09 seconds over the whole search. The observed performance loss is in
the sieve path, not basis construction. In the baseline, sieve events account
for about 89% of end-to-end time; the approximately 3.2-second remainder
includes startup, plan construction, allocations, transfers, synchronization,
GMP verification, and output. The additional wall-clock timers below split
the main components.

## Nsight observations

Nsight Systems on a full height-10M single-GPU search recorded 153
`sieve_kernel<unsigned int,0>` launches totaling 24.613 seconds and 4,896
`basis_kernel` launches totaling 20.0 milliseconds of device execution. CUDA
API durations included 0.579 seconds in 1,683 `cudaFree` calls, 0.207 seconds
in 10,863 `cudaMemcpy` calls, 0.056 seconds in 1,683 `cudaMalloc` calls, and
0.014 seconds in 153 `cudaMemGetInfo` calls. These API durations do not include
all host work or represent additive critical-path wall time. GPU copies took
0.089 seconds H2D and 0.066 seconds D2H. The event-based basis figure was
0.103 seconds; it includes mask metadata transfers as well as kernels.

Wall-clock timers added to the executable measured one further full-box GPU 0
search. These are sums across 153 batches, not overlapping GPU event times:

| Phase | Time (s) |
| :--- | ---: |
| `SievePlan` construction | 0.436 |
| Workspace allocation, metadata upload, kernel configuration | 0.109 |
| Mask basis setup and launch | 0.107 |
| Survivor capacity query and buffer allocation | 0.037 |
| Sieve loop excluding exact callback | 24.771 |
| Exact GMP callback | 1.388 |
| Search wall time | 27.539 |

The four explicit setup phases total 0.689 seconds (2.5% of wall time).
Adding all 0.579 seconds of `cudaFree` API time from the separate Nsight run
gives an approximate upper estimate of 4.6% for repeated setup. It remains
below the issue's 5% trigger for persistent device state; that refactor was
not made. The sieve loop dominates, and exact verification is about 5%.

Nsight Compute profiled one 65,536-block fast sieve launch at height 10M:
256 threads per block, 40 registers per thread, 26.32 KiB dynamic and 512 B
static shared memory per block, no spills, 50% theoretical occupancy and
42.53% achieved occupancy. Its 168.83 ms duration showed 85.96% compute and
L1/TEX throughput, 0.02% DRAM throughput, 56.27% L1 hit rate, 99.52% L2 hit
rate, 96.39% branch efficiency, and 22.84 active threads per warp. The ALU
and load/store pipelines were the busy units. Nsight reported shared-load
bank conflicts in about 17% of shared load wavefronts. Warp stalls were led
by `wait` (1.59) and math-pipe throttling (1.31)
average stalled warps per issue-active cycle, followed by long scoreboard
(0.35) and branch resolving (0.33). The ALU pipeline ran at 83% of peak
and the load/store pipeline at 86%; the instruction mix was dominated by
integer/logic and load/store operations. The achieved occupancy
and throughput differ from the earlier profile in the issue because the
profiled launch and software run are different; the 50% shared-memory limit
is consistent. Reducing shared memory by moving mask rows to global memory
or using fewer initial primes worsened the complete workload.

## Other staged screening

After the full mask/initial-prime sweep, the tuner was extended with `--block`,
`--num-primes`, and `--batch-size`. Each case below used height 10M, the first
1M denominators, one discarded warmup, and three measured runs on GPU 0. The
baseline was `BLOCK=256`, `NUM_PRIMES=28`, `INITIAL_PRIMES=14`, shared rows,
and batch size 65,536. These shorter runs screened candidates; they are not
full-box performance claims. Exact ordered output and exact point counts were
the same for every case. The compact results are in `docs/RTX4090-SCREEN.json`.

For example, reproduce the baseline and the 32-prime candidate with:

```bash
python3 tests/tune_cuda.py --output /tmp/screen-baseline --height 10000000 \
  --denominator-max 1000000 --warmups 1 --repeats 3 --primes 14 --masks shared \
  --num-primes 28
python3 tests/tune_cuda.py --output /tmp/screen-primes32 --height 10000000 \
  --denominator-max 1000000 --warmups 1 --repeats 3 --primes 14 --masks shared \
  --num-primes 32
```

Pass `--block 128` or `--batch-size 16384`/`32768` for the other cases.

| Change from baseline | Median (s) | Modular survivors | Decision |
| :--- | ---: | ---: | :--- |
| Baseline | 3.307 | 14,001,483 | Reference |
| `BLOCK=128` | 3.986 | 14,001,483 | 20.5% slower |
| `NUM_PRIMES=24` | 4.164 | 19,152,464 | Slower |
| `NUM_PRIMES=32` | 3.195 | 13,619,470 | Advance to full-box confirmation |
| Batch size 16,384 | 3.361 | 14,001,483 | Slower |
| Batch size 32,768 | 3.249 | 14,001,483 | 1.75% faster, below the screening threshold |

The short baseline CV was 0.92%, so the issue's threshold was 2.76% for a
performance acceptance decision. Changing `NUM_PRIMES` changes the modular
survivor count by design. Correctness requires byte-identical final ordered
point output and the same exact point count; modular survivor counts are
recorded for diagnosis. The 32-prime result therefore qualified for full-box
confirmation. `BLOCK=512` is unsupported by the current sieve plan: selected
primes must exceed the block size, and the candidate primes are all below 512.
It was rejected before benchmarking.

## Previous validation and scaling at 28 primes

All four requested validation commands passed: `make test-host` (89 mock
checks and sieve-plan boundaries), `make test-multi-gpu DEVICES=0,1` (79 real
CUDA checks), `make test` (CUDA streaming and nine CPU `ratpoints` comparison
cases), and `make record-check` (all 301 published points through height 1M).

The earlier height-10M benchmark after adding phase timers used 28 primes,
14 initial primes, and shared mask rows, with two warmups and five measured
runs per GPU configuration.
The executable SHA-256 was
`2ac81c12647620176eedbdfc6e4d6bcdbaf6018eea3407088b081506e43424c5`.
Compact results are in `docs/RTX4090-MULTIGPU.json`; per-run diagnostics are
in the local benchmark JSON. All ordered outputs and both survivor counts
matched the initial/mask sweep baseline.

| GPUs | Median (s) | Minimum (s) | Mean (s) | Std. dev. (s) | CV | Speedup vs. GPU 1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 27.820 | 27.810 | 27.848 | 0.055 | 0.197% | 0.994× |
| 1 | 27.658 | 27.550 | 27.634 | 0.051 | 0.186% | 1.000× |
| 0,1 | 14.298 | 14.244 | 14.308 | 0.061 | 0.425% | 1.934× |

Two-GPU parallel efficiency is 96.7%. The remaining measured bottleneck is
the fast sieve kernel's ALU/load-store and shared-memory work. Mask placement
and initial-prime count alone do not improve it. The next investigation should
examine shared-load bank conflicts and integer instruction cost without
changing the searched set or exact verification.

## Corrected 28/30/32/34-prime confirmation

The follow-up corrected the acceptance criterion for `NUM_PRIMES`: a different
number of modular survivors is expected when sieve strength changes. Every
candidate must produce identical final ordered point output and the same
exact point count. The tuner now records modular survivors per variant and
can check output byte-for-byte against a separate reference file with
`--reference-output`.

Isolated `-arch=sm_89` builds on GPU 0 used the full height-10M box, two
discarded warmups, and five measured runs each. `INITIAL_PRIMES=14`, shared
rows, `BLOCK=256`, and batch size 65,536 were fixed. The complete compact
results are in `docs/RTX4090-NUM-PRIMES.json`.

| `NUM_PRIMES` | Median (s) | Minimum (s) | CV | Gain vs. 28 | Modular survivors |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 27.794 | 27.739 | 0.160% | reference | 71,497,766 |
| 30 | 27.232 | 27.209 | 0.072% | 2.02% | 68,542,652 |
| 32 | 27.051 | 27.015 | 0.199% | 2.68% | 67,691,763 |
| 34 | 27.075 | 27.037 | 0.143% | 2.59% | 67,404,435 |

All runs had 313 exact points and byte-identical ordered output, SHA-256
`4d30d69412568b0459b9ca0f25a7519f662948dd0b7269a5bb07502d9268eef9`.
The 2.68% median gain of 32 primes exceeds the 2% acceptance threshold;
its difference from 34 primes is only 0.09%, below measurement significance.
`NUM_PRIMES=32` was therefore selected as the default. The kernel and exact
GMP verifier were not modified.

To reproduce the full comparison, run the tuner separately with
`--num-primes 28`, `30`, `32`, and `34`, together with `--primes 14 --masks
shared --warmups 2 --repeats 5`. For the latter three, pass
`--reference-output /path/to/num28/baseline.stdout`.

## Final 32-prime validation and two-GPU scaling

With `NUM_PRIMES=32` as the source default, `make test-host`,
`make test-multi-gpu DEVICES=0,1`, `make test`, and `make record-check` all
passed again. The real-GPU suite passed 79 checks, the CPU comparison covered
nine searches, and the record-curve check found all 301 expected points.

The final default executable SHA-256 was
`9b5559e9c6724ae764c13a83366cb80f5bc6855c2f5db47f4f65f235d452895c`.
At height 10M, two discarded warmups and five measured runs per configuration
gave the following end-to-end results. The compact report is in
`docs/RTX4090-MULTIGPU-32.json`.

| GPUs | Median (s) | Minimum (s) | Mean (s) | Std. dev. (s) | CV | Speedup vs. GPU 1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 27.086 | 27.055 | 27.085 | 0.023 | 0.086% | 0.995× |
| 1 | 26.942 | 26.905 | 26.949 | 0.041 | 0.152% | 1.000× |
| 0,1 | 13.929 | 13.911 | 13.940 | 0.026 | 0.184% | 1.934× |

The two-GPU median improved 2.58% from the previous 14.298 seconds, and
parallel efficiency remained 96.7%. All GPU selections produced identical
ordered output and 313 exact points; 67,691,763 modular survivors were
reported for the 32-prime configuration. The remaining measured bottleneck
is still the fast sieve kernel's ALU/load-store work and shared-memory access.
