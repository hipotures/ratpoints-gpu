# Multi-GPU search and benchmarking

## Build and select devices

Build for the GPUs visible on the build machine:

```bash
make -j"$(nproc)"
./ratpoints_gpu --list-devices
```

For an explicit RTX 4090 / RTX 4070 Ti target (compute capability 8.9):

```bash
make clean
make -j"$(nproc)" ARCH_FLAGS='-arch=sm_89'
```

This selects a compilation target; it is not additional kernel tuning.
NVIDIA's [compute-capability table](https://developer.nvidia.com/cuda/gpus)
lists the supported devices. Use the README's multi-architecture build example
when targeting GPUs of different architectures.

The default is **all devices visible to CUDA**. Device IDs are logical CUDA
ordinals, not necessarily the indices shown by `nvidia-smi`; a visibility mask
can hide and reorder devices. Check names and PCI bus IDs with `--list-devices`
before choosing the two RTX 4090 cards. A third visible card is included by
default unless explicitly excluded.

```bash
CURVE='247747600 -985905640 567207969 2396040466 52485681 -470135160 82342800'
./ratpoints_gpu "$CURVE" 1000000 --devices 0 -v
./ratpoints_gpu "$CURVE" 1000000 --devices 0,1 -v
./ratpoints_gpu "$CURVE" 1000000 --devices all -v
```

Options follow the coefficient string and height. `--list-devices` and `--help`
are standalone commands. Duplicate, negative, malformed, or unavailable device
IDs are errors; no CPU fallback is attempted.

## How work is partitioned

Each wave splits the next denominator interval into disjoint, contiguous
batches, one per selected device, with at most 65,536 denominators per batch.
Small final waves are divided evenly and never launch empty ranges. Each worker
calls `cudaSetDevice` before creating CUDA resources, runs the unchanged modular
sieve, and verifies survivors with its own GMP polynomial and scratch state.
CUDA buffers and events are destroyed on that worker's selected device.

The main thread collects the entire wave, then emits sorted results in
increasing denominator and numerator order. It alone invokes output callbacks.
Infinity is emitted once; custom formats, intervals, both ordinate signs, and
`-1` retain their semantics. A single selected GPU runs synchronously without
an extra worker-thread launch.

The implementation uses bounded waves rather than an unbounded result/reorder
queue. Different GPU models are supported, but equal-size waves wait for the
slowest worker: this is **not** a speed-weighted dynamic scheduler. Use the
benchmark to decide whether including a slower third card helps. Workspaces and
sieve tables are still rebuilt for each batch, as in the original implementation.
Persistent workspaces, weighted scheduling, and Ada-specific kernel tuning are
separate future optimizations.

`--batch-size N` changes the maximum denominators per GPU batch (1..65,536).
Smaller batches reduce work in flight but increase setup and thread-launch
overhead. Host memory retains the verified points for one wave, not the entire
search; very dense curves can still require substantial memory. Modular survivor
buffers keep the existing bounded streaming behavior.

`-1` waits for already launched work in the current wave before returning its
first ordered point. It does not kill running kernels. Exceptions join remaining
workers and produce a nonzero exit status identifying the failing GPU and range.
Earlier successful waves may already have written output; do not treat output
from a failed process as an exhaustive result.

No GPU-to-GPU transfers, NVLink, peer access, NCCL, or pooled VRAM are needed.
Both the host scheduler and sieve-plan iteration use a wide cursor to avoid
signed overflow after an inclusive `INT_MAX` denominator endpoint.

## Correctness tests

```bash
make test-host
make test-multi-gpu DEVICES=0,1
make test
make record-check
```

- `test-host` compiles with the host C++ compiler and GMP. It uses a deliberately
  mocked CUDA API/sieve, plus the real parser, scheduler, exact verification, and
  formatter. It checks concurrency, range coverage, worker failures, invalid
  selection, output ordering, first-point handling, benchmark failure detection,
  and integer endpoints. A separate test exercises the real sieve-plan code.
  **It does not compile or execute CUDA kernels.**
- `test-multi-gpu` requires at least two actual visible GPUs. It checks exact
  point output against an independent Python integer oracle, reversed device
  order, small/uneven/multiple batches, options, and dense streaming survivors.
  It needs Python 3, but not CPU ratpoints.
- `test` retains the original comparison with CPU ratpoints and the CUDA
  streaming test. `record-check` retains the published Stoll-curve check.

The implementation environment had no CUDA toolkit or GPU. Host tests and
sanitizer checks were run; CUDA compilation, kernel execution, and actual
multi-GPU performance still require hardware validation. No scaling result is
claimed by this change.

## Benchmark one GPU against two

```bash
make benchmark DEVICES=0,1 HEIGHT=1000000 REPEATS=3
```

This runs the same Stoll-curve box on GPU 0, GPU 1, and both GPUs. It performs
one discarded warmup per configuration and three measured runs per configuration,
rotating measurement order. Every ordered output and the aggregate modular/exact
survivor counts must match. Any CUDA error, timeout, or mismatch aborts the
benchmark instead of reporting a valid comparison.

For a larger workload and a machine-readable report:

```bash
python3 tests/benchmark_multi_gpu.py \
  --binary ./ratpoints_gpu \
  --devices 0,1 \
  --height 5000000 \
  --repeats 3 \
  --json benchmark-2x4090.json
```

Additional options include `--coefficients`, `--denominator-min`,
`--denominator-max`, `--batch-size`, `--warmups`, and `--timeout`. The default
per-process timeout is 3,600 seconds. Start with the smaller box; enlarging both
bounds multiplies work approximately quadratically. Avoid competing GPU jobs
and record power/clock limits when interpreting results.

The report contains the median and minimum **end-to-end process time**,
bounding-box Gsites/s, and speedup relative to the fastest selected single GPU.
JSON also records the executable hash, GPU listing, visibility mask, search
parameters, individual timing diagnostics, and the common output hash. The
benchmark captures point output, so very dense user-supplied curves can consume
significant host memory; the default record curve is the intended baseline.

`-v` reports `wall_ms` for `PointSearch::run`, excluding argument validation and
device enumeration. `basis_ms` and `sieve_ms` are sums of per-device CUDA event
times; they can exceed elapsed wall time and must not be used directly as
multi-GPU speedup. `initial_mask_gbs` remains the analytical mask-byte estimate
divided by summed sieve event time, not a measured total system bandwidth.
Per-GPU lines show device identity, completed denominators, batches, and event
times. Gsites/s counts the entire requested integer rectangle, including sites
rejected in parallel by the bit sieve; it is not a count of GMP evaluations.
