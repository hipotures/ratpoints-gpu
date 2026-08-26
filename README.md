# ratpoints_gpu

CUDA bounded point search for hyperelliptic curves.

*This project provides a CUDA implementation with a command-line and output
interface compatible with Michael Stoll's
[**ratpoints**](https://github.com/MichaelStollBayreuth/ratpoints). Many thanks
to Michael Stoll for developing and publishing ratpoints. See *Documentation
for the ratpoints program*,
[arXiv:0803.3165](https://arxiv.org/abs/0803.3165).*

For a curve

$$
y^2 = c_0 + c_1x + \cdots + c_nx^n,
$$

the program searches reduced $x=a/b$ in a bounded rectangle.

- CUDA rejects candidates with a modular square sieve.
- GMP verifies every surviving point exactly.

Results are exhaustive within the requested bounds only.

## Performance

Measurements used
[Stoll's record curve](https://www.mathe2.uni-bayreuth.de/stoll/recordcurve.html)
and equal-height boxes $(a,b)\in[-A,A]\times[1,A]$.

GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU (24 SM, 8 GB).

| Backend | $A$ | Sites ($\sim 2A^2$) | Elapsed | Gsite/s |
|---|---:|---:|---:|---:|
| ratpoints 2.1.3 | 300,000 | 180 billion | 54.84 s | 3.28 |
| ratpoints_gpu | 3,750,000 | 28.13 trillion | 64.47 s | 436.3 |

The measured end-to-end speedup was $133\times$. CPU ratpoints can still be faster
for small searches because CUDA has a fixed startup cost.

Enable metrics with `-v` or `RATPOINTS_GPU_BENCHMARK=1`.

## Supported Interface

- Degrees from 1 through 100
- Arbitrary-size integer coefficients
- Exact squarefreeness check before search
- Independent numerator and denominator bounds
- Unions of closed real intervals for $x$
- Weighted projective output compatible with ratpoints
- Points at infinity
- Exact GMP verification
- Ratpoints custom output formats
- Nonzero exit status for invalid input and CUDA errors

Coefficients are listed from constant to leading term.

```bash
./ratpoints_gpu 'c0 c1 ... cn' HEIGHT [OPTIONS]
```

| Option | Action |
|---|---|
| `-dl B`, `-du B` | Set denominator bounds |
| `-l L`, `-u U` | Restrict $x$ to closed intervals |
| `-1` | Stop after the first point |
| `-i`, `-I` | Suppress or restore points at infinity |
| `-y`, `-Y` | Suppress or restore ordinates |
| `-z`, `-Z` | Suppress or restore point output |
| `-q` | Quiet output |
| `-v` | Print timing and survivor metrics |
| `-f FORMAT` | Set the point format with `%x`, `%y`, and `%z` markers |
| `-fs TEXT`, `-fm TEXT`, `-fe TEXT` | Set text before, between, and after points |
| `-s`, `-k`, `-K`, `-j`, `-J` | Accepted with no effect on the CUDA search |
| `--help` | Print usage and supported options |

Interval pairs may be repeated. The first `-l` and final `-u` are optional.
Unsupported options return an error.

## Differences from `ratpoints`

- Sieve-tuning options `-n`, `-N`, `-p`, `-F`, and `-S` are not supported.
- Unchecked-survivor output via `-x` is not supported.
- CUDA is required. There is no CPU fallback.
- One process uses one default GPU. Multi-GPU execution is not supported.
- Ratpoints optimizations based on Sturm isolation, coefficient reversal,
  forbidden divisors, and Jacobi symbols are not implemented. These omissions
  can affect performance and diagnostics, but not the searched set.

## Requirements

Build and runtime dependencies:

- Linux
- NVIDIA GPU with a compatible driver
- CUDA Toolkit with `nvcc` and C++14 support
- GMP headers and library
- GNU Make
- Host C++ compiler supported by the installed CUDA Toolkit

Tests also require Python 3 and ratpoints 2.1.3 or newer.

Debian or Ubuntu packages, excluding CUDA and the NVIDIA driver:

```bash
sudo apt install build-essential make libgmp-dev python3 ratpoints
```

Fedora packages, excluding CUDA, the driver, and ratpoints:

```bash
sudo dnf install gcc-c++ make gmp-devel python3
```

## Build

```bash
make
```

The default `-arch=native` targets GPUs visible during compilation. Override it
for a deployment target:

```bash
make clean
make ARCH_FLAGS='-arch=sm_75'
```

Build a multi-architecture binary with PTX fallback:

```bash
make clean
make ARCH_FLAGS='-gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_80,code=compute_80'
```

Run `make print-config` to inspect active build variables.

Install under `/usr/local/bin`:

```bash
sudo make install
```

Stage a package without root:

```bash
make DESTDIR="$PWD/pkg" PREFIX=/usr install
```

Supported overrides are `NVCC`, `NVCCFLAGS`, `WARNING_FLAGS`, `ARCH_FLAGS`,
`CPPFLAGS`, `LDFLAGS`, `LDLIBS`, `PREFIX`, and `DESTDIR`.

## Example

Michael Stoll's
[record genus-2 curve](https://www.mathe2.uni-bayreuth.de/stoll/recordcurve.html)
is

$$
\begin{aligned}
y^2 ={}& 82342800x^6 - 470135160x^5 + 52485681x^4 \\
       &+ 2396040466x^3 + 567207969x^2 \\
       &- 985905640x + 247747600.
\end{aligned}
$$

Search it to projective height $1{,}000{,}000$:

```bash
./ratpoints_gpu \
  '247747600 -985905640 567207969 2396040466 52485681 -470135160 82342800' \
  1000000 -q
```

The search covers primitive pairs with $\gcd(a,b)=1$. Output uses weighted
projective coordinates $(a:y:b)$. For binary degree $2m$,

$$
y^2 = b^{2m}f(a/b).
$$

Both signs are printed when $y\ne 0$. Points at infinity have $b=0$.

Set `-du` explicitly for large heights. Runtime is approximately linear in the
total number of sites: (numerator range width) times (denominator count).

## Tests

`make test` requires ratpoints 2.1.3 or newer installed and available on `PATH`.

```bash
make test
```

The tests compare complete CPU and GPU point sets for odd and even degrees
through 100, intervals, denominator ranges, points at infinity, and squarefree
input validation.

Check the published record-curve points through height $10^6$:

```bash
make record-check
```

The repository includes Stoll's published list. Of its 321 $x$-coordinates,
301 have projective height at most $10^6$. The GPU search finds all 301.

## Source Layout

- `src/sieve.cu` contains the modular square sieve and CUDA orchestration
- `src/sieve_plan.cpp` builds prime tables and denominator-specific sieve plans
- `src/point_search.cpp` batches searches and performs exact verification
- `src/exact_polynomial.cpp` contains exact GMP polynomial arithmetic
- `src/command_line.cpp` parses and validates command-line arguments
- `src/output_formatter.cpp` implements ratpoints-compatible output
- `include/` contains the interfaces shared between those modules
- `tests/compare_with_ratpoints.py` runs CPU and GPU comparison tests
- `tests/verify_record_curve.py` checks Stoll's published record-curve points

## License

`ratpoints_gpu` was written independently. The original `ratpoints` source code
was never consulted or referenced while writing this repository, and no source
code was copied or adapted. Compatibility is limited to the command-line and
output interface. `ratpoints_gpu` is distributed under GPL-2.0-or-later. See
`COPYING` and `NOTICE`. Original `ratpoints` is copyright (C) 2008, 2009 Michael
Stoll and is licensed GPL-2.0-or-later.

The record curve and bundled point data are documented by J. Steffen Müller and
Michael Stoll in
[*Canonical Heights on Genus Two Jacobians*](https://doi.org/10.2140/ant.2016.10.2153),
*Algebra & Number Theory* **10** (2016), 2153-2234.
