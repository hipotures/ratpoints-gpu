# Issue 14: rank of the fiber at T = -47/80

## Certified arithmetic

The Sage 10.10.beta10 minimal model is

    y^2 + xy = x^3 - 1322791863836678632756897906456499538 x
               + 648069501075857258662767528794908106200954623218194692.

Its rational torsion is trivial. The following points are rational on this model:

| Point | x | y |
| --- | ---: | ---: |
| P0 | -1076035859747408082 | 908597517995823905542912530 |
| PD | 525828187343488308 | 312885388247132101934539950 |
| S | 866040525691648982 | 389911807110387068101800434 |

The first two points are the previously certified saturated rank-2 subgroup.
The third comes from an exact generic section. In the family

    y^2 - (Lx+B)y = x(x+D)(x+E),
    B = pq(L+p+q) - pE - qD,

set `x=-pq`, `y=p(pq-E)`. Substitution gives

    y-(Lx+B) = q(D-pq),
    y[y-(Lx+B)] = -pq(pq-E)(pq-D) = x(x+D)(x+E).

This polynomial identity was checked in Sage over Q(T), and the specialization
was checked on both the integral and minimal models. Sage/eclib saturation of
`[P0, PD, S]` completed with index **1** and numerical regulator
**22656.3901108778**. The three points are independent. Thus **rank E(Q) >= 3**.
No finite rigorous global rank upper bound was obtained; the proven interval is
**[3, infinity)**. Index 1 certifies saturation of this rank-3 subgroup in its
rank-3 span; it does not prove that the full Mordell-Weil rank is 3.

## Bounded experiments

All Sage runs used the pinned `sagemath/sagemath:10.10.beta10` image through
`run_sage_docker.py`, which enforces a wall-clock timeout and removes the
container and its descendants. Full commands, elapsed times, and machine
readable outputs are in the companion `sage-sections*.json` and
`sage-search-*.json` files.

- Generic square-discriminant tests of the natural x expressions `pq`, `-pq`,
  `±p²`, `±q²`, `±D`, `±E`, `-D-E`, and several `pq` plus/minus `D,E`
  expressions found only the known x coordinates `-D`, `-E`, and `-pq`.
- Sage `point_search` on the minimal model, with explicit logarithmic naive
  x-height bounds 10, 15, and 18, found no points. The elapsed times were 2.29,
  2.59, and 31.34 seconds respectively. A height-21 run reached its 45-second
  hard timeout. The visible point x-heights are about 40.8 to 41.5,
  so these searches are exploratory rather than exhaustive for the known
  generators. The height-10 output was generated before the third section was
  added to the search basis; it reports the old lower bound of 2 for that reason.
- The earlier mwrank 2-descent failed on large coefficient conversion, PARI
  `rank_bound` exceeded a bounded 30-second run, and PARI `analytic_rank`
  raised an integer conversion overflow. Those calls were not repeated with
  longer timeouts.

## Continued generic-section and covering work

An exhaustive coefficient-box scan over `x = a*p²+b*pq+c*q²+d*D+e*E`, with
each coefficient in `[-10,10]`, tested 4,084,101 distinct degree-at-most-four
polynomials. Only the known `x=0,-D,-E,-pq` passed the exact generic
square-discriminant test. Repeating the 4,084,101-candidate box in an
LLL-reduced coefficient basis again found only `x=0,-D,-E,-pq`. A separate exact
rational-function scan tested 8,660 distinct nonpolynomial `x` forms built
from two- and three-term combinations of `p²,pq,q²,D,E,pD,qD,pE,qE` over
`p,q,L,p+q,p-q,L+p,L+q`; none passed.
These finite scans are systematic within their stated search spaces and do not
bound the generic rank.

Sage's separate Denis Simon PARI/GP 2-descent was tried with probabilistic
large-prime testing disabled and a hard 90-second Docker timeout. It did not
complete. A second bounded trial on a model translated to an integer adjacent
to the real branch endpoint also timed out at 90 seconds. These are distinct
from the failed eclib and PARI `rank_bound` calls, but yielded no rigorous upper
bound. Exact Sage computations found no rational isogenies of prime degrees
2, 3, 5, or 7, so the common small-isogeny descent routes are unavailable.
A separate bounded PARI `ellrankinit` diagnostic timed out after 30 seconds,
showing that initialization itself is already a bottleneck before its point
search effort parameter matters.
The previously computed conductor has 90 decimal digits and Sage local data
lists 15 bad primes; this helps explain why cubic-field Selmer initialization
is costly, but is not itself a rank bound.

## GPU search on the fixed fiber

For the minimal model, set

    F(x) = (2y+x)^2 = 4x³+x²+4*a4*x+4*a6.

For each integer center `C`, the GPU searches `x=C+n/d` with reduced `n/d`,
`|n|<=H`, and `1<=d<=D`. It sieves the exact cubic `F(C+X)` using primes
coprime to `d`. If `F(C+n/d)` is a rational square, then
`d^4 F(C+n/d)` is an integer square and is a quadratic residue at every
selected prime. Hence the modular sieve has no false negatives within the
explicit rectangle. Every returned square was checked with exact integers,
then on the minimal model in Sage. An exhaustive CPU comparison on 8,004
small grid sites matched the GPU output exactly and recovered `P0`.

The fixed-curve benchmark at `H=1,000,000`, `D=64` (128,000,064 bounding
sites) measured 0.379 s on GPU 0, 0.362 s on GPU 1, and 0.409 s on both,
including startup. All outputs and survivor counts agreed. Scaling to
`H=2,000,000,000`, `D=8192` gave 32.768 trillion bounding sites per center.
At this larger bound, GPU 0 took 4.419 s, GPU 1 took 4.369 s, and both took
2.548 s on identical work; the dual-GPU speedup over the faster single GPU
was 1.715×, with matching ordered output and survivor counts.
The first two-GPU run took 2.56 s, about 12.8 trillion sites/s; nine
utilization samples per GPU averaged 81.3% and 78.6%, with a maximum of
100% on each. Searches around `P0`, `PD`, the third section `S`, and the
integer adjacent to the real branch endpoint yielded only the already known
points (and their negatives) or no points. Sage verified all returned points;
the subgroup lower bound remains 3. Full commands, per-run timings, metrics,
utilization, and outputs are in `rank31-gpu-fixed-*.json` and
`sage-verify-gpu-*.json`.

Increasing to `D=65536` searched 262.144 trillion bounding sites per center,
covering square x-denominators through `256²`. The four two-GPU runs took
16.19–16.32 seconds each, with throughput of 16.1–16.2 trillion bounding
sites/s. At least 57 utilization samples per run averaged 94.9–98.3% on the
two GPUs, with 100% maxima. They again yielded only known points or no points;
Sage verified all six returned signed points. These finite rectangles are
search bounds, not global rank bounds. The wide runs generated roughly
0.38–0.44 million modular survivors per center, all but the known points
rejected by exact integer square checks.

A denominator-focused pass with `H=100,000,000`, `D=1,000,000` covered square
x-denominators through `1000²` in windows closer to each center. Each of the
four two-GPU runs searched 200 trillion bounding sites in 10.2–10.9 seconds,
at 18.3–19.6 trillion sites/s. They yielded only the known signed points or
none; all returned coordinates were checked on the minimal curve in Sage.
The same wide and denominator-focused passes were also run around `PE`, the
third visible `y=0` family point. They returned only `PE` and `-PE`; Sage
verified `PE=-P0-PD` exactly, so neither raises the rank.

The global root number is -1, but it is not used as an algebraic rank proof.
The large Mestre-Nagao score is likewise heuristic only.

The best next route is a covering/Selmer implementation able to handle this
model's large coefficients, or a structural derivation of further sections
outside the tested low-coefficient templates. Unstructured expansion of the
centered GPU rectangles has a low expected point yield because the ordinate
is typically enormous; the finite searches above make no claim of completeness
outside their explicit bounds.
The available bounded local routes were exhausted for this session: eclib's
coefficient conversion fails, PARI cannot complete even its initialization in
30 seconds, two Simon 2-descent models timed out at 90 seconds, small rational
isogenies are absent, the stated generic template spaces produced no new
section, and the GPU rectangles produced no new point. Magma is not installed
on this host. A different covering implementation or new structural formula
would be needed for a stronger rigorous rank result.
