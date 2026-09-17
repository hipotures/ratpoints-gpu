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

The global root number is -1, but it is not used as an algebraic rank proof.
The large Mestre-Nagao score is likewise heuristic only.

The best next route is a covering/Selmer implementation able to handle this
model's coefficients, or a targeted search for further polynomial sections.
