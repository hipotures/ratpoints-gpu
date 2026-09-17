# 02 — Bounded rational points

This experiment searches reduced rational abscissas

```text
x = a/b,  gcd(a,b)=1,  b>0
```

inside configurable numerator and denominator bounds on `y^2=x^3-2`.

For each reduced `a/b`, it evaluates the right-hand side as an exact rational number and tests whether both the reduced numerator and denominator are integer squares. No floating-point test decides point membership.

Example:

```bash
python3 experiments/elliptic-curves/02-rational-points/experiment.py \
  --numerator-bound 5000 --denominator-bound 2500
```

The visible progress line reports raw `(a,b)` pairs processed, throughput, ETA, reduced rational x-values tested, and points found. The result is exhaustive only inside the requested box; it says nothing by itself about all of `E(Q)` or the rank.
