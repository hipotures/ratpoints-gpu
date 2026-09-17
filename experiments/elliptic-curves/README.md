# Elliptic-curve math lab

This directory is a small exact-arithmetic laboratory connected to the mathematics behind `ratpoints_gpu`.

The first three experiments use only the Python standard library. Mathematical point arithmetic uses `fractions.Fraction`; floating point is used only to display logarithmic heights and progress rates.

## Run

From the repository root:

```bash
python3 experiments/elliptic-curves/01-group-law/experiment.py
python3 experiments/elliptic-curves/02-rational-points/experiment.py
python3 experiments/elliptic-curves/03-heights/experiment.py
```

Every experiment emits visible stage/progress information on stderr. Long searches show elapsed time, percentage, throughput, ETA, and useful counters. No Rich or tqdm dependency is required.

Run the focused tests with:

```bash
python3 -m unittest discover -s experiments/elliptic-curves/tests -v
```

## Roadmap

1. `01-group-law`: exact addition, doubling, scalar multiplication, and a finite associativity implementation check.
2. `02-rational-points`: exhaustive search inside a bounded rational box.
3. `03-heights`: growth of exact coordinates and the quadratic-height phenomenon.
4. Later: Mordell-Weil lattices, descent/Selmer groups, and high-rank families.

Finite computation does not prove a global rank, determine all rational points, or replace Mordell-Weil/descent arguments.
