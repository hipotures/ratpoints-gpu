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

## Rank-31 specialization sieve

The prepared Mestre–Nagao pipeline can be run manually from the repository root:

```bash
python3 experiments/elliptic-curves/rank31_mestre_nagao_gpu.py
```

By default, Stage A scores 50,000 new rational parameters plus the known record
control on GPUs 0 and 1 through prime 5000. Stage B scores the top 1,000 new
parameters plus the control through prime 20000. Before scoring, the script
computes one exact CPU oracle and checks both GPUs against it. By default it
checks every odd prime through 1000 and a deterministic sample across the rest
of the configured second range, capped at 256 primes total. Use
`--validation-primes N` to change the cap or `--validation-exhaustive` to check
every odd prime. The script reports the control's rank
and percentile at both stages, prints 50 new leads, writes timestamped and
`latest` JSON reports, then commits and pushes those reports to `master`.

Use `--candidates`, `--first-prime-bound`, `--finalists`,
`--second-prime-bound`, `--devices`, and `--batch-candidates` to configure the
run. `--no-push` writes results without publishing them. The scores rank
candidate specializations heuristically; they do not prove rank.
