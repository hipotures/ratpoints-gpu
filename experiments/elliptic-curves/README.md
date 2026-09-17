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

## Multi-seed campaign

Run the sharded campaign manually from the repository root:

```bash
python3 experiments/elliptic-curves/rank31_mestre_nagao_campaign.py
```

Its defaults use seeds 304–308, one million new candidates per seed, Stage A
through prime 5000, the global top 50,000 distinct new parameters, and Stage B
through prime 100000. It uses both GPUs and the shared bounded CPU oracle from
the single-run tool. Each shard gets compressed complete results and a readable
summary. A compressed global Stage B report contains every finalist with its
source shard, seed, and Stage A rank; a readable campaign summary and `latest`
summary give the control statistics, top 100 leads, timings, and artifact
hashes. Results are committed, pushed, and verified after a successful run.

Configure the campaign with `--seeds` or `--seed-start` and `--shards`,
`--candidates-per-shard`, `--first-prime-bound`, `--global-finalists`,
`--second-prime-bound`, `--denominator-min`, `--denominator-max`, `--t-span`,
`--devices`, and `--batch-candidates`. `--no-push` keeps generated results local.
The campaign is a heuristic search; its scores do not prove curve rank.

The CUDA scorer processes primes in bounded chi-table chunks. Its default
`--chi-chunk-bytes` budget is 256 MiB; the helper's `--plan-only --prime-bound B`
prints the prime count, chunk count, largest chunk, and cumulative chi entries
without allocating chi tables or launching a kernel. The helper rejects bounds
above one million and unsupported memory budgets with an error. Optional
`--counts` output remains ordered by candidate and then prime.
The Python build helper retains only the current source-hash executable in
ignored `build/`, so a stale helper cannot be selected by a filename glob.

## Exact triage of T=-47/80

Run the short, standard-library-only exact computation with:

```bash
python3 experiments/elliptic-curves/rank31_t_minus_47_80.py
```

It writes `results/rank31-t-minus-47-80.json` and a short Markdown summary.
The report proves trivial rational torsion and rank at least 1 from visible
family points. It does not claim their exact subgroup rank or a global rank
upper bound. SageMath, PARI/GP, mwrank and Magma are not installed on the
current host, so minimal-model and descent results remain pending.

On a machine with SageMath, run the following modes separately after reviewing
their expected cost:

```bash
sage -python experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode invariants
sage -python experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode descent
sage -python experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode saturation
sage -python experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode analytic
```

The optional `--mode search --log-height N` searches the **minimal model** with
Sage's logarithmic naive x-height convention. Choose `N` only after inspecting
the minimal model, the known point heights printed by that mode, and the
descent bounds. Increasing `N` can make the search much slower.
