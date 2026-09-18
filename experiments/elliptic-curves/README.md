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
upper bound. The exact Python report does not compute a global rank bound.

The repository runner uses the pinned `sagemath/sagemath:10.10.beta10` Docker
image. It starts Sage as root inside the container, restores host ownership of
the result, enforces a 30-second wall-clock limit by default, and removes the
container and its descendants on timeout. Run from the repository root:

```bash
python3 experiments/elliptic-curves/run_sage_docker.py --timeout 30 experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode invariants
python3 experiments/elliptic-curves/run_sage_docker.py --timeout 30 experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode pari-bound
```

Each run writes `results/rank31-t-minus-47-80-sage-MODE.json` with its status,
elapsed time, Sage version when available, source commit, and result or error.
A timeout exits with code 124 and writes an explicit timeout report. The PARI
bound is optional and can still time out. `--mode mwrank-bound` is separate and
must be requested explicitly; this curve has already triggered an eclib size
limit, so do not use it for exploratory checks. `saturation` and `analytic` are
also explicit modes and may be expensive.

The optional `--mode search --log-height N` searches the **minimal model** with
Sage's logarithmic naive x-height convention. It does not require a global rank
upper bound. Increasing `N` can make the search much slower.

The [issue 14 report](results/rank31-t-minus-47-80-issue14.md) records the
certified rank-3 subgroup and bounded follow-up. Its generic-section scans
are Sage modes `section-scan`, `section-scan-lll`, and
`rational-section-scan`. The fixed-curve GPU search reuses the repository CUDA
modular sieve, with a centered discriminant cubic and exact CPU verification:

```bash
python3 experiments/elliptic-curves/rank31_gpu_fixed_search.py --center P0 --height 2000000000 --denominators 8192 --devices 0,1 --timeout 60 --tag example
python3 experiments/elliptic-curves/run_sage_docker.py --timeout 30 experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode verify-gpu --gpu-report rank31-gpu-fixed-example.json
```

The GPU driver accepts centers `P0`, `PD`, `PE`, `S`, and `R` (the integer next to
the real branch endpoint). Each run writes exact points, throughput, survivor
counts, and sampled utilization under `results/`. The Sage verification mode
checks all returned coordinates on the minimal model and tests additions to
the known rank-3 subgroup.

## Multi-fiber rank search (issue 15)

`rank31_multifiber.py inventory` deduplicates the existing compressed Stage B
campaign and standalone refined files, verifies the 31 record-control witnesses exactly, and writes 30
selected non-control fibers with integral models and all four known sections.
The other modes use the existing CUDA sieve on two RTX 4090s. A search uses
`x = center + stride*n/d` with the rectangle recorded in each JSON result;
every emitted point passes exact integer-square and curve-equation checks.

```bash
python3 experiments/elliptic-curves/rank31_multifiber.py inventory
python3 experiments/elliptic-curves/rank31_multifiber_campaign.py --stage screen
python3 experiments/elliptic-curves/rank31_multifiber_campaign.py --stage promote
python3 experiments/elliptic-curves/rank31_multifiber_campaign.py --stage structured
python3 experiments/elliptic-curves/rank31_multifiber_campaign.py --stage rescale
python3 experiments/elliptic-curves/rank31_multifiber_campaign.py --stage coarse
python3 experiments/elliptic-curves/rank31_multifiber_campaign.py --stage standalone --limit 30
python3 experiments/elliptic-curves/rank31_multifiber_campaign.py --stage deeper
```

The campaign checkpoints a machine-readable scoreboard after each bounded GPU
run. It promotes the strongest Stage B leads to wider windows, then favors
smaller parameter denominators for rescaled family-coordinate grids. New
non-section points trigger bounded Sage rank-growth and saturation attempts.
`run_multifiber_sage.py` runs exact checks in a named Docker container with a
hard timeout and forced container cleanup. `--height-proof` certifies the rank
of `P0, PD, PQ` using exact doubled points, rational logarithm intervals, and
an upward Silverman height-difference bound; it also checks the dependent
`P0, PD, PE` triple as a negative control. `--verify-only` checks every GPU
point on its exact Sage curve.

After the GPU and Sage runs, `rank31_multifiber_finalize.py` merges the
certificates into the scoreboard and writes the concise research report. The
GPU/CPU and Docker/Sage integration checks run with:

```bash
RANK31_TEST_SAGE=1 PYTHONPATH=experiments/elliptic-curves python3 -m unittest test_rank31_multifiber
```
