# Issue #15: multi-candidate fixed-fiber search

Generated: 2026-09-18T01:29:11.067338+00:00

Three completed Stage B campaigns and two standalone refined runs contributed 159,995 distinct non-control finalists. The selected frontier contains 30 specializations; selection includes the five requested strong leads. The rank-31 record is a calibration control, not a discovery candidate. Its 31 published witness coordinates were checked exactly on the recorded minimal model.

## Rigorous results

Best newly certified lower bound: **rank at least 3**. No fourth independent point was found in the completed GPU rectangles. No new generic section was found. The previously known x=-pq section is included in every exact specialization.

| T | Certified subgroup rank | Stage B score | GPU sites | Sage status |
|---|---:|---:|---:|---|
| -47/80 | 3 | 170.19 | 3,938,713,601,015,808 | full Sage/eclib saturation from issue 14 |
| -191989/4040887 | 3 | 167.99 | 2,624,716,800,671,744 | rank 3 certified by exact height intervals; saturation not completed |
| -47/500 | 3 | 167.40 | 11,016,601,602,785,280 | rank 3 certified by exact height intervals; saturation not completed |
| 532929/2579219 | 3 | 166.01 | 2,624,716,800,671,744 | full Sage/eclib saturation |
| -802/2917 | 3 | 165.76 | 3,938,713,601,015,808 | full Sage/eclib saturation |
| -6341/25735 | 3 | 165.72 | 3,938,713,601,015,808 | rank 3 certified by exact height intervals; saturation not completed |
| -183569/2305416 | 3 | 165.58 | 1,313,996,800,344,064 | rank 3 certified by exact height intervals; saturation not completed |
| 141580/460867 | 3 | 165.28 | 1,317,273,600,360,448 | rank 3 certified by exact height intervals; saturation not completed |
| -112991/465710 | 3 | 165.18 | 1,313,996,800,344,064 | rank 3 certified by exact height intervals; saturation not completed |
| -353682/460195 | 3 | 165.01 | 1,317,273,600,360,448 | rank 3 certified by exact height intervals; saturation not completed |
| 1071727/1978006 | 3 | 164.98 | 1,313,996,800,344,064 | full Sage/eclib saturation |
| -600578/1968423 | 3 | 164.70 | 1,313,996,800,344,064 | rank 3 certified by exact height intervals; saturation not completed |
| 3334535/2160657 | 3 | 164.68 | 3,276,800,016,384 | rank 3 certified by exact height intervals; saturation not completed |
| 432533/1540803 | 3 | 164.56 | 1,313,996,800,344,064 | rank 3 certified by exact height intervals; saturation not completed |
| 2112933/2552189 | 3 | 164.41 | 3,276,800,016,384 | full Sage/eclib saturation |

All 30 fibers have exact rank-at-least-three certificates for P0, PD, and PQ. Under pinned Sage 10.10.beta10, the certificates use 4 point-doubling rounds, exact x-coordinates, rational enclosures of logarithms, and an upward bound on Sage’s Silverman height-difference formula; all three leading principal minors are strictly positive. The known relation P0+PD+PE=O is checked as a negative control for both triples and quadruples. Sage/eclib saturation completed on a subset; a saturation timeout leaves the index unknown, not the rank-three lower bound. Good reductions separately prove torsion is trivial for every selected fiber.

## GPU work and limits

| Stage | Rectangles | Sites | Modular survivors | Exact x survivors | GPU time |
|---|---:|---:|---:|---:|---:|
| coarse | 40 | 10,485,760,002,621,440 | 14,463,872 | 32 | 630.4 s |
| deeper | 10 | 10,485,760,002,621,440 | 16,364,056 | 8 | 625.4 s |
| outer | 14 | 3,670,016,000,917,504 | 4,987,042 | 0 | 222.3 s |
| promote | 60 | 15,728,640,003,932,160 | 21,221,815 | 48 | 966.6 s |
| rescale | 65 | 17,039,360,004,259,840 | 23,491,918 | 52 | 1029.2 s |
| screen | 120 | 98,304,000,491,520 | 598,055 | 120 | 56.7 s |
| standalone | 30 | 7,864,320,001,966,080 | 10,390,457 | 24 | 475.6 s |
| structured | 52 | 42,598,400,212,992 | 370,524 | 78 | 24.8 s |

Scheduled scoreboard total (excluding overlapping pilot runs): 65,414,758,417,022,976 bounded sites in 4031.0 summed GPU-run seconds (16.23 trillion sites/s including startup). These sequential GPU jobs alone account for 67.2 minutes of active computation. Bulk stages averaged 16.53 trillion sites/s, consistent with the 16–20 trillion sites/s issue-14 baseline. Recorded bounded Sage CPU time: 2426.6 s across selected fibers. Every bulk invocation used devices 0 and 1; each raw JSON report records their denominator batches and kernel timings. GPU 0: 96.8% mean utilization across 11363 telemetry samples; GPU 1: 96.4% mean utilization across 11363 telemetry samples.

The exact finite regions are listed per candidate in the scoreboard and in each raw GPU JSON report: x = center + stride·n/d, |n| ≤ height, 1 ≤ d ≤ denominators, gcd(n,d)=1. The modular sieve rejects no rational point in that rectangle, as validated against exact CPU enumeration. Every GPU output passed Python integer-square and curve-equation checks. The separate Sage verification artifacts record exact point construction for each completed report. Absence of a new point excludes only these rectangles. A bounded Simon 2-descent for T=-44/43 timed out at 120 seconds; for T=-802/2917 it hit PARI’s 1 GiB bnfinit stack limit before returning a bound. Bounded mwrank bounds on both fibers failed because their 2-descents did not complete. PARI ellrank with known points and zero search effort also timed out at 60 seconds for T=-44/43 and 45 seconds for T=-47/500. None supplies an upper bound.

## Adaptation and next work

The cheap screen returned only known section points. Wide windows therefore went to the strongest 12 Stage B leads; six additional standalone refined leads were screened and widened. Structured x lattices and subsequent rescaled windows favored smaller parameter denominators, which provide better resolution in family coordinates. Candidates with only known points were demoted from further identical-width searches. Fourteen outer windows on two small-denominator fibers tested logarithmically spaced family-coordinate regions and found no points. A separate sparse-coordinate scan tested 2,929,536 low-complexity expressions over Q(T) and 87,886,080 specialized expressions across all 30 fibers. Its modular filter left only the three known generic section forms; exact fixed-fiber checks returned only the 90 known section x-coordinates. This excludes only the explicitly recorded ansatz. Minimal-model diagnostics on six leading fibers found larger maximum coefficient bit sizes than the integral factored models, so repeating the failed descents on these minimal models was not prioritized. Promising follow-up is to derive candidate x-coordinates from covering curves or lattice reduction, then feed those centers to the exact GPU sieve; repeated local rectangles around section points have low yield. Alternative bounded descent algorithms may resolve upper bounds for the smaller-denominator fibers. No global upper bound is claimed.
