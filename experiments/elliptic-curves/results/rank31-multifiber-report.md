# Issue #15: multi-candidate fixed-fiber search

Generated: 2026-09-18T02:39:46.794668+00:00

Three completed Stage B campaigns and two standalone refined runs contributed 159,995 distinct non-control finalists. The selected frontier contains 44 specializations; selection includes the five requested strong leads. The rank-31 record is a calibration control, not a discovery candidate. Its 31 published witness coordinates were checked exactly on the recorded minimal model.

## Rigorous results

Best newly certified lower bound: **rank at least 18**. The leading specialization contains 15 independent generators beyond the three known sections. No new generic section was found. The previously known x=-pq section is included in every exact specialization.

| T | Certified subgroup rank | Stage B score | GPU sites | Sage status |
|---|---:|---:|---:|---|
| 1/4 | 18 | 146.92 | 6,313,324,801,593,896 | through prime 11 (Sage/eclib) |
| -7/12 | 12 | 152.24 | 2,733,292,800,698,888 | through prime 11 (Sage/eclib) |
| -5/36 | 6 | 159.52 | 2,733,292,800,698,888 | through prime 11 (Sage/eclib) |
| -1/19 | 5 | 150.28 | 2,733,292,800,698,888 | through prime 11 (Sage/eclib) |
| -47/80 | 3 | 170.19 | 4,680,153,601,201,168 | full Sage/eclib saturation from issue 14 |
| -191989/4040887 | 3 | 167.99 | 3,366,156,800,857,104 | rank 3 certified by exact height intervals; saturation not completed |
| -47/500 | 3 | 167.40 | 13,240,921,603,341,360 | rank 3 certified by exact height intervals; saturation not completed |
| 532929/2579219 | 3 | 166.01 | 2,995,436,800,764,424 | full Sage/eclib saturation |
| -802/2917 | 3 | 165.76 | 4,680,153,601,201,168 | full Sage/eclib saturation |
| -6341/25735 | 3 | 165.72 | 4,309,433,601,108,488 | rank 3 certified by exact height intervals; saturation not completed |
| -183569/2305416 | 3 | 165.58 | 1,684,716,800,436,744 | rank 3 certified by exact height intervals; saturation not completed |
| 141580/460867 | 3 | 165.28 | 1,687,993,600,453,128 | rank 3 certified by exact height intervals; saturation not completed |
| -112991/465710 | 3 | 165.18 | 1,684,716,800,436,744 | rank 3 certified by exact height intervals; saturation not completed |
| -353682/460195 | 3 | 165.01 | 1,687,993,600,453,128 | rank 3 certified by exact height intervals; saturation not completed |
| 1071727/1978006 | 3 | 164.98 | 1,684,716,800,436,744 | full Sage/eclib saturation |

### New independent generators

- T=1/4: 18 independent generators; basis sample (11063687490, 16441590108001680); (11299301370, 16767506597158440). Full basis: scoreboard; certificate: rank31-multifiber-sage-43.json.
- T=-7/12: 12 independent generators; basis sample (764514769502538, -7924901178007545665472); (787419496369098, -7258898810002096190304). Full basis: scoreboard; certificate: rank31-multifiber-sage-38.json.
- T=-5/36: 6 independent generators; basis sample (-377642887394838, -56139904858384163112192); (-377589920101590, -56131029744777726611040). Full basis: scoreboard; certificate: rank31-multifiber-sage-30-height-extended.json.
- T=-1/19: 5 independent generators; basis sample (2026080229488, -75839388853332307392); (2293683949872, -82763901235170517824). Full basis: scoreboard; certificate: rank31-multifiber-sage-42-height-proof.json.

All 44 fibers have exact rank-at-least-three certificates for P0, PD, and PQ. Under pinned Sage 10.10.beta10, the certificates use 4 point-doubling rounds, exact x-coordinates, rational enclosures of logarithms, and an upward bound on Sage’s Silverman height-difference formula; all three leading principal minors are strictly positive. Additional interval proofs certify rank growth where their principal minor lower bounds are positive. Sage/eclib saturation certifies the higher ranks shown for T=-7/12 and T=1/4 through prime 11. The known relation P0+PD+PE=O is checked as a negative control for both triples and quadruples. A full saturation timeout leaves the index unknown, not the certified subgroup rank. Good reductions separately prove torsion is trivial for every selected fiber.

## GPU work and limits

| Stage | Rectangles | Sites | Modular survivors | Exact x survivors | GPU time |
|---|---:|---:|---:|---:|---:|
| coarse | 40 | 10,485,760,002,621,440 | 14,463,872 | 32 | 630.4 s |
| deeper | 10 | 10,485,760,002,621,440 | 16,364,056 | 8 | 625.4 s |
| discovery | 24 | 5,677,184,001,419,296 | 7,433,729 | 139 | 341.2 s |
| discovery_expand | 8 | 1,790,016,000,447,504 | 2,500,705 | 183 | 108.0 s |
| discovery_expand2 | 8 | 1,790,016,000,447,504 | 2,506,052 | 130 | 107.9 s |
| group | 12 | 2,224,320,000,556,080 | 3,323,753 | 12 | 137.3 s |
| lowden | 40 | 10,485,760,002,621,440 | 13,585,479 | 90 | 631.3 s |
| lowden_remaining | 24 | 6,291,456,001,572,864 | 9,070,657 | 18 | 383.0 s |
| outer | 14 | 3,670,016,000,917,504 | 4,987,042 | 0 | 222.3 s |
| promote | 60 | 15,728,640,003,932,160 | 21,221,815 | 48 | 966.6 s |
| rescale | 65 | 17,039,360,004,259,840 | 23,491,918 | 52 | 1029.2 s |
| screen | 176 | 144,179,200,720,896 | 880,748 | 176 | 83.5 s |
| square | 10 | 1,853,600,000,463,400 | 2,889,646 | 8 | 114.8 s |
| square_frontier | 26 | 4,819,360,001,204,840 | 6,554,628 | 26 | 294.7 s |
| square_rescale | 10 | 1,853,600,000,463,400 | 2,892,103 | 8 | 114.9 s |
| standalone | 30 | 7,864,320,001,966,080 | 10,390,457 | 24 | 475.6 s |
| structured | 52 | 42,598,400,212,992 | 370,524 | 78 | 24.8 s |

Scheduled scoreboard total (excluding overlapping pilot runs): 102,245,945,626,448,680 bounded sites in 6290.9 summed GPU-run seconds (16.25 trillion sites/s including startup). These sequential GPU jobs alone account for 104.8 minutes of active computation. Bulk stages averaged 16.51 trillion sites/s, consistent with the 16–20 trillion sites/s issue-14 baseline. Recorded bounded Sage CPU time: 3561.0 s across selected fibers. Every bulk invocation used devices 0 and 1; each raw JSON report records their denominator batches and kernel timings. GPU 0: 96.5% mean utilization across 19329 telemetry samples; GPU 1: 96.0% mean utilization across 19329 telemetry samples.

The exact finite regions are listed per candidate in the scoreboard and in each raw GPU JSON report: x = center + stride·n/d, |n| ≤ height, gcd(n,d)=1. In consecutive mode 1 ≤ d ≤ denominators; in square mode d=k² with 1 ≤ k ≤ denominators. The modular sieve rejects no rational point in that rectangle, as validated against exact CPU enumeration. Every GPU output passed Python integer-square and curve-equation checks. The separate Sage verification artifacts record exact point construction for each completed report. Absence of a new point excludes only these rectangles. A bounded Simon 2-descent for T=-44/43 timed out at 120 seconds; for T=-802/2917 it hit PARI’s 1 GiB bnfinit stack limit before returning a bound. Bounded mwrank bounds on both fibers failed because their 2-descents did not complete. PARI ellrank with known points and zero search effort also timed out at 60 seconds for T=-44/43 and 45 seconds for T=-47/500. Full Sage/eclib saturation attempts on T=-5/36 and T=1/4 reached their 120-second hard limits; the bounded-prime independence certificates remain valid. None supplies a global upper bound.

## Adaptation and next work

The cheap screen returned only known section points. Wide windows therefore went to the strongest 12 Stage B leads; six additional standalone refined leads were screened and widened. Structured x lattices and subsequent rescaled windows favored smaller parameter denominators, which provide better resolution in family coordinates. Candidates with only known points were demoted from further identical-width searches. Fourteen outer windows on two small-denominator fibers tested logarithmically spaced family-coordinate regions and found no points. A new GPU mode enumerated square denominators k² through k=46,340. Forty-six rectangles across 15 fibers, including rescaled family-coordinate windows, returned only known points; the square mode matched exact CPU enumeration on a small reference rectangle. Exact Sage group arithmetic identified small-height integral x-coordinates of sums of the three known generators on six fibers. Twelve GPU rectangles centered at those subgroup points found no new points. A separate sparse-coordinate scan tested 2,929,536 low-complexity expressions over Q(T) and 128,899,584 specialized expressions across all 44 selected fibers. Its modular filter left only the three known generic section forms; exact fixed-fiber checks returned no new x-coordinate. This excludes only the explicitly recorded ansatz. Minimal-model diagnostics on six leading fibers found larger maximum coefficient bit sizes than the integral factored models, so repeating the failed descents on these minimal models was not prioritized. Fourteen additional small-denominator leads were inventoried; wide windows produced independent point discoveries on T=-5/36, -7/12, -1/19, and 1/4. Discovery-centered windows raised the certified subgroup ranks further on the first two. The T=1/4 point set was reduced numerically to a candidate basis and certified independently by bounded Sage/eclib saturation through prime 11; numerical selection itself is not a rank proof. Promising follow-up is to derive candidate x-coordinates from covering curves or lattice reduction, then feed those centers to the exact GPU sieve; repeated local rectangles around section points have low yield. Alternative bounded descent algorithms may resolve upper bounds for the smaller-denominator fibers. No global upper bound is claimed.
