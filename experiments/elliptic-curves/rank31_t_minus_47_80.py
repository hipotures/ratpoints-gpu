#!/usr/bin/env python3
"""Exact, short arithmetic triage of the T=-47/80 rank-31-family fiber."""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from fractions import Fraction

from rank31_mn_common import (
    ICARM_URL, RESULTS_DIR, discriminant, family_ainvs, family_values,
    odd_primes_up_to, on_curve, transformed_ainvs,
)

T = Fraction(-47, 80)
INTEGRAL_SCALE = 6400
STEM = "rank31-t-minus-47-80"


def c_invariants(a):
    a1, a2, a3, a4, a6 = a
    b2 = a1*a1 + 4*a2
    b4 = a1*a3 + 2*a4
    b6 = a3*a3 + 4*a6
    c4 = b2*b2 - 24*b4
    c6 = -b2**3 + 36*b2*b4 - 216*b6
    delta = discriminant(a)
    if delta == 0:
        raise ArithmeticError("singular specialization")
    if c4**3 - c6**2 != 1728*delta:
        raise AssertionError("Weierstrass invariant identity failed")
    return c4, c6, delta, c4**3/delta


def negate(point, a):
    x, y = point
    return x, -y - a[0]*x - a[2]


def add(p, q, a):
    """Exact chord-and-tangent addition; None denotes the point at infinity."""
    if p is None:
        return q
    if q is None:
        return p
    x1, y1 = p
    x2, y2 = q
    if x1 == x2 and q == negate(p, a):
        return None
    a1, a2, a3, a4, _a6 = a
    if p == q:
        denominator = 2*y1 + a1*x1 + a3
        if denominator == 0:
            return None
        slope = (3*x1*x1 + 2*a2*x1 + a4 - a1*y1)/denominator
    else:
        slope = (y2-y1)/(x2-x1)
    intercept = y1 - slope*x1
    x3 = slope*slope + a1*slope - a2 - x1 - x2
    y3 = -(slope+a1)*x3 - intercept - a3
    result = x3, y3
    if not on_curve(result, a):
        raise AssertionError("point addition left the curve")
    return result


def count_points_mod_p(a, prime):
    if discriminant(a) % prime == 0:
        raise ValueError(f"bad reduction at {prime}")
    coeffs = [int(v) % prime for v in a]
    a1, a2, a3, a4, a6 = coeffs
    count = 1  # point at infinity
    for x in range(prime):
        for y in range(prime):
            if (y*y + a1*x*y + a3*y - x**3 - a2*x*x - a4*x - a6) % prime == 0:
                count += 1
    return count


def small_discriminant_divisors(delta, limit=1000):
    remaining = abs(int(delta))
    factors = []
    for prime in [2] + odd_primes_up_to(limit):
        exponent = 0
        while remaining % prime == 0:
            remaining //= prime
            exponent += 1
        if exponent:
            factors.append({"prime": prime, "valuation_on_integral_model": exponent})
    return factors, remaining


def exact_report():
    started = time.perf_counter()
    L, p, q, D, E, B = family_values(T)
    family_a = family_ainvs(T)
    u = Fraction(1, INTEGRAL_SCALE)
    integral_a = transformed_ainvs(family_a, u, 0, 0, 0)
    if any(value.denominator != 1 for value in integral_a):
        raise AssertionError("chosen model is not integral")
    if family_a != (-L, D+E, -B, D*E, Fraction(0)):
        raise AssertionError("family coefficients disagree")
    c4f, c6f, delta_f, jf = c_invariants(family_a)
    c4i, c6i, delta_i, ji = c_invariants(integral_a)
    if jf != ji or delta_f/u**12 != delta_i:
        raise AssertionError("model invariants do not transform correctly")

    representatives = [("P0", Fraction(0), Fraction(0)),
                       ("PD", -D, Fraction(0)),
                       ("PE", -E, Fraction(0))]
    points = []
    representative_points = []
    for label, x, y in representatives:
        point = x, y
        inverse = negate(point, family_a)
        if not on_curve(point, family_a) or not on_curve(inverse, family_a):
            raise AssertionError(f"{label} is not on the family curve")
        representative_points.append(point)
        for name, (px, py), inverse_of in ((label, point, None),
                                            (f"-{label}", inverse, label)):
            ix, iy = px/u**2, py/u**3
            if ix.denominator != 1 or iy.denominator != 1:
                raise AssertionError(f"{name} did not map to an integral point")
            if not on_curve((ix, iy), integral_a):
                raise AssertionError(f"{name} is not on the integral model")
            points.append({"label": name, "inverse_of": inverse_of,
                           "family_xy": [str(px), str(py)],
                           "integral_xy": [str(ix), str(iy)],
                           "verified_exactly": True})
    if len(set(tuple(point["family_xy"]) for point in points)) != 6:
        raise AssertionError("visible points were not distinct")
    if add(add(representative_points[0], representative_points[1], family_a),
           representative_points[2], family_a) is not None:
        raise AssertionError("the three y=0 points do not sum to infinity")

    reductions = []
    for prime, expected in ((7, 13), (13, 21)):
        count = count_points_mod_p(integral_a, prime)
        if count != expected:
            raise AssertionError(f"unexpected point count at {prime}: {count}")
        reductions.append({"prime": prime, "point_count": count,
                           "integral_discriminant_mod_prime": int(delta_i) % prime})
    if math.gcd(*(row["point_count"] for row in reductions)) != 1:
        raise AssertionError("torsion bound did not become trivial")
    small_factors, cofactor = small_discriminant_divisors(delta_i)
    elapsed = time.perf_counter() - started
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameter_t": str(T),
        "source": ICARM_URL,
        "source_commit": source_head,
        "arithmetic_backend": {"name": "Python fractions.Fraction", "python": platform.python_version()},
        "family_values": {key: str(value) for key, value in
                          zip(("L", "p", "q", "D", "E", "B"), (L, p, q, D, E, B))},
        "models": {
            "family": {"ainvariants": [str(v) for v in family_a],
                       "c4": str(c4f), "c6": str(c6f),
                       "discriminant": str(delta_f), "j_invariant": str(jf)},
            "integral": {"ainvariants": [str(v) for v in integral_a],
                         "c4": str(c4i), "c6": str(c6i),
                         "discriminant": str(delta_i), "j_invariant": str(ji),
                         "minimality": "not established"},
            "minimal": None,
        },
        "family_to_integral_change": {
            "u": str(u), "r": "0", "s": "0", "t": "0",
            "convention": "x_family=u^2*x_integral; y_family=u^3*y_integral",
            "five_ainvariants_verified_exactly": True,
        },
        "visible_points": points,
        "point_relation": "P0 + PD + PE = O, verified by exact group addition",
        "torsion": {"status": "proven trivial", "proof": "good reduction at 7 and 13; torsion order divides gcd(13,21)=1",
                    "good_prime_reductions": reductions},
        "rank": {"rigorous_lower_bound": 1,
                 "proof": "P0 is nonidentity and rational torsion is trivial",
                 "visible_subgroup_rank_range": [1, 2],
                 "visible_subgroup_rank_exact": None,
                 "global_upper_bound": None,
                 "saturation_status": "not attempted; SageMath/mwrank unavailable on this host"},
        "arithmetic_invariants": {
            "conductor": None, "global_root_number": None,
            "analytic_rank": None, "local_reduction_data": None,
            "integral_discriminant_small_prime_divisors_up_to_1000": small_factors,
            "unfactored_discriminant_cofactor": str(cofactor),
            "warning": "Divisors of a nonminimal model discriminant are not a certified bad-prime list."},
        "heuristic_evidence": {
            "mestre_nagao_score_p_500000": 215.984351659,
            "known_rank_31_control_score_p_500000": 227.159765987,
            "rank_claim": "none; these scores are not rank proofs",
            "source": "GitHub issue #12"},
        "external_work": {
            "reason": "SageMath, PARI/GP, mwrank and Magma executables are unavailable on this host",
            "commands": [
                "sage -python experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode invariants",
                "sage -python experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode descent",
                "sage -python experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode saturation",
                "sage -python experiments/elliptic-curves/rank31_t_minus_47_80_sage.py --mode search --log-height N",
            ],
            "search_note": "Run only after inspecting the minimal model, known-point x-heights and descent bounds; N is logarithmic naive x-height on the minimal model.",
        },
        "timings_seconds": {"exact_triage": elapsed},
    }


def summary_text(report):
    a = report["models"]["integral"]["ainvariants"]
    return ("# T = -47/80 exact arithmetic triage\n\n"
            "The family and integral models, their exact invariants, and six visible "
            "points are recorded in the companion JSON report. The integral model "
            "has a1..a6 = " + str(a) + ".\n\n"
            "**Proven:** the visible points lie on the curve; P0 + PD + PE = O; "
            "good-prime counts are #E(F_7)=13 and #E(F_13)=21, so rational torsion "
            "is trivial and the Mordell-Weil rank is at least 1. The exact rank "
            "of the visible subgroup is unresolved (1 or 2).\n\n"
            "**Not computed:** minimal model, conductor, global root number, local "
            "reduction data, analytic rank, a global rank upper bound, and saturation. "
            "The Sage companion script provides separate reproducible modes for these "
            "steps and an optional minimal-model point search. No long arithmetic or GPU "
            "job was run. Mestre-Nagao scores are heuristic evidence, not rank proofs.\n")


def main():
    report = exact_report()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{STEM}.json"
    summary_path = RESULTS_DIR / f"{STEM}.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(summary_text(report), encoding="utf-8")
    print(f"Wrote {json_path} and {summary_path}")
    print("Proven rank lower bound: 1; torsion: trivial; minimal model and upper bound: pending external SageMath")


if __name__ == "__main__":
    main()
