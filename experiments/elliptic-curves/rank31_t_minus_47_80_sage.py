#!/usr/bin/env python3
"""Manual SageMath follow-up for the exact T=-47/80 report.

Run with `sage -python`, one explicit mode at a time. Descent, saturation and
point search may take substantial time and are never launched by the base triage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / "results" / "rank31-t-minus-47-80.json"
MODES = ("invariants", "pari-bound", "pari-init", "mwrank-bound", "simon-bound", "simon-shifted", "small-isogenies", "saturation", "analytic", "sections", "section-scan", "section-scan-lll", "rational-section-scan", "verify-gpu", "search")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--log-height", type=float,
                        help="logarithmic naive x-height bound on the minimal model; required for search")
    parser.add_argument("--gpu-report", action="append", default=[],
                        help="GPU report basename under results/ (repeatable)")
    parser.add_argument("--section-coeff-bound", type=int, default=2,
                        help="absolute coefficient bound for polynomial section scans (1..10)")
    parser.add_argument("--output", type=Path, help="result path (set by the Docker runner)")
    args = parser.parse_args(argv)
    if args.mode == "search" and (args.log_height is None or args.log_height <= 0):
        parser.error("--mode search requires a positive --log-height")
    if args.mode == "verify-gpu" and not args.gpu_report:
        parser.error("--mode verify-gpu requires --gpu-report")
    if not 1 <= args.section_coeff_bound <= 10:
        parser.error("--section-coeff-bound must be in 1..10")
    return args


def point_json(point):
    if point.is_zero():
        return {"infinity": True}
    return {"x": str(point[0]), "y": str(point[1])}


def measured(name, function, classification):
    started = time.perf_counter()
    try:
        value = function()
        return {"status": "completed", "value": value,
                "classification": classification,
                "elapsed_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.perf_counter() - started}


def main():
    args = parse_args()
    from sage.all import EllipticCurve, QQ
    import sage.version
    started = time.perf_counter()
    source_bytes = BASE.read_bytes()
    base = json.loads(source_bytes)
    if base["parameter_t"] != "-47/80":
        raise ValueError("input is not the T=-47/80 exact report")
    a = [QQ(value) for value in base["models"]["integral"]["ainvariants"]]
    integral = EllipticCurve(QQ, a)
    if integral.j_invariant() != QQ(base["models"]["integral"]["j_invariant"]):
        raise ArithmeticError("Sage integral model disagrees with exact Python report")
    minimal = integral.minimal_model()
    iso = integral.isomorphism_to(minimal)
    point_rows = []
    for row in base["visible_points"]:
        x, y = map(QQ, row["integral_xy"])
        point = integral(x, y)
        mapped = iso(point)
        if not mapped.curve() == minimal:
            raise AssertionError("point mapping failed")
        point_rows.append({"label": row["label"], "integral": point_json(point),
                           "minimal": point_json(mapped)})
    params = None
    if hasattr(iso, "tuple"):
        params = [str(value) for value in iso.tuple()]
    result = {
        "mode": args.mode,
        "command": "sage -python " + " ".join(sys.argv),
        "sage_version": sage.version.version,
        "base_report": str(BASE),
        "base_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "integral_ainvariants": [str(value) for value in integral.ainvs()],
        "minimal_ainvariants": [str(value) for value in minimal.ainvs()],
        "minimal_discriminant": str(minimal.discriminant()),
        "integral_to_minimal_isomorphism": {
            "parameters_u_r_s_t": params,
            "description": str(iso),
            "rational_maps": [str(value) for value in iso.rational_maps()],
            "convention": "x_old=u^2*x_new+r; y_old=u^3*y_new+s*u^2*x_new+t",
        },
        "verified_points": point_rows,
        "steps": {},
    }
    if args.mode == "invariants":
        result["steps"]["conductor"] = measured(
            "conductor", lambda: str(minimal.conductor()), "rigorous")
        result["steps"]["global_root_number"] = measured(
            "global_root_number", lambda: int(minimal.root_number()), "rigorous")
        result["steps"]["torsion"] = measured(
            "torsion", lambda: str(minimal.torsion_subgroup()), "rigorous")
        result["steps"]["local_reduction_data"] = measured(
            "local_reduction_data", lambda: [str(item) for item in minimal.local_data()],
            "rigorous")
    elif args.mode in ("pari-bound", "mwrank-bound"):
        algorithm = args.mode.removesuffix("-bound")
        result["steps"][f"rank_upper_bound_{algorithm}"] = measured(
            f"rank_upper_bound_{algorithm}",
            lambda: int(minimal.rank_bound(algorithm=algorithm)),
            "rigorous algebraic upper bound when completed")
    elif args.mode == "pari-init":
        result["steps"]["ellrankinit"] = measured(
            "ellrankinit", lambda: str(minimal.pari_curve().ellrankinit()),
            "PARI 2-descent initialization diagnostic; no rank bound by itself")
    elif args.mode in ("simon-bound", "simon-shifted"):
        known = [minimal(QQ(point_rows[i]["minimal"]["x"]),
                         QQ(point_rows[i]["minimal"]["y"])) for i in (0, 2)]
        curve = minimal
        if args.mode == "simon-shifted":
            center = QQ(-1343557003575107586)
            a4, a6 = minimal.a4(), minimal.a6()
            curve = EllipticCurve(QQ, [1, 3*center, center,
                                      3*center**2+a4,
                                      center**3+a4*center+a6])
            known = [curve(point[0]-center, point[1]) for point in known]
            result["shifted_model"] = {"center": str(center),
                                       "ainvariants": [str(z) for z in curve.ainvs()]}
        def simon_descent():
            lower, upper, points = curve.simon_two_descent(
                known_points=known, limbigprime=0)
            return {"lower": int(lower), "upper": int(upper),
                    "points": [point_json(z) for z in points]}
        result["steps"]["simon_two_descent"] = measured(
            "simon_two_descent", simon_descent,
            "Denis Simon PARI/GP 2-Selmer upper bound if completed")
    elif args.mode == "small-isogenies":
        for degree in (2, 3, 5, 7):
            result["steps"][f"isogenies_degree_{degree}"] = measured(
                f"isogenies_degree_{degree}",
                lambda degree=degree: [
                    {"degree": degree,
                     "codomain_ainvariants": [str(z) for z in phi.codomain().ainvs()]}
                    for phi in minimal.isogenies_prime_degree(l=degree)],
                "exact rational prime-degree isogeny computation")
    elif args.mode == "saturation":
        known = [minimal(QQ(point_rows[i]["minimal"]["x"]),
                         QQ(point_rows[i]["minimal"]["y"])) for i in (0, 2)]
        def saturate_known():
            basis, index, regulator = minimal.saturation(known)
            return {"basis": [point_json(point) for point in basis],
                    "index": str(index), "regulator_numerical": str(regulator),
                    "note": "Sage requires the supplied points to be independent"}
        result["steps"]["visible_pair_saturation"] = measured(
            "visible_pair_saturation", saturate_known,
            "exact eclib saturation if successful; inspect tool output for certification")
    elif args.mode == "analytic":
        result["steps"]["analytic_rank"] = measured(
            "analytic_rank", lambda: int(minimal.analytic_rank(algorithm="pari")),
            "numerical/heuristic, not a Mordell-Weil rank proof")
    elif args.mode in ("sections", "section-scan", "section-scan-lll", "rational-section-scan"):
        from sage.all import PolynomialRing, FractionField
        R = PolynomialRing(QQ, "T")
        T = R.gen()
        L = 446667*T**2 + 471466*T + 239031
        p = 318552*T**2 + 368554*T - 72570
        q = 733413*T**2 - 45082*T - 14960
        D = 5*(7174492962*T**4 - 7114589515*T**3 - 22069002960*T**2 + 3909144679*T - 205134150)
        E = 882769396002*T**4 + 811447034567*T**3 - 1174040743*T**2 - 32493137198*T - 2386325360
        B = p*q*(L+p+q) - p*E - q*D
        K = FractionField(R)
        family = EllipticCurve(K, [-L, D+E, -B, D*E, 0])
        x, y = -p*q, p*(p*q-E)
        generic = family(K(x), K(y))  # exact polynomial identity
        specialized = integral(QQ(x(-QQ(47)/80))*6400**2,
                               QQ(y(-QQ(47)/80))*6400**3)
        mapped = iso(specialized)
        basis = [minimal(QQ(point_rows[i]["minimal"]["x"]),
                         QQ(point_rows[i]["minimal"]["y"])) for i in (0, 2)]
        def certify():
            points, index, regulator = minimal.saturation(basis + [mapped])
            return {"basis": [point_json(z) for z in points], "index": str(index),
                    "regulator_numerical": str(regulator), "rank": len(points)}
        result["generic_section"] = {"x": str(x), "y": str(y),
                                      "verified_exactly": bool(generic in family),
                                      "specialized_integral": point_json(specialized),
                                      "specialized_minimal": point_json(mapped)}
        candidates = {"pq": p*q, "minus_pq": -p*q, "p_squared": p*p,
                      "minus_p_squared": -p*p, "q_squared": q*q,
                      "minus_q_squared": -q*q, "minus_D": -D,
                      "minus_E": -E, "D": D, "E": E,
                      "minus_D_minus_E": -D-E,
                      "p_times_q_plus_D": p*q+D,
                      "p_times_q_plus_E": p*q+E,
                      "minus_pq_minus_D": -p*q-D,
                      "minus_pq_minus_E": -p*q-E}
        result["candidate_x_square_tests"] = {}
        for label, trial_x in candidates.items():
            discriminant_y = (L*trial_x+B)**2 + 4*trial_x*(trial_x+D)*(trial_x+E)
            result["candidate_x_square_tests"][label] = bool(discriminant_y.is_square())
        if args.mode == "sections":
            result["steps"]["section_saturation"] = measured("section_saturation", certify,
                "exact Sage/eclib saturation of independent input points")
        elif args.mode in ("section-scan", "section-scan-lll"):
            from itertools import product
            expressions = (p*p, p*q, q*q, D, E)
            if args.mode == "section-scan-lll":
                from sage.all import Matrix, ZZ
                rows = Matrix(ZZ, [[ZZ(v[i]) for i in range(5)] for v in expressions]).LLL()
                expressions = tuple(R(list(row)) for row in rows.rows())
            seen = set()
            hits = []
            bound = args.section_coeff_bound
            for coefficients in product(range(-bound, bound+1), repeat=len(expressions)):
                trial_x = sum((c*v for c, v in zip(coefficients, expressions)), R.zero())
                if trial_x in seen:
                    continue
                seen.add(trial_x)
                disc_y = (L*trial_x+B)**2 + 4*trial_x*(trial_x+D)*(trial_x+E)
                if not disc_y.is_square():
                    continue
                trial_y = (L*trial_x+B+disc_y.sqrt())/2
                pt = family(K(trial_x), K(trial_y))
                sx, sy = QQ(trial_x(-QQ(47)/80)), QQ(trial_y(-QQ(47)/80))
                special = iso(integral(sx*6400**2, sy*6400**3))
                hits.append({"coefficients": list(coefficients), "x": str(trial_x),
                             "y": str(trial_y), "minimal": point_json(special),
                             "verified_generic": bool(pt in family)})
            result["scan"] = {"basis": [str(v) for v in expressions],
                              "coefficient_range": [-bound, bound],
                              "distinct_polynomials_tested": len(seen), "hits": hits}
        else:
            from itertools import combinations, product
            numerators = (p*p, p*q, q*q, D, E, p*D, q*D, p*E, q*E)
            denominators = (p, q, L, p+q, p-q, L+p, L+q)
            seen = set()
            hits = []
            templates = []
            for i in range(len(numerators)):
                for j in range(i, len(numerators)):
                    for a, b in product((-2, -1, 0, 1, 2), repeat=2):
                        if a != 0 and not (i == j and b != 0):
                            templates.append(((i, j), (a, b)))
            for indices in combinations(range(len(numerators)), 3):
                for coeffs in product((-1, 1), repeat=3):
                    templates.append((indices, coeffs))
            for indices, coeffs in templates:
                numerator = sum((c*numerators[i] for i, c in zip(indices, coeffs)), R.zero())
                for denominator in denominators:
                    trial_x = K(numerator)/denominator
                    if trial_x.denominator() == 1 or trial_x in seen:
                        continue
                    seen.add(trial_x)
                    disc_y = (L*trial_x+B)**2 + 4*trial_x*(trial_x+D)*(trial_x+E)
                    if not disc_y.is_square():
                        continue
                    trial_y = (L*trial_x+B+disc_y.sqrt())/2
                    pt = family(trial_x, trial_y)
                    sx = QQ(trial_x(-QQ(47)/80))
                    sy = QQ(trial_y(-QQ(47)/80))
                    special = iso(integral(sx*6400**2, sy*6400**3))
                    hits.append({"numerator_indices": list(indices),
                                 "numerator_coefficients": list(coeffs),
                                 "denominator": str(denominator),
                                 "x": str(trial_x), "y": str(trial_y),
                                 "minimal": point_json(special),
                                 "verified_generic": bool(pt in family)})
            result["scan"] = {"numerator_basis": [str(z) for z in numerators],
                              "denominators": [str(z) for z in denominators],
                              "distinct_rational_functions_tested": len(seen),
                              "hits": hits}
    elif args.mode == "verify-gpu":
        t = QQ(-47)/80
        p = 318552*t**2 + 368554*t - 72570
        q = 733413*t**2 - 45082*t - 14960
        E = 882769396002*t**4 + 811447034567*t**3 - 1174040743*t**2 - 32493137198*t - 2386325360
        section = iso(integral(QQ(-p*q)*6400**2, QQ(p*(p*q-E))*6400**3))
        basis = [minimal(QQ(point_rows[i]["minimal"]["x"]),
                         QQ(point_rows[i]["minimal"]["y"])) for i in (0, 2)] + [section]
        verified = []
        additions = []
        unresolved = []
        for basename in args.gpu_report:
            if Path(basename).name != basename or not basename.startswith("rank31-gpu-fixed-"):
                raise ValueError("GPU report must be a rank31-gpu-fixed basename")
            report = json.loads((HERE / "results" / basename).read_text())
            if [str(z) for z in report["model"]] != [str(z) for z in minimal.ainvs()]:
                raise ArithmeticError("GPU report uses a different minimal model")
            for row in report["points"]:
                point = minimal(QQ(row["x"]), QQ(row["y"]))
                verified.append({"report": basename, "point": point_json(point)})
                known_pe = -basis[0]-basis[1]
                if point in basis or -point in basis or point == known_pe or -point == known_pe:
                    continue
                try:
                    enlarged, index, regulator = minimal.saturation(basis + [point])
                    if len(enlarged) > len(basis):
                        basis = enlarged
                        additions.append({"point": point_json(point),
                                          "rank": len(basis), "index": str(index),
                                          "regulator_numerical": str(regulator)})
                except (ArithmeticError, ValueError, RuntimeError) as exc:
                    unresolved.append({"point": point_json(point), "error": str(exc)})
        result["gpu_verification"] = {"reports": args.gpu_report,
                                      "verified_points": verified,
                                      "independent_additions": additions,
                                      "unresolved_candidates": unresolved,
                                      "final_basis": [point_json(z) for z in basis],
                                      "rank_lower_bound": len(basis)}
    elif args.mode == "search":
        heights = []
        for row in point_rows:
            x = QQ(row["minimal"]["x"])
            heights.append(math.log(max(abs(int(x.numerator())), int(x.denominator()), 1)))
        result["known_minimal_x_log_heights"] = heights
        print("Known minimal-model point log x-heights:", heights, flush=True)
        print("Requested search log-height bound:", args.log_height, flush=True)
        result["search"] = {
            "model": "Sage minimal Weierstrass model",
            "height_convention": "log(max(abs(reduced x numerator), positive denominator))",
            "log_height_bound": args.log_height,
            "rationale": "Search requires no global rank upper bound; compare with known x-heights.",
        }
        def search_points():
            found = minimal.point_search(args.log_height)
            verified = [minimal(QQ(z[0]), QQ(z[1])) for z in found]
            t = QQ(-47)/80
            p = 318552*t**2 + 368554*t - 72570
            q = 733413*t**2 - 45082*t - 14960
            E = 882769396002*t**4 + 811447034567*t**3 - 1174040743*t**2 - 32493137198*t - 2386325360
            section = iso(integral(QQ(-p*q)*6400**2, QQ(p*(p*q-E))*6400**3))
            known = [minimal(QQ(point_rows[i]["minimal"]["x"]),
                             QQ(point_rows[i]["minimal"]["y"])) for i in (0, 2)] + [section]
            all_points = known[:]
            increases = []
            unresolved = []
            for point in verified:
                if point in all_points or -point in all_points:
                    continue
                try:
                    trial = minimal.saturation(all_points + [point])
                    if len(trial[0]) > len(all_points):
                        all_points = trial[0]
                        increases.append({"point": point_json(point), "rank": len(all_points),
                                          "index": str(trial[1]),
                                          "regulator_numerical": str(trial[2])})
                except (ArithmeticError, ValueError, RuntimeError) as exc:
                    unresolved.append({"point": point_json(point), "reason": str(exc)})
            return {"points": [point_json(z) for z in verified],
                    "independent_additions": increases,
                    "unresolved_candidates": unresolved,
                    "final_basis": [point_json(z) for z in all_points],
                    "final_rank_lower_bound": len(all_points)}
        result["steps"]["point_search"] = measured("point_search", search_points,
            "exact curve verification and Sage/eclib saturated independence checks")
    result["total_elapsed_seconds"] = time.perf_counter() - started
    failed_steps = {key: row["error"] for key, row in result["steps"].items()
                    if row["status"] != "completed"}
    result["status"] = "error" if failed_steps else "completed"
    if failed_steps:
        result["error"] = failed_steps
    target = args.output or HERE / "results" / f"rank31-t-minus-47-80-sage-{args.mode}.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
