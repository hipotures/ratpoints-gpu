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
MODES = ("invariants", "pari-bound", "mwrank-bound", "saturation", "analytic", "search")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--log-height", type=float,
                        help="logarithmic naive x-height bound on the minimal model; required for search")
    parser.add_argument("--output", type=Path, help="result path (set by the Docker runner)")
    args = parser.parse_args(argv)
    if args.mode == "search" and (args.log_height is None or args.log_height <= 0):
        parser.error("--mode search requires a positive --log-height")
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
    elif args.mode == "search":
        bound_reports = [HERE / "results" / f"rank31-t-minus-47-80-sage-{mode}.json"
                         for mode in ("pari-bound", "mwrank-bound")]
        valid_bound = False
        for path in bound_reports:
            if not path.exists():
                continue
            bound = json.loads(path.read_text(encoding="utf-8"))
            if (bound.get("base_sha256") == hashlib.sha256(source_bytes).hexdigest()
                    and bound.get("status", "completed") == "completed"
                    and any(row.get("status") == "completed" for key, row in bound.get("steps", {}).items()
                            if key.startswith("rank_upper_bound_"))):
                valid_bound = True
        if not valid_bound:
            raise RuntimeError("run and review a completed --mode pari-bound or mwrank-bound before point search")
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
            "rationale": "Compare this bound with known_minimal_x_log_heights; search only after model and rank-bound review.",
        }
        result["steps"]["point_search"] = measured(
            "point_search",
            lambda: [point_json(point) for point in minimal.point_search(args.log_height)],
            "exactly verified rational points; completeness only within Sage search method/bound")
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
