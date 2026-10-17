"""
Compares the exact OR-Tools ILP and the ALNS heuristic on the same FSMVRP instances.

Instances are single days from the synthetic generator, with route subsets of
different sizes and demand scaled to stress the ramp (K3) limit. Both solvers use
the same cost function (fleet fixed + per-km cost, spot rental cost), so the
optimality gap is directly comparable.

Usage: python -m benchmarks.solver_benchmark
"""
import json
import random
import time
from pathlib import Path

import numpy as np

from src.data.synthetic_data_generator import LogisticsDataGenerator
from src.optimization.alns_solver import ALNSSolver
from src.optimization.fsmvrp_ilp import FSMVRP_ILPSolver

RESULTS_PATH = Path(__file__).parent / "solver_benchmark_results.json"
ROUTE_COUNTS = [10, 30, 50, 90]
DEMAND_SCALES = [1.0, 2.0, 3.0]
DAYS = [0, 7, 14, 21, 28]
RAMP_WAVES_PER_DAY = 6  # same K3 limit the ILP enforces


def ramp_violations(solution, transfer_centers):
    """Origins whose dispatched vehicle count exceeds ramps * waves (K3)."""
    dispatched = {}
    for key, a in solution["assignments"].items():
        origin = key.split("->")[0]
        dispatched[origin] = dispatched.get(origin, 0) + a["spot_vehicles"] + sum(a["fleet_vehicles"].values())
    return sum(1 for tc_id, n in dispatched.items() if n > transfer_centers[tc_id]["ramps"] * RAMP_WAVES_PER_DAY)


def run():
    gen = LogisticsDataGenerator()
    df_routes, _ = gen.generate_time_series_data(num_days=30)
    ilp, alns = FSMVRP_ILPSolver(), ALNSSolver(max_iterations=200)
    rows = []

    for n_routes in ROUTE_COUNTS:
        for scale in DEMAND_SCALES:
            for day in DAYS:
                day_df = df_routes[df_routes["day_idx"] == day].sort_values(["origin_id", "dest_id"]).head(n_routes)
                demand = {(r.origin_id, r.dest_id): r.actual_desi * scale for r in day_df.itertuples()}
                spot = {(r.origin_id, r.dest_id): r.spot_cost_s_ij for r in day_df.itertuples()}

                t0 = time.perf_counter()
                ilp_sol = ilp.solve(demand, gen.distance_matrix, spot)
                ilp_time = time.perf_counter() - t0

                random.seed(day)
                np.random.seed(day)
                t0 = time.perf_counter()
                alns_sol = alns.solve(demand, gen.distance_matrix, spot)
                alns_time = time.perf_counter() - t0

                rows.append({
                    "routes": n_routes, "demand_scale": scale, "day": day,
                    "ilp_status": ilp_sol["status"], "ilp_cost": ilp_sol["total_cost"], "ilp_seconds": ilp_time,
                    "alns_cost": alns_sol["total_cost"], "alns_seconds": alns_time,
                    "gap_pct": ((alns_sol["total_cost"] - ilp_sol["total_cost"]) / ilp_sol["total_cost"] * 100
                                if ilp_sol["status"] == "OPTIMAL" else None),
                    "alns_k3_violations": ramp_violations(alns_sol, ilp.transfer_centers),
                    "ilp_k3_violations": ramp_violations(ilp_sol, ilp.transfer_centers),
                })

    summary = []
    for n_routes in ROUTE_COUNTS:
        for scale in DEMAND_SCALES:
            group = [r for r in rows if r["routes"] == n_routes and r["demand_scale"] == scale]
            gaps = [r["gap_pct"] for r in group if r["gap_pct"] is not None]
            summary.append({
                "routes": n_routes, "demand_scale": scale, "instances": len(group),
                "ilp_optimal": sum(r["ilp_status"] == "OPTIMAL" for r in group),
                "ilp_infeasible": sum(r["ilp_status"] == "INFEASIBLE_GREEDY_FALLBACK" for r in group),
                "mean_gap_pct": round(float(np.mean(gaps)), 2) if gaps else None,
                "max_gap_pct": round(float(np.max(gaps)), 2) if gaps else None,
                "ilp_mean_seconds": round(float(np.mean([r["ilp_seconds"] for r in group])), 3),
                "alns_mean_seconds": round(float(np.mean([r["alns_seconds"] for r in group])), 3),
                "alns_instances_with_k3_violation": sum(r["alns_k3_violations"] > 0 for r in group),
                "ilp_instances_with_k3_violation": sum(r["ilp_k3_violations"] > 0 for r in group),
            })

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "instances": rows}, f, indent=2)

    def pct(v):
        return "—" if v is None else f"%{v:.2f}"

    print("| Güzergah | Talep | ILP optimal | ILP çözümsüz (K3) | ALNS ort. sapma | ALNS maks. sapma | ILP süre (sn) | ALNS süre (sn) | ALNS K3 ihlali |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in summary:
        print(f"| {s['routes']} | {s['demand_scale']:.0f}x | {s['ilp_optimal']}/{s['instances']} | {s['ilp_infeasible']}/{s['instances']} | "
              f"{pct(s['mean_gap_pct'])} | {pct(s['max_gap_pct'])} | {s['ilp_mean_seconds']:.3f} | "
              f"{s['alns_mean_seconds']:.3f} | {s['alns_instances_with_k3_violation']}/{s['instances']} |")
    return summary


if __name__ == "__main__":
    run()
