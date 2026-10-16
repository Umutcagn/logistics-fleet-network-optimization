class OptimizationAgent:
    """
    Sub-Agent 2: Rota ve Filo Optimizasyonu Sub-Agent'ı
    Executes FSMVRP ILP or ALNS solver, evaluates spot vehicle ratio, and balances fleet composition.
    """
    def __init__(self, ilp_solver, alns_solver):
        self.ilp_solver = ilp_solver
        self.alns_solver = alns_solver

    def run_optimization(self, demand_dict, distance_matrix, spot_costs, backlogs_wit):
        route_count = len(demand_dict)
        
        if route_count < 50:
            sol = self.ilp_solver.solve(demand_dict, distance_matrix, spot_costs, backlogs_wit)
        else:
            sol = self.alns_solver.solve(demand_dict, distance_matrix, spot_costs)

        total_v = sol["total_fleet_vehicles"] + sol["total_spot_vehicles"]
        spot_ratio = (sol["total_spot_vehicles"] / max(1, total_v)) * 100

        summary = {
            "solver_used": "OR-Tools ILP" if route_count < 50 else "ALNS Heuristic",
            "total_cost_tl": sol["total_cost"],
            "total_fleet_vehicles": sol["total_fleet_vehicles"],
            "total_spot_vehicles": sol["total_spot_vehicles"],
            "spot_vehicle_ratio_percent": round(spot_ratio, 1),
            "raw_solution": sol
        }
        return summary
