from .fsmvrp_ilp import FSMVRP_ILPSolver, ILP_ROUTE_LIMIT
from .alns_solver import ALNSSolver

class StochasticScenarioOptimizer:
    """
    Two-Stage Stochastic Programming under Demand Uncertainty (Faz 2).
    Evaluates expected solution costs across Optimistic, Base, and Pessimistic scenario trees.
    
    Reference: Taherkhani et al. (2025) [Two-Stage Stochastic Programming & Benders Decomposition]
    """
    def __init__(self, ilp_solver=None, alns_solver=None):
        self.ilp_solver = ilp_solver or FSMVRP_ILPSolver()
        self.alns_solver = alns_solver or ALNSSolver()

    def optimize_scenarios(self, prediction_intervals: dict, routes_list: list, distance_matrix: dict, spot_costs: dict):
        """
        prediction_intervals: {"optimistic": np.array, "base": np.array, "pessimistic": np.array}
        Scenario probabilities: P(optimistic) = 0.25, P(base) = 0.50, P(pessimistic) = 0.25
        """
        scenario_results = {}
        probabilities = {"optimistic": 0.25, "base": 0.50, "pessimistic": 0.25}
        
        expected_cost = 0.0

        for sc_name, probs in probabilities.items():
            preds = prediction_intervals[sc_name]
            demand_dict = {}
            for idx, (i, j) in enumerate(routes_list):
                demand_dict[(i, j)] = float(preds[idx])
                
            if len(routes_list) < ILP_ROUTE_LIMIT:
                sol = self.ilp_solver.solve(demand_dict, distance_matrix, spot_costs)
            else:
                sol = self.alns_solver.solve(demand_dict, distance_matrix, spot_costs)
                
            cost = sol["total_cost"]
            scenario_results[sc_name] = {
                "probability": probs,
                "demand_total_desi": round(sum(demand_dict.values()), 2),
                "total_cost": cost,
                "spot_vehicles": sol["total_spot_vehicles"],
                "fleet_vehicles": sol["total_fleet_vehicles"]
            }
            
            expected_cost += probs * cost

        return {
            "expected_cost": round(expected_cost, 2),
            "scenario_details": scenario_results
        }
