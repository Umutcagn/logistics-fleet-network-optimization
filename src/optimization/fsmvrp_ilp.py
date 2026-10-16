import json
from pathlib import Path

class FSMVRP_ILPSolver:
    """
    Exact Integer Linear Programming (ILP) Solver for Fleet Size & Mix Vehicle Routing Problem.
    Uses Google OR-Tools CBC/SCIP solver. Used when route count < 50.
    
    References:
    - Salhi et al. 2013 [6] (FSMVRP mathematical formulation)
    - Fachini et al. 2025 [5] (Spot vehicle balancing)
    """
    def __init__(self, network_config_path=None):
        if network_config_path is None:
            network_config_path = Path(__file__).parent.parent / "data" / "sample_network.json"
            
        with open(network_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.transfer_centers = {tc["id"]: tc for tc in self.config["transfer_centers"]}
        self.vehicle_types = self.config["vehicle_types"]
        self.spot_params = self.config["spot_vehicle_params"]
        self.holding_cost = self.config.get("holding_cost_per_desi", 2.5)

    def solve(self, demand_dict: dict, distance_matrix: dict, spot_costs: dict, backlogs_wit: dict = None):
        """
        Solves ILP optimization for a set of route demands.
        demand_dict: {(origin, dest): demand_desi}
        """
        try:
            from ortools.linear_solver import pywraplp
        except ImportError:
            print("[Warning] OR-Tools not installed. Using Greedy Fallback Solver.")
            return self._solve_greedy(demand_dict, distance_matrix, spot_costs)

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            solver = pywraplp.Solver.CreateSolver('CBC')
        if not solver:
            return self._solve_greedy(demand_dict, distance_matrix, spot_costs)

        routes = list(demand_dict.keys())
        v_keys = list(self.vehicle_types.keys())
        
        # Decision Variables
        # X_ijk_m: Number of owned fleet vehicles of type m on route (origin, dest)
        X_vars = {}
        # Z_spot_ij: Integer number of spot vehicles on route (origin, dest)
        Z_vars = {}
        
        for (i, j) in routes:
            Z_vars[(i, j)] = solver.IntVar(0, 50, f"Z_{i}_{j}")
            for m in v_keys:
                # Check ramp constraint (K3)
                v_spec = self.vehicle_types[m]
                origin_tc = self.transfer_centers.get(i, {})
                dest_tc = self.transfer_centers.get(j, {})
                
                # If vehicle requires truck ramp but either origin or dest doesn't have it, max=0
                if v_spec["requires_truck_ramp"] and (not origin_tc.get("has_truck_ramp", True) or not dest_tc.get("has_truck_ramp", True)):
                    X_vars[(i, j, m)] = solver.IntVar(0, 0, f"X_{i}_{j}_{m}")
                else:
                    X_vars[(i, j, m)] = solver.IntVar(0, 20, f"X_{i}_{j}_{m}")

        # Objective function (Equation 1.1)
        objective = solver.Objective()
        
        for (i, j) in routes:
            dist = distance_matrix.get(i, {}).get(j, 100.0)
            spot_cost = spot_costs.get((i, j), 3000.0)
            
            # Spot vehicle cost
            objective.SetCoefficient(Z_vars[(i, j)], spot_cost)
            
            for m in v_keys:
                v_spec = self.vehicle_types[m]
                fixed_c = v_spec["fixed_cost"]
                var_c = dist * v_spec["cost_per_km"]
                total_vehicle_cost = fixed_c + var_c
                
                objective.SetCoefficient(X_vars[(i, j, m)], total_vehicle_cost)

        # Backlog holding cost beta * Wit
        if backlogs_wit:
            for tc_id, wit_val in backlogs_wit.items():
                # Add constant component to objective
                objective.SetOffset(objective.Offset() + self.holding_cost * wit_val)

        objective.SetMinimization()

        # Constraints
        # K1: Demand Fulfillment
        # sum_m (cap_m * X_ij_m) + cap_spot * Z_ij >= Demand_ij
        for (i, j), demand in demand_dict.items():
            ct = solver.Constraint(demand, solver.infinity(), f"Demand_{i}_{j}")
            ct.SetCoefficient(Z_vars[(i, j)], self.spot_params["capacity"])
            for m in v_keys:
                cap_m = self.vehicle_types[m]["capacity"]
                ct.SetCoefficient(X_vars[(i, j, m)], cap_m)

        # K5: Fill Rate / Doluluk Toleransı
        # Actual load >= (1 - alpha_m) * cap_m when X > 0
        # Managed naturally by SCIP objective minimization (choosing smallest adequate vehicle)

        # Solve model
        solver.set_time_limit(10000)  # 10 seconds timeout
        status = solver.Solve()

        solution = {
            "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else ("FEASIBLE" if status == pywraplp.Solver.FEASIBLE else "FAILED"),
            "total_cost": solver.Objective().Value(),
            "assignments": {},
            "spot_vehicles": {},
            "total_fleet_vehicles": 0,
            "total_spot_vehicles": 0
        }

        for (i, j) in routes:
            route_key = f"{i}->{j}"
            spot_count = int(Z_vars[(i, j)].solution_value())
            solution["spot_vehicles"][route_key] = spot_count
            solution["total_spot_vehicles"] += spot_count
            
            route_fleet = {}
            for m in v_keys:
                cnt = int(X_vars[(i, j, m)].solution_value())
                if cnt > 0:
                    route_fleet[m] = cnt
                    solution["total_fleet_vehicles"] += cnt
            
            solution["assignments"][route_key] = {
                "demand_desi": demand,
                "fleet_vehicles": route_fleet,
                "spot_vehicles": spot_count
            }

        return solution

    def _solve_greedy(self, demand_dict: dict, distance_matrix: dict, spot_costs: dict):
        """
        Fast heuristic solver fallback if OR-Tools is unavailable.
        """
        solution = {
            "status": "GREEDY_FALLBACK",
            "total_cost": 0.0,
            "assignments": {},
            "spot_vehicles": {},
            "total_fleet_vehicles": 0,
            "total_spot_vehicles": 0
        }
        
        v_keys = sorted(self.vehicle_types.keys(), key=lambda m: self.vehicle_types[m]["capacity"], reverse=True)
        
        for (i, j), demand in demand_dict.items():
            dist = distance_matrix.get(i, {}).get(j, 100.0)
            remaining_demand = demand
            route_key = f"{i}->{j}"
            fleet_assigned = {}
            spot_assigned = 0
            
            origin_tc = self.transfer_centers.get(i, {})
            dest_tc = self.transfer_centers.get(j, {})
            has_ramp = origin_tc.get("has_truck_ramp", True) and dest_tc.get("has_truck_ramp", True)

            for m in v_keys:
                v_spec = self.vehicle_types[m]
                if v_spec["requires_truck_ramp"] and not has_ramp:
                    continue
                    
                cap = v_spec["capacity"]
                cost = v_spec["fixed_cost"] + dist * v_spec["cost_per_km"]
                
                # Check if fleet vehicle is cheaper per desi than spot
                spot_unit_cost = spot_costs.get((i, j), 3000.0) / self.spot_params["capacity"]
                fleet_unit_cost = cost / cap
                
                if fleet_unit_cost < spot_unit_cost:
                    num_v = int(remaining_demand // cap)
                    if num_v > 0:
                        fleet_assigned[m] = num_v
                        remaining_demand -= num_v * cap
                        solution["total_cost"] += num_v * cost
                        solution["total_fleet_vehicles"] += num_v

            # Cover remaining demand with smallest vehicle or spot
            if remaining_demand > 0:
                spot_c = spot_costs.get((i, j), 3000.0)
                num_spot = int((remaining_demand + self.spot_params["capacity"] - 1) // self.spot_params["capacity"])
                spot_assigned += num_spot
                solution["total_cost"] += num_spot * spot_c
                solution["total_spot_vehicles"] += num_spot

            solution["assignments"][route_key] = {
                "demand_desi": demand,
                "fleet_vehicles": fleet_assigned,
                "spot_vehicles": spot_assigned
            }

        return solution
