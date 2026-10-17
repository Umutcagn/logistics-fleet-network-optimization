import json
from pathlib import Path

# Above this route count the pipeline switches to ALNS. benchmarks/solver_benchmark.py shows the ILP
# solving the full 90-route network to optimality in ~0.02 s while ALNS is slower and 2-17% costlier,
# so the ILP is used for every network size this project generates.
ILP_ROUTE_LIMIT = 1000

class FSMVRP_ILPSolver:
    """
    Exact Integer Linear Programming (ILP) Solver for Fleet Size & Mix Vehicle Routing Problem.
    Uses Google OR-Tools CBC/SCIP solver. Used when route count < ILP_ROUTE_LIMIT.
    
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

        # K3: Ramp slot limit per origin Transfer Center (R_i * daily shift waves)
        origin_routes = {}
        for (i, j) in routes:
            origin_routes.setdefault(i, []).append((i, j))
            
        for orig_id, orig_r_list in origin_routes.items():
            tc_ramps = self.transfer_centers.get(orig_id, {}).get("ramps", 10)
            max_daily_dispatches = tc_ramps * 6  # 6 loading waves per day per ramp (K3)
            ct_ramp = solver.Constraint(0, max_daily_dispatches, f"RampLimit_{orig_id}")
            for (i, j) in orig_r_list:
                ct_ramp.SetCoefficient(Z_vars[(i, j)], 1)
                for m in v_keys:
                    ct_ramp.SetCoefficient(X_vars[(i, j, m)], 1)

        # Solve model
        solver.set_time_limit(10000)  # 10 seconds timeout
        status = solver.Solve()

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            # Usually K3: ramp slots cannot carry the demand. Never return an empty zero-cost plan.
            solution = self._solve_greedy(demand_dict, distance_matrix, spot_costs)
            solution["status"] = "INFEASIBLE_GREEDY_FALLBACK"
            solution["warning"] = (
                "ILP çözümsüz: rampa kapasitesi (K3) talebi karşılamaya yetmiyor. "
                "Gösterilen plan K3 gözetilmeden üretilmiş sezgisel bir plandır."
            )
            return solution

        solution = {
            "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else ("FEASIBLE" if status == pywraplp.Solver.FEASIBLE else "FAILED"),
            "total_cost": round(solver.Objective().Value(), 2),
            "assignments": {},
            "spot_vehicles": {},
            "total_fleet_vehicles": 0,
            "total_spot_vehicles": 0,
            "avg_fill_rate_pct": 0.0,
            "tc_utilization": {}
        }

        fill_rates = []
        tc_outbound_desi = {tc_id: 0.0 for tc_id in self.transfer_centers}
        tc_outbound_vehicles = {tc_id: 0 for tc_id in self.transfer_centers}

        for (i, j) in routes:
            route_key = f"{i}->{j}"
            route_demand = float(demand_dict[(i, j)])
            dist = float(distance_matrix.get(i, {}).get(j, 100.0))
            spot_cost_unit = float(spot_costs.get((i, j), 3000.0))

            spot_count = int(round(Z_vars[(i, j)].solution_value()))
            solution["spot_vehicles"][route_key] = spot_count
            solution["total_spot_vehicles"] += spot_count
            
            route_fleet = {}
            assigned_cap = spot_count * self.spot_params["capacity"]
            route_cost = spot_count * spot_cost_unit

            for m in v_keys:
                cnt = int(round(X_vars[(i, j, m)].solution_value()))
                if cnt > 0:
                    route_fleet[m] = cnt
                    solution["total_fleet_vehicles"] += cnt
                    v_spec = self.vehicle_types[m]
                    assigned_cap += cnt * v_spec["capacity"]
                    route_cost += cnt * (v_spec["fixed_cost"] + dist * v_spec["cost_per_km"])
            
            fill_rate = min(100.0, (route_demand / assigned_cap * 100.0)) if assigned_cap > 0 else 0.0
            fill_rates.append(fill_rate)

            if i in tc_outbound_desi:
                tc_outbound_desi[i] += route_demand
                tc_outbound_vehicles[i] += spot_count + sum(route_fleet.values())

            solution["assignments"][route_key] = {
                "demand_desi": round(route_demand, 1),
                "distance_km": round(dist, 1),
                "fleet_vehicles": route_fleet,
                "spot_vehicles": spot_count,
                "assigned_capacity_desi": assigned_cap,
                "fill_rate_pct": round(fill_rate, 1),
                "route_cost_tl": round(route_cost, 2)
            }

        solution["avg_fill_rate_pct"] = round(sum(fill_rates) / len(fill_rates), 1) if fill_rates else 0.0
        solution["tc_utilization"] = self._compute_tc_utilization(tc_outbound_desi, tc_outbound_vehicles)

        return solution

    def _compute_tc_utilization(self, tc_outbound_desi: dict, tc_outbound_vehicles: dict) -> dict:
        """
        Computes K3 (Ramp Utilization) and K4 (Handling Capacity H_i * Delta_t) metrics per Transfer Center.
        """
        delta_t_hours = 8.0  # Standard line-haul sorting window (hours)
        tc_stats = {}
        for tc_id, tc_info in self.transfer_centers.items():
            out_desi = tc_outbound_desi.get(tc_id, 0.0)
            out_veh = tc_outbound_vehicles.get(tc_id, 0)
            max_handling = tc_info["hourly_handling_capacity"] * delta_t_hours
            handling_pct = (out_desi / max_handling * 100.0) if max_handling > 0 else 0.0
            ramps = tc_info["ramps"]
            tc_stats[tc_id] = {
                "name": tc_info["name"],
                "ramps_R_i": ramps,
                "has_truck_ramp": tc_info.get("has_truck_ramp", True),
                "dispatched_vehicles": out_veh,
                "outbound_desi": round(out_desi, 1),
                "shift_handling_cap_K4": round(max_handling, 0),
                "k4_utilization_pct": round(handling_pct, 1),
                "k4_status": "AŞIM RİSKİ" if handling_pct > 90.0 else ("YOĞUN" if handling_pct > 65.0 else "UYGUN")
            }
        return tc_stats

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
            "total_spot_vehicles": 0,
            "avg_fill_rate_pct": 0.0,
            "tc_utilization": {}
        }
        
        v_keys = sorted(self.vehicle_types.keys(), key=lambda m: self.vehicle_types[m]["capacity"], reverse=True)
        fill_rates = []
        tc_outbound_desi = {tc_id: 0.0 for tc_id in self.transfer_centers}
        tc_outbound_vehicles = {tc_id: 0 for tc_id in self.transfer_centers}
        
        for (i, j), demand in demand_dict.items():
            dist = float(distance_matrix.get(i, {}).get(j, 100.0))
            remaining_demand = float(demand)
            route_key = f"{i}->{j}"
            fleet_assigned = {}
            spot_assigned = 0
            assigned_cap = 0
            route_cost = 0.0
            
            origin_tc = self.transfer_centers.get(i, {})
            dest_tc = self.transfer_centers.get(j, {})
            has_ramp = origin_tc.get("has_truck_ramp", True) and dest_tc.get("has_truck_ramp", True)

            for m in v_keys:
                v_spec = self.vehicle_types[m]
                if v_spec["requires_truck_ramp"] and not has_ramp:
                    continue
                    
                cap = v_spec["capacity"]
                cost = v_spec["fixed_cost"] + dist * v_spec["cost_per_km"]
                
                spot_unit_cost = spot_costs.get((i, j), 3000.0) / self.spot_params["capacity"]
                fleet_unit_cost = cost / cap
                
                if fleet_unit_cost < spot_unit_cost:
                    num_v = int(remaining_demand // cap)
                    if num_v > 0:
                        fleet_assigned[m] = num_v
                        remaining_demand -= num_v * cap
                        assigned_cap += num_v * cap
                        route_cost += num_v * cost
                        solution["total_cost"] += num_v * cost
                        solution["total_fleet_vehicles"] += num_v

            if remaining_demand > 0:
                spot_c = float(spot_costs.get((i, j), 3000.0))
                num_spot = int((remaining_demand + self.spot_params["capacity"] - 1) // self.spot_params["capacity"])
                spot_assigned += num_spot
                assigned_cap += num_spot * self.spot_params["capacity"]
                route_cost += num_spot * spot_c
                solution["total_cost"] += num_spot * spot_c
                solution["total_spot_vehicles"] += num_spot

            fill_rate = min(100.0, (float(demand) / assigned_cap * 100.0)) if assigned_cap > 0 else 0.0
            fill_rates.append(fill_rate)

            if i in tc_outbound_desi:
                tc_outbound_desi[i] += float(demand)
                tc_outbound_vehicles[i] += spot_assigned + sum(fleet_assigned.values())

            solution["assignments"][route_key] = {
                "demand_desi": round(float(demand), 1),
                "distance_km": round(dist, 1),
                "fleet_vehicles": fleet_assigned,
                "spot_vehicles": spot_assigned,
                "assigned_capacity_desi": assigned_cap,
                "fill_rate_pct": round(fill_rate, 1),
                "route_cost_tl": round(route_cost, 2)
            }

        solution["total_cost"] = round(solution["total_cost"], 2)
        solution["avg_fill_rate_pct"] = round(sum(fill_rates) / len(fill_rates), 1) if fill_rates else 0.0
        solution["tc_utilization"] = self._compute_tc_utilization(tc_outbound_desi, tc_outbound_vehicles)

        return solution
