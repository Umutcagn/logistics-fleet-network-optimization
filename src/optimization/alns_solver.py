import random
import copy
import json
import numpy as np
from pathlib import Path

class ALNSSolver:
    """
    Adaptive Large Neighborhood Search (ALNS) solver for large-scale route networks (>= 50 routes).
    Designed to escape local optima by iteratively applying Destroy and Repair operators.
    
    Reference: Ali et al. (2024) [Robust 2-Stage ALNS]
    """
    def __init__(self, network_config_path=None, max_iterations=200):
        if network_config_path is None:
            network_config_path = Path(__file__).parent.parent / "data" / "sample_network.json"
            
        with open(network_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.transfer_centers = {tc["id"]: tc for tc in self.config["transfer_centers"]}
        self.vehicle_types = self.config["vehicle_types"]
        self.spot_params = self.config["spot_vehicle_params"]
        self.max_iterations = max_iterations

    def _calculate_route_cost(self, i, j, demand, fleet_counts, spot_count, distance_matrix, spot_costs):
        dist = distance_matrix.get(i, {}).get(j, 100.0)
        cost = spot_count * spot_costs.get((i, j), 3000.0)
        
        for m, count in fleet_counts.items():
            v_spec = self.vehicle_types[m]
            cost += count * (v_spec["fixed_cost"] + dist * v_spec["cost_per_km"])
            
        return cost

    def initial_solution(self, demand_dict, distance_matrix, spot_costs):
        solution = {}
        total_cost = 0.0
        
        for (i, j), demand in demand_dict.items():
            # Initial greedy assignment
            dist = distance_matrix.get(i, {}).get(j, 100.0)
            spot_c = spot_costs.get((i, j), 3000.0)
            
            origin_tc = self.transfer_centers.get(i, {})
            dest_tc = self.transfer_centers.get(j, {})
            has_ramp = origin_tc.get("has_truck_ramp", True) and dest_tc.get("has_truck_ramp", True)
            
            # Select vehicle type
            best_type = "tir" if has_ramp else "kamyon"
            cap = self.vehicle_types[best_type]["capacity"]
            
            num_fleet = int(demand // cap)
            rem_demand = demand - (num_fleet * cap)
            
            num_spot = 0
            if rem_demand > 0:
                num_spot = 1
                
            fleet_counts = {best_type: num_fleet} if num_fleet > 0 else {}
            cost = self._calculate_route_cost(i, j, demand, fleet_counts, num_spot, distance_matrix, spot_costs)
            
            solution[(i, j)] = {
                "demand": demand,
                "fleet": fleet_counts,
                "spot": num_spot,
                "cost": cost
            }
            total_cost += cost

        return solution, total_cost

    def destroy_random_routes(self, solution, destroy_rate=0.25):
        """
        Destroy operator 1: Random route destruction.
        """
        destroyed = copy.deepcopy(solution)
        routes = list(destroyed.keys())
        num_to_destroy = max(1, int(len(routes) * destroy_rate))
        
        removed_routes = random.sample(routes, num_to_destroy)
        for r in removed_routes:
            destroyed[r] = None
            
        return destroyed, removed_routes

    def destroy_worst_cost_routes(self, solution, destroy_rate=0.20):
        """
        Destroy operator 2: Destroy routes with highest cost per desi.
        """
        destroyed = copy.deepcopy(solution)
        cost_per_desi = []
        for r, data in solution.items():
            cpd = data["cost"] / max(1.0, data["demand"])
            cost_per_desi.append((cpd, r))
            
        cost_per_desi.sort(reverse=True, key=lambda x: x[0])
        num_to_destroy = max(1, int(len(solution) * destroy_rate))
        
        removed_routes = [r for _, r in cost_per_desi[:num_to_destroy]]
        for r in removed_routes:
            destroyed[r] = None
            
        return destroyed, removed_routes

    def repair_greedy(self, partial_solution, removed_routes, demand_dict, distance_matrix, spot_costs):
        """
        Repair operator: Re-assign vehicles trying optimal vehicle combinations (Tır, Kamyon, Kamyonet).
        """
        repaired = copy.deepcopy(partial_solution)
        total_cost = 0.0
        
        for r, data in repaired.items():
            if data is not None:
                total_cost += data["cost"]

        for (i, j) in removed_routes:
            demand = demand_dict[(i, j)]
            dist = distance_matrix.get(i, {}).get(j, 100.0)
            spot_c = spot_costs.get((i, j), 3000.0)
            
            origin_tc = self.transfer_centers.get(i, {})
            dest_tc = self.transfer_centers.get(j, {})
            has_ramp = origin_tc.get("has_truck_ramp", True) and dest_tc.get("has_truck_ramp", True)
            
            best_cost = float('inf')
            best_fleet = {}
            best_spot = 0
            
            # Evaluate combinations
            possible_types = ["tir", "kamyon", "hafif_kamyon", "kamyonet"]
            if not has_ramp:
                possible_types.remove("tir")
                
            for v_type in possible_types:
                cap = self.vehicle_types[v_type]["capacity"]
                num_v = int(np.ceil(demand / cap))
                
                c = self._calculate_route_cost(i, j, demand, {v_type: num_v}, 0, distance_matrix, spot_costs)
                if c < best_cost:
                    best_cost = c
                    best_fleet = {v_type: num_v}
                    best_spot = 0
                    
            # Check spot combination
            num_spot = int(np.ceil(demand / self.spot_params["capacity"]))
            spot_only_cost = num_spot * spot_c
            if spot_only_cost < best_cost:
                best_cost = spot_only_cost
                best_fleet = {}
                best_spot = num_spot

            repaired[(i, j)] = {
                "demand": demand,
                "fleet": best_fleet,
                "spot": best_spot,
                "cost": best_cost
            }
            total_cost += best_cost

        return repaired, total_cost

    def solve(self, demand_dict, distance_matrix, spot_costs):
        """
        Executes ALNS optimization loop.
        """
        current_sol, current_cost = self.initial_solution(demand_dict, distance_matrix, spot_costs)
        best_sol, best_cost = current_sol, current_cost
        
        temperature = current_cost * 0.1
        cooling_rate = 0.95
        
        for iteration in range(self.max_iterations):
            # Select destroy operator
            if random.random() < 0.5:
                destroyed, removed = self.destroy_random_routes(current_sol)
            else:
                destroyed, removed = self.destroy_worst_cost_routes(current_sol)
                
            # Repair
            new_sol, new_cost = self.repair_greedy(destroyed, removed, demand_dict, distance_matrix, spot_costs)
            
            # Acceptance criterion (Simulated Annealing)
            delta = new_cost - current_cost
            if delta < 0 or random.random() < np.exp(-delta / max(1e-5, temperature)):
                current_sol, current_cost = new_sol, new_cost
                if current_cost < best_cost:
                    best_sol, best_cost = copy.deepcopy(current_sol), current_cost
                    
            temperature *= cooling_rate

        # Format output
        assignments = {}
        total_fleet = 0
        total_spot = 0
        fill_rates = []
        tc_outbound_desi = {tc_id: 0.0 for tc_id in self.transfer_centers}
        tc_outbound_vehicles = {tc_id: 0 for tc_id in self.transfer_centers}
        
        for (i, j), data in best_sol.items():
            r_key = f"{i}->{j}"
            dist = float(distance_matrix.get(i, {}).get(j, 100.0))
            assigned_cap = data["spot"] * self.spot_params["capacity"]
            for m, cnt in data["fleet"].items():
                assigned_cap += cnt * self.vehicle_types[m]["capacity"]
                
            fill_rate = min(100.0, (float(data["demand"]) / assigned_cap * 100.0)) if assigned_cap > 0 else 0.0
            fill_rates.append(fill_rate)

            if i in tc_outbound_desi:
                tc_outbound_desi[i] += float(data["demand"])
                tc_outbound_vehicles[i] += data["spot"] + sum(data["fleet"].values())

            assignments[r_key] = {
                "demand_desi": round(float(data["demand"]), 1),
                "distance_km": round(dist, 1),
                "fleet_vehicles": data["fleet"],
                "spot_vehicles": data["spot"],
                "assigned_capacity_desi": assigned_cap,
                "fill_rate_pct": round(fill_rate, 1),
                "route_cost_tl": round(float(data["cost"]), 2)
            }
            total_spot += data["spot"]
            total_fleet += sum(data["fleet"].values())

        delta_t_hours = 8.0
        tc_stats = {}
        for tc_id, tc_info in self.transfer_centers.items():
            out_desi = tc_outbound_desi.get(tc_id, 0.0)
            out_veh = tc_outbound_vehicles.get(tc_id, 0)
            max_handling = tc_info["hourly_handling_capacity"] * delta_t_hours
            handling_pct = (out_desi / max_handling * 100.0) if max_handling > 0 else 0.0
            tc_stats[tc_id] = {
                "name": tc_info["name"],
                "ramps_R_i": tc_info["ramps"],
                "has_truck_ramp": tc_info.get("has_truck_ramp", True),
                "dispatched_vehicles": out_veh,
                "outbound_desi": round(out_desi, 1),
                "shift_handling_cap_K4": round(max_handling, 0),
                "k4_utilization_pct": round(handling_pct, 1),
                "k4_status": "AŞIM RİSKİ" if handling_pct > 90.0 else ("YOĞUN" if handling_pct > 65.0 else "UYGUN")
            }

        return {
            "status": "ALNS_OPTIMAL",
            "total_cost": round(best_cost, 2),
            "assignments": assignments,
            "total_fleet_vehicles": total_fleet,
            "total_spot_vehicles": total_spot,
            "avg_fill_rate_pct": round(sum(fill_rates) / len(fill_rates), 1) if fill_rates else 0.0,
            "tc_utilization": tc_stats
        }
