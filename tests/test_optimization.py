import pytest
from src.data.synthetic_data_generator import LogisticsDataGenerator
from src.optimization.fsmvrp_ilp import FSMVRP_ILPSolver
from src.optimization.alns_solver import ALNSSolver
from src.optimization.hub_consolidation import HubConsolidator

def test_ilp_solver_small_network():
    data_gen = LogisticsDataGenerator()
    ilp_solver = FSMVRP_ILPSolver()
    
    demand_dict = {
        ("TC_IST_AN", "TC_ANK"): 2400.0,
        ("TC_IST_AV", "TC_IZM"): 1200.0
    }
    spot_costs = {
        ("TC_IST_AN", "TC_ANK"): 4000.0,
        ("TC_IST_AV", "TC_IZM"): 3500.0
    }
    
    sol = ilp_solver.solve(demand_dict, data_gen.distance_matrix, spot_costs)
    assert sol["status"] in ["OPTIMAL", "FEASIBLE", "GREEDY_FALLBACK"]
    assert sol["total_cost"] > 0
    assert sol["total_fleet_vehicles"] + sol["total_spot_vehicles"] >= 2

def test_alns_solver_large_network():
    data_gen = LogisticsDataGenerator()
    alns_solver = ALNSSolver(max_iterations=20)
    
    # Generate 10 routes
    df_routes, _ = data_gen.generate_time_series_data(num_days=2)
    single_day = df_routes[df_routes["day_idx"] == 0]
    
    demand_dict = {(r["origin_id"], r["dest_id"]): r["actual_desi"] for _, r in single_day.iterrows()}
    spot_costs = {(r["origin_id"], r["dest_id"]): r["spot_cost_s_ij"] for _, r in single_day.iterrows()}
    
    sol = alns_solver.solve(demand_dict, data_gen.distance_matrix, spot_costs)
    assert sol["status"] == "ALNS_OPTIMAL"
    assert sol["total_cost"] > 0

def test_hub_consolidation():
    data_gen = LogisticsDataGenerator()
    consolidator = HubConsolidator(data_gen.vehicle_types)
    
    demand_dict = {
        ("TC_IST_AN", "TC_ANK"): 1100.0,
        ("TC_IST_AN", "TC_IZM"): 900.0
    }
    
    res = consolidator.evaluate_consolidation(demand_dict, data_gen.distance_matrix)
    assert "active_consolidations_count" in res
    assert "total_estimated_savings_tl" in res
