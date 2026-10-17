import pytest
from src.agents.demand_agent import DemandAgent
from src.agents.optimization_agent import OptimizationAgent
from src.agents.gnn_warning_agent import GNNWarningAgent
from src.agents.main_coordinator import MainCoordinatorAgent
from src.ml.meta_learner import StackingMetaLearner
from src.optimization.fsmvrp_ilp import FSMVRP_ILPSolver
from src.optimization.alns_solver import ALNSSolver
from src.gnn.bottleneck_gnn import GNNBottleneckDetector

def test_main_coordinator_flow():
    meta = StackingMetaLearner()
    ilp = FSMVRP_ILPSolver()
    alns = ALNSSolver(max_iterations=5)
    gnn = GNNBottleneckDetector()
    
    demand_subagent = DemandAgent(meta)
    opt_subagent = OptimizationAgent(ilp, alns)
    gnn_subagent = GNNWarningAgent(gnn)
    
    main_coordinator = MainCoordinatorAgent(demand_subagent, opt_subagent, gnn_subagent)
    
    import pandas as pd
    preds_df = pd.DataFrame({"ensemble_pred": [500.0, 1200.0]})
    demand_dict = {("TC_IST_AN", "TC_ANK"): 500.0, ("TC_IST_AV", "TC_IZM"): 1200.0}
    dist_matrix = {"TC_IST_AN": {"TC_ANK": 450.0}, "TC_IST_AV": {"TC_IZM": 480.0}}
    spot_costs = {("TC_IST_AN", "TC_ANK"): 3000.0, ("TC_IST_AV", "TC_IZM"): 3200.0}
    backlogs = {"TC_IST_AN": 0.0, "TC_ANK": 0.0, "TC_IST_AV": 0.0, "TC_IZM": 0.0}
    incoming = {"TC_ANK": 500.0, "TC_IZM": 1200.0}
    routes = [("TC_IST_AN", "TC_ANK"), ("TC_IST_AV", "TC_IZM")]
    
    report = main_coordinator.process_daily_pipeline(
        preds_df, 0.12, demand_dict, dist_matrix, spot_costs, backlogs, incoming, routes
    )
    
    assert "demand_summary" in report
    assert "optimization_summary" in report
    assert "gnn_summary" in report
    assert "requires_human_approval" in report


def test_target_day_demand_matches_route_forecasts():
    from run_pipeline import run_full_pipeline

    data = run_full_pipeline(num_days=30, verbose=False)
    fc = data["df_forecast"]
    target = fc[fc["date"] == fc["date"].max()]
    expected = {(r.origin_id, r.dest_id): r.ensemble_pred for r in target.itertuples()}
    gnn_incoming = data["gnn_inputs"]["incoming"]
    for dest_id, total in target.groupby("dest_id")["ensemble_pred"].sum().items():
        assert abs(gnn_incoming[dest_id] - total) < 1e-6
    assignments = data["opt_solution"]["assignments"]
    for (orig, dest), demand in expected.items():
        key = f"{orig}->{dest}"
        if key in assignments:
            assert abs(assignments[key]["demand_desi"] - demand) < 1.0
