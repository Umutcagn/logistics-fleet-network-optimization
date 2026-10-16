class MainCoordinatorAgent:
    """
    Main Coordinator Agent.
    Coordinates 3 sub-agents (Demand, Optimization, GNN Warning) and prepares consolidated recommendations
    for the logistics operations manager. Enforces Human-in-the-Loop approval for critical actions.
    """
    def __init__(self, demand_agent, opt_agent, gnn_agent):
        self.demand_agent = demand_agent
        self.opt_agent = opt_agent
        self.gnn_agent = gnn_agent
        self.pending_approvals = []

    def process_daily_pipeline(self, predictions_df, validation_wape, demand_dict, distance_matrix, spot_costs, backlogs_wit, incoming_desi_dict, routes_list):
        # 1. Demand Sub-Agent
        demand_info = self.demand_agent.analyze_forecast(predictions_df, validation_wape)
        
        # 2. Optimization Sub-Agent
        opt_info = self.opt_agent.run_optimization(demand_dict, distance_matrix, spot_costs, backlogs_wit)
        
        # 3. GNN Warning Sub-Agent
        gnn_info = self.gnn_agent.evaluate_network_health(backlogs_wit, incoming_desi_dict, routes_list)

        # Build Master Recommendation Report
        self.pending_approvals = []
        is_critical = False

        if opt_info["spot_vehicle_ratio_percent"] > 35.0:
            is_critical = True
            self.pending_approvals.append({
                "action_id": "ACT_SPOT_HIGH",
                "type": "CRITICAL_SPOT_USAGE",
                "description": f"Spot araç kullanımı %{opt_info['spot_vehicle_ratio_percent']} seviyesine ulaştı! Bütçe aşımı riski onaylanmalı.",
                "status": "WAITING_HUMAN_APPROVAL"
            })

        if gnn_info["high_risk_alerts_count"] > 0:
            is_critical = True
            for alert in gnn_info["alerts"]:
                self.pending_approvals.append({
                    "action_id": f"ACT_REROUTE_{alert['tc_id']}",
                    "type": "CRITICAL_REROUTE",
                    "description": f"{alert['name']} merkezindeki darboğaz sebebiyle ALNS dinamik rota kaydırması uygulanmalı.",
                    "status": "WAITING_HUMAN_APPROVAL"
                })

        return {
            "demand_summary": demand_info,
            "optimization_summary": opt_info,
            "gnn_summary": gnn_info,
            "requires_human_approval": is_critical,
            "pending_approvals": self.pending_approvals
        }

    def approve_action(self, action_id: str):
        for act in self.pending_approvals:
            if act["action_id"] == action_id:
                act["status"] = "APPROVED_BY_HUMAN"
                return True
        return False
