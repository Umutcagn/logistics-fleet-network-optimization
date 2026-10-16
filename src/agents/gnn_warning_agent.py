class GNNWarningAgent:
    """
    Sub-Agent 3: GNN Darboğaz Uyarı Sub-Agent'ı
    Monitors graph network bottlenecks and recommends proactive route rerouting before queues stall.
    """
    def __init__(self, gnn_detector):
        self.gnn_detector = gnn_detector

    def evaluate_network_health(self, backlogs_wit, incoming_desi_dict, routes_list):
        gnn_res = self.gnn_detector.detect_bottlenecks(backlogs_wit, incoming_desi_dict, routes_list)
        high_risk = gnn_res["high_risk_alerts"]
        
        advice = []
        if len(high_risk) > 0:
            for alert in high_risk:
                advice.append(f"ÖNERİ: {alert['name']} kapasite doluluk riski taşıyor! Yük komşu merkezlere ALNS ile kaydırılmalı.")
        else:
            advice.append("Ağ genelinde darboğaz riski düşük, normal operasyon düzeni sürdürülebilir.")

        return {
            "high_risk_alerts_count": len(high_risk),
            "alerts": high_risk,
            "gnn_summary": gnn_res,
            "subagent_advice": advice
        }
