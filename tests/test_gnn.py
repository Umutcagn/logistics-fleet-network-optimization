import pytest
from src.gnn.bottleneck_gnn import GNNBottleneckDetector

def test_gnn_bottleneck_detection():
    detector = GNNBottleneckDetector()
    backlogs = {"TC_IST_AN": 8000.0, "TC_ANK": 100.0}
    incoming = {"TC_IST_AN": 15000.0, "TC_ANK": 2000.0}
    routes = [("TC_IST_AN", "TC_ANK"), ("TC_IST_AV", "TC_IST_AN")]
    
    res = detector.detect_bottlenecks(backlogs, incoming, routes)
    assert "node_risk_scores" in res
    assert "high_risk_alerts" in res
    assert len(res["node_risk_scores"]) == 10
