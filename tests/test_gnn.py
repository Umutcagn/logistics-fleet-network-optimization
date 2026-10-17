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


def test_gnn_scores_are_reproducible():
    args = ({"TC_IST_AN": 8000.0}, {"TC_IST_AN": 15000.0}, [("TC_IST_AN", "TC_ANK")])
    first = GNNBottleneckDetector().detect_bottlenecks(*args)["node_risk_scores"]
    second = GNNBottleneckDetector().detect_bottlenecks(*args)["node_risk_scores"]
    assert first == second


def _capacity(detector, tc_id):
    tc = next(t for t in detector.transfer_centers if t["id"] == tc_id)
    return tc["hourly_handling_capacity"] * 12


def test_trained_weights_are_loaded():
    assert GNNBottleneckDetector().is_trained


def test_overloaded_center_is_critical_and_light_center_is_normal():
    detector = GNNBottleneckDetector()
    incoming = {"TC_BUR": _capacity(detector, "TC_BUR") * 1.5, "TC_ANK": _capacity(detector, "TC_ANK") * 0.3}
    scores = detector.detect_bottlenecks({}, incoming, [("TC_BUR", "TC_ANK")])["node_risk_scores"]
    assert scores["TC_BUR"]["status"] == "CRITICAL_BOTTLENECK"
    assert scores["TC_ANK"]["status"] == "NORMAL"


def test_neighbor_overflow_raises_risk():
    detector = GNNBottleneckDetector()
    routes = [("TC_BUR", "TC_IST_AN")]
    incoming = {"TC_BUR": _capacity(detector, "TC_BUR") * 0.85}
    alone = detector.detect_bottlenecks({}, incoming, routes)["node_risk_scores"]["TC_BUR"]["risk_score"]
    incoming["TC_IST_AN"] = _capacity(detector, "TC_IST_AN") * 2.0
    with_overflow = detector.detect_bottlenecks({}, incoming, routes)["node_risk_scores"]["TC_BUR"]["risk_score"]
    assert alone < 0.45 <= 0.70 <= with_overflow


def test_label_rule_counts_neighbor_overflow():
    import numpy as np
    from src.gnn.train_gnn import label_bottlenecks

    load = np.array([0.8, 1.6, 0.5])
    adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32)
    # node 0: 0.8 + 0.5 * 0.6 = 1.1 -> bottleneck; node 2 is isolated and light
    assert label_bottlenecks(load, adj).tolist() == [1.0, 1.0, 0.0]
