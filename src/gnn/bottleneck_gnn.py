import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

WEIGHTS_PATH = Path(__file__).parent / "bottleneck_gnn_weights.pt"
SHIFT_HOURS = 12


def build_node_features(transfer_centers, backlogs_wit: dict, incoming_desi_dict: dict) -> np.ndarray:
    """Normalized node features shared by training and inference: [backlog, incoming, load ratio, ramps]."""
    features = []
    for tc in transfer_centers:
        wit = backlogs_wit.get(tc["id"], 0.0)
        incoming = incoming_desi_dict.get(tc["id"], 0.0)
        max_cap = tc["hourly_handling_capacity"] * SHIFT_HOURS
        proc_ratio = min(2.0, (incoming + wit) / max(1.0, max_cap))
        features.append([wit / 5000.0, incoming / 20000.0, proc_ratio, float(tc["ramps"]) / 15.0])
    return np.asarray(features, dtype=np.float32)


class GraphNeuralNetworkModel(nn.Module):
    """
    Message Passing Graph Neural Network for Bottleneck Detection.
    Nodes: Transfer Centers (features: [backlog Wit, incoming desi, processing ratio, ramp count])
    Edges: Direct logistics routes (unweighted, undirected adjacency)
    """
    def __init__(self, node_in_dim=4, hidden_dim=16):
        super(GraphNeuralNetworkModel, self).__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.message_agg = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features, adj_matrix):
        """Accepts a single graph (N, F) / (N, N) or a batch (B, N, F) / (B, N, N)."""
        # Step 1: Transform node features
        h = self.node_mlp(node_features)  # (..., N, hidden_dim)

        # Step 2: Mean neighbor aggregation (Message Passing); isolated nodes use their own embedding
        adj = (adj_matrix > 0).float()
        deg = adj.sum(dim=-1, keepdim=True)
        h_neigh = torch.where(deg > 0, (adj @ h) / deg.clamp(min=1.0), h)

        combined = torch.cat([h, h_neigh], dim=-1)
        return self.message_agg(combined).squeeze(-1)

class GNNBottleneckDetector:
    """
    Interpretable Network Bottleneck Early Warning System.
    Receives neighbor congestion signals ("sıkıştım") and predicts bottleneck risk scores.
    
    Reference: Zhang & Pan (2024) MARLHOSA network analysis principle
    """
    def __init__(self, network_config_path=None, seed=42, weights_path=WEIGHTS_PATH):
        if network_config_path is None:
            network_config_path = Path(__file__).parent.parent / "data" / "sample_network.json"
            
        with open(network_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.transfer_centers = self.config["transfer_centers"]
        self.tc_ids = [tc["id"] for tc in self.transfer_centers]
        self.tc_idx_map = {tc_id: i for i, tc_id in enumerate(self.tc_ids)}
        
        torch.manual_seed(seed)
        self.model = GraphNeuralNetworkModel(node_in_dim=4, hidden_dim=16)
        self.is_trained = weights_path is not None and Path(weights_path).exists()
        if self.is_trained:
            self.model.load_state_dict(torch.load(weights_path, weights_only=True))
        else:
            warnings.warn("GNN weights not found; scores come from an untrained model. Run `python -m src.gnn.train_gnn`.")

    def _build_adj_matrix(self, routes_list):
        N = len(self.tc_ids)
        adj = torch.zeros((N, N), dtype=torch.float32)
        for (u, v) in routes_list:
            if u in self.tc_idx_map and v in self.tc_idx_map:
                i, j = self.tc_idx_map[u], self.tc_idx_map[v]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        return adj

    def detect_bottlenecks(self, backlogs_wit: dict, incoming_desi_dict: dict, routes_list: list):
        """
        Evaluates risk score [0.0, 1.0] for every Transfer Center node.
        """
        x_tensor = torch.from_numpy(build_node_features(self.transfer_centers, backlogs_wit, incoming_desi_dict))
        adj_tensor = self._build_adj_matrix(routes_list)

        self.model.eval()
        with torch.no_grad():
            risk_scores = self.model(x_tensor, adj_tensor).numpy()

        results = {}
        alerts = []
        for i, tc_id in enumerate(self.tc_ids):
            score = round(float(risk_scores[i]), 3)
            tc_info = self.transfer_centers[i]
            
            status = "NORMAL"
            if score >= 0.70:
                status = "CRITICAL_BOTTLENECK"
                alerts.append({
                    "tc_id": tc_id,
                    "name": tc_info["name"],
                    "risk_score": score,
                    "backlog_wit": backlogs_wit.get(tc_id, 0.0),
                    "warning": f"Transfer center {tc_info['name']} has severe congestion risk! ({score * 100:.1f}%)"
                })
            elif score >= 0.45:
                status = "WARNING_CONGESTION"
                
            results[tc_id] = {
                "name": tc_info["name"],
                "risk_score": score,
                "status": status,
                "backlog_wit": backlogs_wit.get(tc_id, 0.0),
                "ramps": tc_info["ramps"]
            }

        return {
            "node_risk_scores": results,
            "high_risk_alerts": alerts,
            "model_trained": self.is_trained
        }
