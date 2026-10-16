import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import json
from pathlib import Path

class GraphNeuralNetworkModel(nn.Module):
    """
    Message Passing Graph Neural Network for Bottleneck Detection.
    Nodes: Transfer Centers (features: [backlog Wit, incoming desi, processing ratio, ramp count])
    Edges: Direct logistics routes (features: [distance, allocated capacity])
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
        # Step 1: Transform node features
        h = self.node_mlp(node_features)  # (N, hidden_dim)
        
        # Step 2: Neighbor aggregation (Message Passing)
        N = h.size(0)
        risk_scores = []
        
        for i in range(N):
            # Find neighbors of node i
            neighbors_idx = (adj_matrix[i] > 0).nonzero(as_tuple=True)[0]
            if len(neighbors_idx) == 0:
                h_neigh = h[i]
            else:
                h_neigh = torch.mean(h[neighbors_idx], dim=0)
                
            combined = torch.cat([h[i], h_neigh], dim=0)
            risk = self.message_agg(combined)
            risk_scores.append(risk)
            
        return torch.stack(risk_scores).squeeze(-1)

class GNNBottleneckDetector:
    """
    Interpretable Network Bottleneck Early Warning System.
    Receives neighbor congestion signals ("sıkıştım") and predicts bottleneck risk scores.
    
    Reference: Zhang & Pan (2024) MARLHOSA network analysis principle
    """
    def __init__(self, network_config_path=None):
        if network_config_path is None:
            network_config_path = Path(__file__).parent.parent / "data" / "sample_network.json"
            
        with open(network_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
            
        self.transfer_centers = self.config["transfer_centers"]
        self.tc_ids = [tc["id"] for tc in self.transfer_centers]
        self.tc_idx_map = {tc_id: i for i, tc_id in enumerate(self.tc_ids)}
        
        self.model = GraphNeuralNetworkModel(node_in_dim=4, hidden_dim=16)

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
        N = len(self.tc_ids)
        node_features = []
        
        for tc in self.transfer_centers:
            tc_id = tc["id"]
            wit = backlogs_wit.get(tc_id, 0.0)
            incoming = incoming_desi_dict.get(tc_id, 0.0)
            max_cap = tc["hourly_handling_capacity"] * 12
            proc_ratio = min(2.0, (incoming + wit) / max(1.0, max_cap))
            ramps = float(tc["ramps"])
            
            # Normalize features
            node_features.append([
                wit / 5000.0,
                incoming / 20000.0,
                proc_ratio,
                ramps / 15.0
            ])
            
        x_tensor = torch.tensor(node_features, dtype=torch.float32)
        adj_tensor = self.build_adj_matrix(routes_list) if hasattr(self, 'build_adj_matrix') else self._build_adj_matrix(routes_list)

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
            "high_risk_alerts": alerts
        }
