"""
Trains the bottleneck GNN on synthetic network scenarios.

There is no real labeled bottleneck data, so labels come from an explicit rule:
a transfer center is a bottleneck when its own load ratio plus the overflow it
receives from overloaded neighbors reaches its 12-hour handling capacity:

    load_i      = min(2, (backlog_i + incoming_i) / (hourly_capacity_i * 12))
    overflow_i  = mean over neighbors j of max(0, load_j - 1)
    bottleneck  = load_i + SPILLOVER_SHARE * overflow_i >= 1

The model therefore learns this rule, including the neighbor (graph) term that a
node-only threshold cannot see. It is evaluated against that node-only baseline.

Usage: python -m src.gnn.train_gnn
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.gnn.bottleneck_gnn import (
    SHIFT_HOURS,
    WEIGHTS_PATH,
    GraphNeuralNetworkModel,
    build_node_features,
)

SPILLOVER_SHARE = 0.5
METRICS_PATH = Path(__file__).parent / "training_metrics.json"
NETWORK_PATH = Path(__file__).parent.parent / "data" / "sample_network.json"


def label_bottlenecks(load: np.ndarray, adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(axis=-1)
    neighbor_overflow = (adj @ np.maximum(0.0, load - 1.0)) / np.maximum(deg, 1.0)
    return (load + SPILLOVER_SHARE * neighbor_overflow >= 1.0).astype(np.float32)


def generate_scenarios(transfer_centers, n_scenarios: int, rng: np.random.Generator):
    """Random load levels and random route graphs over the real transfer center list."""
    n = len(transfer_centers)
    caps = np.array([tc["hourly_handling_capacity"] * SHIFT_HOURS for tc in transfer_centers], dtype=np.float64)
    xs, adjs, ys, loads = [], [], [], []

    for _ in range(n_scenarios):
        network_level = rng.uniform(0.1, 1.4)
        load = network_level * rng.lognormal(0.0, 0.35, size=n)
        backlog_share = np.where(rng.random(n) < 0.5, rng.uniform(0.0, 0.5, size=n), 0.0)
        total = load * caps
        backlogs = {tc["id"]: total[i] * backlog_share[i] for i, tc in enumerate(transfer_centers)}
        incoming = {tc["id"]: total[i] * (1 - backlog_share[i]) for i, tc in enumerate(transfer_centers)}

        density = 1.0 if rng.random() < 0.2 else rng.uniform(0.15, 1.0)
        upper = np.triu(rng.random((n, n)) < density, k=1)
        adj = (upper | upper.T).astype(np.float32)

        clipped_load = np.minimum(2.0, load)
        xs.append(build_node_features(transfer_centers, backlogs, incoming))
        adjs.append(adj)
        ys.append(label_bottlenecks(clipped_load, adj))
        loads.append(clipped_load)

    return (torch.tensor(np.stack(xs)), torch.tensor(np.stack(adjs)),
            torch.tensor(np.stack(ys)), np.stack(loads))


def classification_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    return {
        "accuracy": round(float((pred == y).mean()), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
    }


def train(seed: int = 42, epochs: int = 400, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    with open(NETWORK_PATH, "r", encoding="utf-8") as f:
        transfer_centers = json.load(f)["transfer_centers"]

    x_tr, a_tr, y_tr, _ = generate_scenarios(transfer_centers, 6000, rng)
    x_va, a_va, y_va, _ = generate_scenarios(transfer_centers, 1500, rng)
    x_te, a_te, y_te, load_te = generate_scenarios(transfer_centers, 2000, rng)

    model = GraphNeuralNetworkModel(node_in_dim=4, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCELoss()
    best_val, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x_tr, a_tr), y_tr)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_va, a_va), y_va).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and epoch % 50 == 0:
            print(f"epoch {epoch:3d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prob_te = model(x_te, a_te).numpy()

    y = y_te.numpy()
    gnn_pred = (prob_te >= 0.5).astype(np.float32)
    baseline_pred = (load_te >= 1.0).astype(np.float32)
    # Bottlenecks caused only by neighbor overflow: invisible to a node-only threshold
    spillover = (load_te < 1.0) & (y == 1)

    metrics = {
        "label_rule": "load_i + 0.5 * mean_neighbors(max(0, load_j - 1)) >= 1",
        "train_scenarios": 6000, "val_scenarios": 1500, "test_scenarios": 2000,
        "test_nodes": int(y.size),
        "test_positive_rate": round(float(y.mean()), 4),
        "spillover_only_positives": int(spillover.sum()),
        "best_val_bce": round(best_val, 4),
        "gnn": classification_metrics(gnn_pred, y),
        "baseline_node_only_threshold": classification_metrics(baseline_pred, y),
        "spillover_recall": {
            "gnn": round(float(gnn_pred[spillover].mean()), 4) if spillover.any() else None,
            "baseline_node_only_threshold": 0.0,
        },
    }

    torch.save(model.state_dict(), WEIGHTS_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if verbose:
        print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
