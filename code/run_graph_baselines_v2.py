#!/usr/bin/env python3
"""Matched inductive graph baselines for the X-THGNN Version 2 protocol.

The four reference models target the primary entity/node outcome only:

* GCN: homogeneous graph convolution.
* GAT: homogeneous graph attention.
* Temporal-GAT: attention with the standardized temporal/edge vector.
* R-GCN: relation-specific graph convolution.

All models use the same leakage-controlled builder, chronological partitions,
training-only preprocessing, validation-loss early stopping, validation-only
F1 threshold selection, seeds, width, and maximum epoch budget as X-THGNN.
They are parameter-matched reference implementations, not reproductions of a
specific external paper or repository.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, RGCNConv, TransformerConv


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "X_THGNN_Reproduction_v3.py"
SPEC = importlib.util.spec_from_file_location("xthgnn_core_v2", CORE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    positives = labels.sum()
    negatives = labels.numel() - positives
    if positives.item() == 0 or negatives.item() == 0:
        return F.binary_cross_entropy_with_logits(logits, labels)
    return F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=(negatives / positives.clamp_min(1.0)).detach()
    )


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, **_: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        x = F.relu(self.conv1(graph.node_features, graph.edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, graph.edge_index))
        return self.out(x).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, **_: int):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=False, dropout=0.2)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.2)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        x = F.elu(self.conv1(graph.node_features, graph.edge_index))
        x = F.elu(self.conv2(x, graph.edge_index))
        return self.out(x).squeeze(-1)


class TemporalGAT(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, edge_dim: int, **_: int):
        super().__init__()
        self.conv1 = TransformerConv(
            input_dim, hidden_dim, heads=2, concat=False, dropout=0.2,
            edge_dim=edge_dim, beta=True,
        )
        self.conv2 = TransformerConv(
            hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.2,
            edge_dim=edge_dim, beta=True,
        )
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        edge_attr = graph.edge_features
        x = F.elu(self.conv1(graph.node_features, graph.edge_index, edge_attr))
        x = F.elu(self.conv2(x, graph.edge_index, edge_attr))
        return self.out(x).squeeze(-1)


class RGCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_relations: int, **_: int):
        super().__init__()
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, graph):
        x = F.relu(self.conv1(graph.node_features, graph.edge_index, graph.relation_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.conv2(x, graph.edge_index, graph.relation_index))
        return self.out(x).squeeze(-1)


MODELS = {
    "gcn": GCN,
    "gat": GAT,
    "temporal_gat": TemporalGAT,
    "rgcn": RGCN,
}


def load_protocol(path: Path) -> tuple[dict, list[int]]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload["config"], [int(value) for value in payload["training_seeds"]]


def metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    return CORE.binary_metrics(labels, probabilities, threshold)


def run_one(events: Path, protocol_path: Path, output: Path, model_name: str, seed: int) -> Path:
    config, _ = load_protocol(protocol_path)
    set_seed(seed)
    frame = CORE.EventGraphBuilder.load(events)
    data_cfg = config["data"]
    builder = CORE.EventGraphBuilder(
        train_fraction=float(data_cfg["train_fraction"]),
        validation_fraction=float(data_cfg["validation_fraction"]),
        feature_prefix=str(data_cfg["feature_prefix"]),
        history_windows_seconds=tuple(int(v) for v in data_cfg["history_windows_seconds"]),
    )
    train_frame, val_frame, test_frame = builder.split(frame)
    builder.fit(train_frame)
    builder.fit_scaler(train_frame)
    train_graph = builder.build(train_frame).to_torch("cpu")
    val_graph = builder.build(val_frame).to_torch("cpu")
    test_graph = builder.build(test_frame).to_torch("cpu")

    model_cfg = config["model"]
    model = MODELS[model_name](
        input_dim=int(train_graph.node_features.shape[1]),
        hidden_dim=int(model_cfg["embedding_dim"]),
        edge_dim=int(train_graph.edge_features.shape[1]),
        num_relations=len(builder.relation_map),
    )
    opt_cfg = config["optimization"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
    )
    max_epochs = int(opt_cfg["epochs"])
    patience = int(opt_cfg["early_stopping_patience"])
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    stale = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = model(train_graph)[train_graph.node_present]
        train_labels = train_graph.node_labels[train_graph.node_present]
        train_loss = balanced_bce(train_logits, train_labels)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["gradient_clip"]))
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_graph)[val_graph.node_present]
            val_labels = val_graph.node_labels[val_graph.node_present]
            val_loss = balanced_bce(val_logits, val_labels)
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss.detach()),
            "validation_loss": float(val_loss.detach()),
        })
        if float(val_loss) < best_loss - 1e-7:
            best_loss = float(val_loss)
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No graph-baseline state retained.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probability = torch.sigmoid(model(val_graph)[val_graph.node_present]).cpu().numpy()
        test_probability = torch.sigmoid(model(test_graph)[test_graph.node_present]).cpu().numpy()
    val_labels_np = val_graph.node_labels[val_graph.node_present].cpu().numpy().astype(int)
    test_labels_np = test_graph.node_labels[test_graph.node_present].cpu().numpy().astype(int)
    threshold = CORE.select_f1_threshold(val_labels_np, val_probability)
    result = metrics(test_labels_np, test_probability, threshold)
    result.update({
        "model": model_name,
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "epochs_executed": len(history),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "primary_outcome": "entity/node detection on source-defined labels",
        "implementation_scope": "parameter-matched PyTorch Geometric reference implementation",
    })
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    pd.DataFrame({
        "node_id": np.asarray(test_graph.node_ids)[test_graph.node_present.cpu().numpy()],
        "label": test_labels_np,
        "probability": test_probability,
        "prediction": (test_probability >= threshold).astype(int),
    }).to_csv(output / "node_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps({
        "events": str(events),
        "protocol": str(protocol_path),
        "model": model_name,
        "seed": int(seed),
        "torch": torch.__version__,
        "torch_geometric": __import__("torch_geometric").__version__,
        "split": [float(data_cfg["train_fraction"]), float(data_cfg["validation_fraction"]),
                  1.0 - float(data_cfg["train_fraction"]) - float(data_cfg["validation_fraction"])],
    }, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(run_one(args.events, args.protocol, args.output, args.model, args.seed))
