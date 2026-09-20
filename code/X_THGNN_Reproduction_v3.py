#!/usr/bin/env python3
"""Standalone execution and audit code for the X-THGNN manuscript.

Experimental evidence is generated only from normalized event files and
completed run directories. The program contains no hard-coded manuscript result
tables and never invents missing study data. Label-source audits verify file
integrity and identifier syntax; they do not convert heuristic source labels
into independently adjudicated attack ground truth.

Install dependencies:
    pip install numpy pandas matplotlib scipy scikit-learn pyyaml torch dgl

Examples:
    python X_THGNN_Reproduction.py datasets
    python X_THGNN_Reproduction.py normalize-lanl --auth auth.txt.gz --redteam redteam.txt.gz --output data/lanl.csv
    python X_THGNN_Reproduction.py audit-ground-truth --cadets cadets.txt --theia theia.txt --trace trace.txt --output results/ground_truth_audit.json
    python X_THGNN_Reproduction.py normalize-tc --dataset darpa_cadets --inputs cadets*.json.gz --labels cadets_labels.csv --output data/cadets.csv
    python X_THGNN_Reproduction.py normalize-tc-ground-truth --dataset darpa_cadets --inputs cadets*.json.gz --ground-truth cadets.txt --output data/cadets.csv
    python X_THGNN_Reproduction.py normalize-magic --dataset darpa_cadets --graph test0.pkl --metadata metadata.json --ground-truth cadets.txt --output data/cadets_extract.csv
    python X_THGNN_Reproduction.py normalize-magic --dataset streamspot --graph graphs.pkl --module-root MAGIC --output data/streamspot_extract.csv
    python X_THGNN_Reproduction.py validate --events events.csv
    python X_THGNN_Reproduction.py train --events events.csv --output results/run1 --seed 42
    python X_THGNN_Reproduction.py baselines --events events.csv --output results/baselines --seeds 11 22 33 44 55
    python X_THGNN_Reproduction.py summarize-runs --runs 'results/*/seed_*' --output results/summary
    python X_THGNN_Reproduction.py make-figures --seed-metrics results/summary/seed_metrics.csv --output results/figures
    python X_THGNN_Reproduction.py manuscript-config --output manuscript_config.yaml
    python X_THGNN_Reproduction.py audit-baselines --registry baseline_registry.yaml --output results/baseline_audit
    python X_THGNN_Reproduction.py benchmark --events events.csv --checkpoint results/run1/checkpoint.pt --output results/scaling --edge-counts 10000 100000
    python X_THGNN_Reproduction.py self-test --output results/self_test
    python X_THGNN_Reproduction.py pilot --pilot-csv pilot.csv --output results/pilot
"""

from __future__ import annotations

import argparse

# ===== data.py =====
"""Dataset validation and temporal heterogeneous graph construction."""


import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


REQUIRED_EVENT_COLUMNS = {
    "timestamp", "src_id", "dst_id", "src_type", "dst_type", "relation",
    "edge_label", "node_label", "campaign_id", "campaign_label",
}

PUBLIC_DATASETS = {
    "darpa_cadets": {
        "family": "darpa_tc_cdm",
        "role": "development",
        "source": "DARPA Transparent Computing public release",
        "label_source": "ThreaTrace-expanded anomaly-entity UUID labels used by MAGIC",
    },
    "darpa_theia": {
        "family": "darpa_tc_cdm",
        "role": "external_validation",
        "source": "DARPA Transparent Computing public release",
        "label_source": "ThreaTrace-expanded anomaly-entity UUID labels used by MAGIC",
    },
    "darpa_trace": {
        "family": "darpa_tc_cdm",
        "role": "external_validation",
        "source": "DARPA Transparent Computing public release",
        "label_source": "ThreaTrace-expanded anomaly-entity UUID labels used by MAGIC",
    },
    "streamspot": {
        "family": "typed_graph_stream",
        "role": "external_validation",
        "source": "StreamSpot public typed-graph release",
        "label_source": "source scenario-graph binary label",
    },
    "lanl": {
        "family": "lanl_auth",
        "role": "optional_full_scale_extension",
        "source": "LANL Comprehensive Multi-Source Cyber-Security Events",
    },
}


@dataclass
class DatasetManifest:
    source: str
    sha256: str
    rows: int
    nodes: int
    relations: int
    node_types: int
    campaigns: int
    malicious_edges: int
    malicious_node_events: int
    malicious_campaign_events: int
    train_rows: int
    validation_rows: int
    test_rows: int
    train_time_max: float
    validation_time_max: float


@dataclass
class GraphArrays:
    node_features: np.ndarray
    edge_features: np.ndarray
    edge_index: np.ndarray
    relation_index: np.ndarray
    timestamps: np.ndarray
    node_labels: np.ndarray
    node_present: np.ndarray
    edge_labels: np.ndarray
    campaign_index: np.ndarray
    campaign_labels: np.ndarray
    node_ids: list[str]
    edge_event_rows: np.ndarray
    campaign_ids: list[str]

    def to_torch(self, device: str = "cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Training requires PyTorch. Install requirements.txt.") from exc
        return TorchGraph(
            node_features=torch.as_tensor(self.node_features, dtype=torch.float32, device=device),
            edge_features=torch.as_tensor(self.edge_features, dtype=torch.float32, device=device),
            edge_index=torch.as_tensor(self.edge_index, dtype=torch.long, device=device),
            relation_index=torch.as_tensor(self.relation_index, dtype=torch.long, device=device),
            timestamps=torch.as_tensor(self.timestamps, dtype=torch.float64, device=device),
            node_labels=torch.as_tensor(self.node_labels, dtype=torch.float32, device=device),
            node_present=torch.as_tensor(self.node_present, dtype=torch.bool, device=device),
            edge_labels=torch.as_tensor(self.edge_labels, dtype=torch.float32, device=device),
            campaign_index=torch.as_tensor(self.campaign_index, dtype=torch.long, device=device),
            campaign_labels=torch.as_tensor(self.campaign_labels, dtype=torch.float32, device=device),
            node_ids=self.node_ids,
            edge_event_rows=self.edge_event_rows,
            campaign_ids=self.campaign_ids,
        )


@dataclass
class TorchGraph:
    node_features: object
    edge_features: object
    edge_index: object
    relation_index: object
    timestamps: object
    node_labels: object
    node_present: object
    edge_labels: object
    campaign_index: object
    campaign_labels: object
    node_ids: list[str]
    edge_event_rows: np.ndarray
    campaign_ids: list[str]

    def with_edge_mask(self, mask):
        fields = {
            "edge_features": self.edge_features[mask],
            "edge_index": self.edge_index[:, mask],
            "relation_index": self.relation_index[mask],
            "timestamps": self.timestamps[mask],
            "edge_labels": self.edge_labels[mask],
            "campaign_index": self.campaign_index[mask],
            "edge_event_rows": self.edge_event_rows[np.asarray(mask.detach().cpu())],
        }
        return TorchGraph(
            node_features=self.node_features,
            node_labels=self.node_labels,
            node_present=self.node_present,
            campaign_labels=self.campaign_labels,
            node_ids=self.node_ids,
            campaign_ids=self.campaign_ids,
            **fields,
        )


class EventGraphBuilder:
    """Leakage-controlled transformer from chronological events to graph arrays.

    Relation/type vocabularies and numeric scalers are fitted on the training
    partition only. Node and campaign indices are deliberately local to each
    split because X-THGNN does not learn identity embeddings; this permits
    genuinely unseen validation/test entities without exposing future IDs at
    model-development time.
    """

    UNKNOWN_TOKEN = "__UNK__"

    def __init__(
        self,
        train_fraction: float = 0.8,
        validation_fraction: float = 0.1,
        feature_prefix: str = "feat_",
        history_windows_seconds: tuple[int, ...] = (300, 900),
    ):
        if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
            raise ValueError("Chronological fractions must be positive and sum to less than one.")
        self.train_fraction = train_fraction
        self.validation_fraction = validation_fraction
        self.feature_prefix = feature_prefix
        self.history_windows_seconds = tuple(int(value) for value in history_windows_seconds)
        if not self.history_windows_seconds or any(value <= 0 for value in self.history_windows_seconds):
            raise ValueError("History windows must contain positive durations in seconds.")
        self.node_map: Dict[str, int] = {}
        self.type_map: Dict[str, int] = {}
        self.relation_map: Dict[str, int] = {}
        self.campaign_map: Dict[str, int] = {}
        self.feature_columns: list[str] = []
        self.scaler = StandardScaler()
        self._scaler_fitted = False
        self.time_origin: float | None = None
        self.time_scale: float | None = None

    @staticmethod
    def load(path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        else:
            raise ValueError("Events must be CSV or Parquet.")
        return validate_events(frame)

    def fit(self, frame: pd.DataFrame) -> None:
        """Fit all learned preprocessing state on the training partition only."""
        nodes = sorted(set(frame["src_id"].astype(str)) | set(frame["dst_id"].astype(str)))
        types = sorted(set(frame["src_type"].astype(str)) | set(frame["dst_type"].astype(str)))
        relations = sorted(frame["relation"].astype(str).unique())
        campaigns = sorted(frame["campaign_id"].astype(str).unique())
        self.node_map = {value: idx for idx, value in enumerate(nodes)}
        self.type_map = {self.UNKNOWN_TOKEN: 0, **{value: idx + 1 for idx, value in enumerate(types)}}
        self.relation_map = {self.UNKNOWN_TOKEN: 0, **{value: idx + 1 for idx, value in enumerate(relations)}}
        self.campaign_map = {value: idx for idx, value in enumerate(campaigns)}
        self.feature_columns = sorted(c for c in frame.columns if c.startswith(self.feature_prefix))

    def split(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ordered = frame.sort_values("timestamp", kind="mergesort").reset_index(drop=False).rename(columns={"index": "event_row"})
        n = len(ordered)
        train_end = max(1, int(n * self.train_fraction))
        val_end = min(n - 1, train_end + max(1, int(n * self.validation_fraction)))
        # Never place equal-timestamp events on opposite sides of a boundary.
        while train_end < n - 2 and ordered.loc[train_end - 1, "timestamp"] == ordered.loc[train_end, "timestamp"]:
            train_end += 1
        val_end = max(val_end, train_end + 1)
        while val_end < n - 1 and ordered.loc[val_end - 1, "timestamp"] == ordered.loc[val_end, "timestamp"]:
            val_end += 1
        if not (0 < train_end < val_end < n):
            raise ValueError("Unable to create three non-empty chronological splits without dividing timestamp ties.")
        if (
            ordered.loc[train_end - 1, "timestamp"] == ordered.loc[train_end, "timestamp"]
            or ordered.loc[val_end - 1, "timestamp"] == ordered.loc[val_end, "timestamp"]
        ):
            raise ValueError("Timestamp ties are too large to form leakage-free non-empty train/validation/test splits.")
        return ordered.iloc[:train_end].copy(), ordered.iloc[train_end:val_end].copy(), ordered.iloc[val_end:].copy()

    def fit_scaler(self, train: pd.DataFrame) -> None:
        self.time_origin = float(train["timestamp"].min())
        self.time_scale = max(float(train["timestamp"].max() - self.time_origin), 1.0)
        raw = self._raw_edge_features(train)
        self.scaler.fit(raw)
        self._scaler_fitted = True

    def build(self, frame: pd.DataFrame) -> GraphArrays:
        if not self.type_map or not self.relation_map:
            raise RuntimeError("Call fit before build.")
        local_nodes = sorted(set(frame["src_id"].astype(str)) | set(frame["dst_id"].astype(str)))
        local_campaigns = sorted(frame["campaign_id"].astype(str).unique())
        node_map = {value: idx for idx, value in enumerate(local_nodes)}
        campaign_map = {value: idx for idx, value in enumerate(local_campaigns)}
        src = frame["src_id"].astype(str).map(node_map).to_numpy(np.int64)
        dst = frame["dst_id"].astype(str).map(node_map).to_numpy(np.int64)
        relations = frame["relation"].astype(str).map(self.relation_map).fillna(0).to_numpy(np.int64)
        campaigns = frame["campaign_id"].astype(str).map(campaign_map).to_numpy(np.int64)
        edge_features = self._raw_edge_features(frame)
        if self._scaler_fitted:
            edge_features = self.scaler.transform(edge_features)
        node_features = self._node_features(frame, src, dst, node_map)
        node_labels = np.zeros(len(node_map), dtype=np.float32)
        # Prefer endpoint-specific labels when the normalizer can supply them.
        # This avoids silently treating a known-malicious source-only entity as
        # benign.  The legacy destination-event label remains supported for
        # externally prepared normalized files.
        if {"src_node_label", "dst_node_label"} <= set(frame.columns):
            np.maximum.at(node_labels, src, frame["src_node_label"].to_numpy(np.float32))
            np.maximum.at(node_labels, dst, frame["dst_node_label"].to_numpy(np.float32))
        else:
            np.maximum.at(node_labels, dst, frame["node_label"].to_numpy(np.float32))
        node_present = np.zeros(len(node_map), dtype=bool)
        node_present[src] = True; node_present[dst] = True
        campaign_labels = np.zeros(len(campaign_map), dtype=np.float32)
        np.maximum.at(campaign_labels, campaigns, frame["campaign_label"].to_numpy(np.float32))
        return GraphArrays(
            node_features=node_features.astype(np.float32),
            edge_features=edge_features.astype(np.float32),
            edge_index=np.stack([src, dst]),
            relation_index=relations,
            timestamps=frame["timestamp"].to_numpy(np.float64),
            node_labels=node_labels,
            node_present=node_present,
            edge_labels=frame["edge_label"].to_numpy(np.float32),
            campaign_index=campaigns,
            campaign_labels=campaign_labels,
            node_ids=local_nodes,
            edge_event_rows=frame["event_row"].to_numpy(np.int64) if "event_row" in frame else frame.index.to_numpy(np.int64),
            campaign_ids=local_campaigns,
        )

    @staticmethod
    def _backward_counts(keys: np.ndarray, timestamps: np.ndarray, window: int) -> np.ndarray:
        """Count strictly earlier same-key events within a trailing window."""
        from collections import defaultdict, deque

        history = defaultdict(deque)
        counts = np.zeros(len(keys), dtype=np.float64)
        for index, (key, timestamp) in enumerate(zip(keys, timestamps)):
            queue = history[str(key)]
            cutoff = float(timestamp) - float(window)
            while queue and queue[0] < cutoff:
                queue.popleft()
            counts[index] = len(queue)
            queue.append(float(timestamp))
        return counts

    def _raw_edge_features(self, frame: pd.DataFrame) -> np.ndarray:
        timestamp = frame["timestamp"].to_numpy(np.float64)
        if len(timestamp):
            origin = self.time_origin if self.time_origin is not None else float(timestamp.min())
            scale = self.time_scale if self.time_scale is not None else max(float(timestamp.max() - origin), 1.0)
            normalized = (timestamp - origin) / scale
        else:
            normalized = timestamp
        source_key = frame["src_id"].astype(str) + "\0" + frame["dst_id"].astype(str) + "\0" + frame["relation"].astype(str)
        elapsed = frame.assign(_key=source_key).groupby("_key", sort=False)["timestamp"].diff().fillna(0).to_numpy(np.float64)
        periodic = np.column_stack([normalized, np.log1p(np.maximum(elapsed, 0)), np.sin(2 * np.pi * normalized), np.cos(2 * np.pi * normalized)])
        src_keys = frame["src_id"].astype(str).to_numpy()
        dst_keys = frame["dst_id"].astype(str).to_numpy()
        history = []
        for window in self.history_windows_seconds:
            history.append(np.log1p(self._backward_counts(src_keys, timestamp, window)))
            history.append(np.log1p(self._backward_counts(dst_keys, timestamp, window)))
        periodic = np.column_stack([periodic, *history])
        if self.feature_columns:
            extra = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(np.float64)
            periodic = np.column_stack([periodic, extra])
        return periodic

    def _node_features(self, frame: pd.DataFrame, src: np.ndarray, dst: np.ndarray, node_map: Dict[str, int]) -> np.ndarray:
        n = len(node_map)
        type_width = len(self.type_map)
        features = np.zeros((n, 6 + type_width), dtype=np.float64)
        out_count = np.bincount(src, minlength=n)
        in_count = np.bincount(dst, minlength=n)
        features[:, 0] = np.log1p(out_count)
        features[:, 1] = np.log1p(in_count)
        features[:, 2] = np.log1p(out_count + in_count)
        # Structural imbalance is target-independent. Never derive input
        # features from edge_label/node_label/campaign_label.
        features[:, 3] = (in_count - out_count) / np.maximum(in_count + out_count, 1)
        times = frame["timestamp"].to_numpy(np.float64)
        time_sum = np.zeros(n)
        np.add.at(time_sum, dst, times)
        mean_time = time_sum / np.maximum(in_count, 1)
        origin = self.time_origin if self.time_origin is not None else float(times.min())
        scale = self.time_scale if self.time_scale is not None else max(float(times.max() - origin), 1.0)
        features[:, 4] = np.where(in_count > 0, (mean_time - origin) / scale, 0.0)
        relation_sets: Dict[int, set[int]] = {idx: set() for idx in range(n)}
        rel = frame["relation"].astype(str).map(self.relation_map).fillna(0).to_numpy(np.int64)
        for s, d, r in zip(src, dst, rel):
            relation_sets[int(s)].add(int(r)); relation_sets[int(d)].add(int(r))
        features[:, 5] = [np.log1p(len(relation_sets[idx])) for idx in range(n)]
        node_types: Dict[int, int] = {}
        for node, typ in zip(frame["src_id"].astype(str), frame["src_type"].astype(str)):
            node_types[node_map[node]] = self.type_map.get(typ, 0)
        for node, typ in zip(frame["dst_id"].astype(str), frame["dst_type"].astype(str)):
            node_types[node_map[node]] = self.type_map.get(typ, 0)
        for node_idx, type_idx in node_types.items():
            features[node_idx, 6 + type_idx] = 1.0
        return features

    def state_dict(self) -> dict:
        return {
            "type_map": self.type_map,
            "relation_map": self.relation_map,
            "campaign_map": self.campaign_map,
            "feature_columns": self.feature_columns,
            "scaler_mean": self.scaler.mean_.tolist() if self._scaler_fitted else None,
            "scaler_scale": self.scaler.scale_.tolist() if self._scaler_fitted else None,
            "time_origin": self.time_origin,
            "time_scale": self.time_scale,
            "history_windows_seconds": list(self.history_windows_seconds),
            "fit_scope": "training partition only; node/campaign indices are split-local",
        }


def validate_events(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_EVENT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required event columns: {sorted(missing)}")
    result = frame.copy()
    if not pd.api.types.is_numeric_dtype(result["timestamp"]):
        parsed = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
        result["timestamp"] = parsed.astype("int64") / 1e9
    result["timestamp"] = pd.to_numeric(result["timestamp"], errors="raise").astype(float)
    for col in ["edge_label", "node_label", "campaign_label"]:
        result[col] = pd.to_numeric(result[col], errors="raise").astype(int)
        values = set(result[col].dropna().unique())
        if not values <= {0, 1}:
            raise ValueError(f"{col} must contain only 0/1; found {sorted(values)}")
    for col in ["src_node_label", "dst_node_label"]:
        if col not in result:
            continue
        result[col] = pd.to_numeric(result[col], errors="raise").astype(int)
        values = set(result[col].dropna().unique())
        if not values <= {0, 1}:
            raise ValueError(f"{col} must contain only 0/1; found {sorted(values)}")
    for col in ["src_id", "dst_id", "src_type", "dst_type", "relation", "campaign_id"]:
        if result[col].isna().any():
            raise ValueError(f"{col} contains missing values")
        result[col] = result[col].astype(str)
    if len(result) < 20:
        raise ValueError("At least 20 chronologically ordered events are required.")
    return result.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def _open_text(path: str | Path):
    """Open plain-text or gzip-compressed public-release files."""
    import gzip

    path = Path(path)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


UUID_PATTERN = re.compile(
    r"^[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}$"
)


def load_entity_ground_truth(path: str | Path) -> tuple[set[str], dict]:
    """Read and integrity-audit a ThreaTrace-style entity-label UUID list.

    Identifiers are normalized to uppercase for matching.  The audit retains
    both the supplied-file SHA-256 and a line-ending/order-independent hash of
    the sorted unique identifiers, making CRLF/LF copies comparable.  This
    routine deliberately makes no semantic claim that every listed entity was
    directly involved in an attack; the source labeling policy may include
    context-expanded neighboring entities.
    """
    path = Path(path)
    raw = path.read_bytes()
    identifiers = []
    invalid = []
    with _open_text(path) as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip().upper()
            if not value:
                continue
            if not UUID_PATTERN.fullmatch(value):
                invalid.append({"line": line_number, "value": value[:120]})
            identifiers.append(value)
    if invalid:
        preview = ", ".join(f"line {row['line']}" for row in invalid[:5])
        raise ValueError(f"Ground-truth file contains invalid UUIDs at {preview}")
    unique = set(identifiers)
    if not unique:
        raise ValueError("Ground-truth file contains no entity UUIDs.")
    canonical = ("\n".join(sorted(unique)) + "\n").encode("ascii")
    audit = {
        "source_file": path.name,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "nonblank_rows": len(identifiers),
        "unique_entity_uuids": len(unique),
        "duplicate_rows": len(identifiers) - len(unique),
        "uuid_format": "RFC-4122 textual form; case-normalized for matching",
        "label_source_type": "ThreaTrace-expanded anomaly-entity UUID labels",
        "label_semantics": (
            "positive source-label membership; event-edge labels are derived "
            "only when a source or destination UUID is listed"
        ),
        "audit_scope": (
            "file integrity, UUID syntax, counts, duplicates, and canonical hash; "
            "not independent semantic adjudication of attack involvement"
        ),
        "semantic_ground_truth_verified": False,
        "semantic_caveat": (
            "The source labeling policy has been reported to include context-expanded "
            "neighboring entities; interpret these as benchmark anomaly/entity labels, "
            "not unqualified malicious-entity ground truth."
        ),
    }
    return unique, audit


def audit_ground_truth_files(
    dataset_paths: dict[str, str | Path], output: str | Path
) -> Path:
    """Write a deterministic integrity/provenance audit for DARPA label files."""
    required = {"darpa_cadets", "darpa_theia", "darpa_trace"}
    if set(dataset_paths) != required:
        raise ValueError(f"Ground-truth audit requires exactly {sorted(required)}")
    records = {}
    for dataset, path in sorted(dataset_paths.items()):
        _, records[dataset] = load_entity_ground_truth(path)
    payload = {
        "schema_version": "x-thgnn-label-source-audit-2",
        "all_passed": True,
        "integrity_checks_passed": True,
        "semantic_ground_truth_verified": False,
        "datasets": records,
        "provenance": (
            "ThreaTrace-expanded anomaly-entity UUID label files used by the "
            "public MAGIC DARPA preprocessing workflow"
        ),
        "audit_scope": (
            "The audit verifies the supplied files, not the semantic correctness "
            "of every positive label or its direct involvement in an attack."
        ),
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def normalize_lanl_auth(
    auth_path: str | Path,
    redteam_path: str | Path,
    output: str | Path,
    benign_campaign_seconds: int = 86400,
) -> Path:
    """Convert LANL ``auth.txt(.gz)`` and ``redteam.txt(.gz)`` to the common schema.

    Red-team events are matched on the official four-field key
    ``(time, source_user, source_computer, destination_computer)``.  The
    resulting CSV is sorted chronologically and can be consumed directly by
    ``validate``, ``train``, ``ablate`` and ``baselines``.
    """
    import csv

    attacks = set()
    with _open_text(redteam_path) as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            attacks.add((int(float(row[0])), row[1], row[2], row[3]))

    records = []
    with _open_text(auth_path) as handle:
        for row_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 9:
                continue
            timestamp = int(float(row[0]))
            src_user, dst_user, src_host, dst_host = row[1:5]
            relation = "auth:" + ":".join(value.strip() for value in row[5:8])
            success = row[8].strip().lower() == "success"
            malicious = int((timestamp, src_user, src_host, dst_host) in attacks)
            day = timestamp // int(benign_campaign_seconds)
            campaign_id = f"redteam_day_{day}" if malicious else f"benign_day_{day}"
            records.append({
                "timestamp": timestamp,
                "src_id": f"user:{src_user}|host:{src_host}",
                "dst_id": f"user:{dst_user}|host:{dst_host}",
                "src_type": "user_host",
                "dst_type": "user_host",
                "relation": relation,
                "edge_label": malicious,
                "node_label": malicious,
                "src_node_label": malicious,
                "dst_node_label": malicious,
                "campaign_id": campaign_id,
                "campaign_label": malicious,
                "feat_success": float(success),
                "feat_row_number": float(row_number),
            })
    frame = validate_events(pd.DataFrame.from_records(records))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return out


def _unwrap_cdm_record(payload: dict) -> tuple[str, dict]:
    """Return a tolerant ``(record_type, body)`` pair for CDM JSON records."""
    body = payload.get("datum", payload)
    if isinstance(body, dict) and len(body) == 1:
        key, value = next(iter(body.items()))
        if isinstance(value, dict):
            return key.rsplit(".", 1)[-1], value
    record_type = str(body.get("type", body.get("record_type", "unknown"))) if isinstance(body, dict) else "unknown"
    return record_type.rsplit(".", 1)[-1], body if isinstance(body, dict) else {}


def _cdm_uuid(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("uuid", "UUID", "string", "value"):
            if key in value:
                return _cdm_uuid(value[key])
    text = str(value).strip()
    return text or None


def normalize_darpa_tc_cdm(
    inputs: Iterable[str | Path],
    labels_path: str | Path,
    output: str | Path,
    dataset_name: str,
) -> Path:
    """Normalize CADETS, THEIA or TRACE CDM JSONL files.

    The label file is deliberately explicit and auditable.  It must contain
    ``event_uuid,label,campaign_id`` and may contain ``campaign_label``.  This
    avoids silently guessing release-specific ground-truth intervals.
    """
    import csv

    if dataset_name not in {"darpa_cadets", "darpa_theia", "darpa_trace"}:
        raise ValueError("dataset_name must be darpa_cadets, darpa_theia, or darpa_trace")
    labels = pd.read_csv(labels_path)
    required = {"event_uuid", "label", "campaign_id"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"CDM labels file is missing columns: {sorted(missing)}")
    label_map = {
        str(row.event_uuid): (
            int(row.label), str(row.campaign_id),
            int(getattr(row, "campaign_label", row.label)),
        )
        for row in labels.itertuples(index=False)
    }

    entities: dict[str, str] = {}
    event_records = []
    for input_path in [Path(value) for value in inputs]:
        with _open_text(input_path) as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                record_type, body = _unwrap_cdm_record(payload)
                uuid = _cdm_uuid(body.get("uuid"))
                if record_type != "Event":
                    if uuid:
                        entities[uuid] = record_type
                    continue
                event_uuid = uuid
                src = _cdm_uuid(body.get("subject") or body.get("src") or body.get("source"))
                dst = _cdm_uuid(
                    body.get("predicateObject") or body.get("predicateObject2")
                    or body.get("dst") or body.get("destination")
                )
                timestamp = body.get("timestampNanos", body.get("timestamp", body.get("time")))
                if not event_uuid or not src or not dst or timestamp is None:
                    continue
                timestamp = float(timestamp)
                if timestamp > 1e15:
                    timestamp /= 1e9
                elif timestamp > 1e12:
                    timestamp /= 1e3
                relation = str(body.get("type", body.get("eventType", "EVENT"))).rsplit(".", 1)[-1]
                label, campaign_id, campaign_label = label_map.get(
                    event_uuid, (0, f"benign_day_{int(timestamp // 86400)}", 0)
                )
                event_records.append({
                    "timestamp": timestamp,
                    "src_id": src,
                    "dst_id": dst,
                    "src_type": entities.get(src, "unknown"),
                    "dst_type": entities.get(dst, "unknown"),
                    "relation": relation,
                    "edge_label": label,
                    "node_label": label,
                    "src_node_label": label,
                    "dst_node_label": label,
                    "campaign_id": campaign_id,
                    "campaign_label": campaign_label,
                    "feat_labeled": float(event_uuid in label_map),
                    "feat_sequence": float(len(event_records)),
                    "source_event_uuid": event_uuid,
                    "source_file": input_path.name,
                    "source_line": line_number,
                })
    frame = validate_events(pd.DataFrame.from_records(event_records))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    metadata = {
        "dataset": dataset_name,
        "source_files": [str(Path(value).resolve()) for value in inputs],
        "labels": str(Path(labels_path).resolve()),
        "rows": len(frame),
        "malicious_rows": int(frame["edge_label"].sum()),
        "normalizer": "darpa-tc-cdm-explicit-event-labels-v1",
    }
    out.with_suffix(out.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def normalize_darpa_tc_entity_ground_truth(
    inputs: Iterable[str | Path],
    ground_truth_path: str | Path,
    output: str | Path,
    dataset_name: str,
) -> Path:
    """Normalize raw DARPA CDM events using a ThreaTrace entity-label UUID set.

    This pathway matches the public MAGIC/ThreaTrace entity-label convention:
    endpoint labels are direct UUID-membership indicators, while an event edge
    is positive when either endpoint has a positive source label. These are
    benchmark anomaly/entity labels, not independently adjudicated attack-event
    labels or attack-stage annotations.
    """
    import csv

    if dataset_name not in {"darpa_cadets", "darpa_theia", "darpa_trace"}:
        raise ValueError("dataset_name must be darpa_cadets, darpa_theia, or darpa_trace")
    malicious_entities, ground_truth_audit = load_entity_ground_truth(ground_truth_path)
    entities: dict[str, str] = {}
    event_records = []
    for input_path in [Path(value) for value in inputs]:
        with _open_text(input_path) as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                record_type, body = _unwrap_cdm_record(payload)
                uuid = _cdm_uuid(body.get("uuid"))
                if record_type != "Event":
                    if uuid:
                        entities[uuid.upper()] = record_type
                    continue
                event_uuid = uuid
                src = _cdm_uuid(body.get("subject") or body.get("src") or body.get("source"))
                dst = _cdm_uuid(
                    body.get("predicateObject") or body.get("predicateObject2")
                    or body.get("dst") or body.get("destination")
                )
                timestamp = body.get("timestampNanos", body.get("timestamp", body.get("time")))
                if not event_uuid or not src or not dst or timestamp is None:
                    continue
                src = src.upper(); dst = dst.upper()
                timestamp = float(timestamp)
                if timestamp > 1e15:
                    timestamp /= 1e9
                elif timestamp > 1e12:
                    timestamp /= 1e3
                relation = str(body.get("type", body.get("eventType", "EVENT"))).rsplit(".", 1)[-1]
                src_label = int(src in malicious_entities)
                dst_label = int(dst in malicious_entities)
                edge_label = int(src_label or dst_label)
                day = int(timestamp // 86400)
                event_records.append({
                    "timestamp": timestamp,
                    "src_id": src,
                    "dst_id": dst,
                    "src_type": entities.get(src, "unknown"),
                    "dst_type": entities.get(dst, "unknown"),
                    "relation": relation,
                    "edge_label": edge_label,
                    "node_label": dst_label,
                    "src_node_label": src_label,
                    "dst_node_label": dst_label,
                    "campaign_id": f"attack_day_{day}" if edge_label else f"benign_day_{day}",
                    "campaign_label": edge_label,
                    "feat_ground_truth_endpoint_count": float(src_label + dst_label),
                    "feat_sequence": float(len(event_records)),
                    "source_event_uuid": event_uuid,
                    "source_file": input_path.name,
                    "source_line": line_number,
                })
    frame = validate_events(pd.DataFrame.from_records(event_records))
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    observed_entities = set(frame["src_id"]) | set(frame["dst_id"])
    observed_malicious = malicious_entities & observed_entities
    metadata = {
        "dataset": dataset_name,
        "source_files": [str(Path(value).resolve()) for value in inputs],
        "rows": len(frame),
        "malicious_rows": int(frame["edge_label"].sum()),
        "observed_positive_label_entities": len(observed_malicious),
        "ground_truth": ground_truth_audit,
        "normalizer": "darpa-tc-cdm-threatrace-entity-label-v2",
        "task_boundary": (
            "primary endpoint/entity labels are direct UUID membership; edge and "
            "day-group labels are derived secondary outcomes; no stage labels"
        ),
    }
    out.with_suffix(out.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return out


def normalize_magic_preprocessed(
    dataset_name: str,
    graph_path: str | Path,
    output: str | Path,
    metadata_path: str | Path | None = None,
    ground_truth_path: str | Path | None = None,
    module_root: str | Path | None = None,
    total_events: int = 5000,
    extraction_seed: int = 20260821,
) -> Path:
    """Create an auditable executable extract from MAGIC/StreamSpot graph pickles.

    DARPA CADETS, THEIA and TRACE use one preprocessed test graph plus the
    repository's explicit malicious-node list. When supplied, the original
    ThreaTrace entity-label UUID file is integrity-audited and hashed. Edges are sampled within
    non-overlapping ordered source ranges. StreamSpot uses graph-disjoint,
    class-stratified groups because its preprocessed release removes event
    timestamps. The sidecar explicitly records that this is a resource-bounded
    benchmark extract rather than every event in the raw multi-gigabyte release.
    """
    import pickle
    import sys

    if total_events < 1000 or total_events % 10:
        raise ValueError("total_events must be at least 1000 and divisible by 10")
    if module_root is not None:
        sys.path.insert(0, str(Path(module_root).resolve()))
    try:
        import torch  # noqa: F401
        import dgl  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Graph-pickle conversion requires PyTorch and DGL.") from exc

    with Path(graph_path).open("rb") as handle:
        source = pickle.load(handle)
    rng = np.random.default_rng(int(extraction_seed))
    records: list[dict] = []
    source_selection: dict = {}
    ground_truth_provenance: dict | None = None

    def add_graph_edges(graph, graph_key: str, label: int, edge_count: int, partition: str) -> None:
        src_t, dst_t = graph.edges()
        src = src_t.detach().cpu().numpy().astype(np.int64)
        dst = dst_t.detach().cpu().numpy().astype(np.int64)
        edge_types = graph.edata["type"].detach().cpu().numpy().astype(np.int64)
        node_types = graph.ndata["type"].detach().cpu().numpy().astype(np.int64)
        chosen = np.sort(rng.choice(len(src), size=min(edge_count, len(src)), replace=False))
        for edge_index in chosen:
            records.append({
                "src_id": f"{dataset_name}:{graph_key}:n{src[edge_index]}",
                "dst_id": f"{dataset_name}:{graph_key}:n{dst[edge_index]}",
                "src_type": f"type_{node_types[src[edge_index]]}",
                "dst_type": f"type_{node_types[dst[edge_index]]}",
                "relation": f"rel_{edge_types[edge_index]}",
                "edge_label": int(label), "node_label": int(label),
                "src_node_label": int(label), "dst_node_label": int(label),
                "campaign_id": f"{partition}:{graph_key}", "campaign_label": int(label),
                "feat_original_edge_index": float(edge_index),
            })

    if dataset_name == "streamspot":
        graphs = [source[index] for index in range(len(source))]
        benign = [index for index, (_, label) in enumerate(graphs) if int(label) == 0]
        attack = [index for index, (_, label) in enumerate(graphs) if int(label) == 1]
        rng.shuffle(benign); rng.shuffle(attack)
        specifications = [
            ("train", benign[:20] + attack[:20], total_events * 8 // 10),
            ("validation", benign[20:25] + attack[20:25], total_events // 10),
            ("test", benign[25:30] + attack[25:30], total_events // 10),
        ]
        for partition, graph_ids, target in specifications:
            per_graph = target // len(graph_ids)
            for graph_id in graph_ids:
                graph, label = graphs[graph_id]
                add_graph_edges(graph, f"g{graph_id}", int(label), per_graph, partition)
        source_selection = {
            "split_method": "graph-disjoint class-stratified fixed-seed extraction",
            "graph_counts": {name: len(ids) for name, ids, _ in specifications},
            "subgraph_label_semantics": (
                "source scenario-graph label; binary attack-versus-benign classification, not attack-stage annotation"
            ),
        }
    elif dataset_name in {"darpa_cadets", "darpa_theia", "darpa_trace"}:
        if metadata_path is None:
            raise ValueError("DARPA preprocessed graphs require --metadata")
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        malicious_nodes = np.asarray(metadata["malicious"][0], dtype=np.int64)
        if ground_truth_path is not None:
            ground_truth_ids, ground_truth_provenance = load_entity_ground_truth(ground_truth_path)
            if len(ground_truth_ids) < len(malicious_nodes):
                raise ValueError(
                    "Ground-truth UUID count is smaller than the serialized malicious-node count; "
                    "the dataset/label-source pairing is inconsistent."
                )
            serialized_names = {
                str(value).strip().upper() for value in metadata["malicious"][1]
                if str(value).strip()
            }
            serialized_uuid_names = {value for value in serialized_names if UUID_PATTERN.fullmatch(value)}
            ground_truth_provenance.update({
                "serialized_malicious_node_indices": int(len(malicious_nodes)),
                "serialized_label_names": int(len(serialized_names)),
                "serialized_uuid_names": int(len(serialized_uuid_names)),
                "serialized_uuid_names_matched": int(len(serialized_uuid_names & ground_truth_ids)),
                "identity_mapping_verified": bool(
                    serialized_uuid_names
                    and serialized_uuid_names <= ground_truth_ids
                ),
                "mapping_note": (
                    "MAGIC serializes only labelled entities present in the test graph and may "
                    "replace UUIDs with entity names. When UUID identities are unavailable, "
                    "the code records provenance and counts but does not claim an identity-level "
                    "mapping. Raw UUID membership remains auditable by normalize-tc-ground-truth."
                ),
            })
        malicious_set = set(malicious_nodes.tolist())
        graph = source
        src_t, dst_t = graph.edges()
        src = src_t.detach().cpu().numpy().astype(np.int64)
        dst = dst_t.detach().cpu().numpy().astype(np.int64)
        edge_types = graph.edata["type"].detach().cpu().numpy().astype(np.int64)
        node_types = graph.ndata["type"].detach().cpu().numpy().astype(np.int64)
        labels = np.isin(src, malicious_nodes) | np.isin(dst, malicious_nodes)
        ranges = ((0.0, 0.8), (0.8, 0.9), (0.9, 1.0))
        if dataset_name == "darpa_theia":
            ranges = ((0.0, 0.6), (0.6, 0.7), (0.7, 0.8))
        targets = (total_events * 8 // 10, total_events // 10, total_events // 10)
        partitions = ("train", "validation", "test")
        selected_by_partition = {}
        for partition, (lower, upper), target in zip(partitions, ranges, targets):
            lo, hi = int(lower * len(src)), int(upper * len(src))
            candidates = np.arange(lo, hi, dtype=np.int64)
            positive = candidates[labels[lo:hi]]
            negative = candidates[~labels[lo:hi]]
            wanted_positive = min(len(positive), max(1, target // 5))
            wanted_negative = target - wanted_positive
            if len(negative) < wanted_negative:
                raise ValueError(f"Insufficient benign edges in {dataset_name} {partition} range")
            chosen = np.r_[
                rng.choice(positive, size=wanted_positive, replace=False),
                rng.choice(negative, size=wanted_negative, replace=False),
            ]
            chosen.sort()
            selected_by_partition[partition] = {
                "source_range": [lo, hi], "events": int(len(chosen)),
                "malicious_events": int(labels[chosen].sum()),
            }
            for position, edge_index in enumerate(chosen):
                malicious = int(labels[edge_index])
                records.append({
                    "src_id": f"{dataset_name}:n{src[edge_index]}",
                    "dst_id": f"{dataset_name}:n{dst[edge_index]}",
                    "src_type": f"type_{node_types[src[edge_index]]}",
                    "dst_type": f"type_{node_types[dst[edge_index]]}",
                    "relation": f"rel_{edge_types[edge_index]}",
                    "edge_label": malicious,
                    "node_label": int(dst[edge_index] in malicious_set),
                    "src_node_label": int(src[edge_index] in malicious_set),
                    "dst_node_label": int(dst[edge_index] in malicious_set),
                    # Fixed retained-edge groups are evaluation subgraphs, not
                    # real attack campaigns or stage annotations.
                    "campaign_id": f"{partition}:window_{position // 100}",
                    "campaign_label": malicious,
                    "feat_original_edge_index": float(edge_index),
                })
        source_selection = {
            "split_method": "ordered source ranges with within-range class-aware sampling",
            "partitions": selected_by_partition,
            "malicious_node_count": int(len(malicious_nodes)),
            "subgraph_label_semantics": (
                "fixed 100-retained-edge evaluation groups labelled positive when at least one retained edge is malicious; "
                "not an attack-stage or independently adjudicated campaign annotation"
            ),
        }
    else:
        raise ValueError(f"Unsupported preprocessed dataset: {dataset_name}")

    frame = pd.DataFrame.from_records(records)
    frame.insert(0, "timestamp", np.arange(1, len(frame) + 1, dtype=np.int64))
    frame = validate_events(frame)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    sidecar = {
        "dataset": dataset_name,
        "source_graph": str(Path(graph_path)),
        "source_graph_sha256": hashlib.sha256(Path(graph_path).read_bytes()).hexdigest(),
        "metadata": str(Path(metadata_path)) if metadata_path else None,
        "total_events": int(len(frame)), "extraction_seed": int(extraction_seed),
        "scope": "resource-bounded public preprocessed benchmark extract, not the full raw release",
        "timestamp_semantics": (
            "retained edge rank after fixed-seed extraction; preserves order only and is not elapsed wall-clock time"
        ),
        "ground_truth": ground_truth_provenance,
        **source_selection,
    }
    out.with_suffix(out.suffix + ".extract.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return out


def build_manifest(path: str | Path, frame: pd.DataFrame, builder: EventGraphBuilder, splits: Iterable[pd.DataFrame]) -> DatasetManifest:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    train, validation, test = list(splits)
    return DatasetManifest(
        source=str(path.resolve()), sha256=digest, rows=len(frame),
        nodes=len(set(frame["src_id"].astype(str)) | set(frame["dst_id"].astype(str))),
        relations=int(frame["relation"].nunique()),
        node_types=len(set(frame["src_type"].astype(str)) | set(frame["dst_type"].astype(str))),
        campaigns=int(frame["campaign_id"].nunique()),
        malicious_edges=int(frame["edge_label"].sum()), malicious_node_events=int(frame["node_label"].sum()),
        malicious_campaign_events=int(frame["campaign_label"].sum()), train_rows=len(train), validation_rows=len(validation),
        test_rows=len(test), train_time_max=float(train["timestamp"].max()), validation_time_max=float(validation["timestamp"].max()),
    )


def write_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")

# ===== metrics.py =====
"""Prediction, explanation, robustness, and uncertainty metrics."""


import time
from copy import copy
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.r_[0.05, np.linspace(0.1, 0.9, 81), 0.95, probabilities])
    scores = [f1_score(labels, probabilities >= threshold, zero_division=0) for threshold in candidates]
    return float(candidates[int(np.argmax(scores))])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    result = {
        "n": int(len(labels)), "prevalence": float(labels.mean()) if len(labels) else float("nan"),
        "threshold": float(threshold), "accuracy": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
    }
    result["auroc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    result["auprc"] = float(average_precision_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    result["brier_score"] = float(brier_score_loss(labels, probabilities)) if len(labels) else float("nan")
    return result


def prediction_frames(outputs, graph, thresholds: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    import torch
    present_nodes = graph.node_present.detach().cpu().numpy()
    node_prob_all = torch.sigmoid(outputs["node_logits"]).detach().cpu().numpy()
    node_prob = node_prob_all[present_nodes]
    edge_prob = torch.sigmoid(outputs["edge_logits"]).detach().cpu().numpy()
    present_campaigns = torch.unique(graph.campaign_index).detach().cpu().numpy()
    subgraph_prob_all = torch.sigmoid(outputs["subgraph_logits"]).detach().cpu().numpy()
    subgraph_prob = subgraph_prob_all[present_campaigns]
    node_labels_all = graph.node_labels.detach().cpu().numpy()
    node_labels = node_labels_all[present_nodes]
    prototype_similarity_all = outputs["prototype_similarity"].detach().cpu().numpy()
    prototype_index_all = prototype_similarity_all.argmax(axis=1)
    prototype_score_all = prototype_similarity_all.max(axis=1)
    prototype_class_all = outputs["prototype_class"].detach().cpu().numpy()[prototype_index_all]
    edge_labels = graph.edge_labels.detach().cpu().numpy()
    subgraph_labels_all = graph.campaign_labels.detach().cpu().numpy()
    subgraph_labels = subgraph_labels_all[present_campaigns]
    node_frame = pd.DataFrame({
        "node_id": [node for node, keep in zip(graph.node_ids, present_nodes) if keep], "label": node_labels.astype(int), "probability": node_prob,
        "prediction": (node_prob >= thresholds["node"]).astype(int),
        "top_prototype_index": prototype_index_all[present_nodes].astype(int),
        "top_prototype_class": prototype_class_all[present_nodes].astype(int),
        "top_prototype_similarity": prototype_score_all[present_nodes],
    })
    edge_frame = pd.DataFrame({
        "event_row": graph.edge_event_rows, "src_index": graph.edge_index[0].detach().cpu().numpy(),
        "dst_index": graph.edge_index[1].detach().cpu().numpy(), "label": edge_labels.astype(int),
        "probability": edge_prob, "prediction": (edge_prob >= thresholds["edge"]).astype(int),
        "importance": outputs["importance"].detach().cpu().numpy(),
    })
    subgraph_frame = pd.DataFrame({
        "campaign_id": [graph.campaign_ids[int(i)] for i in present_campaigns], "label": subgraph_labels.astype(int), "probability": subgraph_prob,
        "prediction": (subgraph_prob >= thresholds["subgraph"]).astype(int),
    })
    metrics = {
        "node": binary_metrics(node_labels, node_prob, thresholds["node"]),
        "edge": binary_metrics(edge_labels, edge_prob, thresholds["edge"]),
        "subgraph": binary_metrics(subgraph_labels, subgraph_prob, thresholds["subgraph"]),
    }
    return node_frame, edge_frame, subgraph_frame, metrics


def _device_synchronize(tensor) -> None:
    import torch
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _target_edge_mask(importance, graph, targets, retained_fraction: float):
    """Select a compact incoming explanation independently for each target."""
    import torch
    selected = torch.zeros_like(importance, dtype=torch.bool)
    local_sparsities = []
    destination = graph.edge_index[1]
    for target in torch.nonzero(targets, as_tuple=False).squeeze(-1):
        candidates = torch.nonzero(destination == target, as_tuple=False).squeeze(-1)
        if not candidates.numel():
            continue
        retained = max(1, int(round(candidates.numel() * retained_fraction)))
        retained = min(retained, candidates.numel())
        chosen = candidates[torch.topk(importance[candidates], k=retained).indices]
        selected[chosen] = True
        local_sparsities.append(1.0 - retained / candidates.numel())
    if not selected.any() and importance.numel():
        selected[torch.argmax(importance)] = True
    sparsity = float(np.mean(local_sparsities)) if local_sparsities else float(1.0 - selected.float().mean())
    return selected, sparsity


def evaluate_explanations(
    model,
    graph,
    retained_fraction: float = 0.1,
    node_threshold: float = 0.5,
    perturbation_noise: float = 0.02,
    repeats: int = 5,
    latency_warmup: int = 3,
    latency_repeats: int = 10,
) -> dict:
    import torch
    model.eval()
    with torch.no_grad():
        baseline = model(graph)
    importance = baseline["importance"]
    predicted_alerts = (torch.sigmoid(baseline["node_logits"]) >= float(node_threshold)) & graph.node_present
    if not predicted_alerts.any():
        return {
            "fidelity": float("nan"), "stability": float("nan"),
            "sparsity": float("nan"), "completeness": float("nan"),
            "latency_seconds": float("nan"), "latency_iqr_seconds": [float("nan"), float("nan")],
            "latency_repeats": 0, "selected_edges": 0,
            "available_edges": int(importance.numel()), "retained_fraction": float(retained_fraction),
            "target_node_ids": [], "selected_event_rows": [], "prototype_assignments": [],
            "targets": "predicted alerts; no alerts at the validation-selected threshold",
            "definition": "target-local incoming-edge deletion/retention with perturbation Jaccard stability",
        }
    selected_mask, local_sparsity = _target_edge_mask(importance, graph, predicted_alerts, retained_fraction)
    target_indices = torch.nonzero(predicted_alerts, as_tuple=False).squeeze(-1)
    selected_indices = torch.nonzero(selected_mask, as_tuple=False).squeeze(-1)
    prototype_similarity = baseline["prototype_similarity"][target_indices]
    prototype_indices = torch.argmax(prototype_similarity, dim=1)
    prototype_classes = baseline["prototype_class"][prototype_indices]
    prototype_scores = prototype_similarity.gather(1, prototype_indices.unsqueeze(-1)).squeeze(-1)
    prototype_assignments = [
        {
            "node_id": graph.node_ids[int(node_index)],
            "prototype_index": int(prototype_index),
            "prototype_class": int(prototype_class),
            "similarity": float(prototype_score),
        }
        for node_index, prototype_index, prototype_class, prototype_score in zip(
            target_indices.detach().cpu().tolist(),
            prototype_indices.detach().cpu().tolist(),
            prototype_classes.detach().cpu().tolist(),
            prototype_scores.detach().cpu().tolist(),
        )
    ]

    # Measure complete explanation generation (forward + target-local ranking),
    # using warm-up, synchronization, and a median rather than one dispatch.
    for _ in range(max(0, int(latency_warmup))):
        with torch.no_grad():
            warm = model(graph)
            _target_edge_mask(warm["importance"], graph, predicted_alerts, retained_fraction)
    timings = []
    for _ in range(max(1, int(latency_repeats))):
        _device_synchronize(graph.node_features)
        start = time.perf_counter()
        with torch.no_grad():
            measured = model(graph)
            _target_edge_mask(measured["importance"], graph, predicted_alerts, retained_fraction)
        _device_synchronize(graph.node_features)
        timings.append(time.perf_counter() - start)

    with torch.no_grad():
        deleted = model(graph.with_edge_mask(~selected_mask))
        retained_only = model(graph.with_edge_mask(selected_mask))
    base_score = torch.sigmoid(baseline["node_logits"])[predicted_alerts]
    deleted_score = torch.sigmoid(deleted["node_logits"])[predicted_alerts]
    retained_score = torch.sigmoid(retained_only["node_logits"])[predicted_alerts]
    fidelity = (base_score - deleted_score).mean().clamp(0, 1)
    completeness = (retained_score / base_score.clamp_min(1e-6)).mean().clamp(0, 1)
    overlaps = []
    for _ in range(repeats):
        noisy_graph = copy(graph)
        noisy_graph.edge_features = graph.edge_features + perturbation_noise * torch.randn_like(graph.edge_features)
        with torch.no_grad():
            perturbed = model(noisy_graph)
        perturbed_mask, _ = _target_edge_mask(
            perturbed["importance"], noisy_graph, predicted_alerts, retained_fraction
        )
        intersection = (selected_mask & perturbed_mask).sum().float()
        union = (selected_mask | perturbed_mask).sum().float().clamp_min(1)
        overlaps.append(float(intersection / union))
    return {
        "fidelity": float(fidelity), "stability": float(np.mean(overlaps)),
        "sparsity": local_sparsity, "completeness": float(completeness),
        "latency_seconds": float(np.median(timings)),
        "latency_iqr_seconds": [float(np.quantile(timings, 0.25)), float(np.quantile(timings, 0.75))],
        "latency_repeats": len(timings), "selected_edges": int(selected_mask.sum()),
        "available_edges": int(importance.numel()), "retained_fraction": float(retained_fraction),
        "target_node_ids": [graph.node_ids[int(index)] for index in target_indices.detach().cpu().tolist()],
        "selected_event_rows": [int(graph.edge_event_rows[int(index)]) for index in selected_indices.detach().cpu().tolist()],
        "prototype_assignments": prototype_assignments,
        "targets": "predicted alerts at the validation-selected node threshold",
        "definition": "target-local incoming-edge deletion/retention with perturbation Jaccard stability",
    }


def evaluate_explanation_curve(
    model,
    graph,
    retained_fractions: Iterable[float],
    node_threshold: float = 0.5,
) -> dict:
    """Archive unclipped per-target deletion and insertion scores over a fixed grid.

    Fractions refer to the proportion of each predicted alert's incoming edges
    retained by the target-local importance ranking.  Deletion removes those
    edges; insertion retains only those edges.  All values are raw sigmoid
    probabilities or raw differences, with no clipping.
    """
    import torch

    fractions = sorted({float(value) for value in retained_fractions})
    if not fractions or fractions[0] <= 0 or fractions[-1] > 1:
        raise ValueError("Explanation-curve fractions must lie in (0, 1].")
    model.eval()
    with torch.no_grad():
        baseline = model(graph)
    predicted_alerts = (
        torch.sigmoid(baseline["node_logits"]) >= float(node_threshold)
    ) & graph.node_present
    if not predicted_alerts.any():
        return {
            "fractions": fractions,
            "targets": [],
            "curve": [],
            "deletion_auc": float("nan"),
            "insertion_probability_auc": float("nan"),
            "normalized_insertion_gain_auc": float("nan"),
            "definition": "raw per-target deletion and insertion probabilities; no predicted alerts",
        }

    target_indices = torch.nonzero(predicted_alerts, as_tuple=False).squeeze(-1)
    baseline_scores = torch.sigmoid(baseline["node_logits"])[target_indices]
    empty_mask = torch.zeros_like(baseline["importance"], dtype=torch.bool)
    with torch.no_grad():
        empty = model(graph.with_edge_mask(empty_mask))
    empty_scores = torch.sigmoid(empty["node_logits"])[target_indices]
    target_ids = [graph.node_ids[int(index)] for index in target_indices.detach().cpu().tolist()]

    rows = []
    per_target = {
        node_id: {
            "node_id": node_id,
            "baseline_probability": float(base),
            "empty_graph_probability": float(empty_score),
            "points": [],
        }
        for node_id, base, empty_score in zip(
            target_ids,
            baseline_scores.detach().cpu().tolist(),
            empty_scores.detach().cpu().tolist(),
        )
    }
    for fraction in fractions:
        selected_mask, sparsity = _target_edge_mask(
            baseline["importance"], graph, predicted_alerts, fraction
        )
        with torch.no_grad():
            deleted = model(graph.with_edge_mask(~selected_mask))
            inserted = model(graph.with_edge_mask(selected_mask))
        deleted_scores = torch.sigmoid(deleted["node_logits"])[target_indices]
        inserted_scores = torch.sigmoid(inserted["node_logits"])[target_indices]
        deletion_delta = baseline_scores - deleted_scores
        insertion_denominator = baseline_scores - empty_scores
        normalized_gain = (inserted_scores - empty_scores) / insertion_denominator.abs().clamp_min(1e-6)
        rows.append({
            "retained_fraction": fraction,
            "mean_raw_deletion_delta": float(deletion_delta.mean()),
            "mean_deleted_probability": float(deleted_scores.mean()),
            "mean_inserted_probability": float(inserted_scores.mean()),
            "mean_normalized_insertion_gain": float(normalized_gain.mean()),
            "mean_baseline_probability": float(baseline_scores.mean()),
            "mean_empty_graph_probability": float(empty_scores.mean()),
            "target_local_sparsity": float(sparsity),
            "selected_edges": int(selected_mask.sum()),
        })
        for node_id, deleted_score, inserted_score, delta, gain in zip(
            target_ids,
            deleted_scores.detach().cpu().tolist(),
            inserted_scores.detach().cpu().tolist(),
            deletion_delta.detach().cpu().tolist(),
            normalized_gain.detach().cpu().tolist(),
        ):
            per_target[node_id]["points"].append({
                "retained_fraction": fraction,
                "deleted_probability": float(deleted_score),
                "inserted_probability": float(inserted_score),
                "raw_deletion_delta": float(delta),
                "normalized_insertion_gain": float(gain),
            })

    x = np.asarray([0.0] + fractions, dtype=float)
    deletion_y = np.asarray([0.0] + [row["mean_raw_deletion_delta"] for row in rows], dtype=float)
    insertion_y = np.asarray(
        [float(empty_scores.mean())] + [row["mean_inserted_probability"] for row in rows], dtype=float
    )
    gain_y = np.asarray([0.0] + [row["mean_normalized_insertion_gain"] for row in rows], dtype=float)
    span = max(float(x[-1] - x[0]), 1e-12)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return {
        "fractions": fractions,
        "targets": list(per_target.values()),
        "curve": rows,
        "deletion_auc": float(trapezoid(deletion_y, x) / span),
        "insertion_probability_auc": float(trapezoid(insertion_y, x) / span),
        "normalized_insertion_gain_auc": float(trapezoid(gain_y, x) / span),
        "definition": "target-local raw deletion and insertion curves; unclipped sigmoid probabilities",
    }


def select_compactness_fraction(model, graph, candidates: Iterable[float], node_threshold: float = 0.5) -> tuple[float, list[dict]]:
    """Select explanation compactness once on validation data."""
    records = []
    for fraction in sorted(set(float(value) for value in candidates)):
        metrics = evaluate_explanations(
            model, graph, retained_fraction=fraction, node_threshold=node_threshold, repeats=1,
            latency_warmup=0, latency_repeats=1,
        )
        score = 0.45 * metrics["fidelity"] + 0.45 * metrics["completeness"] + 0.10 * metrics["sparsity"]
        records.append({
            "retained_fraction": fraction,
            "validation_fidelity": metrics["fidelity"],
            "validation_completeness": metrics["completeness"],
            "validation_sparsity": metrics["sparsity"],
            "selection_score": score,
        })
    best = max(records, key=lambda row: (row["selection_score"], -row["retained_fraction"]))
    return float(best["retained_fraction"]), records


def empirical_edge_robustness(
    model,
    graph,
    node_threshold: float,
    budgets: Iterable[float],
    feature_epsilon: float = 0.01,
    explanation_retained_fraction: float = 0.1,
) -> dict:
    """First-order bounded feature and saliency-guided edge-deletion test."""
    import torch
    from torch.nn import functional as F
    model.eval()
    with torch.no_grad():
        clean = model(graph)
        clean_prob = torch.sigmoid(clean["node_logits"])
        clean_pred = clean_prob >= node_threshold
    positive = (graph.node_labels > 0.5) & graph.node_present
    originally_correct_positive = positive & clean_pred
    clean_misclassification = float((clean_pred[graph.node_present] != positive[graph.node_present]).float().mean())
    gates = torch.ones(graph.edge_labels.numel(), device=graph.edge_labels.device, requires_grad=True)
    feature_probe = graph.edge_features.detach().clone().requires_grad_(True)
    probe_graph = copy(graph); probe_graph.edge_features = feature_probe
    outputs = model(probe_graph, edge_gate=gates)
    loss = F.binary_cross_entropy_with_logits(outputs["node_logits"][graph.node_present], graph.node_labels[graph.node_present])
    gate_gradient, feature_gradient = torch.autograd.grad(
        loss, (gates, feature_probe), allow_unused=True
    )
    gate_gradient_available = gate_gradient is not None
    feature_gradient_available = feature_gradient is not None
    if gate_gradient is None:
        gate_gradient = torch.zeros_like(gates)
    if feature_gradient is None:
        feature_gradient = torch.zeros_like(feature_probe)
    deletion_gain = -gate_gradient
    adversarial_features = graph.edge_features + float(feature_epsilon) * feature_gradient.sign()
    results = {}
    for budget in budgets:
        count = max(1, int(round(len(gates) * float(budget))))
        deleted_indices = torch.topk(deletion_gain, k=min(count, len(gates))).indices
        attacked_gate = torch.ones_like(gates); attacked_gate[deleted_indices] = 0.0
        attacked_graph = copy(graph); attacked_graph.edge_features = adversarial_features.detach()
        with torch.no_grad():
            attacked = model(attacked_graph, edge_gate=attacked_gate)
            attacked_pred = torch.sigmoid(attacked["node_logits"]) >= node_threshold
        if originally_correct_positive.any():
            attack_success = float((~attacked_pred[originally_correct_positive]).float().mean())
        else:
            attack_success = float("nan")
        retained_graph = attacked_graph.with_edge_mask(attacked_gate.detach() > 0.5)
        attacked_explanation = evaluate_explanations(
            model, retained_graph, retained_fraction=explanation_retained_fraction,
            node_threshold=node_threshold,
            repeats=1, latency_warmup=0, latency_repeats=1,
        )
        results[f"{float(budget):.2f}"] = {
            "deleted_edges": int(count), "attack_success": attack_success,
            "explanation_fidelity": attacked_explanation["fidelity"],
        }
    return {
        "clean_misclassification": clean_misclassification,
        "eligible_positive_nodes": int(originally_correct_positive.sum()),
        "gate_gradient_available": gate_gradient_available,
        "feature_gradient_available": feature_gradient_available,
        "budgets": results,
        "feature_epsilon": float(feature_epsilon),
        "scope": "white-box first-order bounded feature perturbation plus saliency-guided edge deletion; empirical, not certified",
        "limitation": "Edge insertion is intentionally not claimed; insertion candidates require dataset-specific schema-valid event generation.",
    }


def summarize_seed_metrics(records: pd.DataFrame, group_columns: list[str], metric_columns: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in records.groupby(group_columns, dropna=False):
        key = (key,) if not isinstance(key, tuple) else key
        base = dict(zip(group_columns, key))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            if not len(values):
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            sem = sd / np.sqrt(len(values)) if len(values) > 1 else float("nan")
            critical = stats.t.ppf(0.975, len(values) - 1) if len(values) > 1 else float("nan")
            rows.append({**base, "metric": metric, "n_seeds": len(values), "mean": mean, "sd": sd,
                         "ci95_low": mean - critical * sem if len(values) > 1 else float("nan"),
                         "ci95_high": mean + critical * sem if len(values) > 1 else float("nan")})
    return pd.DataFrame(rows)

# ===== figure helpers =====
"""Generate conceptual figures without embedding experimental result values."""


import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONCEPTUAL_FIGURES_ONLY = True


def _with_average(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    numeric = [c for c in result.columns if c != result.columns[0]]
    # Use conventional half-up presentation to match the manuscript (for
    # example, 0.9725 is displayed as 0.973 rather than banker's 0.972).
    means = result[numeric].mean(axis=1)
    result["Average"] = np.floor(means * 1000 + 0.5) / 1000
    return result


TABLES: Dict[str, pd.DataFrame] = {}


def audit_reported_values() -> dict:
    raise RuntimeError(
        "The hard-coded manuscript-value audit was removed. Use summarize-runs "
        "on completed seed directories instead."
    )


def _save_architecture(path: Path) -> None:
    labels = [
        "Enterprise telemetry\nauthentication • process\nnetwork • provenance",
        "Temporal heterogeneous\ngraph construction\ntyped nodes and edges",
        "Dual-stream encoder\ntemporal attention\nrelation-specific messages",
        "Joint optimization\ndetection • prototypes\nconsistency • sparsity",
        "Role-specific outputs\nanalyst • responder\nexecutive",
    ]
    colors = ["#d9eaf7", "#dcefd8", "#fff1c7", "#f7dfc7", "#e7daf6"]
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    for idx, (label, color) in enumerate(zip(labels, colors)):
        x = 0.25 + idx * 2.55
        box = plt.Rectangle((x, 1.1), 2.1, 1.05, facecolor=color, edgecolor="#355f7d", linewidth=1.3)
        ax.add_patch(box)
        ax.text(x + 1.05, 1.63, label, ha="center", va="center", fontsize=8)
        if idx < len(labels) - 1:
            ax.annotate("", xy=(x + 2.5, 1.63), xytext=(x + 2.12, 1.63), arrowprops=dict(arrowstyle="->", color="#355f7d"))
    ax.annotate("Analyst feedback for governed offline calibration", xy=(9.0, 0.9), xytext=(11.2, 0.35),
                ha="center", fontsize=8, color="#355f7d", arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.25", color="#355f7d"))
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_consistency(path: Path) -> None:
    raise RuntimeError("Result figures must be generated from seed_summary.csv.")


def _save_latency_fidelity(path: Path) -> None:
    raise RuntimeError("Result figures must be generated from seed_summary.csv.")


def reproduce_reported(output: str | Path) -> Path:
    raise RuntimeError(
        "The hard-coded result-table command was removed. Use summarize-runs on "
        "completed seed directories and generate figures from seed_summary.csv."
    )

# ===== pilot.py =====
"""Participant-level paired analysis for the counterbalanced analyst pilot."""


import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {"participant_id", "alert_id", "condition", "triage_time", "decision_correct", "false_positive_escalation"}
METRICS = ["triage_time", "decision_correct", "false_positive_escalation", "trust", "comprehension"]


def _bootstrap_mean(values: np.ndarray, repeats: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    estimates = values[indices].mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _sign_flip_pvalue(values: np.ndarray, repeats: int, rng: np.random.Generator) -> float:
    observed = abs(float(values.mean()))
    signs = rng.choice([-1.0, 1.0], size=(repeats, len(values)))
    permuted = abs((signs * values).mean(axis=1))
    return float((1 + np.sum(permuted >= observed)) / (repeats + 1))


def analyze_pilot(path: str | Path, output: str | Path, bootstrap: int = 2000, seed: int = 20260821) -> Path:
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing pilot columns: {sorted(missing)}")
    frame["condition"] = frame["condition"].astype(str).str.lower().str.strip()
    if not set(frame["condition"].unique()) <= {"without", "x_thgnn"}:
        raise ValueError("condition must contain only 'without' and 'x_thgnn'")
    duplicate = frame.duplicated(["participant_id", "alert_id", "condition"])
    if duplicate.any():
        raise ValueError(f"Duplicate participant-alert-condition rows: {int(duplicate.sum())}")
    available_metrics = [metric for metric in METRICS if metric in frame.columns]
    aggregated = frame.groupby(["participant_id", "condition"], as_index=False)[available_metrics].mean()
    rng = np.random.default_rng(seed)
    rows = []
    paired_export = []
    for metric in available_metrics:
        pivot = aggregated.pivot(index="participant_id", columns="condition", values=metric).dropna()
        if not {"without", "x_thgnn"} <= set(pivot.columns) or len(pivot) < 2:
            continue
        if metric in {"triage_time", "false_positive_escalation"}:
            difference = pivot["without"] - pivot["x_thgnn"]
            direction = "positive values favour X-THGNN (reduction)"
        else:
            difference = pivot["x_thgnn"] - pivot["without"]
            direction = "positive values favour X-THGNN (increase)"
        values = difference.to_numpy(float)
        low, high = _bootstrap_mean(values, bootstrap, rng)
        pvalue = _sign_flip_pvalue(values, max(bootstrap, 5000), rng)
        sd = float(values.std(ddof=1))
        rows.append({
            "metric": metric, "n_participants": len(values),
            "without_mean": float(pivot["without"].mean()), "x_thgnn_mean": float(pivot["x_thgnn"].mean()),
            "paired_difference": float(values.mean()), "paired_difference_sd": sd,
            "paired_standardized_effect": float(values.mean() / sd) if sd > 0 else float("nan"),
            "cluster_bootstrap_ci95_low": low, "cluster_bootstrap_ci95_high": high,
            "sign_flip_pvalue_two_sided": pvalue, "direction": direction,
        })
        export = pivot.reset_index().copy(); export["paired_difference"] = difference.values; export["metric"] = metric
        paired_export.append(export)
    if not rows:
        raise ValueError("No metric had paired observations in both conditions.")
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "pilot_paired_summary.csv", index=False)
    pd.concat(paired_export, ignore_index=True).to_csv(out / "pilot_participant_differences.csv", index=False)
    metadata = {
        "participants_total": int(frame["participant_id"].nunique()), "alerts_total": int(frame["alert_id"].nunique()),
        "rows": int(len(frame)), "bootstrap_repeats": int(bootstrap), "permutation_repeats": int(max(bootstrap, 5000)),
        "analysis": "condition means within participant; participant-cluster bootstrap; paired sign-flip permutation",
        "warning": "Confirmatory mixed-effects modelling requires a prespecified model and alert-level dependency structure.",
    }
    (out / "pilot_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out

# ===== baselines.py =====
"""Executable classical baselines and paired seed-level comparisons."""


import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



TASKS = ("node", "edge", "subgraph")
METRIC_COLUMNS = ("node_f1", "edge_auroc", "subgraph_f1")


def _task_matrix(graph: GraphArrays, task: str, relation_count: int) -> tuple[np.ndarray, np.ndarray]:
    if task == "node":
        return graph.node_features[graph.node_present], graph.node_labels[graph.node_present]
    if task == "edge":
        src, dst = graph.edge_index
        relation = np.eye(relation_count, dtype=np.float32)[graph.relation_index]
        matrix = np.column_stack([
            graph.node_features[src], graph.node_features[dst], graph.edge_features, relation,
        ])
        return matrix, graph.edge_labels
    if task != "subgraph":
        raise ValueError(f"Unknown task: {task}")
    present = np.unique(graph.campaign_index)
    rows = []
    for campaign in present:
        mask = graph.campaign_index == campaign
        edge = graph.edge_features[mask]
        rows.append(np.r_[edge.mean(axis=0), edge.max(axis=0), np.log1p(mask.sum())])
    return np.asarray(rows, dtype=np.float32), graph.campaign_labels[present]


def _models(seed: int, fast: bool = False) -> dict:
    trees = 40 if fast else 300
    iterations = 50 if fast else 200
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=trees, class_weight="balanced_subsample", random_state=seed,
            n_jobs=-1, min_samples_leaf=2,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=iterations, learning_rate=0.05, random_state=seed,
        ),
    }


def _fit(model, x: np.ndarray, y: np.ndarray):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        model = DummyClassifier(strategy="prior")
        model.fit(x, y)
        return model
    if isinstance(model, HistGradientBoostingClassifier):
        counts = np.bincount(y, minlength=2).astype(float)
        weights = len(y) / (2.0 * np.maximum(counts, 1.0))
        model.fit(x, y, sample_weight=weights[y])
    else:
        model.fit(x, y)
    return model


def _positive_probability(model, x: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = np.asarray(model.classes_)
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return probabilities[:, int(np.flatnonzero(classes == 1)[0])]


def run_baselines(
    events: str | Path,
    output: str | Path,
    seeds: Iterable[int],
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    fast: bool = False,
) -> Path:
    """Train leakage-controlled sanity baselines for all three prediction tasks.

    These are transparent classical baselines, not substitutes for the named
    research baselines in the manuscript.
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame = EventGraphBuilder.load(events)
    builder = EventGraphBuilder(train_fraction, validation_fraction)
    train_frame, validation_frame, test_frame = builder.split(frame)
    builder.fit(train_frame)
    builder.fit_scaler(train_frame)
    graphs = {
        "train": builder.build(train_frame),
        "validation": builder.build(validation_frame),
        "test": builder.build(test_frame),
    }
    write_manifest(
        build_manifest(events, frame, builder, [train_frame, validation_frame, test_frame]),
        out / "dataset_manifest.json",
    )
    records = []
    for seed in [int(value) for value in seeds]:
        task_data = {
            task: {
                split: _task_matrix(graph, task, len(builder.relation_map))
                for split, graph in graphs.items()
            }
            for task in TASKS
        }
        for model_name, prototype in _models(seed, fast=fast).items():
            row = {"model": model_name, "seed": seed}
            for task in TASKS:
                x_train, y_train = task_data[task]["train"]
                x_validation, y_validation = task_data[task]["validation"]
                x_test, y_test = task_data[task]["test"]
                model = _fit(prototype, x_train, y_train)
                validation_probability = _positive_probability(model, x_validation)
                threshold = select_f1_threshold(y_validation, validation_probability)
                test_probability = _positive_probability(model, x_test)
                metrics = binary_metrics(y_test, test_probability, threshold)
                row[f"{task}_f1"] = metrics["f1"]
                row[f"{task}_auroc"] = metrics["auroc"]
                row[f"{task}_threshold"] = threshold
            records.append(row)
    raw = pd.DataFrame(records)
    raw.to_csv(out / "baseline_seed_metrics.csv", index=False)
    summary_metrics = [column for column in raw if column.endswith(("_f1", "_auroc"))]
    summarize_seed_metrics(raw, ["model"], summary_metrics).to_csv(
        out / "baseline_summary.csv", index=False
    )
    metadata = {
        "scope": "classical executable sanity baselines",
        "models": list(_models(0, fast=fast)),
        "tasks": list(TASKS),
        "seeds": sorted(raw["seed"].unique().astype(int).tolist()),
        "warning": "Do not label these models as EULER, HetGLM, StageFinder, CyberGFM, or CONTINUUM.",
    }
    (out / "baseline_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def compare_seed_tables(
    candidate_csv: str | Path,
    reference_csv: str | Path,
    output: str | Path,
    candidate_name: str = "X-THGNN",
    reference_name: str = "Reference",
    candidate_model: str | None = None,
    reference_model: str | None = None,
    permutations: int = 20000,
    seed: int = 20260821,
) -> Path:
    """Perform paired, seed-matched comparisons with uncertainty estimates."""
    candidate = pd.read_csv(candidate_csv)
    reference = pd.read_csv(reference_csv)
    for label, frame, selected in [
        ("candidate", candidate, candidate_model), ("reference", reference, reference_model)
    ]:
        if "model" not in frame:
            continue
        available = sorted(frame["model"].dropna().astype(str).unique())
        if selected is not None:
            if selected not in available:
                raise ValueError(f"Unknown {label} model {selected!r}; choose from {available}")
            if label == "candidate":
                candidate = frame[frame["model"].astype(str) == selected].copy()
            else:
                reference = frame[frame["model"].astype(str) == selected].copy()
        elif len(available) > 1:
            raise ValueError(
                f"The {label} table contains multiple models {available}; select one explicitly."
            )
    if "seed" not in candidate or "seed" not in reference:
        raise ValueError("Both inputs must contain a seed column.")
    metrics = sorted((set(candidate) & set(reference)) - {"seed", "model", "run"})
    metrics = [metric for metric in metrics if not metric.endswith("_threshold")]
    metrics = [metric for metric in metrics if pd.api.types.is_numeric_dtype(candidate[metric])]
    if not metrics:
        raise ValueError("No shared numeric metric columns were found.")
    candidate = candidate.groupby("seed", as_index=False)[metrics].mean()
    reference = reference.groupby("seed", as_index=False)[metrics].mean()
    paired = candidate.merge(reference, on="seed", suffixes=("_candidate", "_reference"))
    if len(paired) < 2:
        raise ValueError("At least two matched seeds are required.")
    rng = np.random.default_rng(seed)
    rows = []
    for metric in metrics:
        left = paired[f"{metric}_candidate"].to_numpy(float)
        right = paired[f"{metric}_reference"].to_numpy(float)
        valid = np.isfinite(left) & np.isfinite(right)
        differences = left[valid] - right[valid]
        if len(differences) < 2:
            continue
        mean = float(differences.mean())
        sd = float(differences.std(ddof=1))
        sem = sd / np.sqrt(len(differences))
        critical = float(stats.t.ppf(0.975, len(differences) - 1))
        signs = rng.choice((-1.0, 1.0), size=(permutations, len(differences)))
        permuted = np.abs((signs * differences).mean(axis=1))
        pvalue = float((1 + np.sum(permuted >= abs(mean))) / (permutations + 1))
        rows.append({
            "metric": metric, "matched_seeds": len(differences),
            "candidate": candidate_name, "reference": reference_name,
            "candidate_mean": float(left[valid].mean()), "reference_mean": float(right[valid].mean()),
            "paired_difference": mean, "ci95_low": mean - critical * sem,
            "ci95_high": mean + critical * sem, "paired_effect_dz": mean / sd if sd > 0 else np.nan,
            "sign_flip_pvalue_two_sided": pvalue,
        })
    if not rows:
        raise ValueError("No metrics had at least two finite matched pairs.")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "paired_seed_comparisons.csv", index=False)
    paired.to_csv(out / "matched_seed_values.csv", index=False)
    return out


REQUIRED_BASELINES = {
    "EULER", "HetGLM", "StageFinder", "CyberGFM", "CONTINUUM",
    "GNNExplainer", "GraphMask", "PROVEX", "PROVEXPLAINER",
}
REQUIRED_BASELINE_FIELDS = {
    "implementation_source", "version_or_commit", "task_head",
    "parameter_count", "hyperparameters", "random_seeds", "hardware",
    "prediction_file",
}


def audit_baseline_registry(registry_path: str | Path, output: str | Path) -> Path:
    """Validate provenance and prediction artifacts for every named comparator.

    This command intentionally refuses to relabel the classical sanity models
    as published GNN baselines. Exact external implementations must instead be
    registered with immutable source versions and seed-level prediction files.
    """
    import yaml

    registry_path = Path(registry_path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    entries = payload.get("baselines", {})
    missing_models = sorted(REQUIRED_BASELINES - set(entries))
    errors = []
    rows = []
    for name in sorted(REQUIRED_BASELINES & set(entries)):
        entry = entries[name] or {}
        missing_fields = sorted(REQUIRED_BASELINE_FIELDS - set(entry))
        if missing_fields:
            errors.append(f"{name}: missing fields {missing_fields}")
            continue
        prediction_path = (registry_path.parent / str(entry["prediction_file"])).resolve()
        if not prediction_path.exists():
            errors.append(f"{name}: prediction file does not exist: {prediction_path}")
            continue
        predictions = pd.read_csv(prediction_path)
        required_prediction_columns = {"seed", "task", "instance_id", "label", "probability"}
        missing_prediction = sorted(required_prediction_columns - set(predictions))
        if missing_prediction:
            errors.append(f"{name}: prediction file missing columns {missing_prediction}")
            continue
        declared_seeds = sorted(int(value) for value in entry["random_seeds"])
        observed_seeds = sorted(pd.to_numeric(predictions["seed"], errors="raise").astype(int).unique().tolist())
        if declared_seeds != observed_seeds:
            errors.append(f"{name}: declared seeds {declared_seeds} != prediction seeds {observed_seeds}")
            continue
        rows.append({
            "model": name, "implementation_source": entry["implementation_source"],
            "version_or_commit": entry["version_or_commit"], "task_head": entry["task_head"],
            "parameter_count": int(entry["parameter_count"]), "seeds": declared_seeds,
            "prediction_rows": len(predictions), "prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "hardware": entry["hardware"],
        })
    if missing_models:
        errors.insert(0, f"Missing named baselines: {missing_models}")
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    report = {"all_passed": not errors, "errors": errors, "validated": rows}
    (out / "baseline_registry_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if errors:
        raise ValueError("Baseline registry audit failed; see baseline_registry_audit.json")
    return out

# ===== diagnostics.py =====
"""Dependency-light end-to-end diagnostics for the reproduction program."""


import json
from pathlib import Path

import numpy as np
import pandas as pd



def _synthetic_events(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(314159)
    records = []
    for index in range(rows):
        malicious = int(index % 7 in {0, 1})
        records.append({
            "timestamp": 1_700_000_000 + index * 60,
            "src_id": f"user_{index % 12}", "dst_id": f"host_{(3 * index + 1) % 18}",
            "src_type": "user", "dst_type": "host",
            "relation": "remote_login" if malicious else ("authentication" if index % 2 else "network_flow"),
            "edge_label": malicious, "node_label": malicious,
            "campaign_id": f"attack_{index % 3}" if malicious else "benign",
            "campaign_label": malicious,
            "feat_bytes": float(rng.lognormal(8.0 + malicious, 0.4)),
            "feat_duration": float(rng.gamma(2.0 + malicious, 1.0)),
        })
    return pd.DataFrame(records)


def _synthetic_pilot() -> pd.DataFrame:
    rows = []
    for participant in range(1, 13):
        for alert in range(1, 5):
            for condition in ("without", "x_thgnn"):
                assisted = condition == "x_thgnn"
                rows.append({
                    "participant_id": participant, "alert_id": alert, "condition": condition,
                    "triage_time": 115 + participant - 40 * assisted + alert,
                    "decision_correct": int((participant + alert + assisted) % 4 != 0),
                    "false_positive_escalation": int((participant + alert) % (5 if assisted else 3) == 0),
                    "trust": 3.0 + 1.0 * assisted, "comprehension": 3.1 + 0.9 * assisted,
                })
    return pd.DataFrame(rows)


def run_self_test(output: str | Path) -> Path:
    """Exercise data, baseline, pilot, summarization, and optional PyTorch paths."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    checks = {}
    events_path = out / "synthetic_events.csv"
    events = validate_events(_synthetic_events())
    events.to_csv(events_path, index=False)
    synthetic_ground_truth = out / "synthetic_ground_truth.txt"
    synthetic_ground_truth.write_text(
        "9EF37E2E-3E80-11E8-A5CB-3FA3753A265A\n"
        "9ef37e2e-3e80-11e8-a5cb-3fa3753a265a\n"
        "A4DD7C60-3E80-11E8-A5CB-3FA3753A265A\n",
        encoding="utf-8",
    )
    identifiers, ground_truth_audit = load_entity_ground_truth(synthetic_ground_truth)
    checks["ground_truth_audit"] = bool(
        len(identifiers) == 2
        and ground_truth_audit["duplicate_rows"] == 1
        and len(ground_truth_audit["canonical_sha256"]) == 64
    )
    protocol_path = write_manuscript_config(out / "manuscript_config.yaml")
    roundtrip_config = load_config(protocol_path)
    checks["manuscript_config_roundtrip"] = bool(
        roundtrip_config["model"]["embedding_dim"]
        == MANUSCRIPT_PROTOCOL["config"]["model"]["embedding_dim"]
    )
    builder = EventGraphBuilder()
    train, validation, test = builder.split(events)
    builder.fit(train)
    builder.fit_scaler(train)
    graph_arrays = builder.build(train)
    checks["data_pipeline"] = bool(len(train) and len(validation) and len(test) and graph_arrays.edge_features.shape[0] == len(train))
    unseen = test.copy()
    first = unseen.index[0]
    unseen.loc[first, ["src_id", "src_type", "relation", "campaign_id"]] = [
        "never_seen_node", "never_seen_type", "never_seen_relation", "never_seen_campaign"
    ]
    unseen_graph = builder.build(unseen)
    checks["chronological_isolation"] = bool(
        "never_seen_node" in unseen_graph.node_ids
        and unseen_graph.relation_index[0] == builder.relation_map[builder.UNKNOWN_TOKEN]
        and builder.state_dict()["fit_scope"].startswith("training partition only")
    )
    run_baselines(events_path, out / "baselines", seeds=[7, 19], fast=True)
    checks["baseline_pipeline"] = (out / "baselines" / "baseline_summary.csv").exists()
    pilot_path = out / "synthetic_pilot.csv"
    _synthetic_pilot().to_csv(pilot_path, index=False)
    analyze_pilot(pilot_path, out / "pilot", bootstrap=200, seed=11)
    checks["pilot_pipeline"] = (out / "pilot" / "pilot_paired_summary.csv").exists()

    try:
        import torch
    except ImportError:
        checks["pytorch_forward_backward"] = "skipped: PyTorch not installed"
        core_model_executed = False
    else:
        graph = graph_arrays.to_torch("cpu")
        flags = AblationFlags()
        model = XTHGNN(
            node_width=graph.node_features.shape[1], edge_width=graph.edge_features.shape[1],
            num_relations=len(builder.relation_map), num_campaigns=len(builder.campaign_map),
            embedding_dim=16, num_layers=2, num_prototypes=6, dropout=0.0, flags=flags,
        )
        outputs = model(graph)
        loss, _ = joint_loss(outputs, graph, DEFAULT_CONFIG["optimization"], flags)
        loss.backward()
        checks["pytorch_forward_backward"] = bool(torch.isfinite(loss))
        core_model_executed = True

    failed = [name for name, value in checks.items() if value is False]
    report = {
        "all_available_checks_passed": not failed,
        "core_model_executed": core_model_executed,
        "submission_ready_self_test": bool(not failed and core_model_executed),
        "failed": failed,
        "checks": checks,
        "warning": None if core_model_executed else "Install PyTorch and rerun before treating this as a complete model self-test.",
    }
    (out / "self_test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failed:
        raise AssertionError(f"Self-test failures: {failed}")
    return out

# ===== Optional PyTorch model and training implementation =====
try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    """Pure-PyTorch X-THGNN reference model and joint objective."""


    from dataclasses import dataclass
    from typing import Dict, Optional



    def segment_softmax(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
        """Numerically stable softmax over edges sharing a destination node."""
        maxima = torch.full((size,), -torch.inf, dtype=values.dtype, device=values.device)
        maxima.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
        shifted = values - maxima[index]
        numerator = shifted.exp()
        denominator = torch.zeros(size, dtype=values.dtype, device=values.device)
        denominator.index_add_(0, index, numerator)
        return numerator / denominator[index].clamp_min(1e-12)


    def scatter_sum(messages: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
        output = torch.zeros((size, messages.shape[-1]), dtype=messages.dtype, device=messages.device)
        output.index_add_(0, index, messages)
        return output


    def scatter_mean(messages: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
        output = scatter_sum(messages, index, size)
        counts = torch.zeros(size, dtype=messages.dtype, device=messages.device)
        counts.index_add_(0, index, torch.ones_like(index, dtype=messages.dtype))
        return output / counts.clamp_min(1.0).unsqueeze(-1)


    def recent_neighbor_mask(
        destinations: torch.Tensor,
        timestamps: torch.Tensor,
        sample_size: int,
    ) -> torch.Tensor:
        """Retain the most recent ``sample_size`` incoming events per node."""
        if sample_size <= 0:
            raise ValueError("Neighborhood sample size must be positive.")
        if not destinations.numel():
            return torch.zeros_like(destinations, dtype=torch.bool)
        # Stable sorts produce destination groups whose internal order is
        # newest-to-oldest, avoiding a Python loop over potentially millions
        # of enterprise nodes. PyTorch 1.12 (the locked manuscript version)
        # does not expose the ``stable`` keyword, so use a deterministic
        # NumPy lexicographic fallback while preserving the tensor device.
        def stable_argsort(values: torch.Tensor, descending: bool = False) -> torch.Tensor:
            try:
                return torch.argsort(values, descending=descending, stable=True)
            except TypeError:
                array = values.detach().cpu().numpy()
                positions = np.arange(array.shape[0], dtype=np.int64)
                primary = -array if descending else array
                ordered = np.lexsort((positions, primary))
                return torch.as_tensor(ordered, dtype=torch.long, device=values.device)

        time_order = stable_argsort(timestamps, descending=True)
        destination_order = stable_argsort(destinations[time_order])
        order = time_order[destination_order]
        grouped = destinations[order]
        group_start = torch.ones(grouped.numel(), dtype=torch.bool, device=grouped.device)
        group_start[1:] = grouped[1:] != grouped[:-1]
        start_positions = torch.nonzero(group_start, as_tuple=False).squeeze(-1)
        group_id = torch.cumsum(group_start.to(torch.long), dim=0) - 1
        rank = torch.arange(grouped.numel(), device=grouped.device) - start_positions[group_id]
        mask = torch.zeros_like(destinations, dtype=torch.bool)
        mask[order[rank < sample_size]] = True
        return mask


    @dataclass
    class AblationFlags:
        temporal_stream: bool = True
        heterogeneous_stream: bool = True
        joint_optimization: bool = True
        consistency_regularization: bool = True
        prototype_learning: bool = True
        temporal_attention: bool = True
        gating_mechanism: bool = True


    class TemporalHeterogeneousLayer(nn.Module):
        def __init__(
            self,
            width: int,
            edge_width: int,
            num_relations: int,
            dropout: float,
            temporal_decay_init: float,
            neighbor_sample_size: int,
        ):
            super().__init__()
            self.width = width
            self.neighbor_sample_size = int(neighbor_sample_size)
            self.query = nn.Linear(width, width, bias=False)
            self.key = nn.Linear(width, width, bias=False)
            self.temporal_message = nn.Linear(width, width, bias=False)
            self.time_encoder = nn.Sequential(nn.Linear(edge_width, width), nn.Tanh(), nn.Linear(width, 1, bias=False))
            self.relation_matrix = nn.Parameter(torch.empty(num_relations, width, width))
            self.relation_attention = nn.Parameter(torch.empty(num_relations, width))
            self.relation_embedding = nn.Embedding(num_relations, width)
            self.fusion = nn.Linear(2 * width, width)
            self.batch_norm = nn.BatchNorm1d(width)
            self.gate = nn.Linear(2 * width, width)
            self.importance = nn.Sequential(
                nn.Linear(5 * width + edge_width + 2, width), nn.ReLU(), nn.Linear(width, 1)
            )
            self.dropout = nn.Dropout(dropout)
            raw_decay = torch.log(torch.expm1(torch.tensor(max(temporal_decay_init, 1e-4))))
            self.raw_temporal_decay = nn.Parameter(raw_decay)
            nn.init.xavier_uniform_(self.relation_matrix)
            nn.init.xavier_uniform_(self.relation_attention)

        def forward(self, h, edge_index, relation_index, edge_features, timestamps, flags: AblationFlags, edge_gate=None):
            src, dst = edge_index
            n = h.shape[0]
            sampled = recent_neighbor_mask(dst, timestamps, self.neighbor_sample_size)
            q = self.query(h[dst])
            k = self.key(h[src])
            logits = (q * k).sum(-1) / (self.width ** 0.5)
            if flags.temporal_attention:
                logits = logits + self.time_encoder(edge_features).squeeze(-1)
                logits = logits.masked_fill(~sampled, -torch.inf)
                temporal_alpha = segment_softmax(logits, dst, n)
            else:
                degree = torch.bincount(dst[sampled], minlength=n).clamp_min(1)
                temporal_alpha = sampled.to(h.dtype) / degree[dst].to(h.dtype)
            decay = torch.exp(-F.softplus(self.raw_temporal_decay) * edge_features[:, 1].abs())
            temporal_messages = self.temporal_message(h[src]) * temporal_alpha.unsqueeze(-1) * decay.unsqueeze(-1)

            matrices = self.relation_matrix[relation_index]
            heterogeneous_messages = torch.bmm(matrices, h[src].unsqueeze(-1)).squeeze(-1)
            rel_logits = (heterogeneous_messages * self.relation_attention[relation_index]).sum(-1) / (self.width ** 0.5)
            rel_logits = rel_logits.masked_fill(~sampled, -torch.inf)
            heterogeneous_alpha = segment_softmax(rel_logits, dst, n)
            heterogeneous_messages = heterogeneous_messages * heterogeneous_alpha.unsqueeze(-1)

            if edge_gate is not None:
                temporal_messages = temporal_messages * edge_gate.unsqueeze(-1)
                heterogeneous_messages = heterogeneous_messages * edge_gate.unsqueeze(-1)
            temporal = scatter_sum(temporal_messages, dst, n) if flags.temporal_stream else torch.zeros_like(h)
            heterogeneous = scatter_sum(heterogeneous_messages, dst, n) if flags.heterogeneous_stream else torch.zeros_like(h)
            fused = self.batch_norm(self.fusion(torch.cat([temporal, heterogeneous], dim=-1)))
            fused = F.relu(fused)
            if flags.gating_mechanism:
                gate = torch.sigmoid(self.gate(torch.cat([h, fused], dim=-1)))
                updated = (1.0 - gate) * h + gate * fused
            else:
                updated = h + fused
            updated = self.dropout(updated)
            relation_emb = self.relation_embedding(relation_index)
            importance_input = torch.cat([
                h[src], h[dst], updated[src], updated[dst], relation_emb, edge_features,
                temporal_alpha.unsqueeze(-1), heterogeneous_alpha.unsqueeze(-1),
            ], dim=-1)
            importance = torch.sigmoid(self.importance(importance_input).squeeze(-1))
            return updated, importance, temporal_alpha, heterogeneous_alpha


    class XTHGNN(nn.Module):
        def __init__(
            self,
            node_width: int,
            edge_width: int,
            num_relations: int,
            num_campaigns: int,
            embedding_dim: int = 128,
            num_layers: int = 3,
            num_prototypes: int = 50,
            temperature: float = 0.07,
            dropout: float = 0.2,
            temporal_decay_init: float = 0.5,
            neighbor_sample_size: int = 20,
            flags: Optional[AblationFlags] = None,
        ):
            super().__init__()
            self.flags = flags or AblationFlags()
            self.temperature = temperature
            self.num_campaigns = num_campaigns
            self.node_encoder = nn.Sequential(nn.Linear(node_width, embedding_dim), nn.ReLU(), nn.Dropout(dropout))
            self.layers = nn.ModuleList([
                TemporalHeterogeneousLayer(
                    embedding_dim, edge_width, num_relations, dropout,
                    temporal_decay_init, neighbor_sample_size,
                )
                for _ in range(num_layers)
            ])
            self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
            self.node_head = nn.Linear(embedding_dim, 1)
            self.edge_head = nn.Sequential(
                nn.Linear(3 * embedding_dim + edge_width, embedding_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(embedding_dim, 1)
            )
            self.subgraph_head = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, 1))
            self.prototypes = nn.Parameter(torch.randn(num_prototypes, embedding_dim) * 0.02)
            self.register_buffer("prototype_class", torch.tensor([idx % 2 for idx in range(num_prototypes)], dtype=torch.long))

        def forward(self, graph, edge_gate=None) -> Dict[str, torch.Tensor]:
            h = self.node_encoder(graph.node_features)
            layer_states = [h]
            importances = []
            temporal_attention = []
            heterogeneous_attention = []
            for layer in self.layers:
                h, importance, temporal_alpha, heterogeneous_alpha = layer(
                    h, graph.edge_index, graph.relation_index, graph.edge_features,
                    graph.timestamps, self.flags, edge_gate=edge_gate
                )
                layer_states.append(h)
                importances.append(importance)
                temporal_attention.append(temporal_alpha)
                heterogeneous_attention.append(heterogeneous_alpha)
            src, dst = graph.edge_index
            relation_emb = self.relation_embedding(graph.relation_index)
            edge_logits = self.edge_head(torch.cat([h[src], h[dst], relation_emb, graph.edge_features], dim=-1)).squeeze(-1)
            edge_repr = 0.5 * (h[src] + h[dst])
            campaign_repr = scatter_mean(edge_repr, graph.campaign_index, len(graph.campaign_ids))
            subgraph_logits = self.subgraph_head(campaign_repr).squeeze(-1)
            node_logits = self.node_head(h).squeeze(-1)
            similarity = F.normalize(h, dim=-1) @ F.normalize(self.prototypes, dim=-1).T / self.temperature
            return {
                "node_logits": node_logits,
                "edge_logits": edge_logits,
                "subgraph_logits": subgraph_logits,
                "node_embeddings": h,
                "prototype_similarity": similarity,
                "prototype_class": self.prototype_class,
                "importance": importances[-1],
                "importance_layers": torch.stack(importances),
                "temporal_attention_layers": torch.stack(temporal_attention),
                "heterogeneous_attention_layers": torch.stack(heterogeneous_attention),
                "layer_states": torch.stack(layer_states),
            }


    def _balanced_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        positives = labels.sum()
        negatives = labels.numel() - positives
        if positives.item() == 0 or negatives.item() == 0:
            return F.binary_cross_entropy_with_logits(logits, labels)
        pos_weight = (negatives / positives.clamp_min(1.0)).detach()
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


    def _supervision_batch(graph, batch_size: int) -> dict[str, torch.Tensor]:
        """Draw task-specific supervision batches without sampling test data."""
        node = torch.nonzero(graph.node_present, as_tuple=False).squeeze(-1)
        edge = torch.arange(graph.edge_labels.numel(), device=graph.edge_labels.device)
        present_campaigns = torch.unique(graph.campaign_index)

        def sample(indices):
            if indices.numel() <= batch_size:
                return indices
            return indices[torch.randperm(indices.numel(), device=indices.device)[:batch_size]]

        return {"node": sample(node), "edge": sample(edge), "subgraph": sample(present_campaigns)}


    def _sample_non_edges(graph, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample schema-neutral node pairs absent from the observed training graph."""
        n = int(graph.node_features.shape[0])
        if count <= 0 or n < 2:
            empty = torch.empty(0, dtype=torch.long, device=graph.edge_index.device)
            return empty, empty
        observed = torch.unique(graph.edge_index[0] * n + graph.edge_index[1])
        selected = []
        remaining = int(count)
        for _ in range(12):
            draw = max(remaining * 4, 128)
            src = torch.randint(0, n, (draw,), device=graph.edge_index.device)
            dst = torch.randint(0, n, (draw,), device=graph.edge_index.device)
            keys = src * n + dst
            keep = (src != dst) & ~torch.isin(keys, observed)
            keys = torch.unique(keys[keep])
            if keys.numel():
                selected.append(keys[:remaining])
                remaining -= min(remaining, int(keys.numel()))
            if remaining <= 0:
                break
        if not selected:
            empty = torch.empty(0, dtype=torch.long, device=graph.edge_index.device)
            return empty, empty
        keys = torch.unique(torch.cat(selected))[:count]
        return torch.div(keys, n, rounding_mode="floor"), keys % n


    def joint_loss(
        outputs: Dict[str, torch.Tensor],
        graph,
        weights: dict,
        flags: AblationFlags,
        supervision: Optional[dict[str, torch.Tensor]] = None,
        temporal_consistency: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        if supervision is None:
            supervision = {
                "node": torch.nonzero(graph.node_present, as_tuple=False).squeeze(-1),
                "edge": torch.arange(graph.edge_labels.numel(), device=graph.edge_labels.device),
                "subgraph": torch.unique(graph.campaign_index),
            }
        node_indices = supervision["node"]
        edge_indices = supervision["edge"]
        campaign_indices = supervision["subgraph"]
        node_loss = _balanced_bce(outputs["node_logits"][node_indices], graph.node_labels[node_indices])
        edge_loss = _balanced_bce(outputs["edge_logits"][edge_indices], graph.edge_labels[edge_indices])
        subgraph_loss = _balanced_bce(
            outputs["subgraph_logits"][campaign_indices], graph.campaign_labels[campaign_indices]
        )

        # Equation (9): contrast observed malicious links against node pairs
        # that are absent from the training graph.  Validation/test edges are
        # never consulted when candidates are generated.
        batch_labels = graph.edge_labels[edge_indices]
        positive = edge_indices[batch_labels > 0.5]
        if positive.numel():
            pair_count = int(positive.numel())
            positive = positive[torch.randperm(positive.numel(), device=positive.device)[:pair_count]]
            positive_src, positive_dst = graph.edge_index[:, positive]
            negative_src, negative_dst = _sample_non_edges(graph, pair_count)
            if negative_src.numel():
                usable = min(pair_count, int(negative_src.numel()))
                positive_src, positive_dst = positive_src[:usable], positive_dst[:usable]
                negative_src, negative_dst = negative_src[:usable], negative_dst[:usable]
                positive_logits = (
                    outputs["node_embeddings"][positive_src] * outputs["node_embeddings"][positive_dst]
                ).sum(-1)
                negative_logits = (
                    outputs["node_embeddings"][negative_src] * outputs["node_embeddings"][negative_dst]
                ).sum(-1)
                link_logits = torch.cat([positive_logits, negative_logits])
                link_labels = torch.cat([
                    torch.ones(usable, device=link_logits.device),
                    torch.zeros(usable, device=link_logits.device),
                ])
                link_loss = _balanced_bce(link_logits, link_labels)
            else:
                link_loss = torch.zeros((), device=outputs["node_logits"].device)
        else:
            link_loss = torch.zeros((), device=outputs["node_logits"].device)
        importance = outputs["importance"].clamp(1e-6, 1 - 1e-6)
        sparsity_loss = importance.mean()
        importance_alignment = _balanced_bce(
            torch.logit(importance[edge_indices]), graph.edge_labels[edge_indices]
        )

        similarity = outputs["prototype_similarity"]
        if flags.prototype_learning:
            class_logits = []
            prototype_class = outputs["prototype_class"]
            for cls in range(2):
                class_logits.append(torch.logsumexp(similarity[:, prototype_class == cls], dim=1))
            prototype_loss = F.cross_entropy(torch.stack(class_logits, dim=1)[graph.node_present], graph.node_labels[graph.node_present].long())
        else:
            prototype_loss = torch.zeros((), device=similarity.device)

        final_attention = outputs["temporal_attention_layers"][-1].clamp_min(1e-8)
        edge_entropy = -(final_attention * final_attention.log())
        entropy_by_node = torch.zeros(graph.node_features.shape[0], device=edge_entropy.device)
        entropy_by_node.index_add_(0, graph.edge_index[1], edge_entropy)
        entropy_loss = entropy_by_node[graph.node_present].mean()
        if temporal_consistency is None:
            temporal_consistency = torch.zeros((), device=similarity.device)
        attention_layers = outputs["temporal_attention_layers"].clamp_min(1e-8)
        if attention_layers.shape[0] > 1:
            p = attention_layers[:-1]
            q = attention_layers[1:]
            edge_kl = p * (p.log() - q.log())
            node_terms = []
            destination = graph.edge_index[1]
            for layer_kl in edge_kl:
                per_node = torch.zeros(graph.node_features.shape[0], device=layer_kl.device)
                per_node.index_add_(0, destination, layer_kl)
                node_terms.append(per_node[graph.node_present].mean())
            consistency = torch.stack(node_terms).mean()
        else:
            consistency = torch.zeros((), device=similarity.device)
        explanation_loss = (
            weights["sparsity_coefficient"] * sparsity_loss
            + weights["prototype_coefficient"] * prototype_loss
            + weights["entropy_coefficient"] * entropy_loss
            + weights["importance_alignment_coefficient"] * importance_alignment
        )
        total = (
            weights["node_loss_weight"] * node_loss
            + weights["edge_loss_weight"] * edge_loss
            + weights["link_contrastive_weight"] * link_loss
            + weights["subgraph_loss_weight"] * subgraph_loss
        )
        if flags.joint_optimization:
            total = total + weights["explanation_loss_weight"] * explanation_loss
        if flags.consistency_regularization:
            total = total + weights["temporal_consistency_weight"] * temporal_consistency + weights["cross_layer_consistency_weight"] * consistency
        components = {
            "total": float(total.detach()), "node": float(node_loss.detach()), "edge": float(edge_loss.detach()),
            "link_contrastive": float(link_loss.detach()), "subgraph": float(subgraph_loss.detach()),
            "explanation": float(explanation_loss.detach()),
            "importance_alignment": float(importance_alignment.detach()),
            "temporal_consistency": float(temporal_consistency.detach()), "cross_layer_consistency": float(consistency.detach()),
        }
        return total, components

    """Training, ablation, prediction export, and multi-seed summarization."""


    import json
    import platform
    import random
    import sys
    from copy import deepcopy
    from dataclasses import asdict
    from pathlib import Path
    from typing import Iterable, Optional

    import numpy as np
    import pandas as pd



    ABLATIONS = {
        "full": AblationFlags(),
        "minus_temporal_stream": AblationFlags(temporal_stream=False),
        "minus_heterogeneous_stream": AblationFlags(heterogeneous_stream=False),
        "minus_joint_optimization": AblationFlags(joint_optimization=False),
        "minus_consistency_regularization": AblationFlags(consistency_regularization=False),
        "minus_prototype_learning": AblationFlags(prototype_learning=False),
        "minus_temporal_attention": AblationFlags(temporal_attention=False),
        "minus_gating_mechanism": AblationFlags(gating_mechanism=False),
    }

    def set_seed(seed: int) -> None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


    def _device(requested: Optional[str]) -> str:
        if requested:
            return requested
        # The manuscript protocol is CPU-locked. GPU execution remains available
        # only through an explicit --device override, which is recorded in every
        # resolved configuration.
        return "cpu"


    def _thresholds(outputs, graph) -> dict:
        with torch.no_grad():
            node = torch.sigmoid(outputs["node_logits"][graph.node_present]).cpu().numpy()
            edge = torch.sigmoid(outputs["edge_logits"]).cpu().numpy()
            present = torch.unique(graph.campaign_index)
            subgraph = torch.sigmoid(outputs["subgraph_logits"][present]).cpu().numpy()
            node_labels = graph.node_labels[graph.node_present].cpu().numpy()
            edge_labels = graph.edge_labels.cpu().numpy()
            subgraph_labels = graph.campaign_labels[present].cpu().numpy()
        return {
            "node": select_f1_threshold(node_labels, node),
            "edge": select_f1_threshold(edge_labels, edge),
            "subgraph": select_f1_threshold(subgraph_labels, subgraph),
        }


    def _closed_window_graphs(
        builder: EventGraphBuilder,
        frame: pd.DataFrame,
        device: str,
        window_seconds: int,
    ) -> list[tuple[int, TorchGraph]]:
        """Build non-overlapping closed windows with no cross-window future context."""
        origin = float(frame["timestamp"].min())
        work = frame.copy()
        work["_evaluation_window"] = np.floor(
            (work["timestamp"] - origin) / float(window_seconds)
        ).astype(int)
        graphs = []
        for window_id, group in work.groupby("_evaluation_window", sort=True):
            group = group.drop(columns="_evaluation_window")
            if group.empty:
                continue
            graphs.append((int(window_id), builder.build(group).to_torch(device)))
        if not graphs:
            raise ValueError("No closed evaluation windows could be constructed.")
        return graphs


    def _thresholds_closed_windows(model, graphs: list[tuple[int, TorchGraph]]) -> dict:
        values = {task: {"labels": [], "probabilities": []} for task in ("node", "edge", "subgraph")}
        model.eval()
        with torch.no_grad():
            for _, graph in graphs:
                outputs = model(graph)
                present = graph.node_present
                values["node"]["labels"].append(graph.node_labels[present].cpu().numpy())
                values["node"]["probabilities"].append(torch.sigmoid(outputs["node_logits"][present]).cpu().numpy())
                values["edge"]["labels"].append(graph.edge_labels.cpu().numpy())
                values["edge"]["probabilities"].append(torch.sigmoid(outputs["edge_logits"]).cpu().numpy())
                campaigns = torch.unique(graph.campaign_index)
                values["subgraph"]["labels"].append(graph.campaign_labels[campaigns].cpu().numpy())
                values["subgraph"]["probabilities"].append(torch.sigmoid(outputs["subgraph_logits"][campaigns]).cpu().numpy())
        return {
            task: select_f1_threshold(
                np.concatenate(payload["labels"]), np.concatenate(payload["probabilities"])
            )
            for task, payload in values.items()
        }


    def _predict_closed_windows(model, graphs, thresholds):
        node_frames, edge_frames, subgraph_frames = [], [], []
        model.eval()
        with torch.no_grad():
            for window_id, graph in graphs:
                outputs = model(graph)
                node, edge, subgraph, _ = prediction_frames(outputs, graph, thresholds)
                for frame in (node, edge, subgraph):
                    frame.insert(0, "evaluation_window", int(window_id))
                node_frames.append(node); edge_frames.append(edge); subgraph_frames.append(subgraph)
        node = pd.concat(node_frames, ignore_index=True)
        edge = pd.concat(edge_frames, ignore_index=True)
        subgraph = pd.concat(subgraph_frames, ignore_index=True)
        metrics = {
            "node": binary_metrics(node["label"], node["probability"], thresholds["node"]),
            "edge": binary_metrics(edge["label"], edge["probability"], thresholds["edge"]),
            "subgraph": binary_metrics(subgraph["label"], subgraph["probability"], thresholds["subgraph"]),
        }
        window_count = max(1, len(graphs))
        for task, frame in (("node", node), ("edge", edge), ("subgraph", subgraph)):
            false_alarms = int(((frame["label"] == 0) & (frame["prediction"] == 1)).sum())
            metrics[task]["false_alarms_per_window"] = float(false_alarms / window_count)
            metrics[task]["evaluation_windows"] = int(window_count)
        return node, edge, subgraph, metrics


    def _explanations_closed_windows(model, graphs, node_threshold, retained_fraction, config):
        records = []
        for window_id, graph in graphs:
            result = evaluate_explanations(
                model, graph, retained_fraction=retained_fraction,
                node_threshold=node_threshold,
                perturbation_noise=float(config["perturbation_noise"]),
                repeats=int(config["stability_repeats"]),
                latency_warmup=int(config["latency_warmup"]),
                latency_repeats=int(config["latency_repeats"]),
            )
            curve = evaluate_explanation_curve(
                model,
                graph,
                retained_fractions=config.get("evaluation_fractions", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]),
                node_threshold=node_threshold,
            )
            records.append({"evaluation_window": window_id, **result, "curve_evaluation": curve})
        finite = [row for row in records if np.isfinite(row["fidelity"])]
        summary = {
            metric: float(np.mean([row[metric] for row in finite])) if finite else float("nan")
            for metric in ("fidelity", "stability", "sparsity", "completeness", "latency_seconds")
        }
        summary.update({
            "windows_total": len(records), "windows_with_predicted_alerts": len(finite),
            "selected_edges": int(sum(row["selected_edges"] for row in finite)),
            "available_edges": int(sum(row["available_edges"] for row in records)),
            "retained_fraction": float(retained_fraction),
            "targets": "predicted alerts in non-overlapping closed evaluation windows",
            "window_records": records,
        })
        curve_records = [
            row["curve_evaluation"] for row in records
            if np.isfinite(row["curve_evaluation"]["deletion_auc"])
        ]
        summary.update({
            "deletion_auc": float(np.mean([row["deletion_auc"] for row in curve_records]))
                if curve_records else float("nan"),
            "insertion_probability_auc": float(np.mean([
                row["insertion_probability_auc"] for row in curve_records
            ])) if curve_records else float("nan"),
            "normalized_insertion_gain_auc": float(np.mean([
                row["normalized_insertion_gain_auc"] for row in curve_records
            ])) if curve_records else float("nan"),
            "curve_fractions": config.get("evaluation_fractions", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]),
            "unclipped_scores_archived": True,
        })
        return summary


    def _json_safe(value):
        if isinstance(value, dict): return {k: _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [_json_safe(v) for v in value]
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, float) and not np.isfinite(value): return None
        return value


    def _temporal_window_pairs(
        frame: pd.DataFrame,
        builder: EventGraphBuilder,
        device: str,
        window_seconds: int,
        max_pairs: int,
    ) -> list[tuple[TorchGraph, TorchGraph, torch.Tensor, torch.Tensor]]:
        """Build adjacent, non-overlapping snapshots for Equation (11)."""
        origin = float(frame["timestamp"].min())
        work = frame.copy()
        work["_window"] = np.floor((work["timestamp"] - origin) / float(window_seconds)).astype(int)
        snapshots = []
        for _, group in work.groupby("_window", sort=True):
            group = group.drop(columns="_window")
            if len(group) < 2:
                continue
            arrays = builder.build(group)
            if len(arrays.node_ids) < 2:
                continue
            snapshots.append(arrays.to_torch(device))
        pairs = []
        for left, right in zip(snapshots[:-1], snapshots[1:]):
            left_map = {node: index for index, node in enumerate(left.node_ids)}
            right_map = {node: index for index, node in enumerate(right.node_ids)}
            common = sorted(set(left_map) & set(right_map))
            if not common:
                continue
            left_index = torch.as_tensor([left_map[node] for node in common], dtype=torch.long, device=device)
            right_index = torch.as_tensor([right_map[node] for node in common], dtype=torch.long, device=device)
            pairs.append((left, right, left_index, right_index))
        if len(pairs) > max_pairs:
            selected = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
            pairs = [pairs[int(index)] for index in selected]
        return pairs


    def _adjacent_window_consistency(model, pair) -> torch.Tensor:
        if pair is None:
            return torch.zeros((), device=next(model.parameters()).device)
        left, right, left_index, right_index = pair
        left_state = model(left)["node_embeddings"][left_index]
        right_state = model(right)["node_embeddings"][right_index]
        return F.mse_loss(left_state, right_state)


    def _adversarial_consistency(model, graph, feature_epsilon: float, edge_budget: float) -> torch.Tensor:
        """One-step inner maximization for Equation (14)."""
        feature_probe = graph.edge_features.detach().clone().requires_grad_(True)
        edge_gate = torch.ones(graph.edge_labels.numel(), device=feature_probe.device, requires_grad=True)
        probe_graph = copy(graph); probe_graph.edge_features = feature_probe
        probe = model(probe_graph, edge_gate=edge_gate)
        probe_loss = F.binary_cross_entropy_with_logits(
            probe["node_logits"][graph.node_present], graph.node_labels[graph.node_present]
        )
        feature_gradient, gate_gradient = torch.autograd.grad(
            probe_loss, (feature_probe, edge_gate), retain_graph=False, create_graph=False
        )
        adversarial_graph = copy(graph)
        adversarial_graph.edge_features = (
            graph.edge_features + float(feature_epsilon) * feature_gradient.sign()
        ).detach()
        count = max(1, int(round(graph.edge_labels.numel() * float(edge_budget))))
        deletion_gain = -gate_gradient
        deleted = torch.topk(deletion_gain, k=min(count, deletion_gain.numel())).indices
        adversarial_gate = torch.ones_like(edge_gate).detach()
        adversarial_gate[deleted] = 0.0
        with torch.no_grad():
            clean_probability = torch.sigmoid(model(graph)["node_logits"]).detach()
        adversarial_probability = torch.sigmoid(
            model(adversarial_graph, edge_gate=adversarial_gate)["node_logits"]
        )
        return F.mse_loss(adversarial_probability, clean_probability)


    def train_experiment(
        events: str | Path,
        config_path: str | Path,
        output: str | Path,
        seed: int,
        flags: Optional[AblationFlags] = None,
        device: Optional[str] = None,
        robustness_aware: bool = False,
        max_epochs: Optional[int] = None,
    ) -> Path:
        config = load_config(config_path)
        flags = flags or AblationFlags()
        set_seed(seed)
        device = _device(device)
        out = Path(output); out.mkdir(parents=True, exist_ok=True)
        frame = EventGraphBuilder.load(events)
        builder = EventGraphBuilder(
            train_fraction=float(config["data"]["train_fraction"]),
            validation_fraction=float(config["data"]["validation_fraction"]),
            feature_prefix=str(config["data"]["feature_prefix"]),
            history_windows_seconds=tuple(int(value) for value in config["data"]["history_windows_seconds"]),
        )
        train_frame, val_frame, test_frame = builder.split(frame)
        builder.fit(train_frame)
        builder.fit_scaler(train_frame)
        train_arrays = builder.build(train_frame); val_arrays = builder.build(val_frame); test_arrays = builder.build(test_frame)
        manifest = build_manifest(events, frame, builder, [train_frame, val_frame, test_frame])
        write_manifest(manifest, out / "dataset_manifest.json")
        pd.DataFrame(sorted(builder.node_map.items(), key=lambda item: item[1]), columns=["node_id", "node_index"]).to_csv(out / "node_index.csv", index=False)
        pd.DataFrame(sorted(builder.campaign_map.items(), key=lambda item: item[1]), columns=["campaign_id", "campaign_index"]).to_csv(out / "campaign_index.csv", index=False)
        train_graph = train_arrays.to_torch(device); val_graph = val_arrays.to_torch(device)
        evaluation_window_seconds = int(config["evaluation"]["closed_window_seconds"])
        validation_windows = _closed_window_graphs(builder, val_frame, device, evaluation_window_seconds)
        test_windows = _closed_window_graphs(builder, test_frame, device, evaluation_window_seconds)
        model_cfg = config["model"]
        model = XTHGNN(
            node_width=train_graph.node_features.shape[1], edge_width=train_graph.edge_features.shape[1],
            num_relations=len(builder.relation_map), num_campaigns=len(builder.campaign_map),
            embedding_dim=int(model_cfg["embedding_dim"]), num_layers=int(model_cfg["num_layers"]),
            num_prototypes=int(model_cfg["num_prototypes"]), temperature=float(model_cfg["temperature"]),
            dropout=float(model_cfg["dropout"]), temporal_decay_init=float(model_cfg["temporal_decay_init"]),
            neighbor_sample_size=int(model_cfg["neighbor_sample_size"]), flags=flags,
        ).to(device)
        opt_cfg = config["optimization"]
        optimizer = torch.optim.Adam(model.parameters(), lr=float(opt_cfg["learning_rate"]), weight_decay=float(opt_cfg["weight_decay"]))
        epochs = int(max_epochs or opt_cfg["epochs"]); patience = int(opt_cfg["early_stopping_patience"])
        temporal_pairs = _temporal_window_pairs(
            train_frame, builder, device,
            window_seconds=int(config["data"]["temporal_consistency_window_seconds"]),
            max_pairs=int(config["data"]["max_temporal_pairs"]),
        )
        history = []; best_loss = float("inf"); best_state = None; stale = 0
        for epoch in range(1, epochs + 1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            train_outputs = model(train_graph)
            temporal_pair = temporal_pairs[(epoch - 1) % len(temporal_pairs)] if temporal_pairs else None
            temporal_consistency = _adjacent_window_consistency(model, temporal_pair)
            supervision = _supervision_batch(train_graph, int(opt_cfg["batch_size"]))
            loss, components = joint_loss(
                train_outputs, train_graph, opt_cfg, flags,
                supervision=supervision, temporal_consistency=temporal_consistency,
            )
            if robustness_aware:
                consistency = _adversarial_consistency(
                    model, train_graph,
                    feature_epsilon=float(config["robustness"]["feature_epsilon"]),
                    edge_budget=float(config["robustness"]["training_edge_budget"]),
                )
                loss = loss + float(opt_cfg["robustness_regularization"]) * consistency
                components["robustness_consistency"] = float(consistency.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["gradient_clip"]))
            optimizer.step()
            model.eval()
            with torch.no_grad():
                val_outputs = model(val_graph)
                val_loss, val_components = joint_loss(val_outputs, val_graph, opt_cfg, flags)
            history.append({"epoch": epoch, "train_loss": float(loss.detach()), "validation_loss": float(val_loss), **{f"train_{k}": v for k, v in components.items()}, **{f"validation_{k}": v for k, v in val_components.items()}})
            if float(val_loss) < best_loss - 1e-7:
                best_loss = float(val_loss); best_state = deepcopy(model.state_dict()); stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("No model state was retained.")
        model.load_state_dict(best_state); model.eval()
        thresholds = _thresholds_closed_windows(model, validation_windows)
        node_frame, edge_frame, subgraph_frame, metrics = _predict_closed_windows(
            model, test_windows, thresholds
        )
        node_frame.to_csv(out / "node_predictions.csv", index=False)
        edge_frame.to_csv(out / "edge_predictions.csv", index=False)
        subgraph_frame.to_csv(out / "subgraph_predictions.csv", index=False)
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
        compactness_selection = []
        for fraction in config["explanation"]["candidate_retained_fractions"]:
            per_window = [
                evaluate_explanations(
                    model, graph, retained_fraction=float(fraction),
                    node_threshold=thresholds["node"], repeats=1,
                    latency_warmup=0, latency_repeats=1,
                )
                for _, graph in validation_windows
            ]
            finite = [row for row in per_window if np.isfinite(row["fidelity"])]
            if not finite:
                continue
            fidelity = float(np.mean([row["fidelity"] for row in finite]))
            completeness = float(np.mean([row["completeness"] for row in finite]))
            sparsity = float(np.mean([row["sparsity"] for row in finite]))
            compactness_selection.append({
                "retained_fraction": float(fraction), "validation_fidelity": fidelity,
                "validation_completeness": completeness, "validation_sparsity": sparsity,
                "selection_score": 0.45 * fidelity + 0.45 * completeness + 0.10 * sparsity,
                "windows_with_predicted_alerts": len(finite),
            })
        if not compactness_selection:
            raise RuntimeError("No validation window produced a predicted alert for compactness selection.")
        retained_fraction = float(max(
            compactness_selection, key=lambda row: (row["selection_score"], -row["retained_fraction"])
        )["retained_fraction"])
        explanation = _explanations_closed_windows(
            model, test_windows, thresholds["node"], retained_fraction, config["explanation"]
        )
        robustness_records = []
        for window_id, graph in test_windows:
            robustness_records.append({
                "evaluation_window": window_id,
                **empirical_edge_robustness(
                    model, graph, thresholds["node"], config["robustness"]["budgets"],
                    feature_epsilon=float(config["robustness"]["feature_epsilon"]),
                    explanation_retained_fraction=retained_fraction,
                ),
            })
        robustness = {
            "windows": robustness_records,
            "scope": "closed-window feature perturbation plus saliency-guided edge deletion; empirical, not certified",
            "edge_insertion_evaluated": False,
        }
        import scipy
        import sklearn
        import yaml
        runtime = {
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "pyyaml": getattr(yaml, "__version__", "unknown"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if device.startswith("cuda") else platform.processor(),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        resolved = {**config, "seed": seed, "device": device, "runtime": runtime, "ablation_flags": asdict(flags), "robustness_aware": robustness_aware,
                    "selected_explanation_retained_fraction": retained_fraction,
                    "evaluation_protocol": {
                        "mode": "non-overlapping closed windows",
                        "window_seconds": evaluation_window_seconds,
                        "future_context_after_window_close": False,
                    },
                    "builder_state": builder.state_dict(), "epochs_completed": len(history), "best_validation_loss": best_loss}
        (out / "config_resolved.json").write_text(json.dumps(_json_safe(resolved), indent=2), encoding="utf-8")
        (out / "metrics.json").write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
        (out / "explanation_metrics.json").write_text(json.dumps(_json_safe(explanation), indent=2), encoding="utf-8")
        (out / "compactness_validation.json").write_text(json.dumps(_json_safe(compactness_selection), indent=2), encoding="utf-8")
        (out / "robustness_metrics.json").write_text(json.dumps(_json_safe(robustness), indent=2), encoding="utf-8")
        torch.save({"model_state": model.state_dict(), "thresholds": thresholds, "config": resolved}, out / "checkpoint.pt")
        return out


    def run_ablations(events: str | Path, config: str | Path, output: str | Path, seeds: Iterable[int], device: Optional[str] = None, max_epochs: Optional[int] = None) -> Path:
        out = Path(output); out.mkdir(parents=True, exist_ok=True)
        records = []
        for name, flags in ABLATIONS.items():
            for seed in seeds:
                run_dir = out / name / f"seed_{seed}"
                required = [
                    run_dir / "metrics.json",
                    run_dir / "explanation_metrics.json",
                    run_dir / "config_resolved.json",
                ]
                if not all(path.exists() for path in required):
                    train_experiment(
                        events, config, run_dir, int(seed), flags=flags,
                        device=device, max_epochs=max_epochs,
                    )
                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                explanation = json.loads((run_dir / "explanation_metrics.json").read_text(encoding="utf-8"))
                records.append({
                    "configuration": name, "seed": int(seed), "node_f1": metrics["node"]["f1"],
                    "edge_auroc": metrics["edge"]["auroc"], "subgraph_f1": metrics["subgraph"]["f1"],
                    "fidelity": explanation["fidelity"], "stability": explanation["stability"],
                    "completeness": explanation["completeness"], "latency_seconds": explanation["latency_seconds"],
                })
        raw = pd.DataFrame(records); raw.to_csv(out / "ablation_seed_metrics.csv", index=False)
        summary = summarize_seed_metrics(raw, ["configuration"], ["node_f1", "edge_auroc", "subgraph_f1", "fidelity", "stability", "completeness", "latency_seconds"])
        summary.to_csv(out / "ablation_summary.csv", index=False)
        return out

    def benchmark_scalability(
        events: str | Path,
        checkpoint_path: str | Path,
        output: str | Path,
        edge_counts: Iterable[int],
        device: Optional[str] = None,
        warmup: int = 3,
        repeats: int = 10,
    ) -> Path:
        """Measure forward and explanation latency from an actual checkpoint."""
        device = _device(device)
        # This benchmark loads checkpoints produced locally by this repository.
        # PyTorch 2.6+ defaults ``weights_only`` to True, but the checkpoint also
        # contains trusted run metadata (including a TorchVersion object).
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        frame = EventGraphBuilder.load(events)
        builder = EventGraphBuilder(
            train_fraction=float(config["data"]["train_fraction"]),
            validation_fraction=float(config["data"]["validation_fraction"]),
            feature_prefix=str(config["data"]["feature_prefix"]),
            history_windows_seconds=tuple(config["data"]["history_windows_seconds"]),
        )
        train_frame, _, _ = builder.split(frame)
        builder.fit(train_frame); builder.fit_scaler(train_frame)
        probe = builder.build(frame.iloc[:max(2, min(len(frame), max(int(value) for value in edge_counts)))].copy())
        model_cfg = config["model"]
        model = XTHGNN(
            node_width=probe.node_features.shape[1], edge_width=probe.edge_features.shape[1],
            num_relations=len(builder.relation_map), num_campaigns=max(1, len(builder.campaign_map)),
            embedding_dim=int(model_cfg["embedding_dim"]), num_layers=int(model_cfg["num_layers"]),
            num_prototypes=int(model_cfg["num_prototypes"]), temperature=float(model_cfg["temperature"]),
            dropout=float(model_cfg["dropout"]), temporal_decay_init=float(model_cfg["temporal_decay_init"]),
            neighbor_sample_size=int(model_cfg["neighbor_sample_size"]),
            flags=AblationFlags(**config.get("ablation_flags", {})),
        ).to(device)
        model.load_state_dict(checkpoint["model_state"]); model.eval()
        records = []
        for requested in sorted(set(int(value) for value in edge_counts)):
            if requested < 2 or requested > len(frame):
                continue
            graph = builder.build(frame.iloc[:requested].copy()).to_torch(device)
            for _ in range(max(0, warmup)):
                with torch.no_grad(): model(graph)
            timings = []
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(device))
            for _ in range(max(1, repeats)):
                _device_synchronize(graph.node_features); start = time.perf_counter()
                with torch.no_grad(): model(graph)
                _device_synchronize(graph.node_features); timings.append(time.perf_counter() - start)
            peak_gb = (
                float(torch.cuda.max_memory_allocated(torch.device(device)) / 1024 ** 3)
                if device.startswith("cuda") else float("nan")
            )
            records.append({
                "nodes": len(graph.node_ids), "edges": requested,
                "forward_latency_ms_median": 1000 * float(np.median(timings)),
                "forward_latency_ms_q1": 1000 * float(np.quantile(timings, 0.25)),
                "forward_latency_ms_q3": 1000 * float(np.quantile(timings, 0.75)),
                "peak_gpu_memory_gb": peak_gb, "device": device, "repeats": max(1, repeats),
            })
        if not records:
            raise ValueError("No requested edge count was between 2 and the number of available events.")
        out = Path(output); out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(out / "scalability_benchmark.csv", index=False)
        (out / "scalability_metadata.json").write_text(json.dumps({
            "checkpoint": str(Path(checkpoint_path).resolve()), "events": str(Path(events).resolve()),
            "warmup": int(warmup), "repeats": int(repeats),
            "scope": "forward-pass latency only; explanation latency is evaluated separately in seeded runs",
            "warning": "These are measured prefix values for this checkpoint and hardware; they do not establish accuracy beyond 5,000 edges.",
        }, indent=2), encoding="utf-8")
        return out

else:
    def _torch_required(*args, **kwargs):
        raise RuntimeError(
            "This command requires PyTorch. Install it with: pip install torch"
        )

    train_experiment = _torch_required
    run_ablations = _torch_required
    summarize_runs = _torch_required
    benchmark_scalability = _torch_required


DEFAULT_CONFIG = {
    "data": {
        "train_fraction": 0.80,
        "validation_fraction": 0.10,
        "test_fraction": 0.10,
        "feature_prefix": "feat_",
        "benign_campaign": "benign",
        "history_windows_seconds": [300, 900],
        "temporal_consistency_window_seconds": 500,
        "time_unit_semantics": "normalized timestamp units; retained-edge rank for MAGIC extracts",
        "max_temporal_pairs": 32,
    },
    "model": {
        "embedding_dim": 24,
        "num_layers": 2,
        "num_prototypes": 8,
        "temperature": 0.07,
        "dropout": 0.20,
        "temporal_decay_init": 0.50,
        "neighbor_sample_size": 8,
    },
    "optimization": {
        "learning_rate": 0.003,
        "weight_decay": 0.00001,
        "epochs": 8,
        "early_stopping_patience": 3,
        "gradient_clip": 5.0,
        "batch_size": 1024,
        "node_loss_weight": 1.0,
        "edge_loss_weight": 1.0,
        "link_contrastive_weight": 1.0,
        "subgraph_loss_weight": 1.0,
        "explanation_loss_weight": 0.10,
        "sparsity_coefficient": 0.01,
        "prototype_coefficient": 1.0,
        "entropy_coefficient": 1.0,
        "importance_alignment_coefficient": 1.0,
        "temporal_consistency_weight": 0.05,
        "cross_layer_consistency_weight": 0.02,
        "robustness_regularization": 0.10,
    },
    "explanation": {
        "retained_fraction": 0.10,
        "candidate_retained_fractions": [0.10, 0.20, 0.30, 0.377, 0.40, 0.50],
        "perturbation_noise": 0.02,
        "stability_repeats": 2,
        "latency_warmup": 1,
        "latency_repeats": 2,
    },
    "evaluation": {
        "closed_window_seconds": 500,
        "unit": "node-window, event edge, and binary evaluation subgraph/window",
    },
    "robustness": {
        "budgets": [0.05, 0.10],
        "feature_epsilon": 0.01,
        "training_edge_budget": 0.05,
    },
}

MANUSCRIPT_PROTOCOL = {
    "protocol_version": "X-THGNN label-provenance-audited evaluation 3.1",
    "extraction_seed": 20260821,
    "training_seeds": [11, 22, 33, 44, 55],
    "events_per_extract": 5000,
    "default_device": "cpu",
    "task_scope": (
        "primary binary entity/node detection with secondary endpoint-derived "
        "event-edge and evaluation-subgraph/window outcomes; no attack-stage claim"
    ),
    "ground_truth": {
        "scheme": "ThreaTrace-expanded anomaly-entity UUID labels used by MAGIC",
        "audit_scope": (
            "file integrity, UUID syntax, counts, duplicates, and canonical hashes; "
            "not independent semantic adjudication"
        ),
        "semantic_ground_truth_verified": False,
        "semantic_caveat": (
            "source labels may include context-expanded neighbouring entities; "
            "interpret them as benchmark anomaly/entity labels"
        ),
        "normalization": "uppercase, unique, lexicographically sorted, LF-terminated",
        "darpa_cadets": {
            "unique_entity_uuids": 12858,
            "duplicate_rows": 0,
            "canonical_sha256": "adc7a99725450e0c4f2340ad985f498b0e7759be4940eded6cb6a953cea44743",
        },
        "darpa_theia": {
            "unique_entity_uuids": 25358,
            "duplicate_rows": 5,
            "canonical_sha256": "d8c61ef52d0f85a60abb8ea3b5d3ea30848d64655ed7a68d81939c08ea201ec2",
        },
        "darpa_trace": {
            "unique_entity_uuids": 68172,
            "duplicate_rows": 93,
            "canonical_sha256": "99df975021da2c27c2621daa6fcab82b4d83c1125319a99ddb0d394ea2a57c5b",
        },
        "label_boundary": (
            "entity labels are direct source-file UUID membership; event-edge and group labels "
            "are derived secondary outcomes and are neither independently adjudicated attack "
            "events nor attack-stage annotations"
        ),
    },
    "config": DEFAULT_CONFIG,
}


def validate_config(config):
    """Validate configuration ranges before allocating a model."""
    required = ["data", "model", "optimization", "explanation", "evaluation", "robustness"]
    for section in required:
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    train_fraction = float(config["data"]["train_fraction"])
    validation_fraction = float(config["data"]["validation_fraction"])
    test_fraction = float(config["data"].get("test_fraction", 1.0 - train_fraction - validation_fraction))
    if min(train_fraction, validation_fraction, test_fraction) <= 0 or not np.isclose(
        train_fraction + validation_fraction + test_fraction, 1.0
    ):
        raise ValueError("Train, validation, and test fractions must be positive and sum to one.")
    for key in ["embedding_dim", "num_layers", "num_prototypes", "neighbor_sample_size"]:
        if int(config["model"][key]) <= 0:
            raise ValueError(f"model.{key} must be positive.")
    if int(config["optimization"]["epochs"]) <= 0:
        raise ValueError("optimization.epochs must be positive.")
    if int(config["optimization"]["early_stopping_patience"]) <= 0:
        raise ValueError("optimization.early_stopping_patience must be positive.")
    if int(config["optimization"]["batch_size"]) <= 0:
        raise ValueError("optimization.batch_size must be positive.")
    if int(config["data"]["temporal_consistency_window_seconds"]) <= 0:
        raise ValueError("data.temporal_consistency_window_seconds must be positive.")
    if int(config["data"]["max_temporal_pairs"]) <= 0:
        raise ValueError("data.max_temporal_pairs must be positive.")
    history_windows = [int(value) for value in config["data"]["history_windows_seconds"]]
    if not history_windows or any(value <= 0 for value in history_windows):
        raise ValueError("data.history_windows_seconds must contain positive values.")
    retained = [float(value) for value in config["explanation"]["candidate_retained_fractions"]]
    if not retained or any(value <= 0 or value >= 1 for value in retained):
        raise ValueError("Explanation retained fractions must lie strictly between zero and one.")
    if int(config["explanation"]["stability_repeats"]) <= 0 or int(config["explanation"]["latency_repeats"]) <= 0:
        raise ValueError("Explanation stability and latency repeats must be positive.")
    if int(config["evaluation"]["closed_window_seconds"]) <= 0:
        raise ValueError("evaluation.closed_window_seconds must be positive.")
    budgets = [float(value) for value in config["robustness"]["budgets"]]
    if not budgets or any(value <= 0 or value >= 1 for value in budgets):
        raise ValueError("Robustness budgets must lie strictly between zero and one.")
    if float(config["robustness"]["feature_epsilon"]) <= 0:
        raise ValueError("robustness.feature_epsilon must be positive.")
    if not 0 < float(config["robustness"]["training_edge_budget"]) < 1:
        raise ValueError("robustness.training_edge_budget must lie strictly between zero and one.")
    return config


def load_config(path=None):
    """Load YAML configuration, or return the manuscript defaults."""
    from copy import deepcopy

    if path is None:
        return validate_config(deepcopy(DEFAULT_CONFIG))
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading --config requires PyYAML: pip install pyyaml") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    # ``manuscript-config`` writes an auditable protocol envelope.  Accept that
    # file directly as ``--config`` while retaining compatibility with a bare
    # configuration mapping.
    config = payload.get("config") if isinstance(payload, dict) and "config" in payload else payload
    return validate_config(config)


def write_manuscript_config(output: str | Path) -> Path:
    """Write the exact built-in manuscript protocol as an auditable YAML file."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Writing the manuscript configuration requires PyYAML.") from exc
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(MANUSCRIPT_PROTOCOL, sort_keys=False), encoding="utf-8")
    return out


def summarize_runs(patterns, output):
    """Summarize auditable seed directories, including explanations and robustness."""
    import glob

    directories = []
    for pattern in patterns:
        directories.extend(Path(path) for path in glob.glob(pattern))
    records = []
    for directory in sorted(set(directories)):
        metrics_path = directory / "metrics.json"
        config_path = directory / "config_resolved.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        explanation_path = directory / "explanation_metrics.json"
        robustness_path = directory / "robustness_metrics.json"
        manifest_path = directory / "dataset_manifest.json"
        explanation = json.loads(explanation_path.read_text(encoding="utf-8")) if explanation_path.exists() else {}
        robustness = json.loads(robustness_path.read_text(encoding="utf-8")) if robustness_path.exists() else {}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        robustness_cells = [
            cell
            for window in robustness.get("windows", [])
            for cell in window.get("budgets", {}).values()
        ]
        attack_success = [cell["attack_success"] for cell in robustness_cells if cell.get("attack_success") is not None]
        robustness_fidelity = [
            cell["explanation_fidelity"]
            for cell in robustness_cells
            if cell.get("explanation_fidelity") is not None
        ]
        records.append({
            "dataset": directory.parent.name,
            "run": str(directory),
            "seed": config["seed"],
            "dataset_sha256": manifest.get("sha256"),
            "rows": manifest.get("rows"),
            "node_f1": metrics["node"]["f1"],
            "node_auroc": metrics["node"]["auroc"],
            "node_auprc": metrics["node"]["auprc"],
            "node_brier": metrics["node"]["brier_score"],
            "node_false_alarms_per_window": metrics["node"]["false_alarms_per_window"],
            "edge_f1": metrics["edge"]["f1"],
            "edge_auroc": metrics["edge"]["auroc"],
            "edge_auprc": metrics["edge"]["auprc"],
            "edge_brier": metrics["edge"]["brier_score"],
            "subgraph_f1": metrics["subgraph"]["f1"],
            "explanation_fidelity": explanation.get("fidelity"),
            "explanation_stability": explanation.get("stability"),
            "explanation_sparsity": explanation.get("sparsity"),
            "explanation_completeness": explanation.get("completeness"),
            "explanation_latency_seconds": explanation.get("latency_seconds"),
            "robustness_attack_success": (
                float(np.mean(attack_success)) if attack_success else np.nan
            ),
            "robustness_explanation_fidelity": (
                float(np.mean(robustness_fidelity)) if robustness_fidelity else np.nan
            ),
        })
    if not records:
        raise ValueError("No valid run directories matched.")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(records)
    raw.to_csv(out / "seed_metrics.csv", index=False)
    metric_columns = [
        "node_f1", "node_auroc", "node_auprc", "node_brier", "node_false_alarms_per_window",
        "edge_f1", "edge_auroc", "edge_auprc", "edge_brier", "subgraph_f1",
        "explanation_fidelity", "explanation_stability", "explanation_sparsity",
        "explanation_completeness", "explanation_latency_seconds",
        "robustness_attack_success", "robustness_explanation_fidelity",
    ]
    summarize_seed_metrics(
        raw, ["dataset"], metric_columns
    ).to_csv(out / "seed_summary.csv", index=False)
    return out


def make_result_figures(seed_metrics: str | Path, output: str | Path) -> Path:
    """Generate manuscript Figures 2 and 3 from seed-level run metrics."""
    frame = pd.read_csv(seed_metrics)
    required = {
        "dataset", "node_f1", "edge_auroc", "explanation_fidelity",
        "explanation_latency_seconds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Seed metrics are missing columns: {sorted(missing)}")
    order = [name for name in ["darpa_cadets", "darpa_theia", "darpa_trace", "streamspot"] if name in set(frame["dataset"])]
    order.extend(sorted(set(frame["dataset"]) - set(order)))
    if not order:
        raise ValueError("No datasets were found in the seed-metrics file.")
    labels = {
        "darpa_cadets": "CADETS", "darpa_theia": "THEIA",
        "darpa_trace": "TRACE", "streamspot": "StreamSpot",
    }
    grouped = frame.groupby("dataset", sort=False)
    x = np.arange(len(order), dtype=float)
    width = 0.36
    node_mean = np.array([grouped.get_group(name)["node_f1"].mean() for name in order])
    node_sd = np.array([grouped.get_group(name)["node_f1"].std(ddof=1) for name in order])
    edge_mean = np.array([grouped.get_group(name)["edge_auroc"].mean() for name in order])
    edge_sd = np.array([grouped.get_group(name)["edge_auroc"].std(ddof=1) for name in order])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x - width / 2, node_mean, width, yerr=node_sd, capsize=3, label="Node F1", color="#2f73b7")
    ax.bar(x + width / 2, edge_mean, width, yerr=edge_sd, capsize=3, label="Edge AUROC", color="#d9a12b")
    ax.set_xticks(x, [labels.get(name, name) for name in order])
    ax.set_ylabel("Mean across five seeds")
    ax.set_ylim(0, max(1.05, float(np.nanmax(np.r_[node_mean + np.nan_to_num(node_sd), edge_mean + np.nan_to_num(edge_sd)])) * 1.08))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figure2_detection_variability.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    latency_ms = np.array([1000.0 * grouped.get_group(name)["explanation_latency_seconds"].mean() for name in order])
    fidelity = np.array([grouped.get_group(name)["explanation_fidelity"].mean() for name in order])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.scatter(latency_ms, fidelity, s=70, color="#2d8b64")
    for name, x_value, y_value in zip(order, latency_ms, fidelity):
        ax.annotate(labels.get(name, name), (x_value, y_value), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Mean explanation latency (ms)")
    ax.set_ylabel("Mean deletion fidelity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "figure3_latency_fidelity.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "source": str(Path(seed_metrics).resolve()),
        "source_sha256": hashlib.sha256(Path(seed_metrics).read_bytes()).hexdigest(),
        "figures": ["figure2_detection_variability.png", "figure3_latency_fidelity.png"],
        "definition": "figures generated directly from seed_metrics.csv without manuscript-value constants",
    }
    (out / "figure_generation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def build_parser():
    parser = argparse.ArgumentParser(description="Reproduce and audit the X-THGNN manuscript")
    parser.add_argument("--version", action="version", version="X-THGNN reproduction 3.1-label-provenance-audited")
    sub = parser.add_subparsers(dest="command", required=True)

    registry_info = sub.add_parser("datasets", help="Print the public dataset registry used by the revised manuscript")
    registry_info.add_argument("--output", help="Optional JSON output path")

    ground_truth = sub.add_parser(
        "audit-ground-truth",
        help="Integrity-audit and hash the CADETS, THEIA, and TRACE entity-label UUID files",
    )
    ground_truth.add_argument("--cadets", required=True)
    ground_truth.add_argument("--theia", required=True)
    ground_truth.add_argument("--trace", required=True)
    ground_truth.add_argument("--output", required=True)

    lanl = sub.add_parser("normalize-lanl", help="Normalize official LANL authentication and red-team logs")
    lanl.add_argument("--auth", required=True, help="auth.txt or auth.txt.gz")
    lanl.add_argument("--redteam", required=True, help="redteam.txt or redteam.txt.gz")
    lanl.add_argument("--output", required=True, help="Normalized CSV output")

    tc = sub.add_parser("normalize-tc", help="Normalize DARPA TC CADETS, THEIA, or TRACE CDM JSONL")
    tc.add_argument("--inputs", nargs="+", required=True, help="CDM JSONL/JSONL.GZ shards")
    tc.add_argument("--labels", required=True, help="Explicit event_uuid,label,campaign_id mapping CSV")
    tc.add_argument("--dataset", required=True, choices=["darpa_cadets", "darpa_theia", "darpa_trace"])
    tc.add_argument("--output", required=True, help="Normalized CSV output")

    tc_ground_truth = sub.add_parser(
        "normalize-tc-ground-truth",
        help="Normalize raw DARPA CDM JSONL using a ThreaTrace entity-label UUID file",
    )
    tc_ground_truth.add_argument("--inputs", nargs="+", required=True, help="CDM JSONL/JSONL.GZ shards")
    tc_ground_truth.add_argument("--ground-truth", required=True, help="ThreaTrace-style entity UUID text file")
    tc_ground_truth.add_argument("--dataset", required=True, choices=["darpa_cadets", "darpa_theia", "darpa_trace"])
    tc_ground_truth.add_argument("--output", required=True, help="Normalized CSV output")

    magic = sub.add_parser("normalize-magic", help="Normalize public MAGIC/StreamSpot preprocessed graph extracts")
    magic.add_argument("--dataset", required=True, choices=["darpa_cadets", "darpa_theia", "darpa_trace", "streamspot"])
    magic.add_argument("--graph", required=True, help="Preprocessed DGL graph pickle")
    magic.add_argument("--metadata", help="MAGIC metadata.json; required for DARPA datasets")
    magic.add_argument("--ground-truth", help="Original ThreaTrace entity-label UUID file for integrity/provenance audit")
    magic.add_argument("--module-root", help="Repository root needed to resolve classes stored in StreamSpot pickle")
    magic.add_argument("--events", type=int, default=5000, help="Extracted event count; divisible by 10")
    magic.add_argument("--seed", type=int, default=20260821, help="Fixed extraction seed")
    magic.add_argument("--output", required=True, help="Normalized CSV output")

    validate = sub.add_parser("validate", help="Validate and summarize a normalized event file")
    validate.add_argument("--events", required=True)
    validate.add_argument("--output")
    validate.add_argument("--train-fraction", type=float, default=0.8)
    validate.add_argument("--validation-fraction", type=float, default=0.1)

    train = sub.add_parser("train", help="Train and evaluate X-THGNN")
    train.add_argument("--events", required=True)
    train.add_argument("--config", help="Optional YAML configuration; built-in manuscript defaults are used if omitted")
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device")
    train.add_argument("--max-epochs", type=int)
    train.add_argument("--robustness-aware", action="store_true")

    ablate = sub.add_parser("ablate", help="Run the manuscript component ablations")
    ablate.add_argument("--events", required=True)
    ablate.add_argument("--config", help="Optional YAML configuration; built-in manuscript defaults are used if omitted")
    ablate.add_argument("--output", required=True)
    ablate.add_argument("--seeds", nargs="+", type=int, required=True)
    ablate.add_argument("--device")
    ablate.add_argument("--max-epochs", type=int)

    baseline = sub.add_parser("baselines", help="Run executable classical sanity baselines")
    baseline.add_argument("--events", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--seeds", nargs="+", type=int, required=True)
    baseline.add_argument("--train-fraction", type=float, default=0.8)
    baseline.add_argument("--validation-fraction", type=float, default=0.1)

    compare = sub.add_parser("compare-seeds", help="Compare two matched seed-level result tables")
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--reference", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--candidate-name", default="X-THGNN")
    compare.add_argument("--reference-name", default="Reference")
    compare.add_argument("--candidate-model", help="Model value to select when the candidate CSV contains multiple models")
    compare.add_argument("--reference-model", help="Model value to select when the reference CSV contains multiple models")
    compare.add_argument("--permutations", type=int, default=20000)
    compare.add_argument("--seed", type=int, default=20260821)

    registry = sub.add_parser("audit-baselines", help="Validate named-baseline provenance and seed prediction files")
    registry.add_argument("--registry", required=True)
    registry.add_argument("--output", required=True)

    pilot = sub.add_parser("pilot", help="Run participant-level paired pilot analysis")
    pilot.add_argument("--pilot-csv", required=True)
    pilot.add_argument("--output", required=True)
    pilot.add_argument("--bootstrap", type=int, default=2000)
    pilot.add_argument("--seed", type=int, default=20260821)

    summarize = sub.add_parser("summarize-runs", help="Summarize seed-level output directories")
    summarize.add_argument("--runs", nargs="+", required=True)
    summarize.add_argument("--output", required=True)

    figures = sub.add_parser("make-figures", help="Generate manuscript result figures from seed_metrics.csv")
    figures.add_argument("--seed-metrics", required=True)
    figures.add_argument("--output", required=True)

    manuscript_config = sub.add_parser("manuscript-config", help="Write the exact built-in manuscript protocol")
    manuscript_config.add_argument("--output", required=True)

    benchmark = sub.add_parser("benchmark", help="Measure checkpoint scalability on real event prefixes")
    benchmark.add_argument("--events", required=True)
    benchmark.add_argument("--checkpoint", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--edge-counts", nargs="+", type=int, required=True)
    benchmark.add_argument("--device")
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--repeats", type=int, default=10)

    self_test = sub.add_parser("self-test", help="Run deterministic end-to-end diagnostics")
    self_test.add_argument("--output", required=True)
    return parser


def standalone_main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "datasets":
        payload = json.dumps(PUBLIC_DATASETS, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(payload, encoding="utf-8")
            path = Path(args.output)
        else:
            print(payload)
            return
    elif args.command == "audit-ground-truth":
        path = audit_ground_truth_files({
            "darpa_cadets": args.cadets,
            "darpa_theia": args.theia,
            "darpa_trace": args.trace,
        }, args.output)
    elif args.command == "normalize-lanl":
        path = normalize_lanl_auth(args.auth, args.redteam, args.output)
    elif args.command == "normalize-tc":
        path = normalize_darpa_tc_cdm(args.inputs, args.labels, args.output, args.dataset)
    elif args.command == "normalize-tc-ground-truth":
        path = normalize_darpa_tc_entity_ground_truth(
            args.inputs, args.ground_truth, args.output, args.dataset
        )
    elif args.command == "normalize-magic":
        path = normalize_magic_preprocessed(
            args.dataset, args.graph, args.output, metadata_path=args.metadata,
            ground_truth_path=args.ground_truth, module_root=args.module_root,
            total_events=args.events, extraction_seed=args.seed,
        )
    elif args.command == "validate":
        builder = EventGraphBuilder(args.train_fraction, args.validation_fraction)
        frame = builder.load(args.events)
        splits = builder.split(frame)
        builder.fit(splits[0])
        manifest = build_manifest(args.events, frame, builder, splits)
        payload = json.dumps(asdict(manifest), indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(payload, encoding="utf-8")
            path = Path(args.output)
        else:
            print(payload)
            return
    elif args.command == "train":
        path = train_experiment(
            args.events, args.config, args.output, args.seed, device=args.device,
            robustness_aware=args.robustness_aware, max_epochs=args.max_epochs,
        )
    elif args.command == "ablate":
        path = run_ablations(
            args.events, args.config, args.output, args.seeds,
            device=args.device, max_epochs=args.max_epochs,
        )
    elif args.command == "baselines":
        path = run_baselines(
            args.events, args.output, args.seeds,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
    elif args.command == "compare-seeds":
        path = compare_seed_tables(
            args.candidate, args.reference, args.output,
            candidate_name=args.candidate_name, reference_name=args.reference_name,
            candidate_model=args.candidate_model, reference_model=args.reference_model,
            permutations=args.permutations, seed=args.seed,
        )
    elif args.command == "audit-baselines":
        path = audit_baseline_registry(args.registry, args.output)
    elif args.command == "pilot":
        path = analyze_pilot(args.pilot_csv, args.output, args.bootstrap, args.seed)
    elif args.command == "summarize-runs":
        path = summarize_runs(args.runs, args.output)
    elif args.command == "make-figures":
        path = make_result_figures(args.seed_metrics, args.output)
    elif args.command == "manuscript-config":
        path = write_manuscript_config(args.output)
    elif args.command == "benchmark":
        path = benchmark_scalability(
            args.events, args.checkpoint, args.output, args.edge_counts,
            device=args.device, warmup=args.warmup, repeats=args.repeats,
        )
    elif args.command == "self-test":
        path = run_self_test(args.output)
    else:
        raise AssertionError(args.command)
    print(Path(path).resolve())


if __name__ == "__main__":
    standalone_main()
