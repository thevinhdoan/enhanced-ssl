#!/usr/bin/env python3
"""
Walk a saved_models tree, collect cluster purity from partition_stats JSONs,
and pair them with the train/test accuracies from the matching log.txt.

Example:
    python scripts/summarize_partition_stats.py --root "saved_models/(v2)" > summary.csv
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def compute_cluster_purity(json_path: Path) -> Tuple[Optional[int], Optional[float]]:
    """Compute purity = sum(max class count per cluster) / total points."""
    with json_path.open("r") as f:
        data = json.load(f)

    clusters: List[dict] = data.get("clusters", [])
    total_points = 0
    max_sums = 0

    for cluster in clusters:
        counts: Dict[str, int] = cluster.get("class_counts_including_centroid") or {}
        if not counts:
            # fall back to assigned counts if needed
            counts = cluster.get("assigned_unlabeled_class_counts") or {}
        if not counts:
            continue

        size = cluster.get("size_including_centroid")
        if size is None:
            size = sum(counts.values())

        total_points += size
        max_sums += max(counts.values())

    purity = max_sums / total_points if total_points else None
    epoch = data.get("epoch")
    return epoch, purity


def parse_log_epochs(log_path: Path) -> Dict[int, Tuple[Optional[float], Optional[float]]]:
    """
    Parse log.txt and return mapping: epoch -> (train_acc, test_acc).
    Train acc is taken from train/acc_gt; test acc from test/top-1-acc.
    """
    epoch_re = re.compile(r"Epoch\s+(\d+)/")
    acc_re = re.compile(r"train/acc_gt:\s*([0-9.]+).*test/top-1-acc:\s*([0-9.]+)")

    epoch = None
    metrics: Dict[int, Tuple[Optional[float], Optional[float]]] = {}

    with log_path.open("r", errors="ignore") as f:
        for line in f:
            m_epoch = epoch_re.search(line)
            if m_epoch:
                epoch = int(m_epoch.group(1))
            m_acc = acc_re.search(line)
            if m_acc and epoch is not None:
                train_acc = float(m_acc.group(1))
                test_acc = float(m_acc.group(2))
                metrics[epoch] = (train_acc, test_acc)

    return metrics


def iter_partition_dirs(root: Path) -> Iterable[Path]:
    """Yield every partition_stats directory under root."""
    for path in root.rglob("partition_stats"):
        if path.is_dir():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize cluster purity + accuracies for PET runs.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Root folder to search (e.g., saved_models/(v2)).",
    )
    args = parser.parse_args()

    root: Path = args.root
    log_cache: Dict[Path, Dict[int, Tuple[Optional[float], Optional[float]]]] = {}

    # CSV header
    print("run_root,epoch,json_path,purity,train_acc,test_acc")

    for stats_dir in iter_partition_dirs(root):
        log_path = stats_dir.parent / "log.txt"
        if not log_path.exists():
            continue

        # cache log parsing per run
        if log_path not in log_cache:
            log_cache[log_path] = parse_log_epochs(log_path)
        epoch_metrics = log_cache[log_path]

        for json_file in sorted(stats_dir.glob("*.json")):
            epoch_from_json, purity = compute_cluster_purity(json_file)

            # fallback: try to parse epoch from filename, pattern ..._eXYZ.json
            if epoch_from_json is None:
                match = re.search(r"_e(\d+)", json_file.stem)
                if match:
                    epoch_from_json = int(match.group(1))

            train_acc, test_acc = (None, None)
            if epoch_from_json is not None:
                if epoch_from_json in epoch_metrics:
                    train_acc, test_acc = epoch_metrics[epoch_from_json]
                else:
                    # Fallback: if metrics are only logged on later epochs, use the
                    # nearest later epoch's metrics for this JSON.
                    later_epochs = sorted(e for e in epoch_metrics if e > epoch_from_json)
                    if later_epochs:
                        nearest = later_epochs[0]
                        train_acc, test_acc = epoch_metrics[nearest]

            run_root = stats_dir.parent
            print(
                f"{run_root},{epoch_from_json},{json_file},{purity},{train_acc},{test_acc}"
            )


if __name__ == "__main__":
    main()
