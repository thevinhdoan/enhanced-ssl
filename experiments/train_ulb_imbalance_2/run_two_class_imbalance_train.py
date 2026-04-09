#!/usr/bin/env python3
"""
Create a two-class VTAB-style dataset view with unlabeled-only imbalance and launch training.

The script works by:
- reading an existing training config as the base run definition
- filtering a VTAB dataset down to two classes
- keeping labeled training balanced across the two classes
- downsampling only the unlabeled portion of the second class by a requested
  class-0:class-1 ratio
- precomputing labeled / unlabeled index files for the filtered training split
- generating a new config that points `train.py` at the synthesized dataset root

This keeps the actual training stack unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from ruamel.yaml import YAML


YAML_RT = YAML()
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitEntry:
    rel_path: str
    label: int


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = YAML_RT.load(f)
    return data if data is not None else {}


def save_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        YAML_RT.dump(data, f)


def default_python_exec() -> str:
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def ratio_tag(value: float) -> str:
    tag = f"{value:g}"
    tag = tag.replace("-", "neg")
    tag = tag.replace(".", "p")
    return tag


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a two-class VTAB dataset view with unlabeled-only imbalance and launch train.py."
    )
    parser.add_argument("--base-config", type=Path, required=True, help="Base training config yaml.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset name. Defaults to the dataset in --base-config.")
    parser.add_argument("--classes", type=int, nargs=2, default=[0, 1], metavar=("CLASS0", "CLASS1"), help="Two class labels to keep.")
    parser.add_argument("--ratio", type=float, required=True, help="Keep unlabeled class 1 at roughly 1/ratio of unlabeled class 0.")
    parser.add_argument("--lb-head-count", type=int, default=None, help="Override the labeled count per kept class. Defaults to the per-class labeled count implied by the base config.")
    parser.add_argument("--min-tail-count", type=int, default=0, help="Minimum number of unlabeled class-1 examples to keep.")
    parser.add_argument("--generated-data-root", type=Path, default=Path("data/generated_two_class_imbalance"))
    parser.add_argument("--generated-config-root", type=Path, default=Path("config/generated/two_class_imbalance"))
    parser.add_argument("--save-root", type=Path, default=Path("saved_models/two_class_imbalance"))
    parser.add_argument("--python", type=str, default=default_python_exec())
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the base config seed for both split generation and training.")
    parser.add_argument("--lambda-1", type=float, default=None, help="Override lambda_1 in the generated training config.")
    parser.add_argument("--lambda-2", type=float, default=None, help="Override lambda_2 in the generated training config.")
    parser.add_argument("--grouping-update-interval", type=int, default=None, help="Override grouping_update_interval in the generated training config.")
    parser.add_argument("--uratio", type=int, default=None, help="Override uratio in the generated training config.")
    parser.add_argument("--partition-log-dir", type=str, default=None, help="Override partition_log_dir in the generated training config.")
    parser.add_argument("--log-partition-stats", dest="log_partition_stats", action="store_true", help="Enable partition statistic logging in the generated training config.")
    parser.add_argument("--no-log-partition-stats", dest="log_partition_stats", action="store_false", help="Disable partition statistic logging in the generated training config.")
    parser.set_defaults(log_partition_stats=None)
    parser.add_argument("--no-rescale-schedule", dest="rescale_schedule", action="store_false", help="Keep the original num_train_iter / num_eval_iter / num_log_iter / num_warmup_iter from the base config.")
    parser.set_defaults(rescale_schedule=True)
    parser.add_argument("--dry-run", action="store_true", help="Generate dataset/config, print the train command, but do not launch training.")
    return parser.parse_args()


def ensure_supported_ratio(ratio: float) -> float:
    if ratio < 1.0:
        raise ValueError("--ratio must be >= 1.0 so class 1 is not larger than class 0.")
    return ratio


def infer_labeled_head_count(base_cfg: Dict) -> int:
    lb_ratio = float(base_cfg.get("lb_imb_ratio", 1))
    num_labels = int(base_cfg["num_labels"])
    if lb_ratio != 1.0:
        return num_labels

    num_classes = int(base_cfg["num_classes"])
    if num_labels % num_classes != 0:
        raise ValueError(
            "Cannot infer per-class labeled count from the base config because "
            f"num_labels={num_labels} is not divisible by num_classes={num_classes}. "
            "Pass --lb-head-count explicitly."
        )
    return num_labels // num_classes


def source_dataset_root(base_cfg: Dict, dataset: str) -> Path:
    return Path(base_cfg.get("data_dir", "./data")) / "vtab" / dataset.lower()


def read_split(path: Path) -> List[SplitEntry]:
    entries: List[SplitEntry] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(" ", 1)
            entries.append(SplitEntry(rel_path=rel_path, label=int(label)))
    return entries


def count_by_label(entries: Iterable[SplitEntry]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for entry in entries:
        counts[entry.label] = counts.get(entry.label, 0) + 1
    return counts


def tail_count_for_ratio(head_count: int, ratio: float, min_tail_count: int) -> int:
    if head_count <= 0:
        return 0
    target = int(math.floor(head_count / ratio))
    if min_tail_count > 0:
        target = max(target, min_tail_count)
    return target


def max_feasible_head_count(
    max_available_head: int,
    available_tail: int,
    ratio: float,
    min_tail_count: int,
) -> int:
    low = 0
    high = max_available_head
    best = -1
    while low <= high:
        mid = (low + high) // 2
        tail_needed = tail_count_for_ratio(mid, ratio, min_tail_count)
        if tail_needed <= available_tail:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def sample_positions(
    positions: Sequence[int],
    count: int,
    seed: int,
) -> List[int]:
    if count < 0:
        raise ValueError(f"Cannot sample a negative number of positions: {count}")
    if count > len(positions):
        raise ValueError(f"Requested {count} items from a pool of {len(positions)}.")
    if count == len(positions):
        return list(positions)
    rng = random.Random(seed)
    shuffled = list(positions)
    rng.shuffle(shuffled)
    selected = sorted(shuffled[:count])
    return selected


def filter_positions_by_class(entries: Sequence[SplitEntry], target_label: int) -> List[int]:
    return [idx for idx, entry in enumerate(entries) if entry.label == target_label]


def build_train_subset(
    train_entries: Sequence[SplitEntry],
    class0: int,
    class1: int,
    ratio: float,
    lb_head_count: int,
    min_tail_count: int,
    seed: int,
) -> Tuple[List[SplitEntry], List[int], List[int], Dict[str, int]]:
    class0_positions = filter_positions_by_class(train_entries, class0)
    class1_positions = filter_positions_by_class(train_entries, class1)

    if not class0_positions:
        raise ValueError(f"Training split has no samples for class {class0}.")
    if not class1_positions:
        raise ValueError(f"Training split has no samples for class {class1}.")
    if lb_head_count > len(class0_positions):
        raise ValueError(
            f"Requested lb head count {lb_head_count}, but training split only has "
            f"{len(class0_positions)} samples for class {class0}."
        )
    if lb_head_count > len(class1_positions):
        raise ValueError(
            f"Requested lb head count {lb_head_count}, but training split only has "
            f"{len(class1_positions)} samples for class {class1}."
        )

    labeled_class0 = set(sample_positions(class0_positions, lb_head_count, seed + 11))
    labeled_class1 = set(sample_positions(class1_positions, lb_head_count, seed + 21))

    remaining_class0 = sorted(set(class0_positions) - labeled_class0)
    remaining_class1 = sorted(set(class1_positions) - labeled_class1)

    selected_unlabeled_class0_count = max_feasible_head_count(
        max_available_head=len(remaining_class0),
        available_tail=len(remaining_class1),
        ratio=ratio,
        min_tail_count=min_tail_count,
    )
    if selected_unlabeled_class0_count < 0:
        raise ValueError(
            f"Could not construct a feasible unlabeled split for classes {class0}/{class1} "
            f"with ratio={ratio} after reserving labeled data."
        )
    unlabeled_tail_count = tail_count_for_ratio(selected_unlabeled_class0_count, ratio, min_tail_count)

    selected_unlabeled_class1 = set(sample_positions(remaining_class1, unlabeled_tail_count, seed + 101))
    selected_unlabeled_class0 = set(sample_positions(remaining_class0, selected_unlabeled_class0_count, seed + 91))

    selected_positions = labeled_class0 | labeled_class1 | selected_unlabeled_class0 | selected_unlabeled_class1
    position_to_new_index: Dict[int, int] = {}
    new_entries: List[SplitEntry] = []
    for original_idx, entry in enumerate(train_entries):
        if original_idx not in selected_positions:
            continue
        position_to_new_index[original_idx] = len(new_entries)
        new_entries.append(entry)

    labeled_indices = [position_to_new_index[idx] for idx in sorted(labeled_class0 | labeled_class1)]
    unlabeled_indices = [position_to_new_index[idx] for idx in sorted(selected_unlabeled_class0 | selected_unlabeled_class1)]

    stats = {
        "train_class0_total_available": len(class0_positions),
        "train_class1_total_available": len(class1_positions),
        "labeled_class0": len(labeled_class0),
        "labeled_class1": len(labeled_class1),
        "unlabeled_class0_available_after_lb": len(remaining_class0),
        "unlabeled_class1_available_after_lb": len(remaining_class1),
        "unlabeled_only_class0": len(selected_unlabeled_class0),
        "unlabeled_only_class1": len(selected_unlabeled_class1),
        "selected_class0_total": len(labeled_class0) + len(selected_unlabeled_class0),
        "selected_class1_total": len(labeled_class1) + len(selected_unlabeled_class1),
    }

    return new_entries, labeled_indices, unlabeled_indices, stats


def build_eval_subset(
    entries: Sequence[SplitEntry],
    class0: int,
    class1: int,
) -> Tuple[List[SplitEntry], Dict[str, int]]:
    class0_positions = filter_positions_by_class(entries, class0)
    class1_positions = filter_positions_by_class(entries, class1)

    if not class0_positions:
        raise ValueError(f"Split has no samples for class {class0}.")
    if not class1_positions:
        raise ValueError(f"Split has no samples for class {class1}.")

    selected_positions = set(class0_positions) | set(class1_positions)
    filtered_entries = [entry for idx, entry in enumerate(entries) if idx in selected_positions]

    stats = {
        "class0_total": len(class0_positions),
        "class1_total": len(class1_positions),
    }
    return filtered_entries, stats


def write_split(path: Path, entries: Sequence[SplitEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(f"{entry.rel_path} {entry.label}\n")


def write_idx_list(path: Path, entries: Sequence[SplitEntry], indices: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in indices:
            entry = entries[idx]
            f.write(f"{entry.rel_path} {entry.label} {idx}\n")


def ensure_images_link(source_root: Path, generated_root: Path) -> None:
    source_images = source_root / "images"
    target_images = generated_root / "images"
    if target_images.is_symlink() or target_images.exists():
        return
    target_images.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(source_images.resolve(), target_images, target_is_directory=True)


def relative_config_parent(base_config_path: Path) -> Path:
    try:
        return base_config_path.parent.relative_to(Path("config"))
    except ValueError:
        return base_config_path.parent


def class_filter_tag(classes: Sequence[int], remap_selected_classes: bool = True) -> str:
    tag = "_".join([f"c{int(label)}" for label in classes])
    if remap_selected_classes:
        tag = f"{tag}_remap"
    return tag


def map_supervised_source_to_two_class(source_config_path: str, classes: Sequence[int]) -> str:
    source_path = Path(source_config_path)
    if not source_path.exists():
        raise FileNotFoundError(f"PET source config not found: {source_path}")

    source_cfg = load_yaml(source_path)
    try:
        rel_parent = source_path.parent.relative_to(Path("config"))
    except ValueError:
        rel_parent = source_path.parent

    source_seed = int(source_cfg.get("seed", 0))
    mapped_path = (
        Path("config/generated/two_class_supervised")
        / f"classes_{int(classes[0])}_{int(classes[1])}"
        / f"seed_{source_seed}"
        / rel_parent
        / "config.yaml"
    )
    if not mapped_path.exists():
        raise FileNotFoundError(
            f"Mapped two-class teacher config not found: {mapped_path}. "
            "Run run_two_class_supervised_flow.py for this class pair before PET/V-PET training."
        )
    return str(mapped_path)


def rewrite_pet_sources_for_two_class(base_cfg: Dict, classes: Sequence[int]) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    pet_sources = cfg.get("pet_sources")
    if not pet_sources:
        return cfg

    mapped_sources = [map_supervised_source_to_two_class(source, classes) for source in pet_sources]
    cfg["pet_sources"] = mapped_sources
    return cfg


def build_generated_config(
    base_cfg: Dict,
    dataset: str,
    classes: Sequence[int],
    data_dir: Path,
    save_dir: Path,
    ratio: float,
    lb_total_count: int,
    ulb_head_count: int,
    seed: int,
    train_split: str,
    metadata_path: Path,
    lambda_1: float | None,
    lambda_2: float | None,
    grouping_update_interval: int | None,
    uratio: int | None,
    log_partition_stats: bool | None,
    partition_log_dir: str | None,
    schedule_overrides: Dict[str, int],
) -> Dict:
    cfg = rewrite_pet_sources_for_two_class(base_cfg, classes)
    cfg["dataset"] = dataset
    cfg["data_dir"] = str(data_dir)
    cfg["save_dir"] = str(save_dir)
    cfg["save_name"] = "log"
    cfg["resume"] = False
    cfg["load_path"] = None
    cfg["overwrite"] = True
    cfg["seed"] = seed
    cfg["train_split"] = train_split
    cfg["selected_classes"] = [int(label) for label in classes]
    cfg["remap_selected_classes"] = True
    cfg["num_classes"] = len(classes)
    cfg["num_labels"] = lb_total_count
    cfg["lb_imb_ratio"] = 1
    cfg["ulb_num_labels"] = ulb_head_count
    cfg["ulb_imb_ratio"] = ratio
    cfg["include_lb_to_ulb"] = False
    if lambda_1 is not None:
        cfg["lambda_1"] = lambda_1
    if lambda_2 is not None:
        cfg["lambda_2"] = lambda_2
    if grouping_update_interval is not None:
        cfg["grouping_update_interval"] = grouping_update_interval
    if uratio is not None:
        cfg["uratio"] = uratio
    if log_partition_stats is not None:
        cfg["log_partition_stats"] = bool(log_partition_stats)
    if partition_log_dir is not None:
        cfg["partition_log_dir"] = partition_log_dir
    for key, value in schedule_overrides.items():
        cfg[key] = value
    cfg["generated_two_class_imbalance_meta"] = str(metadata_path)
    return cfg


def compute_schedule_overrides(
    base_cfg: Dict,
    original_train_size: int,
    reduced_train_size: int,
) -> Dict[str, int]:
    if reduced_train_size <= 0:
        raise ValueError("Reduced train size must be positive to compute a schedule.")

    base_epochs = int(base_cfg["epoch"])
    base_num_train_iter = int(base_cfg["num_train_iter"])
    base_num_eval_iter = int(base_cfg.get("num_eval_iter", 1))
    base_num_log_iter = int(base_cfg.get("num_log_iter", 1))
    base_num_warmup_iter = int(base_cfg.get("num_warmup_iter", 0))

    if base_epochs <= 0 or base_num_train_iter <= 0:
        raise ValueError("Base config must have positive epoch and num_train_iter.")

    base_steps_per_epoch = max(1, base_num_train_iter // base_epochs)
    scale = float(reduced_train_size) / float(max(original_train_size, 1))
    scaled_steps_per_epoch = max(1, int(round(base_steps_per_epoch * scale)))
    scaled_num_train_iter = scaled_steps_per_epoch * base_epochs
    iter_scale = float(scaled_num_train_iter) / float(base_num_train_iter)

    scaled_num_eval_iter = max(1, int(round(base_num_eval_iter * iter_scale)))
    scaled_num_log_iter = max(1, int(round(base_num_log_iter * iter_scale)))
    scaled_num_warmup_iter = max(0, int(round(base_num_warmup_iter * iter_scale)))

    scaled_num_eval_iter = min(scaled_num_eval_iter, scaled_num_train_iter)
    scaled_num_log_iter = min(scaled_num_log_iter, scaled_num_train_iter)
    scaled_num_warmup_iter = min(scaled_num_warmup_iter, scaled_num_train_iter)

    return {
        "num_train_iter": scaled_num_train_iter,
        "num_eval_iter": scaled_num_eval_iter,
        "num_log_iter": scaled_num_log_iter,
        "num_warmup_iter": scaled_num_warmup_iter,
        "reduced_steps_per_epoch": scaled_steps_per_epoch,
        "base_steps_per_epoch": base_steps_per_epoch,
    }


def launch_command(command: Sequence[str], env: Dict[str, str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, env=env, check=True)


def main() -> None:
    args = parse_args()
    ratio = ensure_supported_ratio(args.ratio)

    base_cfg = load_yaml(args.base_config)
    dataset = (args.dataset or str(base_cfg.get("dataset", ""))).lower()
    if not dataset:
        raise ValueError("Dataset must be provided either via --dataset or in --base-config.")
    if args.dataset is not None and str(base_cfg.get("dataset", "")).lower() != dataset:
        raise ValueError(
            f"--dataset={dataset} does not match the dataset in the base config "
            f"({base_cfg.get('dataset')}). Use a matching base config."
        )

    train_split = str(base_cfg.get("train_split", "train"))
    if train_split not in ("train", "trainval"):
        raise ValueError(f"Unsupported train_split={train_split}. Expected 'train' or 'trainval'.")

    class0, class1 = args.classes
    if class0 == class1:
        raise ValueError("--classes must contain two distinct labels.")

    seed = int(args.seed if args.seed is not None else base_cfg.get("seed", 0))
    lb_head_count = int(args.lb_head_count if args.lb_head_count is not None else infer_labeled_head_count(base_cfg))

    source_root = source_dataset_root(base_cfg, dataset)
    if not source_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {source_root}")
    required_splits = set(SPLITS) | {train_split}
    for split in sorted(required_splits):
        split_path = source_root / f"{split}.list"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing required split file: {split_path}")

    source_splits = {split: read_split(source_root / f"{split}.list") for split in required_splits}

    train_entries, labeled_indices, ulb_indices, train_stats = build_train_subset(
        train_entries=source_splits[train_split],
        class0=class0,
        class1=class1,
        ratio=ratio,
        lb_head_count=lb_head_count,
        min_tail_count=args.min_tail_count,
        seed=seed,
    )

    val_entries, val_stats = build_eval_subset(
        entries=source_splits["val"],
        class0=class0,
        class1=class1,
    )
    test_entries, test_stats = build_eval_subset(
        entries=source_splits["test"],
        class0=class0,
        class1=class1,
    )

    head_unlabeled_only = train_stats["unlabeled_only_class0"]
    if head_unlabeled_only < 0:
        raise ValueError("Computed a negative unlabeled head count, which indicates an invalid split.")

    variant_tag = f"classes_{class0}_{class1}/ratio_{ratio_tag(ratio)}/seed_{seed}"
    generated_data_dir = args.generated_data_root / dataset / variant_tag
    dataset_root = generated_data_dir / "vtab" / dataset
    ensure_images_link(source_root, dataset_root)

    write_split(dataset_root / f"{train_split}.list", train_entries)
    if train_split != "train":
        write_split(dataset_root / "train.list", train_entries)
    write_split(dataset_root / "val.list", val_entries)
    write_split(dataset_root / "test.list", test_entries)
    write_split(dataset_root / "trainval.list", list(train_entries) + list(val_entries))

    lb_total_count = len(labeled_indices)
    labeled_idx_dir = dataset_root / "labeled_idx"
    lb_idx_path = labeled_idx_dir / f"lb_labels{lb_total_count}_1_seed{seed}_{train_split}_idx.list"
    ulb_idx_path = labeled_idx_dir / f"ulb_labels{lb_total_count}_{ratio}_seed{seed}_{train_split}_idx.list"
    class_appendix = f"{train_split}_{class_filter_tag([class0, class1])}_"
    lb_idx_path_class = labeled_idx_dir / f"lb_labels{lb_total_count}_1_seed{seed}_{class_appendix}idx.list"
    ulb_idx_path_class = labeled_idx_dir / f"ulb_labels{lb_total_count}_{ratio}_seed{seed}_{class_appendix}idx.list"
    write_idx_list(lb_idx_path, train_entries, labeled_indices)
    write_idx_list(ulb_idx_path, train_entries, ulb_indices)
    write_idx_list(lb_idx_path_class, train_entries, labeled_indices)
    write_idx_list(ulb_idx_path_class, train_entries, ulb_indices)

    rel_cfg_parent = relative_config_parent(args.base_config)
    generated_cfg_path = (
        args.generated_config_root
        / rel_cfg_parent
        / f"classes_{class0}_{class1}"
        / f"ratio_{ratio_tag(ratio)}"
        / f"seed_{seed}"
        / "config.yaml"
    )
    save_dir = (
        args.save_root
        / rel_cfg_parent
        / f"classes_{class0}_{class1}"
        / f"ratio_{ratio_tag(ratio)}"
        / f"seed_{seed}"
    )
    metadata_path = generated_cfg_path.parent / "metadata.json"
    if args.rescale_schedule:
        schedule_info = compute_schedule_overrides(
            base_cfg=base_cfg,
            original_train_size=len(source_splits[train_split]),
            reduced_train_size=len(train_entries),
        )
        schedule_overrides = {
            "num_train_iter": schedule_info["num_train_iter"],
            "num_eval_iter": schedule_info["num_eval_iter"],
            "num_log_iter": schedule_info["num_log_iter"],
            "num_warmup_iter": schedule_info["num_warmup_iter"],
        }
    else:
        schedule_info = {
            "num_train_iter": int(base_cfg["num_train_iter"]),
            "num_eval_iter": int(base_cfg.get("num_eval_iter", 1)),
            "num_log_iter": int(base_cfg.get("num_log_iter", 1)),
            "num_warmup_iter": int(base_cfg.get("num_warmup_iter", 0)),
            "base_steps_per_epoch": int(base_cfg["num_train_iter"]) // int(base_cfg["epoch"]),
            "reduced_steps_per_epoch": int(base_cfg["num_train_iter"]) // int(base_cfg["epoch"]),
        }
        schedule_overrides = {}
    metadata = {
        "base_config": str(args.base_config),
        "dataset": dataset,
        "source_root": str(source_root),
        "generated_dataset_root": str(dataset_root),
        "classes": [class0, class1],
        "ratio": ratio,
        "seed": seed,
        "train_split": train_split,
        "lb_head_count": lb_head_count,
        "lb_total_count": lb_total_count,
        "ulb_head_count": head_unlabeled_only,
        "include_lb_to_ulb": False,
        "training_overrides": {
            "lambda_1": args.lambda_1,
            "lambda_2": args.lambda_2,
            "grouping_update_interval": args.grouping_update_interval,
            "uratio": args.uratio,
            "log_partition_stats": args.log_partition_stats,
            "partition_log_dir": args.partition_log_dir,
        },
        "schedule": {
            "rescaled": bool(args.rescale_schedule),
            "original_train_size": len(source_splits[train_split]),
            "reduced_train_size": len(train_entries),
            "base_num_train_iter": int(base_cfg["num_train_iter"]),
            "base_num_eval_iter": int(base_cfg.get("num_eval_iter", 1)),
            "base_num_log_iter": int(base_cfg.get("num_log_iter", 1)),
            "base_num_warmup_iter": int(base_cfg.get("num_warmup_iter", 0)),
            "base_steps_per_epoch": int(schedule_info["base_steps_per_epoch"]),
            "num_train_iter": int(schedule_info["num_train_iter"]),
            "num_eval_iter": int(schedule_info["num_eval_iter"]),
            "num_log_iter": int(schedule_info["num_log_iter"]),
            "num_warmup_iter": int(schedule_info["num_warmup_iter"]),
            "steps_per_epoch": int(schedule_info["reduced_steps_per_epoch"]),
        },
        "counts": {
            "train": train_stats,
            "val": val_stats,
            "test": test_stats,
            "actual_labeled_total": len(labeled_indices),
            "actual_unlabeled_only_total": len(ulb_indices),
            "actual_unlabeled_with_lb_total": len(ulb_indices) + len(labeled_indices),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    generated_cfg = build_generated_config(
        base_cfg=base_cfg,
        dataset=dataset,
        classes=[class0, class1],
        data_dir=generated_data_dir,
        save_dir=save_dir,
        ratio=ratio,
        lb_total_count=lb_total_count,
        ulb_head_count=head_unlabeled_only,
        seed=seed,
        train_split=train_split,
        metadata_path=metadata_path,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        grouping_update_interval=args.grouping_update_interval,
        uratio=args.uratio,
        log_partition_stats=args.log_partition_stats,
        partition_log_dir=args.partition_log_dir,
        schedule_overrides=schedule_overrides,
    )
    save_yaml(generated_cfg_path, generated_cfg)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"Source dataset root: {source_root}")
    print(f"Generated dataset root: {dataset_root}")
    print(f"Generated config: {generated_cfg_path}")
    print(f"Counts: {json.dumps(metadata['counts'], indent=2)}")

    command = [args.python, "train.py", "--c", str(generated_cfg_path)]
    launch_command(command, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
