#!/usr/bin/env python3
"""
Launch unlabeled-imbalance PET / v-PET experiments for DTD 3-shot DINOv2.

Why this exists:
- `train.py` re-applies the YAML after CLI parsing, so config values like
  `save_dir`, `resume`, and `load_path` cannot be reliably overridden from the
  command line.
- This launcher generates fresh configs per run, then calls `train.py --c ...`.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ruamel.yaml import YAML


YAML_RT = YAML()

BASE_CONFIGS: Tuple[Tuple[str, str, Path], ...] = (
    ("adaptformer", "pet", Path("config/adaptformer/pet-ensembled/dtd/3-shot/dinov2/config.yaml")),
    ("adaptformer", "v-pet", Path("config/adaptformer/pet-ensembled-across-nets/dtd/3-shot/dinov2/config.yaml")),
    ("lora", "pet", Path("config/lora/pet-ensembled/dtd/3-shot/dinov2/config.yaml")),
    ("lora", "v-pet", Path("config/lora/pet-ensembled-across-nets/dtd/3-shot/dinov2/config.yaml")),
)


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = YAML_RT.load(f)
    return data if data is not None else {}


def save_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        YAML_RT.dump(data, f)


def ratio_tag(value: float) -> str:
    tag = str(value)
    tag = tag.replace("-", "neg")
    tag = tag.replace(".", "p")
    return tag


def read_train_targets(dataset_root: Path, train_split: str) -> List[int]:
    targets: List[int] = []
    list_path = dataset_root / f"{train_split}.list"
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label = line.rsplit(" ", 1)
            targets.append(int(label))
    return targets


def read_labeled_indices(dataset_root: Path, num_labels: int, seed: int, train_split: str) -> List[int]:
    idx_path = dataset_root / "labeled_idx" / f"lb_labels{num_labels}_1_seed{seed}_{train_split}_idx.list"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing labeled split file: {idx_path}")

    labeled_indices: List[int] = []
    with idx_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            labeled_indices.append(int(line.split()[-1]))
    return labeled_indices


def infer_ulb_head_count(dataset_root: Path, num_labels: int, seed: int, train_split: str) -> int:
    train_targets = read_train_targets(dataset_root, train_split)
    labeled_indices = set(read_labeled_indices(dataset_root, num_labels, seed, train_split))

    remaining_per_class: Dict[int, int] = {}
    for idx, target in enumerate(train_targets):
        if idx in labeled_indices:
            continue
        remaining_per_class[target] = remaining_per_class.get(target, 0) + 1

    if not remaining_per_class:
        raise RuntimeError("Could not infer unlabeled class counts from the split.")
    return max(remaining_per_class.values())


def default_python_exec() -> str:
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def build_run_config(
    base_config: Dict,
    base_config_path: Path,
    save_dir: Path,
    ulb_imb_ratio: float,
    ulb_head_count: int,
    lambda_1: float,
    lambda_2: float,
    grouping_update_interval: int,
    uratio: int,
    log_partition_stats: bool,
) -> Dict:
    cfg = copy.deepcopy(base_config)

    cfg["save_dir"] = str(save_dir)
    cfg["save_name"] = "log"
    cfg["overwrite"] = True
    cfg["resume"] = False
    cfg["load_path"] = None

    cfg["ulb_imb_ratio"] = ulb_imb_ratio
    cfg["ulb_num_labels"] = ulb_head_count

    cfg["lambda_1"] = lambda_1
    cfg["lambda_2"] = lambda_2
    cfg["grouping_update_interval"] = grouping_update_interval
    cfg["uratio"] = uratio
    cfg["log_partition_stats"] = bool(log_partition_stats)

    cfg["generated_from"] = str(base_config_path)
    return cfg


def launch_command(command: Sequence[str], env: Dict[str, str], dry_run: bool) -> None:
    print(" ".join(command))
    if not dry_run:
        subprocess.run(command, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and launch AdaptFormer/LoRA PET and v-PET runs for unlabeled imbalance on DTD 3-shot DINOv2."
    )
    parser.add_argument("--ratios", type=float, nargs="+", required=True, help="Unlabeled imbalance ratios to run, e.g. 2 5 10.")
    parser.add_argument("--ulb-head-count", type=int, default=None, help="Head-class unlabeled count. Defaults to the max remaining per-class count from the existing DTD 3-shot split.")
    parser.add_argument("--generated-config-root", type=Path, default=Path("config/generated/ulb_imbalance_dtd_dinov2"))
    parser.add_argument("--save-root", type=Path, default=Path("saved_models/ulb_imbalance_dtd_dinov2"))
    parser.add_argument("--python", type=str, default=default_python_exec())
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--peft", type=str, choices=["adaptformer", "lora"], default=None, help="Run only one PEFT family.")
    parser.add_argument("--variant", type=str, choices=["pet", "v-pet"], default=None, help="Run only one PET variant.")
    parser.add_argument("--lambda-1", type=float, default=0.01)
    parser.add_argument("--lambda-2", type=float, default=0.1)
    parser.add_argument("--grouping-update-interval", type=int, default=5)
    parser.add_argument("--uratio", type=int, default=12)
    parser.add_argument("--no-log-partition-stats", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_cfg = load_yaml(BASE_CONFIGS[0][2])
    dataset = str(base_cfg["dataset"]).lower()
    train_split = str(base_cfg.get("train_split", "train"))
    dataset_root = Path(base_cfg.get("data_dir", "./data")) / "vtab" / dataset
    num_labels = int(base_cfg["num_labels"])
    seed = int(base_cfg["seed"])

    ulb_head_count = args.ulb_head_count
    if ulb_head_count is None:
        ulb_head_count = infer_ulb_head_count(dataset_root, num_labels, seed, train_split)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(f"Dataset root: {dataset_root}")
    print(f"Train split: {train_split}")
    print(f"Labeled seed / num_labels: {seed} / {num_labels}")
    print(f"Derived unlabeled head count: {ulb_head_count}")

    selected_configs = [
        (peft_name, variant_name, base_cfg_path)
        for peft_name, variant_name, base_cfg_path in BASE_CONFIGS
        if (args.peft is None or peft_name == args.peft)
        and (args.variant is None or variant_name == args.variant)
    ]

    if not selected_configs:
        raise ValueError("No runs selected. Check --peft and --variant.")

    print("Selected runs:")
    for peft_name, variant_name, _ in selected_configs:
        print(f"- {peft_name} / {variant_name}")

    for ulb_ratio in args.ratios:
        ratio_key = ratio_tag(ulb_ratio)
        for peft_name, variant_name, base_cfg_path in selected_configs:
            base_config = load_yaml(base_cfg_path)

            generated_cfg_path = (
                args.generated_config_root
                / peft_name
                / variant_name
                / f"ulb_imb_{ratio_key}"
                / "config.yaml"
            )
            save_dir = (
                args.save_root
                / peft_name
                / variant_name
                / dataset
                / "3-shot"
                / "dinov2"
                / f"ulb_imb_{ratio_key}"
            )

            run_config = build_run_config(
                base_config=base_config,
                base_config_path=base_cfg_path,
                save_dir=save_dir,
                ulb_imb_ratio=ulb_ratio,
                ulb_head_count=ulb_head_count,
                lambda_1=args.lambda_1,
                lambda_2=args.lambda_2,
                grouping_update_interval=args.grouping_update_interval,
                uratio=args.uratio,
                log_partition_stats=not args.no_log_partition_stats,
            )
            save_yaml(generated_cfg_path, run_config)

            command = [
                args.python,
                "train.py",
                "--c",
                str(generated_cfg_path),
            ]
            launch_command(command, env=env, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
