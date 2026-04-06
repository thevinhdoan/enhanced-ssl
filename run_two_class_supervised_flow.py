#!/usr/bin/env python3
"""
Generate and run two-class supervised teacher configs from the uncommented commands
in scripts/clip/run_supervised.sh and scripts/dinov2/run_supervised.sh.

The flow is:
1. Read the selected supervised configs from the shell scripts.
2. Clone each config into a generated two-class variant.
3. Train each generated config with train.py.
4. Run eval_gen_list.py on just those generated configs.
5. Run eval.py to produce pseudo-label artifacts for those trained models.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

from ruamel.yaml import YAML


YAML_RT = YAML()


def default_python_exec() -> str:
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-class supervised teacher training + eval flow."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. dtd.")
    parser.add_argument(
        "--classes",
        type=int,
        nargs=2,
        default=[0, 1],
        metavar=("CLASS0", "CLASS1"),
        help="The two original class labels to keep.",
    )
    parser.add_argument(
        "--run-script",
        action="append",
        default=None,
        help="Supervised run script(s) to read selected configs from. Defaults to scripts/clip/run_supervised.sh and scripts/dinov2/run_supervised.sh.",
    )
    parser.add_argument(
        "--generated-config-root",
        type=Path,
        default=Path("config/generated/two_class_supervised"),
        help="Root for generated two-class supervised configs.",
    )
    parser.add_argument(
        "--generated-save-root",
        type=Path,
        default=Path("saved_models/two_class_supervised"),
        help="Root for generated two-class supervised checkpoints.",
    )
    parser.add_argument(
        "--eval-list-path",
        type=Path,
        default=Path("eval_list.two_class_supervised.pkl"),
        help="Path for the temporary eval list consumed by eval.py.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device passed to eval.py.")
    parser.add_argument("--python", type=str, default=default_python_exec())
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override for all generated teacher configs.")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs and print commands, but do not run training/eval.")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        data = YAML_RT.load(f)
    return data if data is not None else {}


def save_yaml(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        YAML_RT.dump(data, f)


def class_tag(classes: Sequence[int]) -> str:
    return "classes_" + "_".join([str(label) for label in classes])


def default_run_scripts() -> List[Path]:
    return [
        Path("scripts/clip/run_supervised.sh"),
        Path("scripts/dinov2/run_supervised.sh"),
    ]


def extract_config_path_from_command(line: str) -> Path:
    parts = shlex.split(line.strip())
    for idx, token in enumerate(parts):
        if token == "--c" and idx + 1 < len(parts):
            return Path(parts[idx + 1])
    raise ValueError(f"Could not find --c in line: {line}")


def discover_selected_configs(script_paths: Sequence[Path]) -> List[Path]:
    config_paths: List[Path] = []
    for script_path in script_paths:
        if not script_path.exists():
            raise FileNotFoundError(f"Run script not found: {script_path}")
        with script_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if not stripped.startswith("python "):
                    continue
                if "train.py" not in stripped or " --c " not in stripped:
                    continue
                config_paths.append(extract_config_path_from_command(stripped))
    deduped = []
    seen = set()
    for path in config_paths:
        path_key = str(path)
        if path_key in seen:
            continue
        seen.add(path_key)
        deduped.append(path)
    return deduped


def infer_per_class_label_count(config: Dict) -> int:
    num_labels = int(config["num_labels"])
    num_classes = int(config["num_classes"])
    if num_labels % num_classes != 0:
        raise ValueError(
            f"Cannot infer per-class labeled count from num_labels={num_labels} and num_classes={num_classes}."
        )
    return num_labels // num_classes


def config_relative_parent(config_path: Path) -> Path:
    try:
        return config_path.parent.relative_to(Path("config"))
    except ValueError:
        return config_path.parent


def build_generated_paths(
    base_config_path: Path,
    classes: Sequence[int],
    seed: int,
    generated_config_root: Path,
    generated_save_root: Path,
) -> tuple[Path, Path, Path]:
    rel_parent = config_relative_parent(base_config_path)
    variant_root = Path(class_tag(classes)) / f"seed_{seed}" / rel_parent
    generated_config_path = generated_config_root / variant_root / "config.yaml"
    generated_save_dir = generated_save_root / variant_root
    metadata_path = generated_config_path.parent / "metadata.json"
    return generated_config_path, generated_save_dir, metadata_path


def build_generated_config(
    base_config: Dict,
    classes: Sequence[int],
    generated_save_dir: Path,
    metadata_path: Path,
    seed: int,
) -> Dict:
    cfg = copy.deepcopy(base_config)
    per_class_label_count = infer_per_class_label_count(base_config)
    save_name = str(cfg.get("save_name", "log"))

    cfg["selected_classes"] = [int(label) for label in classes]
    cfg["remap_selected_classes"] = True
    cfg["num_classes"] = len(classes)
    cfg["num_labels"] = per_class_label_count * len(classes)
    cfg["seed"] = seed
    cfg["save_dir"] = str(generated_save_dir)
    cfg["load_path"] = str(generated_save_dir / save_name / "latest_model.pth")
    cfg["generated_two_class_supervised_meta"] = str(metadata_path)
    return cfg


def launch(command: Sequence[str], env: Dict[str, str], dry_run: bool) -> None:
    printable = " ".join([shlex.quote(token) for token in command])
    print(printable)
    if not dry_run:
        subprocess.run(command, env=env, check=True)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.lower()
    classes = list(args.classes)
    if len(set(classes)) != 2:
        raise ValueError("--classes must contain exactly two distinct labels.")

    run_scripts = [Path(path) for path in (args.run_script or default_run_scripts())]
    selected_config_paths = discover_selected_configs(run_scripts)
    if not selected_config_paths:
        raise ValueError("No uncommented supervised train.py commands were found in the selected run scripts.")

    generated_config_paths: List[Path] = []
    manifest_entries = []
    for base_config_path in selected_config_paths:
        if not base_config_path.exists():
            raise FileNotFoundError(f"Selected config not found: {base_config_path}")
        base_config = load_yaml(base_config_path)
        if str(base_config.get("dataset", "")).lower() != dataset:
            continue
        if str(base_config.get("algorithm", "")).lower() != "supervised":
            continue

        seed = int(args.seed if args.seed is not None else base_config.get("seed", 0))
        generated_config_path, generated_save_dir, metadata_path = build_generated_paths(
            base_config_path=base_config_path,
            classes=classes,
            seed=seed,
            generated_config_root=args.generated_config_root,
            generated_save_root=args.generated_save_root,
        )
        generated_config = build_generated_config(
            base_config=base_config,
            classes=classes,
            generated_save_dir=generated_save_dir,
            metadata_path=metadata_path,
            seed=seed,
        )
        save_yaml(generated_config_path, generated_config)

        metadata = {
            "base_config": str(base_config_path),
            "generated_config": str(generated_config_path),
            "generated_save_dir": str(generated_save_dir),
            "dataset": dataset,
            "classes": classes,
            "seed": seed,
            "num_classes": int(generated_config["num_classes"]),
            "num_labels": int(generated_config["num_labels"]),
            "load_path": generated_config["load_path"],
            "run_scripts": [str(path) for path in run_scripts],
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        generated_config_paths.append(generated_config_path)
        manifest_entries.append(metadata)

    if not generated_config_paths:
        raise ValueError(
            f"No uncommented supervised configs for dataset={dataset} were found in the selected run scripts."
        )

    manifest_root = args.generated_config_root / class_tag(classes) / f"seed_{manifest_entries[0]['seed']}"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2, sort_keys=True)

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("Selected generated teacher configs:")
    for generated_config_path in generated_config_paths:
        print(f"- {generated_config_path}")

    for generated_config_path in generated_config_paths:
        launch(
            [args.python, "train.py", "--c", str(generated_config_path)],
            env=env,
            dry_run=args.dry_run,
        )

    launch(
        [
            args.python,
            "eval_gen_list.py",
            "--config-list",
            *[str(path) for path in generated_config_paths],
            "--output",
            str(args.eval_list_path),
        ],
        env=env,
        dry_run=args.dry_run,
    )
    launch(
        [
            args.python,
            "eval.py",
            "--eval_list",
            str(args.eval_list_path),
            "--device",
            args.device,
        ],
        env=env,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
