import argparse
import os
import pickle
from pathlib import Path

from ruamel.yaml import YAML


_yaml = YAML()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the config list consumed by eval.py."
    )
    parser.add_argument(
        "--config-list",
        nargs="*",
        default=None,
        help="Explicit config.yaml paths to consider. When unset, scan --config-root.",
    )
    parser.add_argument(
        "--config-root",
        action="append",
        default=None,
        help="Config root(s) to scan recursively. Defaults to ./config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_list.pkl",
        help="Output pickle path for the generated eval list.",
    )
    return parser.parse_args()


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        config = _yaml.load(f)
    return config if config is not None else {}


def discover_config_paths(args):
    if args.config_list:
        return [Path(path) for path in args.config_list]

    config_roots = args.config_root or ["./config"]
    config_paths = []
    for root in config_roots:
        root_path = Path(root)
        if root_path.is_file():
            config_paths.append(root_path)
            continue
        for current_root, _, files in os.walk(root_path):
            if "pet" in current_root:
                continue
            if "config.yaml" not in files:
                continue
            config_paths.append(Path(current_root) / "config.yaml")
    return sorted(set(config_paths))


if __name__ == "__main__":
    args = parse_args()

    eval_list = []
    for config_path in discover_config_paths(args):
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        config = load_config(config_path)
        load_path = config.get("load_path")
        if load_path and os.path.exists(load_path):
            eval_list.append((load_path, config))

    output_path = Path(args.output)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(eval_list, f)
    print(f"Saved {len(eval_list)} models to {output_path}")
