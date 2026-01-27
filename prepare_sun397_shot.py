#!/usr/bin/env python3
"""
Utility to generate SUN397 supervised configs for a new shot count and refresh the run scripts.

Steps performed (default is generate only; add --run to execute training/eval):
1) Clone the four SUN397 supervised template configs to the requested shot, scaling
   num_labels/num_train_iter/num_warmup_iter/num_log_iter/num_eval_iter and
   updating save_dir/load_path.
2) Rewrite scripts/clip/run_supervised.sh and scripts/dinov2/run_supervised.sh so
   the original commands are commented and the new shot-specific commands are appended.
3) Optionally run the refreshed .sh scripts, eval_gen_list.py, and eval.py.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

from ruamel.yaml import YAML

yaml = YAML()
yaml.indent(mapping=2, sequence=4, offset=2)

# SUN397 constants
NUM_CLASSES = 397
DATASET = "sun397"

# template configs to clone: (label, run_script_key, template_path)
TEMPLATES: List[Tuple[str, str, Path]] = [
    (
        "adaptformer_clip",
        "clip",
        Path("config/adaptformer/supervised/sun397/3-shot/clip/dpr-0/train-aug-strong/{'adapter_scaler': 0.1, 'adapter_bottleneck': 16}/lr-1e-3/config.yaml"),
    ),
    (
        "lora_clip",
        "clip",
        Path("config/lora/supervised/sun397/3-shot/clip/dpr-0/train-aug-strong/{'lora_bottleneck': 16}/lr-1e-3/config.yaml"),
    ),
    (
        "adaptformer_dinov2",
        "dinov2",
        Path("config/adaptformer/supervised/sun397/3-shot/dinov2/dpr-0.2/train-aug-strong/{'adapter_scaler': 0.1, 'adapter_bottleneck': 4}/lr-1e-3/config.yaml"),
    ),
    (
        "lora_dinov2",
        "dinov2",
        Path("config/lora/supervised/sun397/3-shot/dinov2/dpr-0/train-aug-strong/{'lora_bottleneck': 16}/lr-1e-3/config.yaml"),
    ),
]

RUN_SCRIPTS: Dict[str, Path] = {
    "clip": Path("scripts/clip/run_supervised.sh"),
    "dinov2": Path("scripts/dinov2/run_supervised.sh"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare SUN397 configs and run scripts for a new shot count.")
    p.add_argument("--shot", type=int, required=True, help="Shot count to generate (e.g., 16).")
    p.add_argument("--run", action="store_true", help="Also execute the run scripts, eval_gen_list.py, and eval.py.")
    return p.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open() as f:
        return yaml.load(f)


def dump_yaml(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.dump(obj, f)


def scaled_value(value: int, factor: float) -> int:
    # Use rounding to keep parity with 3->6-shot scaling in the repo.
    return int(math.floor(value * factor + 0.5))


def generate_config(template_path: Path, shot: int) -> Path:
    # Derive base shot from the template path segment "<shot>-shot".
    parts = template_path.parts
    shot_parts = [p for p in parts if p.endswith("-shot")]
    if not shot_parts:
        raise ValueError(f"Cannot find '*-shot' segment in {template_path}")
    base_shot = int(shot_parts[0].split("-")[0])
    factor = shot / base_shot

    data = load_yaml(template_path)

    # Update numeric fields.
    data["num_labels"] = NUM_CLASSES * shot
    for key in ["num_train_iter", "num_warmup_iter", "num_log_iter", "num_eval_iter"]:
        if key in data:
            data[key] = scaled_value(int(data[key]), factor)

    # Clean/save paths with new shot count.
    for key in ["save_dir", "load_path"]:
        if key in data and data[key]:
            updated = str(data[key]).replace(f"{base_shot}-shot", f"{shot}-shot")
            updated = updated.replace("\n", "").strip()
            data[key] = updated

    target_path = Path(str(template_path).replace(f"{base_shot}-shot", f"{shot}-shot"))
    dump_yaml(data, target_path)
    return target_path


def rewrite_run_script(path: Path, new_commands: List[str], shot: int) -> None:
    shebang = "#!/usr/bin/env bash"
    lines: List[str] = []
    if path.exists():
        existing = path.read_text().splitlines()
        if existing and existing[0].startswith("#!"):
            shebang = existing[0].rstrip()
            existing = existing[1:]
        # Comment out prior lines to preserve history.
        for line in existing:
            if line.lstrip().startswith("#") or not line.strip():
                lines.append(line.rstrip())
            else:
                lines.append(f"# {line.rstrip()}")

    banner = f"# Auto-generated for SUN397 {shot}-shot; previous lines preserved above."
    content = [shebang, banner] + lines + [""] + new_commands + [""]
    path.write_text("\n".join(content))


def main() -> None:
    args = parse_args()
    shot = args.shot

    generated_paths: Dict[str, List[str]] = {"clip": [], "dinov2": []}

    # Step 1: configs
    for label, run_key, template in TEMPLATES:
        new_path = generate_config(template, shot)
        generated_paths[run_key].append(str(new_path))
        print(f"[generated] {label}: {new_path}")

    # Step 2: run scripts
    clip_cmds = [
        f'python train.py --c "config/adaptformer/supervised/{DATASET}/{shot}-shot/clip/dpr-0/train-aug-strong/{{\'adapter_scaler\': 0.1, \'adapter_bottleneck\': 16}}/lr-1e-3/config.yaml"',
        f'python train.py --c "config/lora/supervised/{DATASET}/{shot}-shot/clip/dpr-0/train-aug-strong/{{\'lora_bottleneck\': 16}}/lr-1e-3/config.yaml"',
    ]
    dinov2_cmds = [
        f'python train.py --c "config/adaptformer/supervised/{DATASET}/{shot}-shot/dinov2/dpr-0.2/train-aug-strong/{{\'adapter_scaler\': 0.1, \'adapter_bottleneck\': 4}}/lr-1e-3/config.yaml"',
        f'python train.py --c "config/lora/supervised/{DATASET}/{shot}-shot/dinov2/dpr-0/train-aug-strong/{{\'lora_bottleneck\': 16}}/lr-1e-3/config.yaml"',
    ]

    rewrite_run_script(RUN_SCRIPTS["clip"], clip_cmds, shot)
    rewrite_run_script(RUN_SCRIPTS["dinov2"], dinov2_cmds, shot)
    print(f"[updated] run scripts for {shot}-shot")

    # Optional: run training + eval
    if args.run:
        import subprocess

        cmds = [
            "bash scripts/clip/run_supervised.sh",
            "bash scripts/dinov2/run_supervised.sh",
            "python3 eval_gen_list.py",
            "python3 eval.py",
        ]
        for cmd in cmds:
            print(f"[running] {cmd}")
            subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
