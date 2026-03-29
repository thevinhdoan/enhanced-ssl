import argparse
import csv
import math
import os
import pickle
import zipfile

import torch
import torch.nn.functional as F

from infer_corrupted import (
    _YAML,
    _apply_overrides,
    _build_model,
    _build_test_loader,
    _denorm_imagenet,
    _infer_peft_config_from_state_dict,
    _infer_timm_net_from_state_dict,
    _load_checkpoint,
    _load_maybe_yaml,
    _load_yaml,
    _looks_normalized,
    _merge_peft_config,
    _normalize_net_name,
    _pick_checkpoint,
    _renorm_imagenet,
    _resolve_path,
    _str2bool,
    _validate_cfg,
    run_inference,
)


_ROOT_DIR = os.path.abspath(os.curdir)
_SUPPORTED_ATTACKS = ("pgd", "square")


def _parse_attack_list(attack_list):
    if attack_list:
        attacks = [attack.strip().lower() for attack in attack_list.split(",") if attack.strip()]
    else:
        attacks = ["pgd"]
    invalid = [attack for attack in attacks if attack not in _SUPPORTED_ATTACKS]
    if invalid:
        raise ValueError(f"Unsupported attacks: {', '.join(invalid)}. Supported: {', '.join(_SUPPORTED_ATTACKS)}")
    return attacks


def _parse_float_list(values):
    if values is None:
        return [8.0 / 255.0]
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def _find_data_zip(data_zip):
    if data_zip:
        return _resolve_path(data_zip)
    for candidate in ["data.zip", "data_vtab.zip", "vtab.zip"]:
        path = _resolve_path(candidate)
        if os.path.exists(path):
            return path
    return None


def _extract_data_zip(zip_path, data_dir):
    with zipfile.ZipFile(zip_path) as zf:
        roots = {
            name.split("/")[0]
            for name in zf.namelist()
            if name and not name.endswith("/")
        }
        if "data" in roots:
            extract_root = os.path.dirname(data_dir) or "."
        elif "vtab" in roots:
            extract_root = data_dir
        else:
            extract_root = data_dir
        os.makedirs(extract_root, exist_ok=True)
        zf.extractall(extract_root)
    if "data" in roots:
        return os.path.join(extract_root, "data")
    return data_dir


def _expected_vtab_list(data_dir, dataset, split):
    return os.path.join(data_dir, "vtab", dataset.lower(), f"{split}.list")


def _ensure_data_available(data_dir, dataset, data_zip=None):
    expected = _expected_vtab_list(data_dir, dataset, "test")
    if os.path.exists(expected):
        return data_dir

    base = os.path.basename(os.path.normpath(data_dir))
    for subdir in ("data", "vtab", "vtab_release"):
        if base == subdir:
            continue
        candidate = os.path.join(data_dir, subdir)
        expected = _expected_vtab_list(candidate, dataset, "test")
        if os.path.exists(expected):
            return candidate

    zip_path = _find_data_zip(data_zip)
    if zip_path and os.path.exists(zip_path):
        data_dir = _extract_data_zip(zip_path, data_dir)
        expected = _expected_vtab_list(data_dir, dataset, "test")
        if os.path.exists(expected):
            return data_dir
        for subdir in ("data", "vtab", "vtab_release"):
            candidate = os.path.join(data_dir, subdir)
            expected = _expected_vtab_list(candidate, dataset, "test")
            if os.path.exists(expected):
                return candidate

    raise FileNotFoundError(
        f"VTAB test list not found at {expected}. Provide --data_dir or --data_zip."
    )


def _float_key(value):
    if value is None:
        return ""
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return ""
    return f"{float(value):.10g}"


def _attack_key(dataset, net, checkpoint, attack, norm, eps, pgd_steps, pgd_step_size, square_queries, square_p_init, square_seed):
    return (
        str(dataset or "").strip(),
        str(net or "").strip(),
        str(checkpoint or "").strip(),
        str(attack or "").strip(),
        str(norm or "").strip(),
        _float_key(eps),
        str(pgd_steps or "").strip(),
        _float_key(pgd_step_size),
        str(square_queries or "").strip(),
        _float_key(square_p_init),
        str(square_seed or "").strip(),
    )


def _attack_key_from_row(row):
    if not row:
        return None
    return (
        str(row.get("dataset", "")).strip(),
        str(row.get("net", "")).strip(),
        str(row.get("checkpoint", "")).strip(),
        str(row.get("attack", "")).strip(),
        str(row.get("norm", "")).strip(),
        _float_key(row.get("eps")),
        str(row.get("pgd_steps", "")).strip(),
        _float_key(row.get("pgd_step_size")),
        str(row.get("square_queries", "")).strip(),
        _float_key(row.get("square_p_init")),
        str(row.get("square_seed", "")).strip(),
    )


def _load_existing_attack_keys(path):
    existing = set()
    if not path or not os.path.exists(path):
        return existing
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return existing
        for row in reader:
            key = _attack_key_from_row(row)
            if key:
                existing.add(key)
    return existing


def _prepare_attack_inputs(x, assume_imagenet_norm):
    was_norm = bool(assume_imagenet_norm and _looks_normalized(x))
    x_pixel = _denorm_imagenet(x) if was_norm else x
    return x_pixel.clamp(0.0, 1.0), was_norm


def _forward_model_on_pixel_inputs(model, x_pixel, was_norm):
    x_model = _renorm_imagenet(x_pixel) if was_norm else x_pixel
    out = model(x_model)
    return out["logits"] if isinstance(out, dict) else out


def _run_pgd_attack(
    model,
    x,
    y,
    eps,
    steps,
    step_size,
    random_start,
    assume_imagenet_norm,
):
    if steps <= 0:
        raise ValueError("--pgd_steps must be positive")

    x_pixel, was_norm = _prepare_attack_inputs(x, assume_imagenet_norm)
    x_orig = x_pixel.detach()
    if step_size is None:
        step_size = eps / max(steps // 2, 1)

    if random_start:
        x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
        x_adv = x_adv.clamp(0.0, 1.0)
    else:
        x_adv = x_orig.clone()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = _forward_model_on_pixel_inputs(model, x_adv, was_norm)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0.0, 1.0)

    return _renorm_imagenet(x_adv) if was_norm else x_adv


def _untargeted_margin_loss(logits, y):
    true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other_logits = logits.clone()
    other_logits.scatter_(1, y.view(-1, 1), -torch.inf)
    max_other_logits = other_logits.max(dim=1).values
    return true_logits - max_other_logits


def _square_p_selection(p_init, i_iter, n_iters):
    if n_iters <= 0:
        return p_init

    scaled_iter = int(i_iter / n_iters * 10000)
    if 10 < scaled_iter <= 50:
        return p_init / 2
    if 50 < scaled_iter <= 200:
        return p_init / 4
    if 200 < scaled_iter <= 500:
        return p_init / 8
    if 500 < scaled_iter <= 1000:
        return p_init / 16
    if 1000 < scaled_iter <= 2000:
        return p_init / 32
    if 2000 < scaled_iter <= 4000:
        return p_init / 64
    if 4000 < scaled_iter <= 6000:
        return p_init / 128
    if 6000 < scaled_iter <= 8000:
        return p_init / 256
    if 8000 < scaled_iter <= 10000:
        return p_init / 512
    return p_init


def _cpu_randint(low, high, size, generator):
    return torch.randint(low, high, size, generator=generator, device="cpu")


def _random_linf_values(shape, eps, dtype, device, generator):
    signs = _cpu_randint(0, 2, shape, generator=generator).mul_(2).sub_(1)
    return signs.to(device=device, dtype=dtype) * eps


def _run_square_attack(
    model,
    x,
    y,
    eps,
    n_queries,
    p_init,
    seed,
    assume_imagenet_norm,
):
    if n_queries <= 0:
        raise ValueError("--square_queries must be positive")
    if p_init <= 0:
        raise ValueError("--square_p_init must be positive")

    x_pixel, was_norm = _prepare_attack_inputs(x, assume_imagenet_norm)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    with torch.inference_mode():
        logits_clean = _forward_model_on_pixel_inputs(model, x_pixel, was_norm)
    correct_mask = logits_clean.argmax(dim=1).eq(y)
    if not bool(correct_mask.any()):
        return _renorm_imagenet(x_pixel) if was_norm else x_pixel

    x_orig = x_pixel[correct_mask].clone()
    y_orig = y[correct_mask].clone()
    n_active, c, h, w = x_orig.shape
    n_features = c * h * w
    max_side = max(min(h, w) - 1, 1)

    init_delta = _random_linf_values((n_active, c, 1, w), eps, x_orig.dtype, x_orig.device, generator)
    x_best = (x_orig + init_delta).clamp(0.0, 1.0)

    with torch.inference_mode():
        logits_best = _forward_model_on_pixel_inputs(model, x_best, was_norm)
    loss_min = _untargeted_margin_loss(logits_best, y_orig)
    margin_min = loss_min.clone()

    for i_iter in range(max(n_queries - 1, 0)):
        idx_to_fool = margin_min > 0
        if not bool(idx_to_fool.any()):
            break

        active_idx = idx_to_fool.nonzero(as_tuple=True)[0]
        x_curr = x_orig[active_idx]
        x_best_curr = x_best[active_idx].clone()
        y_curr = y_orig[active_idx]
        loss_min_curr = loss_min[active_idx]
        margin_min_curr = margin_min[active_idx]
        deltas = x_best_curr - x_curr

        p = _square_p_selection(p_init, i_iter, n_queries)
        side = int(round(math.sqrt(p * n_features / c)))
        side = min(max(side, 1), max_side)

        for sample_idx in range(x_best_curr.shape[0]):
            max_h = max(h - side, 1)
            max_w = max(w - side, 1)
            center_h = int(_cpu_randint(0, max_h, (1,), generator).item())
            center_w = int(_cpu_randint(0, max_w, (1,), generator).item())

            x_curr_window = x_curr[sample_idx, :, center_h:center_h + side, center_w:center_w + side]
            x_best_window = x_best_curr[sample_idx, :, center_h:center_h + side, center_w:center_w + side]

            for _ in range(32):
                square_delta = _random_linf_values((c, 1, 1), eps, deltas.dtype, deltas.device, generator)
                deltas[sample_idx, :, center_h:center_h + side, center_w:center_w + side] = square_delta.expand(c, side, side)
                candidate_window = (
                    x_curr_window + deltas[sample_idx, :, center_h:center_h + side, center_w:center_w + side]
                ).clamp(0.0, 1.0)
                if not bool(torch.all((candidate_window - x_best_window).abs() < 1e-7)):
                    break

        x_new = (x_curr + deltas).clamp(0.0, 1.0)
        with torch.inference_mode():
            logits_new = _forward_model_on_pixel_inputs(model, x_new, was_norm)
        loss = _untargeted_margin_loss(logits_new, y_curr)
        margin = loss
        improved = loss < loss_min_curr

        loss_min[active_idx] = torch.where(improved, loss, loss_min_curr)
        margin_min[active_idx] = torch.where(improved, margin, margin_min_curr)
        if bool(improved.any()):
            x_best[active_idx[improved]] = x_new[improved]

    x_adv = x_pixel.clone()
    x_adv[correct_mask] = x_best
    return _renorm_imagenet(x_adv) if was_norm else x_adv


def run_inference_adversarial(
    model,
    loader,
    device,
    attack,
    eps,
    attack_norm="Linf",
    pgd_steps=20,
    pgd_step_size=None,
    pgd_random_start=True,
    square_queries=1000,
    square_p_init=0.05,
    square_seed=0,
    assume_imagenet_norm=True,
):
    model.eval()
    total = 0
    correct = 0
    has_labels = False

    for batch_idx, batch in enumerate(loader):
        x = None
        y = None
        for key in ("x", "x_lb", "x_ulb", "x_ulb_w"):
            if key in batch:
                x = batch[key]
                break
        for key in ("y", "y_lb", "y_ulb"):
            if key in batch:
                y = batch[key]
                break

        if x is None:
            continue
        if y is None:
            raise ValueError("Adversarial evaluation requires labels, but the loader did not provide them.")

        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()

        if attack == "pgd":
            x_adv = _run_pgd_attack(
                model,
                x,
                y,
                eps=eps,
                steps=pgd_steps,
                step_size=pgd_step_size,
                random_start=pgd_random_start,
                assume_imagenet_norm=assume_imagenet_norm,
            )
        elif attack == "square":
            x_adv = _run_square_attack(
                model,
                x,
                y,
                eps=eps,
                n_queries=square_queries,
                p_init=square_p_init,
                seed=square_seed + batch_idx,
                assume_imagenet_norm=assume_imagenet_norm,
            )
        else:
            raise ValueError(f"Unsupported attack: {attack}")

        with torch.inference_mode():
            out = model(x_adv)
            logits = out["logits"] if isinstance(out, dict) else out
            pred = logits.argmax(dim=-1)

        total += pred.shape[0]
        has_labels = True
        correct += pred.eq(y).sum().item()

    acc = (correct / total) if (has_labels and total > 0) else None
    return {"num_samples": total, "acc": acc}


def main():
    parser = argparse.ArgumentParser(description="Run clean and adversarial inference on the VTAB test set.")
    parser.add_argument("-c", "--config", default=None, help="Path to config yaml used for training.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    parser.add_argument("--data_dir", default=None, help="Override data root.")
    parser.add_argument("--data_zip", default=None, help="Optional zip containing data/ or vtab/ contents.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu.")
    parser.add_argument("--output", default=None, help="Optional pickle output path for clean inference results.")
    parser.add_argument("--save_probs", action="store_true", help="Store softmax probabilities for clean inference.")
    parser.add_argument("--save_logits", action="store_true", help="Store raw logits for clean inference.")
    parser.add_argument("--dataset", default=None, help="Dataset name (e.g. dtd).")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes.")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labeled samples.")
    parser.add_argument("--net", default=None, help="Network name.")
    parser.add_argument("--net_from_name", type=_str2bool, default=None, help="Use torchvision net by name.")
    parser.add_argument("--use_pretrain", type=_str2bool, default=None, help="Load base pretrained weights if available.")
    parser.add_argument("--pretrain_path", default=None, help="Optional pretrained checkpoint path.")
    parser.add_argument("--img_size", type=int, default=None, help="Image size.")
    parser.add_argument("--crop_ratio", type=float, default=None, help="Crop ratio.")
    parser.add_argument("--train_split", default=None, help="Train split name.")
    parser.add_argument("--train_aug", default=None, help="Train augmentation (weak/strong/none).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--ulb_num_labels", type=int, default=None, help="Unlabeled labels count.")
    parser.add_argument("--lb_imb_ratio", type=int, default=None, help="Labeled imbalance ratio.")
    parser.add_argument("--ulb_imb_ratio", type=int, default=None, help="Unlabeled imbalance ratio.")
    parser.add_argument("--peft_config", default=None, help="PEFT config yaml or inline dict.")
    parser.add_argument("--vit_config", default=None, help="ViT config yaml or inline dict.")

    parser.add_argument("--eval_attacks", action="store_true", help="Evaluate adversarial attacks and log to CSV.")
    parser.add_argument("--attack_out", default="attack_results.csv", help="CSV output path for adversarial results.")
    parser.add_argument("--attack_list", default="pgd", help="Comma-separated attacks to run (pgd,square).")
    parser.add_argument("--attack_norm", default="Linf", help="Attack norm. PGD and Square Attack currently support Linf.")
    parser.add_argument("--eps_list", default=None, help="Comma-separated eps values in [0,1]. Default: 8/255.")
    parser.add_argument("--pgd_steps", type=int, default=20, help="Number of PGD steps.")
    parser.add_argument("--pgd_step_size", type=float, default=None, help="PGD step size. Default derives from eps.")
    parser.add_argument("--pgd_random_start", type=_str2bool, default=True, help="Use random start for PGD.")
    parser.add_argument("--square_queries", type=int, default=1000, help="Square Attack query budget per example.")
    parser.add_argument("--square_p_init", type=float, default=0.05, help="Initial square size fraction for Square Attack.")
    parser.add_argument("--square_seed", type=int, default=0, help="Random seed for Square Attack.")
    parser.add_argument(
        "--assume_imagenet_norm",
        type=_str2bool,
        default=True,
        help="If True, heuristically denorm to [0,1] before attacking and renorm before model forward.",
    )

    args = parser.parse_args()

    cfg = _load_yaml(args.config) if args.config else {}
    cfg = _apply_overrides(cfg, args)
    _validate_cfg(cfg)

    data_dir = args.data_dir or cfg.get("data_dir", "./data")
    data_dir = _resolve_path(data_dir)
    data_dir = _ensure_data_available(data_dir, cfg["dataset"], data_zip=args.data_zip)

    checkpoint_path = _pick_checkpoint(cfg, args.checkpoint)
    if checkpoint_path is None:
        raise ValueError("Checkpoint path not found. Use --checkpoint or set load_path in config.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state_dict = _load_checkpoint(checkpoint_path)
    cfg["net"] = _normalize_net_name(cfg.get("net"))
    inferred_net = _infer_timm_net_from_state_dict(cfg["net"], state_dict)
    if inferred_net != cfg["net"]:
        print(f"Adjusting net from {cfg['net']} to {inferred_net} based on checkpoint.")
        cfg["net"] = inferred_net
    inferred_peft = _infer_peft_config_from_state_dict(state_dict)
    cfg = _merge_peft_config(cfg, inferred_peft)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        device = "cpu"

    model = _build_model(cfg)
    model.load_state_dict(state_dict)
    model = model.to(device)

    batch_size = args.batch_size or cfg.get("eval_batch_size", cfg.get("batch_size", 16))
    loader = _build_test_loader(cfg, data_dir, batch_size, args.num_workers)

    results = run_inference(model, loader, device, save_probs=args.save_probs, save_logits=args.save_logits)
    if results["acc"] is None:
        print(f"Test samples: {results['num_samples']}")
    else:
        print(f"Test samples: {results['num_samples']}, clean accuracy: {results['acc']:.4f}")

    if args.output:
        output_path = _resolve_path(args.output)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved clean inference results to {output_path}")

    if not args.eval_attacks:
        return

    if args.attack_norm != "Linf":
        raise ValueError("This script currently supports only --attack_norm Linf.")

    attacks = _parse_attack_list(args.attack_list)
    eps_values = _parse_float_list(args.eps_list)

    out_path = _resolve_path(args.attack_out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    existing_keys = _load_existing_attack_keys(out_path)
    fieldnames = [
        "dataset",
        "net",
        "checkpoint",
        "attack",
        "norm",
        "eps",
        "pgd_steps",
        "pgd_step_size",
        "square_queries",
        "square_p_init",
        "square_seed",
        "num_samples",
        "acc",
    ]
    needs_header = (not os.path.exists(out_path)) or (os.path.getsize(out_path) == 0)
    skipped = 0
    written = 0

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()

        for attack in attacks:
            for eps in eps_values:
                key = _attack_key(
                    cfg["dataset"],
                    cfg["net"],
                    checkpoint_path,
                    attack,
                    args.attack_norm,
                    eps,
                    args.pgd_steps if attack == "pgd" else "",
                    args.pgd_step_size if attack == "pgd" else "",
                    args.square_queries if attack == "square" else "",
                    args.square_p_init if attack == "square" else "",
                    args.square_seed if attack == "square" else "",
                )
                if key in existing_keys:
                    skipped += 1
                    print(f"[SKIP] {attack:>10s} eps={eps:.6f} already logged")
                    continue

                result = run_inference_adversarial(
                    model,
                    loader,
                    device,
                    attack=attack,
                    eps=eps,
                    attack_norm=args.attack_norm,
                    pgd_steps=args.pgd_steps,
                    pgd_step_size=args.pgd_step_size,
                    pgd_random_start=args.pgd_random_start,
                    square_queries=args.square_queries,
                    square_p_init=args.square_p_init,
                    square_seed=args.square_seed,
                    assume_imagenet_norm=args.assume_imagenet_norm,
                )
                row = {
                    "dataset": cfg["dataset"],
                    "net": cfg["net"],
                    "checkpoint": checkpoint_path,
                    "attack": attack,
                    "norm": args.attack_norm,
                    "eps": float(eps),
                    "pgd_steps": (args.pgd_steps if attack == "pgd" else ""),
                    "pgd_step_size": (None if attack != "pgd" else (args.pgd_step_size if args.pgd_step_size is not None else "")),
                    "square_queries": (args.square_queries if attack == "square" else ""),
                    "square_p_init": (args.square_p_init if attack == "square" else ""),
                    "square_seed": (args.square_seed if attack == "square" else ""),
                    "num_samples": result["num_samples"],
                    "acc": (None if result["acc"] is None else float(result["acc"])),
                }
                writer.writerow(row)
                f.flush()
                existing_keys.add(key)
                written += 1
                print(f"[ATTACK] {attack:>10s} eps={eps:.6f} acc={row['acc']}")

    print(f"Saved adversarial results to {out_path} (new: {written}, skipped: {skipped})")


if __name__ == "__main__":
    main()
