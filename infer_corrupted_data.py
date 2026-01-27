import argparse
import csv
import os
import pickle
import numpy as np

import torch
from dotwiz import DotWiz
from ruamel.yaml import YAML
from torch.utils.data import DataLoader

from semilearn.core.utils import VTAB_DSETS, get_net_builder, get_peft_config
from semilearn.datasets.cv_datasets.vtab import get_vtab


_ROOT_DIR = os.path.abspath(os.curdir)
_YAML = YAML()

_TIMM_PRETRAIN_WEIGHTS = {
    "vit_base_patch16_224.augreg_in21k": "pretrain_weight/vit_base_patch16_224_augreg_in21k.bin",
    "vit_base_patch14_reg4_dinov2.lvd142m": "pretrain_weight/vit_base_patch14_reg4_dinov2_lvd142m.bin",
    "vit_base_patch16_clip_224.openai": "pretrain_weight/vit_base_patch16_clip_224_openai.bin",
    "ViT-B-16-SigLIP": "pretrain_weight/siglip.bin",
    "vit_large_patch14_clip_224.openai": "pretrain_weight/vit_large_patch14_clip_224.openai.bin",
    "vit_large_patch14_reg4_dinov2.lvd142m": "pretrain_weight/vit_large_patch14_reg4_dinov2.lvd142m.bin",
    "vit_large_patch14_dinov2.lvd142m": "pretrain_weight/vit_large_patch14_dinov2.lvd142m.bin",
    "open_clip_ViT-B-16": "pretrain_weight/open_clip_ViT-B-16.bin",
    "metaclip_400m": "pretrain_weight/vit_base_patch16_clip_224.metaclip_400m.bin",
}

_TIMM_VARIANTS = {
    "vit_base_patch16_224.augreg_in21k": "vit_base_patch16_224_in21k_petl",
    "vit_base_patch14_reg4_dinov2.lvd142m": "vit_base_patch14_dinov2_petl",
    "vit_base_patch16_clip_224.openai": "vit_base_patch16_clip_224_petl",
    "ViT-B-16-SigLIP": "vit_base_patch16_siglip_224_petl",
    "vit_large_patch14_clip_224.openai": "vit_large_patch14_clip_224_petl",
    "vit_large_patch14_reg4_dinov2.lvd142m": "vit_base_patch14_dinov2_large_petl",
    "vit_large_patch14_dinov2.lvd142m": "vit_base_patch14_dinov2_large_petl",
    "open_clip_ViT-B-16": "vit_base_patch16_metaclip_224_petl",
    "metaclip_400m": "vit_base_patch16_metaclip_224_petl",
}

# 19 corruptions (15 "common" + 4 extras) in the usual CIFAR/ImageNet-C family.
CORRUPTIONS_19 = [
    # noise
    "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    # blur
    "gaussian_blur", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    # weather
    "snow", "frost", "fog", "spatter",
    # digital
    "brightness", "contrast", "saturate", "jpeg_compression", "pixelate", "elastic_transform",
]
CORRUPTION_SEVERITIES = (1, 2, 3, 4, 5)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGECORRUPTIONS_PATCHED = False
_NUMPY_FLOAT_PATCHED = False


def _looks_normalized(x: torch.Tensor) -> bool:
    """
    Heuristic: if values aren't roughly in [0, 1], it's probably normalized.
    Works well for ImageNet normalization where values are ~[-2.2, 2.6].
    """
    # x shape: (B, C, H, W)
    xmin = float(x.min().item())
    xmax = float(x.max().item())
    return (xmin < -0.1) or (xmax > 1.1)


def _denorm_imagenet(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """
    x: (B,C,H,W) normalized -> returns (B,C,H,W) in [0,1]
    """
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = x * std_t + mean_t
    return x.clamp(0.0, 1.0)


def _renorm_imagenet(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """
    x: (B,C,H,W) in [0,1] -> returns normalized
    """
    mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean_t) / std_t


def _parse_corruption_list(corruption_list):
    if corruption_list:
        return [c.strip() for c in corruption_list.split(",") if c.strip()]
    return CORRUPTIONS_19


def _corruption_key(dataset, net, checkpoint, corruption, severity):
    return (
        str(dataset or "").strip(),
        str(net or "").strip(),
        str(checkpoint or "").strip(),
        str(corruption or "").strip(),
        int(severity),
    )


def _corruption_key_from_row(row):
    try:
        return (
            str(row.get("dataset", "")).strip(),
            str(row.get("net", "")).strip(),
            str(row.get("checkpoint", "")).strip(),
            str(row.get("corruption", "")).strip(),
            int(str(row.get("severity", "")).strip()),
        )
    except (ValueError, TypeError, AttributeError):
        return None


def _load_existing_corruption_keys(path):
    existing = set()
    if not path or not os.path.exists(path):
        return existing
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return existing
        for row in reader:
            key = _corruption_key_from_row(row)
            if key:
                existing.add(key)
    return existing


def _apply_corruption_to_batch(
    x: torch.Tensor,
    corruption: str,
    severity: int,
    assume_imagenet_norm: bool = True,
) -> torch.Tensor:
    """
    Apply an ImageNet-C style corruption to a batch tensor.
    Input x expected shape: (B,C,H,W)
    Returns tensor with same shape, on CPU (float32).
    """
    try:
        from imagecorruptions import corrupt
    except Exception as e:
        raise RuntimeError(
            "Missing dependency `imagecorruptions`. Install with: pip install imagecorruptions"
        ) from e
    _patch_numpy_float_alias()
    _patch_imagecorruptions_for_skimage()

    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"Expected x shape (B,3,H,W). Got: {tuple(x.shape)}")

    x_cpu = x.detach().cpu()

    was_norm = False
    if assume_imagenet_norm and _looks_normalized(x_cpu):
        was_norm = True
        x_cpu = _denorm_imagenet(x_cpu)  # -> [0,1]

    # Convert to uint8 HWC per sample, corrupt, convert back
    out = []
    for img in x_cpu:
        img = img.clamp(0.0, 1.0)
        img_u8 = (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)  # HWC uint8
        img_cor = corrupt(img_u8, corruption_name=corruption, severity=severity)  # HWC uint8
        img_cor = torch.from_numpy(img_cor).permute(2, 0, 1).float() / 255.0  # CHW float [0,1]
        out.append(img_cor)

    x_out = torch.stack(out, dim=0)

    if was_norm:
        x_out = _renorm_imagenet(x_out)

    return x_out


def _patch_imagecorruptions_for_skimage():
    global _IMAGECORRUPTIONS_PATCHED
    if _IMAGECORRUPTIONS_PATCHED:
        return
    try:
        import inspect
        import imagecorruptions.corruptions as ic_corruptions
    except Exception:
        return
    try:
        sig = inspect.signature(ic_corruptions.gaussian)
    except (TypeError, ValueError):
        return
    if "multichannel" in sig.parameters:
        _IMAGECORRUPTIONS_PATCHED = True
        return

    original = ic_corruptions.gaussian

    def _gaussian_compat(*args, **kwargs):
        if "multichannel" in kwargs:
            multichannel = kwargs.pop("multichannel")
            if "channel_axis" not in kwargs:
                kwargs["channel_axis"] = -1 if multichannel else None
        return original(*args, **kwargs)

    ic_corruptions.gaussian = _gaussian_compat
    _IMAGECORRUPTIONS_PATCHED = True


def _patch_numpy_float_alias():
    global _NUMPY_FLOAT_PATCHED
    if _NUMPY_FLOAT_PATCHED:
        return
    try:
        _ = np.float_
        _NUMPY_FLOAT_PATCHED = True
        return
    except AttributeError:
        pass
    np.float_ = np.float64
    _NUMPY_FLOAT_PATCHED = True


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        data = _YAML.load(f)
    return data if data is not None else {}


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_ROOT_DIR, path))


def _load_maybe_yaml(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    value = str(value)
    if os.path.exists(value) and os.path.isfile(value):
        return _load_yaml(value)
    data = _YAML.load(value)
    return data if data is not None else {}


def _normalize_load_path(load_path):
    if isinstance(load_path, list):
        if not load_path:
            return None
        return load_path[0]
    return load_path


def _normalize_net_name(net):
    if not net:
        return net
    if net.startswith("timm/"):
        return net
    if net in _TIMM_PRETRAIN_WEIGHTS or net in _TIMM_VARIANTS:
        return f"timm/{net}"
    return net


def _infer_embed_dim(state_dict):
    for key in ("model.cls_token", "cls_token", "model.pos_embed", "pos_embed"):
        if key in state_dict:
            return state_dict[key].shape[-1]
    for key in ("model.patch_embed.proj.weight", "patch_embed.proj.weight"):
        if key in state_dict:
            return state_dict[key].shape[0]
    return None


def _infer_timm_net_from_state_dict(net, state_dict):
    if not net or not net.startswith("timm/"):
        return net
    model_name = net.split("/", 1)[1]
    embed_dim = _infer_embed_dim(state_dict)
    if embed_dim is None:
        return net
    if "dinov2" in model_name and "patch14" in model_name:
        if embed_dim == 768:
            return "timm/vit_base_patch14_reg4_dinov2.lvd142m"
        if embed_dim == 1024:
            return "timm/vit_large_patch14_reg4_dinov2.lvd142m"
    return net


def _pick_checkpoint(cfg, override):
    if override:
        return _resolve_path(override)
    load_path = _normalize_load_path(cfg.get("load_path"))
    if load_path is None:
        save_dir = cfg.get("save_dir")
        save_name = cfg.get("save_name", "log")
        if save_dir:
            load_path = os.path.join(save_dir, save_name, "latest_model.pth")
    return _resolve_path(load_path)


def _strip_state_dict_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    if any(key.startswith("backbone.") for key in cleaned.keys()):
        cleaned = {k.replace("backbone.", ""): v for k, v in cleaned.items() if "rot_" not in k}
    return cleaned


def _load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    return _strip_state_dict_prefix(state_dict)


def _infer_peft_config_from_state_dict(state_dict):
    peft_config = {}
    keys = list(state_dict.keys())

    for key in ("model.head.0.weight", "head.0.weight"):
        if key in state_dict:
            peft_config["method_name"] = "stronger_head"
            peft_config["head_h"] = state_dict[key].shape[0]
            break

    lora_key = next((k for k in keys if k.endswith("lora_a.weight")), None)
    if lora_key:
        peft_config["lora_bottleneck"] = state_dict[lora_key].shape[0]

    ft_mlp_key = next((k for k in keys if "ft_mlp_module.down_proj.weight" in k), None)
    if ft_mlp_key:
        peft_config["method_name"] = peft_config.get("method_name", "adaptformer")
        peft_config["ft_mlp_module"] = "adapter"
        peft_config["adapter_bottleneck"] = state_dict[ft_mlp_key].shape[0]
    ft_mlp_key = next((k for k in keys if "ft_mlp_module.adapter_down.weight" in k), None)
    if ft_mlp_key:
        peft_config["method_name"] = peft_config.get("method_name", "adaptformer")
        peft_config["ft_mlp_module"] = "convpass"
        peft_config["convpass_bottleneck"] = state_dict[ft_mlp_key].shape[0]
    ft_mlp_key = next((k for k in keys if "ft_mlp_module.conv_A.weight" in k), None)
    if ft_mlp_key:
        peft_config["method_name"] = peft_config.get("method_name", "adaptformer")
        peft_config["ft_mlp_module"] = "repadapter"
        peft_config["repadapter_bottleneck"] = state_dict[ft_mlp_key].shape[0]

    ft_attn_key = next((k for k in keys if "ft_attn_module.down_proj.weight" in k), None)
    if ft_attn_key:
        peft_config["method_name"] = peft_config.get("method_name", "adaptformer")
        peft_config["ft_attn_module"] = "adapter"
        peft_config["adapter_bottleneck"] = state_dict[ft_attn_key].shape[0]
    ft_attn_key = next((k for k in keys if "ft_attn_module.adapter_down.weight" in k), None)
    if ft_attn_key:
        peft_config["method_name"] = peft_config.get("method_name", "adaptformer")
        peft_config["ft_attn_module"] = "convpass"
        peft_config["convpass_bottleneck"] = state_dict[ft_attn_key].shape[0]
    ft_attn_key = next((k for k in keys if "ft_attn_module.conv_A.weight" in k), None)
    if ft_attn_key:
        peft_config["method_name"] = peft_config.get("method_name", "adaptformer")
        peft_config["ft_attn_module"] = "repadapter"
        peft_config["repadapter_bottleneck"] = state_dict[ft_attn_key].shape[0]

    return peft_config


def _expected_vtab_list(data_dir, dataset, split):
    return os.path.join(data_dir, "vtab", dataset.lower(), f"{split}.list")


def _ensure_data_available(data_dir, dataset):
    expected = _expected_vtab_list(data_dir, dataset, "test")
    if os.path.exists(expected):
        return data_dir
    base = os.path.basename(os.path.normpath(data_dir))
    if base != "data":
        candidate = os.path.join(data_dir, "data")
        expected = _expected_vtab_list(candidate, dataset, "test")
        if os.path.exists(expected):
            return candidate
    if base != "vtab":
        candidate = os.path.join(data_dir, "vtab")
        expected = _expected_vtab_list(candidate, dataset, "test")
        if os.path.exists(expected):
            return candidate
    if base != "vtab_release":
        candidate = os.path.join(data_dir, "vtab_release")
        expected = _expected_vtab_list(candidate, dataset, "test")
        if os.path.exists(expected):
            return candidate
    raise FileNotFoundError(
        f"VTAB test list not found at {expected}. Ensure --data_dir points to a data/, vtab/, or vtab_release/ folder."
    )


def _build_timm_without_pretrain(model_name, num_classes, peft_config, vit_config):
    import timm
    from semilearn.nets.timm_wrapper import TimmViTWrapper, petl_builder

    variant = _TIMM_VARIANTS.get(model_name)
    if not variant:
        raise ValueError(f"Unsupported timm model: {model_name}")
    vit_kwargs = {**vit_config, "img_size": 224, "num_classes": num_classes}
    model = timm.create_model(variant, pretrained=False, peft_config=peft_config, **vit_kwargs)
    if hasattr(peft_config, "method_name"):
        model, _ = petl_builder(model, peft_config)
    return TimmViTWrapper(model)


def _build_model(cfg):
    net = _normalize_net_name(cfg["net"])
    num_classes = cfg["num_classes"]
    net_from_name = cfg.get("net_from_name", False)
    use_pretrain = cfg.get("use_pretrain", True)
    pretrained_path = cfg.get("pretrain_path", cfg.get("pretrained_path"))
    peft_config = get_peft_config(cfg.get("peft_config"))
    vit_config = cfg.get("vit_config", {})

    if net.startswith("timm/") and not net_from_name:
        model_name = net.split("/", 1)[1]
        if not use_pretrain:
            return _build_timm_without_pretrain(model_name, num_classes, peft_config, vit_config)
        weight_path = _TIMM_PRETRAIN_WEIGHTS.get(model_name)
        if weight_path:
            resolved = _resolve_path(weight_path)
            if not os.path.exists(resolved):
                print(f"Missing base weights at {resolved}, building without pretrain weights.")
                return _build_timm_without_pretrain(model_name, num_classes, peft_config, vit_config)

    net_builder = get_net_builder(net, net_from_name, peft_config=peft_config, vit_config=vit_config)
    return net_builder(num_classes=num_classes, pretrained=use_pretrain, pretrained_path=pretrained_path)


def _build_test_loader(cfg, data_dir, batch_size, num_workers):
    dataset = cfg["dataset"]
    num_labels = cfg["num_labels"]
    num_classes = cfg["num_classes"]
    if dataset not in VTAB_DSETS:
        raise ValueError(f"Dataset {dataset} is not a VTAB dataset. Found datasets: {VTAB_DSETS}")
    args = DotWiz({
        "seed": cfg.get("seed", 0),
        "num_labels": num_labels,
        "dataset": dataset,
        "ulb_num_labels": cfg.get("ulb_num_labels"),
        "lb_imb_ratio": cfg.get("lb_imb_ratio", 1),
        "ulb_imb_ratio": cfg.get("ulb_imb_ratio", 1),
        "net": cfg.get("net"),
        "train_split": cfg.get("train_split", "train"),
        "crop_ratio": cfg.get("crop_ratio", 0.875),
        "img_size": cfg.get("img_size", 224),
        "train_aug": cfg.get("train_aug", "weak"),
    })
    _, _, _, test_dset, _ = get_vtab(
        args,
        "extract_pl",
        dataset,
        num_labels,
        num_classes,
        data_dir,
        include_lb_to_ulb=True,
    )
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        test_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )


def _select_batch(batch, keys):
    for key in keys:
        if key in batch:
            return batch[key]
    return None


def _concat_or_none(tensors):
    if not tensors:
        return None
    return torch.cat(tensors, dim=0).numpy()


def run_inference(model, loader, device, save_probs=False, save_logits=False):
    model.eval()
    total = 0
    correct = 0
    has_labels = False

    preds = []
    labels = []
    probs = []
    logits = []
    indices = []

    with torch.inference_mode():
        for batch in loader:
            idx = _select_batch(batch, ["idx", "idx_lb", "idx_ulb"])
            x = _select_batch(batch, ["x", "x_lb", "x_ulb", "x_ulb_w"])
            y = _select_batch(batch, ["y", "y_lb", "y_ulb"])

            x = x.to(device, non_blocking=True).float()
            out = model(x)
            batch_logits = out["logits"] if isinstance(out, dict) else out
            batch_probs = torch.softmax(batch_logits, dim=-1)
            batch_pred = batch_probs.argmax(dim=-1)

            total += batch_pred.shape[0]
            if y is not None:
                has_labels = True
                correct += batch_pred.cpu().eq(y).sum().item()
                labels.append(y.cpu())

            preds.append(batch_pred.cpu())
            if idx is not None:
                indices.append(idx.cpu())
            if save_probs:
                probs.append(batch_probs.cpu())
            if save_logits:
                logits.append(batch_logits.cpu())

    acc = (correct / total) if (has_labels and total > 0) else None
    output = {
        "num_samples": total,
        "acc": acc,
        "y_pred": _concat_or_none(preds),
        "y_true": _concat_or_none(labels),
        "idx": _concat_or_none(indices),
    }
    if save_probs:
        output["probs"] = _concat_or_none(probs)
    if save_logits:
        output["logits"] = _concat_or_none(logits)
    return output


def run_inference_corrupted(
    model,
    loader,
    device,
    corruption: str,
    severity: int,
    assume_imagenet_norm: bool = True,
):
    model.eval()
    total = 0
    correct = 0
    has_labels = False

    with torch.inference_mode():
        for batch in loader:
            x = _select_batch(batch, ["x", "x_lb", "x_ulb", "x_ulb_w"])
            y = _select_batch(batch, ["y", "y_lb", "y_ulb"])

            if x is None:
                continue

            # Apply corruption on CPU, then move to device
            x_cor = _apply_corruption_to_batch(
                x, corruption=corruption, severity=severity, assume_imagenet_norm=assume_imagenet_norm
            ).to(device, non_blocking=True)

            out = model(x_cor)
            batch_logits = out["logits"] if isinstance(out, dict) else out
            batch_pred = batch_logits.argmax(dim=-1)

            total += batch_pred.shape[0]
            if y is not None:
                has_labels = True
                correct += batch_pred.cpu().eq(y).sum().item()

    acc = (correct / total) if (has_labels and total > 0) else None
    return {"num_samples": total, "acc": acc}


def _apply_overrides(cfg, args):
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.num_classes is not None:
        cfg["num_classes"] = args.num_classes
    if args.num_labels is not None:
        cfg["num_labels"] = args.num_labels
    if args.net:
        cfg["net"] = _normalize_net_name(args.net)
    if args.net_from_name is not None:
        cfg["net_from_name"] = args.net_from_name
    if args.use_pretrain is not None:
        cfg["use_pretrain"] = args.use_pretrain
    if args.pretrain_path:
        cfg["pretrain_path"] = args.pretrain_path
    if args.img_size is not None:
        cfg["img_size"] = args.img_size
    if args.crop_ratio is not None:
        cfg["crop_ratio"] = args.crop_ratio
    if args.train_split:
        cfg["train_split"] = args.train_split
    if args.train_aug:
        cfg["train_aug"] = args.train_aug
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.ulb_num_labels is not None:
        cfg["ulb_num_labels"] = args.ulb_num_labels
    if args.lb_imb_ratio is not None:
        cfg["lb_imb_ratio"] = args.lb_imb_ratio
    if args.ulb_imb_ratio is not None:
        cfg["ulb_imb_ratio"] = args.ulb_imb_ratio
    if args.peft_config is not None:
        cfg["peft_config"] = _load_maybe_yaml(args.peft_config)
    if args.vit_config is not None:
        cfg["vit_config"] = _load_maybe_yaml(args.vit_config)
    return cfg


def _validate_cfg(cfg):
    required = ["dataset", "num_classes", "num_labels", "net"]
    missing = [key for key in required if cfg.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")


def _merge_peft_config(cfg, inferred_peft):
    if not inferred_peft:
        return cfg
    existing = cfg.get("peft_config")
    if existing in (None, {}):
        cfg["peft_config"] = inferred_peft
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Run inference on the VTAB test set.")
    parser.add_argument("-c", "--config", default=None, help="Path to config yaml used for training.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Override data root (data/, vtab/, or vtab_release/ folder, or their parent).",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Override eval batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu.")
    parser.add_argument("--output", default=None, help="Optional pickle output path.")
    parser.add_argument("--save_probs", action="store_true", help="Store softmax probabilities.")
    parser.add_argument("--save_logits", action="store_true", help="Store raw logits.")
    parser.add_argument("--dataset", default=None, help="Dataset name (e.g. dtd).")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes.")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labeled samples.")
    parser.add_argument("--net", default=None, help="Network name (e.g. timm/vit_base_patch14_reg4_dinov2.lvd142m).")
    parser.add_argument("--net_from_name", type=_str2bool, default=None, help="Use torchvision net by name.")
    parser.add_argument("--use_pretrain", type=_str2bool, default=None, help="Load base pretrained weights if available.")
    parser.add_argument("--pretrain_path", default=None, help="Optional pretrained checkpoint path.")
    parser.add_argument("--img_size", type=int, default=None, help="Image size.")
    parser.add_argument("--crop_ratio", type=float, default=None, help="Crop ratio.")
    parser.add_argument("--train_split", default=None, help="Train split name (e.g. train).")
    parser.add_argument("--train_aug", default=None, help="Train augmentation (weak/strong/none).")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--ulb_num_labels", type=int, default=None, help="Unlabeled labels count.")
    parser.add_argument("--lb_imb_ratio", type=int, default=None, help="Labeled imbalance ratio.")
    parser.add_argument("--ulb_imb_ratio", type=int, default=None, help="Unlabeled imbalance ratio.")
    parser.add_argument("--peft_config", default=None, help="PEFT config yaml or inline dict.")
    parser.add_argument("--vit_config", default=None, help="ViT config yaml or inline dict.")

    parser.add_argument("--eval_corruptions", action="store_true",
                        help="Evaluate 19 corruptions x 5 severities and log to CSV.")
    parser.add_argument("--corruption_out", default="corruption_results.csv",
                        help="CSV output path for corruption results.")
    parser.add_argument("--corruption_list", default=None,
                        help="Optional comma-separated corruptions to run (default: all 19).")
    parser.add_argument("--assume_imagenet_norm", type=_str2bool, default=True,
                        help="If True, heuristically detect ImageNet normalization and denorm/renorm for corruption.")

    args = parser.parse_args()

    cfg = _load_yaml(args.config) if args.config else {}
    cfg = _apply_overrides(cfg, args)
    _validate_cfg(cfg)
    data_dir = args.data_dir or cfg.get("data_dir", "./data")
    data_dir = _resolve_path(data_dir)
    data_dir = _ensure_data_available(data_dir, cfg["dataset"])
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
        print(f"Test samples: {results['num_samples']}, accuracy: {results['acc']:.4f}")

    if args.output:
        output_path = _resolve_path(args.output)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved inference results to {output_path}")

    if args.eval_corruptions:
        corruptions = _parse_corruption_list(args.corruption_list)

        out_path = _resolve_path(args.corruption_out)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        existing_keys = _load_existing_corruption_keys(out_path)
        fieldnames = ["dataset", "net", "checkpoint", "corruption", "severity", "num_samples", "acc"]
        needs_header = (not os.path.exists(out_path)) or (os.path.getsize(out_path) == 0)
        skipped = 0
        written = 0

        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()

            for c in corruptions:
                for s in CORRUPTION_SEVERITIES:
                    # skip s=5 for glass_blur due to extreme slowness
                    if c == "glass_blur" and s == 5:
                        print(f"[SKIP] {c:>18s} sev={s} too slow to run")
                        continue
                    key = _corruption_key(cfg["dataset"], cfg["net"], checkpoint_path, c, s)
                    if key in existing_keys:
                        skipped += 1
                        print(f"[SKIP] {c:>18s} sev={s} already logged")
                        continue
                    r = run_inference_corrupted(
                        model,
                        loader,
                        device,
                        corruption=c,
                        severity=s,
                        assume_imagenet_norm=args.assume_imagenet_norm,
                    )
                    row = {
                        "dataset": cfg["dataset"],
                        "net": cfg["net"],
                        "checkpoint": checkpoint_path,
                        "corruption": c,
                        "severity": s,
                        "num_samples": r["num_samples"],
                        "acc": (None if r["acc"] is None else float(r["acc"])),
                    }
                    writer.writerow(row)
                    f.flush()
                    existing_keys.add(key)
                    written += 1
                    print(f"[CORR] {c:>18s} sev={s}  acc={row['acc']}")
        print(f"Saved corruption results to {out_path} (new: {written}, skipped: {skipped})")


if __name__ == "__main__":
    main()
