#!/usr/bin/env python3
"""
Label Propagation baseline for list-backed VTAB datasets in this repository.

This follows the Iscen et al. training pattern:
1. Warm-start on labeled data
2. Extract embeddings for train = labeled + unlabeled
3. Build a cosine k-NN graph and propagate labels
4. Fine-tune on labeled + pseudo-labeled train data
5. Repeat
"""

import argparse
import csv
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import faiss
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from dotwiz import DotWiz
from PIL import Image
from ruamel.yaml import YAML
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms

from semilearn.core.utils import get_net_builder, get_peft_config
from semilearn.datasets.augmentation import RandAugment


_YAML = YAML()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG = logging.getLogger("labelprop")
CSV_FIELDS = [
    "stage",
    "round",
    "epoch",
    "global_epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "test_loss",
    "test_acc",
    "pseudo_selected",
    "pseudo_total",
    "pseudo_acc_all",
    "pseudo_acc_selected",
    "pseudo_conf_mean",
    "graph_sigma",
    "graph_prop_steps",
    "graph_prop_delta",
]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def setup_logger(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.setLevel(logging.INFO)
    LOG.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOG.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_dir / "log.txt")
    file_handler.setFormatter(formatter)
    LOG.addHandler(file_handler)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> DotWiz:
    with path.open("r", encoding="utf-8") as f:
        data = _YAML.load(f)
    return DotWiz(data or {})


def save_run_config(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        _YAML.dump(payload, f)


def read_split_list(list_path: Path) -> List[Tuple[str, int]]:
    entries: List[Tuple[str, int]] = []
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(" ", 1)
            entries.append((rel_path, int(label)))
    return entries


def read_labeled_indices(dataset_root: Path, num_labels: int, seed: int, train_split: str) -> np.ndarray:
    idx_path = dataset_root / "labeled_idx" / f"lb_labels{num_labels}_1_seed{seed}_{train_split}_idx.list"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing labeled split file: {idx_path}")

    indices: List[int] = []
    with idx_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            indices.append(int(parts[-1]))
    return np.asarray(indices, dtype=np.int64)


def make_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    train_weak = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    train_strong = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return train_weak, train_strong, eval_transform


class IndexedImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        entries: Sequence[Tuple[str, int]],
        indices: Sequence[int],
        transform: transforms.Compose,
        override_labels: Optional[Sequence[int]] = None,
        sample_weights: Optional[Sequence[float]] = None,
        return_index: bool = False,
    ) -> None:
        self.dataset_root = dataset_root
        self.entries = list(entries)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.override_labels = None if override_labels is None else np.asarray(override_labels, dtype=np.int64)
        self.sample_weights = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float32)
        self.return_index = return_index

        if self.override_labels is not None and len(self.override_labels) != len(self.indices):
            raise ValueError("override_labels length does not match indices")
        if self.sample_weights is not None and len(self.sample_weights) != len(self.indices):
            raise ValueError("sample_weights length does not match indices")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        global_idx = int(self.indices[item])
        rel_path, label = self.entries[global_idx]
        if self.override_labels is not None:
            label = int(self.override_labels[item])
        weight = 1.0 if self.sample_weights is None else float(self.sample_weights[item])

        image = Image.open(self.dataset_root / rel_path).convert("RGB")
        image = self.transform(image)

        if self.return_index:
            return image, int(label), global_idx
        return image, int(label), weight


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(key.startswith(prefix) for key in keys):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def build_model(cfg: DotWiz, checkpoint: Optional[Path]) -> nn.Module:
    peft_cfg = get_peft_config(cfg.get("peft_config", {}))
    vit_cfg = cfg.get("vit_config", {}) or {}
    net_builder = get_net_builder(
        cfg.net,
        from_name=cfg.get("net_from_name", False),
        peft_config=peft_cfg,
        vit_config=vit_cfg,
    )
    model = net_builder(
        num_classes=cfg.num_classes,
        pretrained=cfg.get("use_pretrain", True),
        pretrained_path=cfg.get("pretrained_path", ""),
    )

    if checkpoint is not None:
        checkpoint = checkpoint.resolve()
        LOG.info("Loading checkpoint from %s", checkpoint)
        raw_state = torch.load(checkpoint, map_location="cpu")
        state_dict = raw_state.get("model", raw_state)
        state_dict = strip_prefix_if_present(state_dict, "module.")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        LOG.info("Checkpoint load finished: missing=%d unexpected=%d", len(missing), len(unexpected))
    else:
        LOG.info("No checkpoint provided. Starting from the backbone pretrain only.")

    return model.to(DEVICE)


def build_optimizer(model: nn.Module, optim_name: str, lr: float, weight_decay: float, momentum: float) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    if optim_name.lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optim_name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {optim_name}")


def evaluate(model: nn.Module, loader: DataLoader, split: str) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels, _weights in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)["logits"]
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += batch_size

    metrics = {
        f"{split}_loss": total_loss / max(total_examples, 1),
        f"{split}_acc": total_correct / max(total_examples, 1),
    }
    return metrics


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels, weights in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        weights = weights.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)["logits"]
        losses = F.cross_entropy(logits, labels, reduction="none")
        loss = (losses * weights).mean()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return {
        "train_loss": total_loss / max(total_examples, 1),
        "train_acc": total_correct / max(total_examples, 1),
    }


@torch.no_grad()
def compute_embeddings(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    indices: List[np.ndarray] = []

    for images, batch_labels, batch_indices in loader:
        images = images.to(DEVICE, non_blocking=True)
        feats = model.forward_features(images)
        features.append(feats.detach().cpu().numpy().astype(np.float32))
        labels.append(batch_labels.numpy().astype(np.int64))
        indices.append(batch_indices.numpy().astype(np.int64))

    return (
        np.concatenate(features, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(indices, axis=0),
    )


def build_knn_graph(features: np.ndarray, k: int, sigma: Optional[float]) -> Tuple[sp.csr_matrix, float]:
    num_samples, feat_dim = features.shape
    normalized = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
    normalized = normalized.astype(np.float32)

    index = faiss.IndexFlatIP(feat_dim)
    index.add(normalized)
    search_k = min(k + 1, num_samples)
    similarities, neighbors = index.search(normalized, search_k)

    rows: List[int] = []
    cols: List[int] = []
    distances: List[float] = []

    for src in range(num_samples):
        for dst, sim in zip(neighbors[src], similarities[src]):
            dst = int(dst)
            if dst == src:
                continue
            rows.append(src)
            cols.append(dst)
            distances.append(float(max(0.0, 1.0 - sim)))

    edge_distances = np.asarray(distances, dtype=np.float32)
    if sigma is None:
        positive_distances = edge_distances[edge_distances > 0]
        sigma_value = float(np.median(positive_distances)) if positive_distances.size else 0.1
    else:
        sigma_value = float(sigma)
    sigma_value = max(sigma_value, 1e-6)

    weights = np.exp(-(edge_distances ** 2) / (2.0 * sigma_value * sigma_value)).astype(np.float32)
    w = sp.coo_matrix((weights, (rows, cols)), shape=(num_samples, num_samples), dtype=np.float32).tocsr()
    w = 0.5 * (w + w.T)

    degree = np.asarray(w.sum(axis=1)).reshape(-1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.clip(degree, 1e-12, None))
    d_inv_sqrt = sp.diags(inv_sqrt_degree.astype(np.float32))
    s = (d_inv_sqrt @ w @ d_inv_sqrt).tocsr()

    return s, sigma_value


def label_propagation(
    graph: sp.csr_matrix,
    labeled_targets: np.ndarray,
    num_classes: int,
    num_labeled: int,
    alpha: float,
    max_iters: int,
    tol: float,
    temperature: float,
) -> Tuple[np.ndarray, int, float]:
    num_nodes = graph.shape[0]
    y = np.zeros((num_nodes, num_classes), dtype=np.float32)
    y[np.arange(num_labeled), labeled_targets] = 1.0

    f = y.copy()
    delta = math.inf

    for step in range(1, max_iters + 1):
        f_new = alpha * graph.dot(f) + (1.0 - alpha) * y
        delta = float(np.mean(np.abs(f_new - f)))
        f = f_new
        if delta < tol:
            return normalize_probabilities(f, num_labeled, temperature), step, delta

    return normalize_probabilities(f, num_labeled, temperature), max_iters, delta


def normalize_probabilities(scores: np.ndarray, num_labeled: int, temperature: float) -> np.ndarray:
    scores = np.maximum(scores, 1e-12)
    scores = scores / np.clip(scores.sum(axis=1, keepdims=True), 1e-12, None)
    if temperature != 1.0:
        unlabeled_scores = scores[num_labeled:] ** (1.0 / temperature)
        unlabeled_scores = unlabeled_scores / np.clip(unlabeled_scores.sum(axis=1, keepdims=True), 1e-12, None)
        scores[num_labeled:] = unlabeled_scores
    return scores


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, meta: Dict[str, float]) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "meta": meta,
        },
        path,
    )


def append_metrics(csv_path: Path, row: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    normalized_row = {field: row.get(field, "") for field in CSV_FIELDS}
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized_row)


def infer_output_dir(cfg: DotWiz, seed: int, explicit_output_dir: Optional[Path]) -> Path:
    if explicit_output_dir is not None:
        return explicit_output_dir

    method_name = cfg.get("peft_config", {}).get("method_name", "full")
    shot = cfg.num_labels // cfg.num_classes if cfg.num_labels % cfg.num_classes == 0 else cfg.num_labels

    net_name = str(cfg.net).lower()
    if "dinov2" in net_name:
        backbone = "dinov2"
    elif "clip" in net_name:
        backbone = "clip"
    else:
        backbone = net_name.split("/")[-1].replace(".", "_")

    return Path("saved_models") / "labelprop" / method_name / cfg.dataset / f"{shot}-shot" / backbone / f"seed-{seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a label propagation baseline using the dataset defined by the config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional model checkpoint to resume from. Default is no checkpoint.")
    parser.add_argument("--data-root", type=Path, default=None, help="Defaults to cfg.data_dir from the config.")
    parser.add_argument("--seed", type=int, default=None, help="Defaults to config seed.")
    parser.add_argument("--train-split", type=str, default=None, help="Defaults to config train_split or train.")
    parser.add_argument("--train-aug", type=str, default="weak", choices=["none", "weak", "strong"])
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--round-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--pseudo-loss-weight", type=float, default=1.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-weighting", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--max-prop-iters", type=int, default=50)
    parser.add_argument("--prop-tol", type=float, default=1e-6)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    seed = cfg.seed if args.seed is None else args.seed
    set_seed(seed)

    output_dir = infer_output_dir(cfg, seed, args.output_dir)
    setup_logger(output_dir)
    LOG.info("Outputs will be written to %s", output_dir.resolve())

    train_split = args.train_split or cfg.get("train_split", "train")
    batch_size = args.batch_size or cfg.get("batch_size", 16)
    eval_batch_size = args.eval_batch_size or cfg.get("eval_batch_size", 32)
    lr = args.lr if args.lr is not None else cfg.get("lr", 1e-4)
    weight_decay = args.weight_decay if args.weight_decay is not None else cfg.get("weight_decay", 5e-4)
    momentum = args.momentum if args.momentum is not None else cfg.get("momentum", 0.9)
    optim_name = args.optim or cfg.get("optim", "AdamW")

    default_warmup = max(5, int(cfg.get("epoch", 30) // 3))
    warmup_epochs = default_warmup if args.warmup_epochs is None else args.warmup_epochs
    remaining_epochs = max(1, int(cfg.get("epoch", 30)) - warmup_epochs)
    default_round_epochs = max(1, int(math.ceil(remaining_epochs / max(args.num_rounds, 1))))
    round_epochs = default_round_epochs if args.round_epochs is None else args.round_epochs

    base_data_root = Path(cfg.get("data_dir", "data")) if args.data_root is None else args.data_root
    dataset_root = base_data_root / "vtab" / str(cfg.dataset).lower()
    train_entries = read_split_list(dataset_root / f"{train_split}.list")
    val_entries = read_split_list(dataset_root / "val.list")
    test_entries = read_split_list(dataset_root / "test.list")

    labeled_indices = read_labeled_indices(dataset_root, cfg.num_labels, seed, train_split)
    all_train_indices = np.arange(len(train_entries), dtype=np.int64)
    labeled_mask = np.zeros(len(train_entries), dtype=bool)
    labeled_mask[labeled_indices] = True
    unlabeled_indices = all_train_indices[~labeled_mask]
    ordered_train_indices = np.concatenate([labeled_indices, unlabeled_indices], axis=0)

    train_weak, train_strong, eval_transform = make_transforms(cfg.img_size)
    if args.train_aug == "none":
        train_transform = eval_transform
    elif args.train_aug == "weak":
        train_transform = train_weak
    else:
        train_transform = train_strong

    labeled_train_ds = IndexedImageDataset(dataset_root, train_entries, labeled_indices, train_transform)
    val_ds = IndexedImageDataset(dataset_root, val_entries, np.arange(len(val_entries)), eval_transform)
    test_ds = IndexedImageDataset(dataset_root, test_entries, np.arange(len(test_entries)), eval_transform)
    train_eval_ds = IndexedImageDataset(dataset_root, train_entries, ordered_train_indices, eval_transform, return_index=True)

    labeled_loader = DataLoader(labeled_train_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.get("num_workers", 4), pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=torch.cuda.is_available())
    train_eval_loader = DataLoader(train_eval_ds, batch_size=eval_batch_size, shuffle=False, num_workers=cfg.get("num_workers", 4), pin_memory=torch.cuda.is_available())

    model = build_model(cfg, args.checkpoint)
    optimizer = build_optimizer(model, optim_name, lr, weight_decay, momentum)

    metrics_csv = output_dir / "metrics.csv"
    pseudo_dir = output_dir / "pseudo_labels"
    pseudo_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(
        output_dir / "run_config.yaml",
        {
            "config_path": str(args.config),
            "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            "seed": seed,
            "train_split": train_split,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "optimizer": optim_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "train_aug": args.train_aug,
            "num_rounds": args.num_rounds,
            "warmup_epochs": warmup_epochs,
            "round_epochs": round_epochs,
            "pseudo_loss_weight": args.pseudo_loss_weight,
            "confidence_threshold": args.confidence_threshold,
            "confidence_weighting": args.confidence_weighting,
            "temperature": args.temperature,
            "k": args.k,
            "alpha": args.alpha,
            "sigma": args.sigma,
            "max_prop_iters": args.max_prop_iters,
            "prop_tol": args.prop_tol,
        },
    )

    LOG.info("Using dataset=%s dataset_root=%s", cfg.dataset, dataset_root)
    LOG.info("Using labeled split file for seed=%d with %d labeled / %d unlabeled train samples", seed, len(labeled_indices), len(unlabeled_indices))
    LOG.info("Warmup epochs=%d, round epochs=%d, rounds=%d, optimizer=%s, lr=%.6f", warmup_epochs, round_epochs, args.num_rounds, optim_name, lr)
    LOG.info("Graph settings: k=%d alpha=%.3f sigma=%s temperature=%.3f confidence_threshold=%.3f", args.k, args.alpha, "median-heuristic" if args.sigma is None else f"{args.sigma:.6f}", args.temperature, args.confidence_threshold)

    best_val_acc = -1.0
    global_epoch = 0

    def evaluate_and_log(stage: str, round_idx: int, epoch_idx: int, extra: Optional[Dict[str, object]] = None) -> None:
        nonlocal best_val_acc
        val_metrics = evaluate(model, val_loader, "val")
        test_metrics = evaluate(model, test_loader, "test")
        row: Dict[str, object] = {
            "stage": stage,
            "round": round_idx,
            "epoch": epoch_idx,
            "global_epoch": global_epoch,
            **val_metrics,
            **test_metrics,
        }
        if extra is not None:
            row.update(extra)
        append_metrics(metrics_csv, row)
        LOG.info(
            "%s round=%d epoch=%d val_acc=%.4f test_acc=%.4f",
            stage,
            round_idx,
            epoch_idx,
            val_metrics["val_acc"],
            test_metrics["test_acc"],
        )
        if val_metrics["val_acc"] > best_val_acc:
            best_val_acc = val_metrics["val_acc"]
            save_checkpoint(
                output_dir / "best_model.pth",
                model,
                optimizer,
                {
                    "best_val_acc": best_val_acc,
                    "round": round_idx,
                    "epoch": epoch_idx,
                    "global_epoch": global_epoch,
                },
            )

    start_time = time.time()

    LOG.info("Starting supervised warmup")
    for epoch_idx in range(1, warmup_epochs + 1):
        train_metrics = train_one_epoch(model, labeled_loader, optimizer)
        global_epoch += 1
        LOG.info(
            "warmup epoch=%d/%d train_loss=%.4f train_acc=%.4f",
            epoch_idx,
            warmup_epochs,
            train_metrics["train_loss"],
            train_metrics["train_acc"],
        )
        evaluate_and_log("warmup", 0, epoch_idx, train_metrics)
        save_checkpoint(
            output_dir / "latest_model.pth",
            model,
            optimizer,
            {"best_val_acc": best_val_acc, "round": 0, "epoch": epoch_idx, "global_epoch": global_epoch},
        )

    for round_idx in range(1, args.num_rounds + 1):
        LOG.info("Propagation round %d/%d", round_idx, args.num_rounds)

        features, train_targets, extracted_indices = compute_embeddings(model, train_eval_loader)
        if not np.array_equal(extracted_indices, ordered_train_indices):
            raise RuntimeError("Train embedding order does not match the expected labeled+unlabeled order.")

        graph, sigma_value = build_knn_graph(features, args.k, args.sigma)
        propagated_scores, prop_steps, prop_delta = label_propagation(
            graph=graph,
            labeled_targets=train_targets[: len(labeled_indices)],
            num_classes=cfg.num_classes,
            num_labeled=len(labeled_indices),
            alpha=args.alpha,
            max_iters=args.max_prop_iters,
            tol=args.prop_tol,
            temperature=args.temperature,
        )

        unlabeled_scores = propagated_scores[len(labeled_indices) :]
        unlabeled_preds = unlabeled_scores.argmax(axis=1)
        unlabeled_conf = unlabeled_scores.max(axis=1)
        unlabeled_true = train_targets[len(labeled_indices) :]
        pseudo_acc_all = float((unlabeled_preds == unlabeled_true).mean())

        selected_mask = unlabeled_conf >= args.confidence_threshold
        selected_indices = unlabeled_indices[selected_mask]
        selected_labels = unlabeled_preds[selected_mask]
        selected_conf = unlabeled_conf[selected_mask]
        selected_true = unlabeled_true[selected_mask]
        pseudo_acc_selected = float((selected_labels == selected_true).mean()) if len(selected_indices) else 0.0

        np.savez(
            pseudo_dir / f"round_{round_idx:02d}.npz",
            unlabeled_indices=unlabeled_indices,
            pseudo_labels=unlabeled_preds,
            pseudo_confidence=unlabeled_conf,
            selected_indices=selected_indices,
            selected_labels=selected_labels,
            selected_confidence=selected_conf,
        )

        LOG.info(
            "Propagation stats round=%d sigma=%.6f steps=%d delta=%.8f pseudo_acc_all=%.4f selected=%d/%d pseudo_acc_selected=%.4f conf_mean=%.4f",
            round_idx,
            sigma_value,
            prop_steps,
            prop_delta,
            pseudo_acc_all,
            len(selected_indices),
            len(unlabeled_indices),
            pseudo_acc_selected,
            float(selected_conf.mean()) if len(selected_conf) else 0.0,
        )

        if len(selected_indices):
            pseudo_weights = np.ones(len(selected_indices), dtype=np.float32) * args.pseudo_loss_weight
            if args.confidence_weighting:
                pseudo_weights = pseudo_weights * selected_conf.astype(np.float32)
            pseudo_train_ds = IndexedImageDataset(
                dataset_root=dataset_root,
                entries=train_entries,
                indices=selected_indices,
                transform=train_transform,
                override_labels=selected_labels,
                sample_weights=pseudo_weights,
            )
            round_train_ds = ConcatDataset([labeled_train_ds, pseudo_train_ds])
        else:
            round_train_ds = labeled_train_ds

        round_loader = DataLoader(
            round_train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=torch.cuda.is_available(),
        )

        for epoch_idx in range(1, round_epochs + 1):
            train_metrics = train_one_epoch(model, round_loader, optimizer)
            global_epoch += 1
            LOG.info(
                "round=%d epoch=%d/%d train_loss=%.4f train_acc=%.4f pseudo_used=%d",
                round_idx,
                epoch_idx,
                round_epochs,
                train_metrics["train_loss"],
                train_metrics["train_acc"],
                len(selected_indices),
            )
            evaluate_and_log(
                "round_train",
                round_idx,
                epoch_idx,
                {
                    **train_metrics,
                    "pseudo_selected": int(len(selected_indices)),
                    "pseudo_total": int(len(unlabeled_indices)),
                    "pseudo_acc_all": pseudo_acc_all,
                    "pseudo_acc_selected": pseudo_acc_selected,
                    "pseudo_conf_mean": float(selected_conf.mean()) if len(selected_conf) else 0.0,
                    "graph_sigma": sigma_value,
                    "graph_prop_steps": prop_steps,
                    "graph_prop_delta": prop_delta,
                },
            )
            save_checkpoint(
                output_dir / "latest_model.pth",
                model,
                optimizer,
                {
                    "best_val_acc": best_val_acc,
                    "round": round_idx,
                    "epoch": epoch_idx,
                    "global_epoch": global_epoch,
                },
            )

    total_minutes = (time.time() - start_time) / 60.0
    LOG.info("Finished training in %.2f minutes. Best val acc=%.4f", total_minutes, best_val_acc)


if __name__ == "__main__":
    main()
