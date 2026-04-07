import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image


REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_SRC_ROOT = REPO_ROOT / "Qwen3-VL-Embedding"
DEFAULT_MODEL_PATH = REPO_ROOT / "Qwen3-VL-Embedding-2B"
DEFAULT_INSTRUCTION = "Represent the user's input."
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


if str(QWEN_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_SRC_ROOT))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402


class _PostAdapter(nn.Module):
    def __init__(self, hidden_size, bottleneck, scaler):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        self.scaler = scaler

    def forward(self, x):
        return x + self.scaler * self.up(self.act(self.down(x)))


class _LoRAHead(nn.Module):
    def __init__(self, hidden_size, num_classes, rank):
        super().__init__()
        self.base = nn.Linear(hidden_size, num_classes)
        self.lora_a = nn.Linear(hidden_size, rank, bias=False)
        self.lora_b = nn.Linear(rank, num_classes, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x))


class QwenEmbeddingWrapper(nn.Module):
    def __init__(self, embedder, num_features, num_classes, peft_config=None, instruction=DEFAULT_INSTRUCTION):
        super().__init__()
        self.embedder = embedder
        self.model = embedder.model
        self.processor = embedder.processor
        self.num_features = num_features
        self.instruction = instruction
        self.peft_config = peft_config
        self.adapter = None
        self.head = nn.Linear(num_features, num_classes)

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if peft_config is not None and getattr(peft_config, "method_name", None) == "adaptformer":
            bottleneck = int(getattr(peft_config, "adapter_bottleneck", 64))
            scaler = float(getattr(peft_config, "adapter_scaler", 0.1))
            self.adapter = _PostAdapter(num_features, bottleneck, scaler)
        elif peft_config is not None and getattr(peft_config, "method_name", None) in ["lora", "lora_1"]:
            rank = int(getattr(peft_config, "lora_bottleneck", 8))
            rank = max(rank, 1)
            self.head = _LoRAHead(num_features, num_classes, rank)

    def _to_pil_images(self, x: torch.Tensor):
        mean = IMAGENET_MEAN.to(device=x.device, dtype=x.dtype)
        std = IMAGENET_STD.to(device=x.device, dtype=x.dtype)
        x = (x * std + mean).clamp(0.0, 1.0)
        return [to_pil_image(image.detach().cpu()) for image in x]

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        images = self._to_pil_images(x)
        inputs = [{"image": image, "instruction": self.instruction} for image in images]
        with torch.inference_mode():
            features = self.embedder.process(inputs)
        return features.float()

    def forward_head(self, features: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.adapter is not None:
            features = self.adapter(features)
        return features if pre_logits else self.head(features)

    def forward(self, x, only_fc=False, only_feat=False):
        assert not (only_fc and only_feat), "only_fc and only_feat cannot be True at the same time"

        if (not only_fc) and (not only_feat):
            feat = self.forward_features(x)
            logits = self.forward_head(feat)
            return {"feat": feat, "logits": logits}
        if only_feat:
            return self.forward_features(x)
        return self.forward_head(x)


def qwen_builder(pretrained=True, pretrained_path="", peft_config=None, **kwargs):
    assert pretrained, "Qwen embedding wrapper expects pretrained weights."

    model_path = Path(pretrained_path) if pretrained_path not in ["", None] else DEFAULT_MODEL_PATH
    num_classes = kwargs["num_classes"]

    print(
        f"Building Qwen3-VL-Embedding model with pretrained={pretrained}, "
        f"pretrained_path={model_path}, kwargs={kwargs}"
    )

    embedder = Qwen3VLEmbedder(
        model_name_or_path=str(model_path),
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    hidden_size = int(embedder.model.config.text_config.hidden_size)
    return QwenEmbeddingWrapper(
        embedder=embedder,
        num_features=hidden_size,
        num_classes=num_classes,
        peft_config=peft_config,
    )
