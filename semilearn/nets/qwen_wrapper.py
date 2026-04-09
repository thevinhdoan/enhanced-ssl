import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_SRC_ROOT = REPO_ROOT / "Qwen3-VL-Embedding"
DEFAULT_MODEL_PATH = REPO_ROOT / "Qwen3-VL-Embedding-2B"
DEFAULT_INSTRUCTION = "Represent the user's input."
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


if str(QWEN_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(QWEN_SRC_ROOT))

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402


class _PostAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck: int, scaler: float):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size)
        self.scaler = scaler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scaler * self.up(self.act(self.down(x)))


class _LoRAHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, rank: int):
        super().__init__()
        self.base = nn.Linear(hidden_size, num_classes)
        self.lora_a = nn.Linear(hidden_size, rank, bias=False)
        self.lora_b = nn.Linear(rank, num_classes, bias=False)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x))


class QwenEmbeddingWrapper(nn.Module):
    def __init__(
        self,
        embedder: Qwen3VLEmbedder,
        num_features: int,
        num_classes: int,
        peft_config: Optional[Any] = None,
        instruction: str = DEFAULT_INSTRUCTION,
    ):
        super().__init__()
        self.embedder = embedder
        self.model = embedder.model
        self.processor = embedder.processor
        self.num_features = num_features
        self.instruction = instruction
        self.peft_config = peft_config
        self.adapter: Optional[nn.Module] = None
        self.head: nn.Module = nn.Linear(num_features, num_classes)

        self.register_buffer("imagenet_mean", IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer("imagenet_std", IMAGENET_STD.clone(), persistent=False)

        self.image_prompt_template = self.processor.apply_chat_template(
            [[
                {"role": "system", "content": [{"type": "text", "text": self.instruction}]},
                {"role": "user", "content": [{"type": "image"}]},
            ]],
            add_generation_prompt=True,
            tokenize=False,
        )[0]

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        method_name = getattr(peft_config, "method_name", None) if peft_config is not None else None
        if method_name == "adaptformer":
            bottleneck = int(getattr(peft_config, "adapter_bottleneck", 64))
            scaler = float(getattr(peft_config, "adapter_scaler", 0.1))
            self.adapter = _PostAdapter(num_features, bottleneck, scaler)
        elif method_name in ["lora", "lora_1"]:
            rank = max(int(getattr(peft_config, "lora_bottleneck", 8)), 1)
            self.head = _LoRAHead(num_features, num_classes, rank)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _build_texts(self, batch_size: int) -> list[str]:
        return [self.image_prompt_template] * batch_size

    def _to_pixel_tensors(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype)
        std = self.imagenet_std.to(device=x.device, dtype=x.dtype)
        pixels = (x * std + mean).clamp(0.0, 1.0)
        return pixels.detach().to(device="cpu", dtype=torch.float32, non_blocking=False).contiguous()

    def _move_processed_inputs(self, processed_inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_device = self.device
        moved: Dict[str, Any] = {}
        for key, value in processed_inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(model_device, non_blocking=True)
            else:
                moved[key] = value
        return moved

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        images = self._to_pixel_tensors(x)
        texts = self._build_texts(images.shape[0])

        with torch.inference_mode():
            processed_inputs = self.processor(
                text=texts,
                images=images,
                padding=True,
                do_resize=False,
                return_tensors="pt",
            )
            processed_inputs = self._move_processed_inputs(processed_inputs)
            outputs = self.embedder.forward(processed_inputs)
            features = self.embedder._pooling_last(
                outputs["last_hidden_state"],
                outputs["attention_mask"],
            )

        return features.float()

    def forward_head(self, features: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.adapter is not None:
            features = self.adapter(features)
        return features if pre_logits else self.head(features)

    def forward(self, x: torch.Tensor, only_fc: bool = False, only_feat: bool = False):
        assert not (only_fc and only_feat), "only_fc and only_feat cannot be True at the same time"

        if (not only_fc) and (not only_feat):
            feat = self.forward_features(x)
            logits = self.forward_head(feat)
            return {"feat": feat, "logits": logits}
        if only_feat:
            return self.forward_features(x)
        return self.forward_head(x)



def qwen_builder(pretrained: bool = True, pretrained_path: str = "", peft_config: Optional[Any] = None, **kwargs):
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
