"""Waiv robust encoders — Phaet and Mascaret — as LazySlide ``ImageModel``s.

Phaet and Mascaret are the fine-tuned, acquisition-robust encoders released with
Filiot et al., "Robustifying pathology foundation models via fine-tuning"
(arXiv:2607.22861, Waiv). Phaet is a fine-tuned Phikon-v2 (1024-d); Mascaret is a
fine-tuned Midnight-12k (1536-d, L2-normalised CLS). Both ship custom HF code
(``trust_remote_code=True``) and are gated under a non-commercial academic license,
so they are not in LazySlide's registry — we register them here.

Load/transform/feature contract matches each model's HuggingFace card (retrieved
2026-07-31): ``AutoModel.from_pretrained(repo, trust_remote_code=True)`` exposing
``.encode(pixel_values) -> (B, D)``; inputs resized to 224, centre-cropped, and
normalised with the model's own ``pixel_mean``/``pixel_std``.
"""

from __future__ import annotations

import torch

from lazyslide.models._model_registry import register
from lazyslide.models._utils import hf_access
from lazyslide.models.base import ImageModel, ModelTask


def _resize_crop_normalize(mean, std):
    from torchvision.transforms import v2

    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(224),
            v2.CenterCrop(224),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


@register(
    key="phaet",
    task=ModelTask.vision,
    is_gated=True,
    license="Waiv non-commercial academic license",
    license_url="https://huggingface.co/wearewaiv/phaet/blob/main/LICENSE.pdf",
    description="Phaet: acquisition-robust fine-tuned Phikon-v2 (Waiv)",
    commercial=False,
    hf_url="https://huggingface.co/wearewaiv/phaet",
    paper_url="https://doi.org/10.48550/arXiv.2607.22861",
    encode_dim=1024,
    vision_encoder="phikon-v2",
)
class Phaet(ImageModel):
    def __init__(self, model_path=None, token=None):
        from transformers import AutoModel

        with hf_access("wearewaiv/phaet"):
            self.model = AutoModel.from_pretrained(
                "wearewaiv/phaet", trust_remote_code=True, token=token
            )

    def get_transform(self):
        return _resize_crop_normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    @torch.inference_mode()
    def encode_image(self, image) -> torch.Tensor:
        return self.model.encode(image).float()


@register(
    key="mascaret",
    task=ModelTask.vision,
    is_gated=True,
    license="Waiv non-commercial academic license",
    license_url="https://huggingface.co/wearewaiv/mascaret/blob/main/LICENSE.pdf",
    description="Mascaret: acquisition-robust fine-tuned Midnight-12k (Waiv)",
    commercial=False,
    hf_url="https://huggingface.co/wearewaiv/mascaret",
    paper_url="https://doi.org/10.48550/arXiv.2607.22861",
    encode_dim=1536,
    vision_encoder="midnight",
)
class Mascaret(ImageModel):
    def __init__(self, model_path=None, token=None):
        from transformers import AutoModel

        with hf_access("wearewaiv/mascaret"):
            self.model = AutoModel.from_pretrained(
                "wearewaiv/mascaret", trust_remote_code=True, token=token
            )

    def get_transform(self):
        return _resize_crop_normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    @torch.inference_mode()
    def encode_image(self, image) -> torch.Tensor:
        return self.model.encode(image).float()
