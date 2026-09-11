from __future__ import annotations
import warnings
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from .constants import IMAGENET_MEAN, IMAGENET_STD

class TimmFeatureBackbone(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError(
                "timm is required. Run: pip install -r requirements.txt"
            ) from exc
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )
        self.num_features = int(self.model.num_features)

    def forward(self, x):
        out = self.model(x)
        if out.ndim > 2:
            out = out.flatten(1)
        return out

def load_backbone(name: str, pretrained: bool = True, dino_source: str = "auto"):
    name = name.lower()
    if name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        return model, dim, "torchvision:resnet50"

    if name == "convnextv2_tiny":
        model_name = "convnextv2_tiny.fcmae_ft_in1k"
        model = TimmFeatureBackbone(model_name, pretrained)
        return model, model.num_features, f"timm:{model_name}"

    if name == "dinov2_vits14":
        source = dino_source.lower()
        if source not in {"hub", "timm", "auto"}:
            raise ValueError("dino_source must be hub, timm, or auto")

        if source in {"hub", "auto"}:
            try:
                model = torch.hub.load(
                    "facebookresearch/dinov2",
                    "dinov2_vits14",
                    pretrained=pretrained,
                )
                return model, 384, "meta-pytorch-hub:dinov2_vits14"
            except Exception as exc:
                if source == "hub":
                    raise
                warnings.warn(
                    "Meta Hub load failed; falling back to timm. "
                    f"Original error: {exc}"
                )

        model_name = "vit_small_patch14_dinov2.lvd142m"
        model = TimmFeatureBackbone(model_name, pretrained)
        return model, model.num_features, f"timm:{model_name}"

    raise ValueError(
        "model must be resnet50, convnextv2_tiny, or dinov2_vits14"
    )

class FrozenFeatureClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int = 10,
        adapter_dim: int = 0,
        dropout: float = 0.10,
        pretrained: bool = True,
        dino_source: str = "auto",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.adapter_dim = int(adapter_dim)
        self.dino_source_requested = dino_source

        backbone, feature_dim, actual_source = load_backbone(
            backbone_name, pretrained, dino_source
        )
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.actual_source = actual_source

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        if adapter_dim > 0:
            self.adapter = nn.Sequential(
                nn.Linear(self.feature_dim, adapter_dim),
                nn.LayerNorm(adapter_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_dim = adapter_dim
        else:
            self.adapter = nn.Identity()
            head_dim = self.feature_dim

        self.classifier = nn.Linear(head_dim, num_classes)

        self.register_buffer(
            "pixel_mean",
            torch.tensor(IMAGENET_MEAN).view(1,3,1,1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(IMAGENET_STD).view(1,3,1,1),
            persistent=False,
        )

    def normalize(self, x):
        return (x - self.pixel_mean) / self.pixel_std

    def encode(self, x):
        x = self.normalize(x)
        with torch.no_grad():
            feat = self.backbone(x)
        if isinstance(feat, dict):
            if "x_norm_clstoken" in feat:
                feat = feat["x_norm_clstoken"]
            else:
                raise RuntimeError(f"Unsupported feature dict: {feat.keys()}")
        if feat.ndim > 2:
            feat = feat.flatten(1)
        return self.adapter(feat)

    def forward(self, x, return_embedding: bool = False):
        emb = self.encode(x)
        logits = self.classifier(emb)
        return (logits, emb) if return_embedding else logits

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def checkpoint_config(self):
        return {
            "backbone_name": self.backbone_name,
            "num_classes": self.num_classes,
            "adapter_dim": self.adapter_dim,
            "dino_source_requested": self.dino_source_requested,
            "actual_source": self.actual_source,
            "feature_dim": self.feature_dim,
        }
