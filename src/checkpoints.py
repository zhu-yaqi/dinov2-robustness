from __future__ import annotations
from typing import Any
import torch
from .models import FrozenFeatureClassifier
from .utils import ensure_parent

def save_checkpoint(
    path: str,
    model: FrozenFeatureClassifier,
    epoch: int,
    val_accuracy: float,
    label: str,
    training_config: dict[str, Any],
):
    ensure_parent(path)
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "val_accuracy": float(val_accuracy),
        "label": label,
        "model_config": model.checkpoint_config(),
        "training_config": training_config,
        # The frozen backbone is intentionally not saved.
        "trainable_state": {
            "adapter": model.adapter.state_dict(),
            "classifier": model.classifier.state_dict(),
        },
    }
    torch.save(payload, path)

def load_checkpoint(
    path: str,
    device: torch.device,
    override_dino_source: str | None = None,
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["model_config"]
    source = override_dino_source or cfg.get("dino_source_requested", "auto")

    model = FrozenFeatureClassifier(
        backbone_name=cfg["backbone_name"],
        num_classes=int(cfg.get("num_classes", 10)),
        adapter_dim=int(cfg.get("adapter_dim", 0)),
        dino_source=source,
        pretrained=True,
    )
    model.adapter.load_state_dict(payload["trainable_state"]["adapter"])
    model.classifier.load_state_dict(payload["trainable_state"]["classifier"])
    model.to(device).eval()
    return model, payload
