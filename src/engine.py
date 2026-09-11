from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .corruptions import apply_corruption, random_occlusion

@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    consistency_loss: float = 0.0

def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
        enabled=enabled and device.type == "cuda",
    )

def train_baseline_epoch(model, loader, optimizer, device, amp=True):
    model.train()
    total_loss = total_correct = total_count = 0
    pbar = tqdm(loader, desc="train", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with _autocast(device, amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        n = labels.size(0)
        total_loss += float(loss.item()) * n
        total_correct += int((logits.argmax(1) == labels).sum().item())
        total_count += n
        pbar.set_postfix(
            loss=f"{total_loss/total_count:.4f}",
            acc=f"{100*total_correct/total_count:.2f}",
        )

    return EpochMetrics(total_loss/total_count, total_correct/total_count)

def train_consistency_epoch(
    model,
    loader,
    optimizer,
    device,
    lambda_consistency: float,
    occlusion_min: float,
    occlusion_max: float,
    amp=True,
):
    model.train()
    total_loss = total_cons = total_correct = total_count = 0
    pbar = tqdm(loader, desc="train-consistency", leave=False)

    for clean, labels in pbar:
        clean = clean.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        occluded = random_occlusion(clean, occlusion_min, occlusion_max)

        optimizer.zero_grad(set_to_none=True)
        paired = torch.cat([clean, occluded], dim=0)

        with _autocast(device, amp):
            logits_all, emb_all = model(paired, return_embedding=True)
            b = clean.size(0)
            clean_logits = logits_all[:b]
            clean_emb = emb_all[:b]
            occ_emb = emb_all[b:]

            cls_loss = F.cross_entropy(clean_logits, labels)
            clean_norm = F.normalize(clean_emb.float(), dim=1)
            occ_norm = F.normalize(occ_emb.float(), dim=1)
            cons_loss = 1.0 - (clean_norm * occ_norm).sum(dim=1).mean()
            loss = cls_loss + lambda_consistency * cons_loss

        loss.backward()
        optimizer.step()

        n = labels.size(0)
        total_loss += float(loss.item()) * n
        total_cons += float(cons_loss.item()) * n
        total_correct += int((clean_logits.argmax(1) == labels).sum().item())
        total_count += n
        pbar.set_postfix(
            loss=f"{total_loss/total_count:.4f}",
            cons=f"{total_cons/total_count:.4f}",
            acc=f"{100*total_correct/total_count:.2f}",
        )

    return EpochMetrics(
        total_loss/total_count,
        total_correct/total_count,
        total_cons/total_count,
    )

@torch.inference_mode()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    amp=True,
    corruption: Optional[str] = None,
    level: Optional[int] = None,
    corruption_seed: int = 1234,
):
    model.eval()
    total_loss = total_correct = total_count = 0
    generator = torch.Generator(device="cpu").manual_seed(corruption_seed)

    for images, labels in tqdm(
        loader, desc=f"eval:{corruption or 'clean'}", leave=False
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if corruption is not None:
            if level is None:
                raise ValueError("level is required for corrupted evaluation")
            images, _ = apply_corruption(
                images, corruption, level, generator
            )

        with _autocast(device, amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        n = labels.size(0)
        total_loss += float(loss.item()) * n
        total_correct += int((logits.argmax(1) == labels).sum().item())
        total_count += n

    return EpochMetrics(total_loss/total_count, total_correct/total_count)
