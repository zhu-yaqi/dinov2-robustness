from __future__ import annotations
import argparse, platform, sys
import torch, torchvision

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--check-models", action="store_true")
    p.add_argument("--dino-source", choices=["hub","timm","auto"], default="auto")
    return p.parse_args()

def main():
    args = parse_args()
    print("="*60)
    print("Python:", sys.version.replace("\n"," "))
    print("OS:", platform.platform())
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch CUDA runtime:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    import timm
    print("timm:", timm.__version__)
    print("device:", device)

    x = torch.rand(1,3,224,224, device=device)
    if not args.check_models:
        print("Basic environment check passed.")
        return

    from src.models import FrozenFeatureClassifier
    from src.utils import format_count, total_parameter_count, trainable_parameter_count

    for name in ["resnet50","convnextv2_tiny","dinov2_vits14"]:
        print("\nChecking", name)
        model = FrozenFeatureClassifier(
            name, adapter_dim=0, dino_source=args.dino_source
        ).to(device).eval()
        with torch.inference_mode():
            y = model(x)
        print("output:", tuple(y.shape))
        print("source:", model.actual_source)
        print("total:", format_count(total_parameter_count(model)))
        print("trainable:", format_count(trainable_parameter_count(model)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\nAll model checks passed.")

if __name__ == "__main__":
    main()
