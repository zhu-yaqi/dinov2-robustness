from pathlib import Path
from torchvision import datasets

def main():
    root = Path("data")
    root.mkdir(parents=True, exist_ok=True)
    print("Downloading/checking CIFAR-10 training set...")
    datasets.CIFAR10(root=str(root), train=True, download=True)
    print("Downloading/checking CIFAR-10 test set...")
    datasets.CIFAR10(root=str(root), train=False, download=True)
    print("Done:", root.resolve())

if __name__ == "__main__":
    main()
