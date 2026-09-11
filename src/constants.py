CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CORRUPTION_SPECS = {
    "occlusion": {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50},
    "gaussian_noise": {1: 0.05, 2: 0.10, 3: 0.15, 4: 0.20, 5: 0.25},
    "gaussian_blur": {1: 0.50, 2: 1.00, 3: 1.50, 4: 2.00, 5: 2.50},
    "brightness": {1: 0.80, 2: 0.60, 3: 0.40, 4: 0.25, 5: 0.10},
}
