import torch
from src.corruptions import apply_corruption, fixed_ratio_occlusion

def test_occlusion_preserves_shape_and_range():
    x = torch.ones(4, 3, 32, 32)
    g = torch.Generator().manual_seed(123)
    y = fixed_ratio_occlusion(x, ratio=0.25, generator=g)
    assert y.shape == x.shape
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 1.0
    assert (y == 0).any()

def test_all_corruptions_preserve_shape_and_range():
    x = torch.rand(2, 3, 64, 64)
    for name in ["occlusion","gaussian_noise","gaussian_blur","brightness"]:
        g = torch.Generator().manual_seed(123)
        y, _ = apply_corruption(x, name=name, level=3, generator=g)
        assert y.shape == x.shape
        assert float(y.min()) >= 0.0
        assert float(y.max()) <= 1.0
