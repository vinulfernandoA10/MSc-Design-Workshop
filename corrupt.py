"""Test-time image corruptions at 5 severities, following CIFAR-10-C (Hendrycks & Dietterich, 2019).

Operates on [0,1] image tensors (N, C, H, W). Noise uses a fixed seed per severity, so every model
is evaluated on exactly the same corrupted images.
"""
import math
import torch
import torchvision.transforms.functional as TF
from torchvision.io import decode_jpeg, encode_jpeg

SEVERITY = {  # parameter per severity 1..5; values from CIFAR-10-C, brightness adapted to grayscale
    "noise": [.04, .06, .08, .09, .10],         # Gaussian noise std
    "blur": [.4, .6, .7, .8, 1.0],              # Gaussian blur sigma (pixels)
    "contrast": [.75, .5, .4, .3, .15],         # contrast factor (1 = unchanged)
    "brightness": [.05, .1, .15, .2, .3],       # additive brightness shift
    "jpeg": [80, 65, 58, 50, 40],               # JPEG quality
}


def corrupt(x, name, sev):
    v = SEVERITY[name][sev - 1]
    if name == "noise":
        x = x + torch.randn(x.shape, generator=torch.Generator().manual_seed(sev)) * v
    elif name == "blur":
        x = TF.gaussian_blur(x, 2 * math.ceil(3 * v) + 1, v)
    elif name == "contrast":
        m = x.mean((1, 2, 3), keepdim=True); x = (x - m) * v + m
    elif name == "brightness":
        x = x + v
    elif name == "jpeg":
        u8 = (x * 255).round().to(torch.uint8)
        x = torch.stack([decode_jpeg(encode_jpeg(i, quality=v)) for i in u8]).float() / 255
    return x.clamp(0, 1)


if __name__ == "__main__":
    # self-check: shape and range preserved, and damage grows with severity for every corruption
    x = torch.rand(8, 3, 28, 28) * .6 + .2
    for c in SEVERITY:
        diffs = [(corrupt(x, c, s) - x).abs().mean().item() for s in range(1, 6)]
        assert corrupt(x, c, 3).shape == x.shape and 0 <= corrupt(x, c, 5).min() and corrupt(x, c, 5).max() <= 1
        assert all(a <= b + 1e-3 for a, b in zip(diffs, diffs[1:])), (c, diffs)
        print(f"{c:<10} mean |change| by severity: " + " ".join(f"{d:.3f}" for d in diffs))
    print("ok")
