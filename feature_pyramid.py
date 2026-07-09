"""
Feature Pyramid Generator Module

Implements shared-weight initial convolution blocks that process multi-resolution
inputs to generate aligned feature pyramids for UNet++ encoder integration.

Key Design:
- Shared weights across all resolution levels (parameter efficient)
- Resolution-invariant feature learning
- Spatial alignment for downstream fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class SharedConvBlock(nn.Module):
    """
    Shared-weight convolution block applied to all resolution levels.

    This learns resolution-invariant features while being parameter efficient.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        num_blocks: int = 2
    ):
        """
        Args:
            in_channels: Input channels (3 for RGB)
            out_channels: Output feature channels
            num_blocks: Number of conv-bn-relu blocks
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        # Build sequential conv blocks
        layers = []
        current_channels = in_channels

        for i in range(num_blocks):
            layers.extend([
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            current_channels = out_channels

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            Feature tensor (B, out_channels, H, W)
        """
        return self.conv_block(x)


class FeaturePyramidModule(nn.Module):
    """
    Feature Pyramid Module that processes multi-resolution inputs through
    shared-weight convolutions to generate aligned feature pyramids.

    Architecture:
        Input: {1024: (B,3,1024,1024), 512: (B,3,512,512), 256: (B,3,256,256)}
                              |
                   Shared Conv Block (same weights)
                              |
        Output: {1024: (B,64,1024,1024), 512: (B,64,512,512), 256: (B,64,256,256)}
    """

    def __init__(
        self,
        in_channels: int = 3,
        pyramid_channels: int = 64,
        resolutions: List[int] = [1024, 512, 256],
        num_conv_blocks: int = 2
    ):
        """
        Args:
            in_channels: Input image channels (3 for RGB)
            pyramid_channels: Output channels for pyramid features
            resolutions: List of resolution levels
            num_conv_blocks: Number of shared conv blocks
        """
        super().__init__()

        self.in_channels = in_channels
        self.pyramid_channels = pyramid_channels
        self.resolutions = sorted(resolutions, reverse=True)  # [1024, 512, 256]
        self.num_conv_blocks = num_conv_blocks

        # Shared weight convolution block
        # Applied to all resolutions with same parameters
        self.shared_conv = SharedConvBlock(
            in_channels=in_channels,
            out_channels=pyramid_channels,
            num_blocks=num_conv_blocks
        )

    def forward(self, images: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Process multi-resolution images through shared convolutions.

        Args:
            images: Dictionary mapping resolution -> image tensor
                   {1024: (B,3,1024,1024), 512: (B,3,512,512), 256: (B,3,256,256)}

        Returns:
            Dictionary mapping resolution -> feature tensor
                   {1024: (B,64,1024,1024), 512: (B,64,512,512), 256: (B,64,256,256)}
        """
        features = {}

        for res in self.resolutions:
            if res not in images:
                raise ValueError(f"Resolution {res} not found in input images")

            # Apply shared convolution to this resolution
            features[res] = self.shared_conv(images[res])

        return features

    def get_num_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdaptiveFeaturePyramid(nn.Module):
    """
    Advanced Feature Pyramid with resolution-specific adaptation layers.

    This variant adds lightweight adaptation layers after shared convolutions
    to allow resolution-specific feature refinement.
    """

    def __init__(
        self,
        in_channels: int = 3,
        pyramid_channels: int = 64,
        resolutions: List[int] = [1024, 512, 256],
        num_conv_blocks: int = 2,
        use_adaptation: bool = True
    ):
        """
        Args:
            in_channels: Input image channels
            pyramid_channels: Output channels for pyramid features
            resolutions: List of resolution levels
            num_conv_blocks: Number of shared conv blocks
            use_adaptation: Whether to use resolution-specific adaptation
        """
        super().__init__()

        self.in_channels = in_channels
        self.pyramid_channels = pyramid_channels
        self.resolutions = sorted(resolutions, reverse=True)
        self.use_adaptation = use_adaptation

        # Shared feature extraction
        self.shared_conv = SharedConvBlock(
            in_channels=in_channels,
            out_channels=pyramid_channels,
            num_blocks=num_conv_blocks
        )

        # Optional: Resolution-specific adaptation layers
        if use_adaptation:
            self.adaptation_layers = nn.ModuleDict({
                str(res): nn.Sequential(
                    nn.Conv2d(pyramid_channels, pyramid_channels, 1, bias=False),
                    nn.BatchNorm2d(pyramid_channels),
                    nn.ReLU(inplace=True)
                )
                for res in self.resolutions
            })

    def forward(self, images: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Process images with shared conv + optional adaptation.

        Args:
            images: Dict mapping resolution -> image tensor

        Returns:
            Dict mapping resolution -> adapted feature tensor
        """
        features = {}

        for res in self.resolutions:
            if res not in images:
                raise ValueError(f"Resolution {res} not found in input")

            # Shared feature extraction
            feat = self.shared_conv(images[res])

            # Optional resolution-specific adaptation
            if self.use_adaptation:
                feat = self.adaptation_layers[str(res)](feat)

            features[res] = feat

        return features

    def get_num_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing and example usage
if __name__ == '__main__':
    print("Testing Feature Pyramid Modules...")

    batch_size = 2
    resolutions = [1024, 512, 256]

    # Create dummy multi-resolution input
    images = {
        1024: torch.randn(batch_size, 3, 1024, 1024),
        512: torch.randn(batch_size, 3, 512, 512),
        256: torch.randn(batch_size, 3, 256, 256)
    }

    print("\nInput:")
    for res, img in images.items():
        print(f"  {res}x{res}: {img.shape}")

    # Test basic Feature Pyramid
    print("\n1. Testing FeaturePyramidModule (Shared Conv Only)...")
    pyramid = FeaturePyramidModule(
        in_channels=3,
        pyramid_channels=64,
        resolutions=resolutions,
        num_conv_blocks=2
    )

    features = pyramid(images)
    print("\nOutput:")
    for res, feat in features.items():
        print(f"  {res}x{res}: {feat.shape}")

    num_params = pyramid.get_num_parameters()
    print(f"\nParameters: {num_params:,}")

    # Test adaptive pyramid
    print("\n2. Testing AdaptiveFeaturePyramid (Shared + Adaptation)...")
    adaptive_pyramid = AdaptiveFeaturePyramid(
        in_channels=3,
        pyramid_channels=64,
        resolutions=resolutions,
        num_conv_blocks=2,
        use_adaptation=True
    )

    adaptive_features = adaptive_pyramid(images)
    print("\nOutput:")
    for res, feat in adaptive_features.items():
        print(f"  {res}x{res}: {feat.shape}")

    adaptive_params = adaptive_pyramid.get_num_parameters()
    print(f"\nParameters: {adaptive_params:,}")
    print(f"Additional params from adaptation: {adaptive_params - num_params:,}")

    # Verify gradient flow
    print("\n3. Testing gradient flow...")
    loss = features[1024].mean()
    loss.backward()

    has_gradients = all(p.grad is not None for p in pyramid.parameters())
    print(f"  All parameters have gradients: {has_gradients}")

    print("\n✓ Feature Pyramid Module tests PASSED")
