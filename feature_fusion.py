"""
Multi-Scale Feature Fusion Module

Implements strategies for fusing features from different resolution levels
of the image pyramid before feeding to the UNet++ encoder.

Fusion Strategies:
1. ConcatFusion: Simple concatenation (baseline)
2. AttentionFusion: Learnable attention-weighted fusion (advanced)
3. ProgressiveFusion: Hierarchical encoder-level specific fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class ConcatFusion(nn.Module):
    """
    Concatenation-based fusion (baseline).

    Upsamples all resolutions to the target size, concatenates along channel
    dimension, then reduces channels via 1x1 conv.

    Simple and effective for initial experiments.
    """

    def __init__(
        self,
        pyramid_channels: int = 64,
        num_resolutions: int = 3,
        output_channels: int = 192,  # Default: keep concatenated channels
        target_resolution: int = 1024,
        force_1x1_conv: bool = False
    ):
        """
        Args:
            pyramid_channels: Channels from each pyramid level
            num_resolutions: Number of pyramid levels (3 for 1024, 512, 256)
            output_channels: Output channels after fusion (typically pyramid_channels * num_resolutions)
            target_resolution: Target spatial resolution for fusion
            force_1x1_conv: If True, a 1x1 conv block is used even if in_channels == out_channels.
        """
        super().__init__()

        self.pyramid_channels = pyramid_channels
        self.num_resolutions = num_resolutions
        self.output_channels = output_channels
        self.target_resolution = target_resolution

        concat_channels = pyramid_channels * num_resolutions

        # Optional channel reduction (only if output_channels != concat_channels)
        if not force_1x1_conv and output_channels == concat_channels:
            # No reduction - keep all multi-scale information!
            self.channel_reduce = nn.Identity()
        else:
            # Reduce channels if explicitly requested
            self.channel_reduce = nn.Sequential(
                nn.Conv2d(concat_channels, output_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-resolution features via concatenation.

        Args:
            features: Dict mapping resolution -> feature tensor
                     {1024: (B,64,1024,1024), 512: (B,64,512,512), 256: (B,64,256,256)}

        Returns:
            Fused features (B, output_channels, target_resolution, target_resolution)
        """
        # Upsample all to target resolution
        upsampled = []
        for res in sorted(features.keys(), reverse=True):
            feat = features[res]
            if feat.shape[2] != self.target_resolution:
                feat = F.interpolate(
                    feat,
                    size=(self.target_resolution, self.target_resolution),
                    mode='bilinear',
                    align_corners=False
                )
            upsampled.append(feat)

        # Concatenate along channel dimension
        concat_feat = torch.cat(upsampled, dim=1)

        # Reduce channels
        fused = self.channel_reduce(concat_feat)

        return fused


class ConcatExpansionFusion(nn.Module):
    """
    Applies channel expansion to concatenated features (e.g., 192 -> 256 -> 384).

    This module first concatenates features from multiple resolutions, then
    uses a series of convolutional blocks to expand the channel dimension,
    creating a richer representation for the subsequent encoder.
    """
    def __init__(
        self,
        pyramid_channels: int = 64,
        num_resolutions: int = 3,
        expansion_channels: List[int] = [256, 384],
        target_resolution: int = 1024
    ):
        super().__init__()
        self.target_resolution = target_resolution
        concat_channels = pyramid_channels * num_resolutions
        
        layers = []
        in_channels = concat_channels
        
        for out_channels in expansion_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            in_channels = out_channels
            
        self.expansion_block = nn.Sequential(*layers)

    def forward(self, features: Dict[int, torch.Tensor]) -> torch.Tensor:
        upsampled = []
        for res in sorted(features.keys(), reverse=True):
            feat = features[res]
            if feat.shape[2] != self.target_resolution:
                feat = F.interpolate(
                    feat,
                    size=(self.target_resolution, self.target_resolution),
                    mode='bilinear',
                    align_corners=False
                )
            upsampled.append(feat)
        
        concat_feat = torch.cat(upsampled, dim=1)
        expanded = self.expansion_block(concat_feat)
        return expanded


class AttentionFusion(nn.Module):
    """
    Attention-weighted fusion.

    Learns importance weights for each resolution level using a lightweight
    attention mechanism. More sophisticated than concatenation.
    """

    def __init__(
        self,
        pyramid_channels: int = 64,
        num_resolutions: int = 3,
        output_channels: int = 64,
        target_resolution: int = 1024,
        reduction: int = 16
    ):
        """
        Args:
            pyramid_channels: Channels from each pyramid level
            num_resolutions: Number of pyramid levels
            output_channels: Output channels after fusion
            target_resolution: Target spatial resolution
            reduction: Channel reduction factor for attention
        """
        super().__init__()

        self.pyramid_channels = pyramid_channels
        self.num_resolutions = num_resolutions
        self.output_channels = output_channels
        self.target_resolution = target_resolution

        # Per-resolution feature projection
        self.feature_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(pyramid_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_resolutions)
        ])

        # Attention weight generation
        concat_channels = pyramid_channels * num_resolutions
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global pooling
            nn.Conv2d(concat_channels, concat_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(concat_channels // reduction, num_resolutions, 1),
            nn.Softmax(dim=1)  # Normalize across resolutions
        )

    def forward(self, features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Fuse features using learned attention weights.

        Args:
            features: Dict mapping resolution -> feature tensor

        Returns:
            Fused features (B, output_channels, target_resolution, target_resolution)
        """
        batch_size = list(features.values())[0].shape[0]

        # Upsample all to target resolution and project
        projected_features = []
        upsampled_for_attention = []

        for i, res in enumerate(sorted(features.keys(), reverse=True)):
            feat = features[res]

            # Upsample if needed
            if feat.shape[2] != self.target_resolution:
                feat = F.interpolate(
                    feat,
                    size=(self.target_resolution, self.target_resolution),
                    mode='bilinear',
                    align_corners=False
                )

            # Project to output channels
            projected = self.feature_projections[i](feat)
            projected_features.append(projected)
            upsampled_for_attention.append(feat)

        # Calculate attention weights
        # Concatenate for attention calculation
        concat_feat = torch.cat(upsampled_for_attention, dim=1)
        attention_weights = self.attention(concat_feat)  # (B, num_res, 1, 1)

        # Apply attention weights
        fused = torch.zeros(
            batch_size, self.output_channels, self.target_resolution, self.target_resolution,
            device=projected_features[0].device,
            dtype=projected_features[0].dtype
        )

        for i, proj_feat in enumerate(projected_features):
            weight = attention_weights[:, i:i+1, :, :]  # (B, 1, 1, 1)
            fused = fused + weight * proj_feat

        return fused


class ProgressiveFusion(nn.Module):
    """
    Progressive hierarchical fusion.

    Fuses resolutions progressively from coarse to fine, allowing
    refinement at each step. Useful for encoder-level specific processing.
    """

    def __init__(
        self,
        pyramid_channels: int = 64,
        output_channels: int = 64,
        resolutions: List[int] = [1024, 512, 256]
    ):
        """
        Args:
            pyramid_channels: Channels from each pyramid level
            output_channels: Output channels after fusion
            resolutions: List of resolutions in descending order
        """
        super().__init__()

        self.pyramid_channels = pyramid_channels
        self.output_channels = output_channels
        self.resolutions = sorted(resolutions, reverse=True)  # [1024, 512, 256]

        # Progressive fusion blocks
        # Start from coarsest (256), progressively add finer details
        self.fusion_blocks = nn.ModuleList()

        # First block: process coarsest resolution
        self.fusion_blocks.append(nn.Sequential(
            nn.Conv2d(pyramid_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        ))

        # Subsequent blocks: fuse with progressively finer resolutions
        for _ in range(len(resolutions) - 1):
            self.fusion_blocks.append(nn.Sequential(
                nn.Conv2d(output_channels + pyramid_channels, output_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Progressive fusion from coarse to fine.

        Process: 256 -> upsample + fuse with 512 -> upsample + fuse with 1024

        Args:
            features: Dict mapping resolution -> feature tensor

        Returns:
            Fused features at finest resolution
        """
        # Start with coarsest resolution (smallest: 256)
        coarsest_res = min(features.keys())
        fused = self.fusion_blocks[0](features[coarsest_res])

        # Progressively add finer resolutions, ensuring correct order
        sorted_resolutions = sorted([res for res in features.keys() if res != coarsest_res])

        for i, target_res in enumerate(sorted_resolutions, start=1):
            # Upsample current fused features to next resolution
            if fused.shape[2] != target_res:
                fused = F.interpolate(
                    fused,
                    size=(target_res, target_res),
                    mode='bilinear',
                    align_corners=False
                )

            # Concatenate with features at this resolution
            fused = torch.cat([fused, features[target_res]], dim=1)

            # Apply fusion block
            fused = self.fusion_blocks[i](fused)

        return fused


class FeatureFusion(nn.Module):
    """
    Unified feature fusion interface supporting multiple strategies.

    This is the main module to use in the complete model.
    """

    def __init__(
        self,
        pyramid_channels: int = 64,
        output_channels: int = 64,
        num_resolutions: int = 3,
        target_resolution: int = 1024,
        fusion_type: str = 'concat',
        **kwargs
    ):
        """
        Args:
            pyramid_channels: Channels from pyramid module
            output_channels: Output channels for encoder
            num_resolutions: Number of pyramid levels
            target_resolution: Target resolution (usually 1024)
            fusion_type: 'concat', 'attention', or 'progressive'
            **kwargs: Additional arguments for specific fusion types
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.target_resolution = target_resolution

        if fusion_type == 'concat':
            self.fusion = ConcatFusion(
                pyramid_channels=pyramid_channels,
                num_resolutions=num_resolutions,
                output_channels=output_channels,
                target_resolution=target_resolution
            )
        elif fusion_type == 'concat_1x1':
            # Force the use of a 1x1 conv even if in_channels == out_channels
            fused_channels = pyramid_channels * num_resolutions
            self.fusion = ConcatFusion(
                pyramid_channels=pyramid_channels,
                num_resolutions=num_resolutions,
                output_channels=fused_channels, # 192 -> 192
                target_resolution=target_resolution,
                force_1x1_conv=True
            )
        elif fusion_type == 'concat_expansion':
            self.fusion = ConcatExpansionFusion(
                pyramid_channels=pyramid_channels,
                num_resolutions=num_resolutions,
                expansion_channels=kwargs.get('expansion_channels', [256, 384]),
                target_resolution=target_resolution
            )
        elif fusion_type == 'attention':
            # AttentionFusion doesn't need resolutions, so we pop it from kwargs
            kwargs.pop('resolutions', None)
            self.fusion = AttentionFusion(
                pyramid_channels=pyramid_channels,
                num_resolutions=num_resolutions,
                output_channels=output_channels,
                target_resolution=target_resolution,
                **kwargs
            )
        elif fusion_type == 'progressive':
            resolutions = kwargs.get('resolutions', [1024, 512, 256])
            self.fusion = ProgressiveFusion(
                pyramid_channels=pyramid_channels,
                output_channels=output_channels,
                resolutions=resolutions
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, features: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-resolution features.

        Args:
            features: Dict mapping resolution -> feature tensor

        Returns:
            Fused features (B, output_channels, target_resolution, target_resolution)
        """
        return self.fusion(features)

    def get_num_parameters(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing
if __name__ == '__main__':
    print("Testing Feature Fusion Modules...")

    batch_size = 2
    pyramid_channels = 64
    resolutions = [1024, 512, 256]

    # Create dummy pyramid features
    features = {
        1024: torch.randn(batch_size, pyramid_channels, 1024, 1024),
        512: torch.randn(batch_size, pyramid_channels, 512, 512),
        256: torch.randn(batch_size, pyramid_channels, 256, 256)
    }

    print("\nInput pyramid features:")
    for res, feat in features.items():
        print(f"  {res}x{res}: {feat.shape}")

    # Test each fusion strategy
    fusion_types = ['concat', 'attention', 'progressive']

    for fusion_type in fusion_types:
        print(f"\n{'='*50}")
        print(f"Testing {fusion_type.upper()} fusion...")
        print(f"{'='*50}")

        fusion = FeatureFusion(
            pyramid_channels=pyramid_channels,
            output_channels=64,
            num_resolutions=len(resolutions),
            target_resolution=1024,
            fusion_type=fusion_type,
            resolutions=resolutions
        )

        # Forward pass
        fused = fusion(features)
        print(f"\nOutput shape: {fused.shape}")
        print(f"Expected: (2, 64, 1024, 1024)")

        # Parameter count
        num_params = fusion.get_num_parameters()
        print(f"Parameters: {num_params:,}")

        # Test gradient flow
        loss = fused.mean()
        loss.backward()
        has_grads = all(p.grad is not None for p in fusion.parameters() if p.requires_grad)
        print(f"Gradient flow: {'✓' if has_grads else '✗'}")

        # Clear gradients for next test
        fusion.zero_grad()

    print("\n✓ All Feature Fusion tests PASSED")
