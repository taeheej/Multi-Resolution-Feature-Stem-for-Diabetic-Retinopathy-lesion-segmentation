"""
A standalone nn.Module that encapsulates the multi-resolution
feature extraction and fusion pipeline.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional

# Add parent directories to path for standalone execution/testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multi_resolution.feature_pyramid import FeaturePyramidModule, AdaptiveFeaturePyramid
from multi_resolution.feature_fusion import FeatureFusion

class MultiResolutionStem(nn.Module):
    """
    A module that takes multi-resolution images and produces a single,
    fused feature tensor ready for a segmentation backbone.

    Combines:
    1. Feature Pyramid: Shared-weight processing at multiple resolutions.
    2. Feature Fusion: Multi-scale feature integration.
    """
    def __init__(
        self,
        resolutions: List[int] = [1024, 512, 256],
        pyramid_channels: int = 64,
        fusion_type: str = 'concat',
        fusion_output_channels: Optional[int] = None,
        expansion_channels: Optional[List[int]] = None,
        use_adaptive_pyramid: bool = False
    ):
        super().__init__()

        # --- 1. Feature Pyramid Module ---
        if use_adaptive_pyramid:
            self.pyramid = AdaptiveFeaturePyramid(
                in_channels=3,
                pyramid_channels=pyramid_channels,
                resolutions=resolutions,
                num_conv_blocks=2,
                use_adaptation=True
            )
        else:
            self.pyramid = FeaturePyramidModule(
                in_channels=3,
                pyramid_channels=pyramid_channels,
                resolutions=resolutions,
                num_conv_blocks=2
            )

        # --- 2. Feature Fusion Module ---
        if fusion_output_channels is None:
            # Default behavior for direct passthrough (e.g., 192 channels)
            final_fusion_channels = pyramid_channels * len(resolutions)
        else:
            # For reduction (e.g., 64) or expansion (e.g., 384)
            final_fusion_channels = fusion_output_channels

        self.fusion = FeatureFusion(
            pyramid_channels=pyramid_channels,
            output_channels=final_fusion_channels,
            num_resolutions=len(resolutions),
            target_resolution=max(resolutions),
            fusion_type=fusion_type,
            resolutions=resolutions,
            expansion_channels=expansion_channels
        )
        
        # Store the output channels of this module for easy access
        self.output_channels = final_fusion_channels

    def forward(self, images: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: Dict mapping resolution -> image tensor
        Returns:
            A single, fused feature tensor.
        """
        pyramid_features = self.pyramid(images)
        fused_features = self.fusion(pyramid_features)
        return fused_features
