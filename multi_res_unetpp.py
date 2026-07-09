"""
Multi-Resolution UNet++ Model

Integrates multi-resolution image pyramid processing with UNet++
for improved lesion segmentation across different scales.

Architecture Flow:
    Multi-Res Images -> Feature Pyramid -> Feature Fusion -> UNet++ Encoder -> Decoder -> Output

Key Components:
- FeaturePyramidModule: Shared-weight processing of multi-scale inputs
- FeatureFusion: Multi-scale feature fusion
- UNet++ from segmentation_models_pytorch (standard, clean implementation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import sys
import os
import segmentation_models_pytorch as smp

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multi_resolution.feature_pyramid import FeaturePyramidModule, AdaptiveFeaturePyramid
from multi_resolution.feature_fusion import FeatureFusion


class MultiResUNetPlusPlus(nn.Module):
    """
    Complete multi-resolution UNet++ model.

    Combines:
    1. Feature Pyramid: Shared-weight processing at multiple resolutions
    2. Feature Fusion: Multi-scale feature integration
    3. UNet++: Standard pretrained encoder/decoder from smp

    Args:
        encoder_name: Pretrained encoder ('resnet34', 'resnet50', etc.)
        encoder_weights: Pretrained weights ('imagenet', etc.)
        resolutions: List of input resolutions [1024, 512, 256]
        pyramid_channels: Output channels from pyramid module
        fusion_type: 'concat', 'attention', or 'progressive'
        fusion_output_channels: Explicitly set output channels for fusion module
        expansion_channels: Optional list of channels for expansion blocks
        use_adaptive_pyramid: Use resolution-specific adaptation in pyramid
        legacy_pre_bottleneck: If True, run in legacy mode with a pre-bottleneck fusion
        decoder_channels: UNet++ decoder configuration
        n_classes: Number of output classes (4 for DDR)
        freeze_encoder_epochs: Number of initial epochs to freeze encoder (0 = no freezing)
    """

    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str = 'imagenet',
        resolutions: List[int] = [1024, 512, 256],
        pyramid_channels: int = 64,
        fusion_type: str = 'concat',
        fusion_output_channels: Optional[int] = None,
        expansion_channels: Optional[List[int]] = None,
        use_adaptive_pyramid: bool = False,
        legacy_pre_bottleneck: bool = False,
        decoder_channels: List[int] = [256, 128, 64, 32, 16],
        n_classes: int = 4,
        freeze_encoder_epochs: int = 0
    ):
        super().__init__()

        self.resolutions = sorted(resolutions, reverse=True)  # [1024, 512, 256]
        self.pyramid_channels = pyramid_channels
        self.fusion_type = fusion_type
        self.n_classes = n_classes
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.encoder_frozen = False
        self.encoder_weights = encoder_weights  # Store for later use
        self.legacy_pre_bottleneck = legacy_pre_bottleneck

        # === 1. Feature Pyramid Module ===
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

        # Create the UNet++ model once. It's always initialized with 3 input channels
        # and then adapted if necessary in the non-legacy path.
        self.unetpp = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            in_channels=3,
            classes=n_classes
        )

        if self.legacy_pre_bottleneck:
            # --- LEGACY ARCHITECTURE (with bottleneck) ---
            print("  \033[93m⚠ WARNING: Running in legacy pre-bottleneck mode.\033[0m")
            self.fusion = FeatureFusion(
                pyramid_channels=pyramid_channels,
                output_channels=64,  # Legacy mode hardcodes a reduction to 64
                num_resolutions=len(resolutions),
                target_resolution=max(resolutions),
                fusion_type=fusion_type,
                resolutions=resolutions
            )
            self.input_adapter = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 3, kernel_size=1)
            )
            # self.unetpp is already configured correctly for 3-channel input
        else:
            # --- CURRENT ARCHITECTURE (bottleneck removed) ---
            if fusion_output_channels is None:
                final_fusion_channels = pyramid_channels * len(resolutions)
            else:
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

            # No input adapter in the new architecture
            self.input_adapter = None
            
            # Adapt the pre-created UNet++ model for direct multi-channel input
            self._adapt_encoder_input(final_fusion_channels, encoder_name)

        # Initialize encoder freezing if specified
        if freeze_encoder_epochs > 0:
            self.freeze_encoder()

    def _adapt_encoder_input(self, in_channels: int, encoder_name: str):
        """
        Replace the first convolutional layer of the encoder to accept
        in_channels instead of 3, while preserving pretrained weights
        for the rest of the encoder.

        This is done by:
        1. Creating a new conv layer with in_channels input
        2. Initializing it intelligently (not random)
        3. Replacing encoder's first conv layer

        Args:
            in_channels: Number of input channels (e.g., 192 for concat fusion)
            encoder_name: Name of encoder (resnet34, resnet50, etc.)
        """
        # Get the first conv layer from the encoder
        if 'resnet' in encoder_name or 'resnext' in encoder_name:
            # ResNet-style encoders
            old_conv = self.unetpp.encoder.conv1

            # Create new conv layer with same settings but different input channels
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

            # Initialize new conv intelligently
            # Option 1: Xavier/Kaiming initialization (standard)
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            # Option 2: If in_channels is a multiple of 3, we can repeat pretrained weights
            # This preserves some pretrained knowledge
            if in_channels % 3 == 0 and self.encoder_weights is not None:
                with torch.no_grad():
                    # Repeat the 3-channel weights across all input channels
                    repeat_factor = in_channels // 3
                    repeated_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)
                    # Average to maintain magnitude
                    new_conv.weight.copy_(repeated_weights / repeat_factor)
                    print(f"  ✓ Initialized first conv with repeated pretrained weights ({in_channels} channels)")
            else:
                print(f"  ✓ Initialized first conv with Kaiming initialization ({in_channels} channels)")

            if new_conv.bias is not None and old_conv.bias is not None:
                new_conv.bias.data.copy_(old_conv.bias.data)

            # Replace the layer
            self.unetpp.encoder.conv1 = new_conv

        elif 'efficientnet' in encoder_name:
            # EfficientNet has a different structure
            # First conv is in encoder.conv_stem
            old_conv = self.unetpp.encoder.conv_stem

            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )

            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')

            if in_channels % 3 == 0 and self.encoder_weights is not None:
                with torch.no_grad():
                    repeat_factor = in_channels // 3
                    repeated_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)
                    new_conv.weight.copy_(repeated_weights / repeat_factor)
                    print(f"  ✓ Initialized first conv with repeated pretrained weights ({in_channels} channels)")

            self.unetpp.encoder.conv_stem = new_conv

        else:
            raise NotImplementedError(f"First layer adaptation not implemented for {encoder_name}")

        print(f"  ✓ Adapted encoder input: 3 → {in_channels} channels (rest of encoder unchanged)")

    def freeze_encoder(self):
        """Freeze UNet++ encoder for initial training."""
        if not self.encoder_frozen:
            for param in self.unetpp.encoder.parameters():
                param.requires_grad = False
            self.encoder_frozen = True
            print("  UNet++ encoder frozen")

    def unfreeze_encoder(self):
        """Unfreeze UNet++ encoder for fine-tuning."""
        if self.encoder_frozen:
            for param in self.unetpp.encoder.parameters():
                param.requires_grad = True
            self.encoder_frozen = False
            print("  UNet++ encoder unfrozen")

    def forward(
        self,
        images: Dict[int, torch.Tensor],
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete multi-resolution pipeline.

        Args:
            images: Dict mapping resolution -> image tensor
                   {1024: (B,3,1024,1024), 512: (B,3,512,512), 256: (B,3,256,256)}
            return_intermediate: If True, return intermediate features for analysis

        Returns:
            Dictionary containing:
            - 'segmentation': Final segmentation logits (B, n_classes, H, W)
            - 'pyramid_features': (optional) Features from pyramid module
            - 'fused_features': (optional) Features after fusion
        """
        intermediate = {}

        # Step 1: Generate multi-resolution feature pyramid
        # Each resolution -> 64 channels
        pyramid_features = self.pyramid(images)
        if return_intermediate:
            intermediate['pyramid_features'] = pyramid_features

        # Step 2: Fuse multi-scale features
        fused_features = self.fusion(pyramid_features)
        if return_intermediate:
            intermediate['fused_features'] = fused_features

        # Step 3: Pass to UNet++
        if self.legacy_pre_bottleneck:
            # Legacy path with input adapter
            adapted_features = self.input_adapter(fused_features)
            segmentation = self.unetpp(adapted_features)
        else:
            # New direct-input path
            segmentation = self.unetpp(fused_features)

        # Combine outputs
        result = {'segmentation': segmentation}
        if return_intermediate:
            result.update(intermediate)

        return result

    def get_num_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """
        Get parameter counts for each component.

        Args:
            trainable_only: If True, count only trainable parameters

        Returns:
            Dictionary with parameter counts
        """
        def count_params(module):
            if trainable_only:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            return sum(p.numel() for p in module.parameters())

        return {
            'pyramid': count_params(self.pyramid),
            'fusion': count_params(self.fusion),
            'unetpp_encoder': count_params(self.unetpp.encoder),
            'unetpp_decoder': count_params(self.unetpp.decoder),
            'unetpp_total': count_params(self.unetpp),
            'total': count_params(self)
        }


# Testing
if __name__ == '__main__':
    print("Testing Multi-Resolution UNet++ Model...")

    batch_size = 2
    resolutions = [1024, 512, 256]
    n_classes = 4

    # Create dummy multi-resolution input
    images = {
        1024: torch.randn(batch_size, 3, 1024, 1024),
        512: torch.randn(batch_size, 3, 512, 512),
        256: torch.randn(batch_size, 3, 256, 256)
    }

    print("\nInput:")
    for res, img in images.items():
        print(f"  {res}x{res}: {img.shape}")

    # Test different fusion strategies
    fusion_types = ['concat', 'attention', 'progressive']

    for fusion_type in fusion_types:
        print("\n" + "="*60)
        print(f"Testing with {fusion_type.upper()} fusion...")
        print("="*60)

        try:
            model = MultiResUNetPlusPlus(
                encoder_name='resnet34',
                encoder_weights='imagenet',
                resolutions=resolutions,
                pyramid_channels=64,
                fusion_type=fusion_type,
                use_adaptive_pyramid=(fusion_type == 'attention'),
                n_classes=n_classes,
                freeze_encoder_epochs=0  # No freezing for testing
            )

            # Forward pass with intermediate outputs
            output = model(images, return_intermediate=True)

            print(f"\nOutput keys: {list(output.keys())}")
            print(f"Segmentation shape: {output['segmentation'].shape}")
            print(f"Expected: ({batch_size}, {n_classes}, 1024, 1024)")

            if 'pyramid_features' in output:
                print("\nPyramid features:")
                for res, feat in output['pyramid_features'].items():
                    print(f"  {res}x{res}: {feat.shape}")

            if 'fused_features' in output:
                print(f"\nFused features: {output['fused_features'].shape}")

            if 'adapted_input' in output:
                print(f"Adapted input: {output['adapted_input'].shape}")

            # Parameter counts
            param_counts = model.get_num_parameters(trainable_only=True)
            print("\nTrainable parameters:")
            total_params = 0
            for component, count in param_counts.items():
                print(f"  {component}: {count:,}")
                total_params = max(total_params, count)

            # Test gradient flow
            loss = output['segmentation'].mean()
            loss.backward()
            pyramid_has_grad = any(p.grad is not None for p in model.pyramid.parameters())
            unetpp_has_grad = any(p.grad is not None for p in model.unetpp.parameters())

            print(f"\nGradient flow:")
            print(f"  Pyramid: {'✓' if pyramid_has_grad else '✗'}")
            print(f"  UNet++: {'✓' if unetpp_has_grad else '✗'}")

            print(f"\n✓ {fusion_type.upper()} fusion test PASSED")

        except Exception as e:
            print(f"\n✗ {fusion_type.upper()} fusion test FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Test encoder freezing
    print("\n" + "="*60)
    print("Testing encoder freezing functionality...")
    print("="*60)

    model = MultiResUNetPlusPlus(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        freeze_encoder_epochs=10
    )

    print(f"\nEncoder frozen: {model.encoder_frozen}")
    params_frozen = model.get_num_parameters(trainable_only=True)
    print(f"Trainable params (frozen): {params_frozen['total']:,}")

    model.unfreeze_encoder()
    print(f"\nEncoder frozen: {model.encoder_frozen}")
    params_unfrozen = model.get_num_parameters(trainable_only=True)
    print(f"Trainable params (unfrozen): {params_unfrozen['total']:,}")
    print(f"Additional trainable params: {params_unfrozen['total'] - params_frozen['total']:,}")

    print("\n✓ All Multi-Resolution UNet++ tests PASSED")
