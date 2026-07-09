"""
FPN Multi-Resolution Ensemble Model

Processes multiple resolution inputs through FPN encoders and fuses predictions.
Supports both shared-weight and separate FPN encoders for ablation studies.

Architecture:
    Input: {1024: [B,3,1024,1024], 512: [B,3,512,512], 256: [B,3,256,256]}
      ↓
    Three FPN Encoders (shared or separate weights)
      ↓
    Ensemble/Fusion Strategy
      ↓
    Final Segmentation: [B, n_classes, 1024, 1024]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from typing import Dict, List, Optional


class FPNMultiResEnsemble(nn.Module):
    """
    Multi-Resolution FPN Ensemble for lesion segmentation.

    Processes each resolution level through FPN encoder(s) and fuses outputs.

    Args:
        encoder_name (str): Encoder architecture (e.g., 'resnet34', 'resnet50')
        encoder_weights (str): Pretrained weights ('imagenet' or None)
        n_classes (int): Number of output classes (4 for DDR lesions)
        resolutions (List[int]): Resolution levels [1024, 512, 256]
        shared_fpn_weights (bool): Use same FPN for all resolutions (shared weights)
            vs separate FPN per resolution. Default False (separate).
        fusion_strategy (str): How to combine predictions
            - 'average': Simple average of upsampled predictions
            - 'learned': Learnable weighted fusion
            - 'max': Max pooling across predictions
        output_resolution (int): Target output resolution (default 1024)
    """

    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: Optional[str] = 'imagenet',
        n_classes: int = 4,
        resolutions: List[int] = [1024, 512, 256],
        shared_fpn_weights: bool = False,
        fusion_strategy: str = 'average',
        output_resolution: int = 1024
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.n_classes = n_classes
        self.resolutions = sorted(resolutions, reverse=True)  # [1024, 512, 256]
        self.shared_fpn_weights = shared_fpn_weights
        self.fusion_strategy = fusion_strategy
        self.output_resolution = output_resolution

        # Create FPN model(s)
        if shared_fpn_weights:
            # Single FPN used for all resolutions
            self.fpn = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=n_classes,
                activation=None
            )
            self.fpn_models = None
        else:
            # Separate FPN for each resolution
            self.fpn = None
            self.fpn_models = nn.ModuleDict({
                str(res): smp.FPN(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=3,
                    classes=n_classes,
                    activation=None
                )
                for res in self.resolutions
            })

        # Fusion module
        if fusion_strategy == 'learned':
            # Learnable weights for each resolution
            self.fusion_weights = nn.Parameter(
                torch.ones(len(self.resolutions)) / len(self.resolutions)
            )
        else:
            self.fusion_weights = None

    def forward(
        self,
        images: Dict[int, torch.Tensor],
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-resolution FPN ensemble.

        Args:
            images: Dict mapping resolution -> image tensor
                {1024: [B,3,1024,1024], 512: [B,3,512,512], 256: [B,3,256,256]}
            return_intermediate: Return intermediate predictions per resolution

        Returns:
            Dict containing:
                - 'segmentation': Final fused prediction [B, n_classes, output_res, output_res]
                - 'predictions': Dict of per-resolution predictions (if return_intermediate)
        """
        batch_size = images[self.resolutions[0]].shape[0]
        device = images[self.resolutions[0]].device

        # Process each resolution through FPN
        predictions = {}

        for res in self.resolutions:
            if res not in images:
                raise ValueError(f"Resolution {res} not found in input images")

            img = images[res]

            # Forward through appropriate FPN
            if self.shared_fpn_weights:
                pred = self.fpn(img)
            else:
                pred = self.fpn_models[str(res)](img)

            predictions[res] = pred

        # Upsample all predictions to output resolution
        upsampled_predictions = []

        for res in self.resolutions:
            pred = predictions[res]

            if pred.shape[-1] != self.output_resolution:
                pred_upsampled = F.interpolate(
                    pred,
                    size=(self.output_resolution, self.output_resolution),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                pred_upsampled = pred

            upsampled_predictions.append(pred_upsampled)

        # Fuse predictions
        if self.fusion_strategy == 'average':
            # Simple average
            fused = torch.stack(upsampled_predictions, dim=0).mean(dim=0)

        elif self.fusion_strategy == 'learned':
            # Learned weighted average
            # Apply softmax to ensure weights sum to 1
            weights = F.softmax(self.fusion_weights, dim=0)

            fused = torch.zeros_like(upsampled_predictions[0])
            for i, pred in enumerate(upsampled_predictions):
                fused += weights[i] * pred

        elif self.fusion_strategy == 'max':
            # Max pooling across resolutions
            fused = torch.stack(upsampled_predictions, dim=0).max(dim=0)[0]

        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # Prepare output
        output = {'segmentation': fused}

        if return_intermediate:
            output['predictions'] = predictions
            output['upsampled_predictions'] = {
                res: pred for res, pred in zip(self.resolutions, upsampled_predictions)
            }

        return output

    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for analysis."""
        if self.shared_fpn_weights:
            fpn_params = sum(p.numel() for p in self.fpn.parameters())
            return {
                'fpn': fpn_params,
                'fusion': 0 if self.fusion_weights is None else self.fusion_weights.numel(),
                'total': fpn_params + (0 if self.fusion_weights is None else self.fusion_weights.numel())
            }
        else:
            fpn_params_per_res = {
                res: sum(p.numel() for p in self.fpn_models[str(res)].parameters())
                for res in self.resolutions
            }
            total_fpn = sum(fpn_params_per_res.values())
            fusion_params = 0 if self.fusion_weights is None else self.fusion_weights.numel()

            return {
                **{f'fpn_{res}': params for res, params in fpn_params_per_res.items()},
                'fusion': fusion_params,
                'total': total_fpn + fusion_params
            }


# Test/Example usage
if __name__ == '__main__':
    print("Testing FPNMultiResEnsemble...")

    # Test 1: Shared weights
    print("\n[Test 1] Shared FPN weights")
    model_shared = FPNMultiResEnsemble(
        encoder_name='resnet34',
        encoder_weights=None,
        n_classes=4,
        resolutions=[1024, 512, 256],
        shared_fpn_weights=True,
        fusion_strategy='average'
    )

    # Create dummy input
    batch_size = 2
    images = {
        1024: torch.randn(batch_size, 3, 1024, 1024),
        512: torch.randn(batch_size, 3, 512, 512),
        256: torch.randn(batch_size, 3, 256, 256)
    }

    output = model_shared(images, return_intermediate=True)
    print(f"Output shape: {output['segmentation'].shape}")
    print(f"Expected: torch.Size([{batch_size}, 4, 1024, 1024])")
    print(f"Intermediate predictions: {list(output['predictions'].keys())}")

    params = model_shared.get_num_parameters()
    print(f"Parameters: {params['total']:,}")

    # Test 2: Separate weights
    print("\n[Test 2] Separate FPN weights")
    model_separate = FPNMultiResEnsemble(
        encoder_name='resnet34',
        encoder_weights=None,
        n_classes=4,
        resolutions=[1024, 512, 256],
        shared_fpn_weights=False,
        fusion_strategy='learned'
    )

    output = model_separate(images)
    print(f"Output shape: {output['segmentation'].shape}")

    params = model_separate.get_num_parameters()
    print(f"Parameters: {params['total']:,}")
    print(f"  FPN 1024: {params['fpn_1024']:,}")
    print(f"  FPN 512: {params['fpn_512']:,}")
    print(f"  FPN 256: {params['fpn_256']:,}")
    print(f"  Fusion weights: {params['fusion']}")

    print("\n✓ All tests passed!")
