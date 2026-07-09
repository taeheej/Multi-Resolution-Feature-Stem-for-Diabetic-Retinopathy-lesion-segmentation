"""
Multi-Resolution Input Processing for Lesion Segmentation

This module implements multi-resolution image pyramid processing for UNet++
lesion segmentation, designed to improve detection of small lesions (MA, HE)
while maintaining performance on larger lesions (EX, SE).

Key Components:
- MultiResolutionDDRDataset: Data loader generating multi-scale image pyramids
- FeaturePyramidModule: Shared-weight initial convolutions
- FeatureFusion: Multi-scale feature fusion strategies
- MultiResUNetPlusPlus: Complete multi-resolution UNet++ model

Evaluation Focus:
- Primary metrics: AP (Average Precision), IoU (Intersection over Union)
- Per-class metrics: mAP, mIoU
- Small lesion focus: AP@IoU=0.5, AP@IoU=0.75 for MA and HE
"""

__version__ = "0.1.0"
__author__ = "DDR Project - Thesis Implementation"

from .multi_res_dataset import MultiResolutionDDRDataset
from .feature_pyramid import FeaturePyramidModule
from .feature_fusion import FeatureFusion
from .multi_res_unetpp import MultiResUNetPlusPlus
from .fpn_multires import FPNMultiResEnsemble

__all__ = [
    'MultiResolutionDDRDataset',
    'FeaturePyramidModule',
    'FeatureFusion',
    'MultiResUNetPlusPlus',
    'FPNMultiResEnsemble',
]
