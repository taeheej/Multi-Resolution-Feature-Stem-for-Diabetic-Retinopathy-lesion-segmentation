"""
Multi-Resolution DDR Dataset

Extends DDRPatchDataset to generate multi-scale image pyramids for
improved lesion segmentation across different scales.

Key Features:
- Generates 3 resolution levels: 1024x1024, 512x512, 256x256
- Synchronized augmentations across all resolutions
- Memory-efficient lazy evaluation
- Compatible with existing DDR dataset structure
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A
import cv2
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from ddr_dataset import DDRPatchDataset
from torch.nn import functional as F


class MultiResolutionDDRDataset(DDRPatchDataset):
    """
    Multi-resolution dataset that generates image pyramids at three scales.

    Resolution Levels:
    - Level 0: 1024x1024 (fine details, small lesions like MA)
    - Level 1: 512x512 (medium details, HE)
    - Level 2: 256x256 (coarse details, large lesions like EX, SE)

    Args:
        base_dir (str): Path to DDR lesion_segmentation directory
        split (str): 'train', 'valid', or 'test'
        resolutions (List[int]): List of resolution levels [1024, 512, 256]
        transform (callable): Standard torchvision transforms
        use_augmentations (bool): Apply augmentations (for training)
        augment_before_pyramid (bool): If True, augment at full resolution then downsample.
            If False, downsample then apply synchronized augmentation to each level.
            Default True (better quality, preserves details).
        verbose (bool): Print initialization info
    """

    def __init__(
        self,
        base_dir: str = '../DDR-dataset/lesion_segmentation',
        split: str = 'train',
        resolutions: List[int] = [1024, 512, 256],
        transform: Optional[callable] = None,
        use_augmentations: bool = True,
        augment_before_pyramid: bool = True,
        verbose: bool = True,
        **kwargs
    ):
        # Initialize parent with largest resolution as base patch size
        self.resolutions = sorted(resolutions, reverse=True)  # [1024, 512, 256]
        self.base_resolution = self.resolutions[0]
        self.augment_before_pyramid = augment_before_pyramid
        self.return_untransformed_highest_res = kwargs.get('return_untransformed_highest_res', False)

        super().__init__(
            base_dir=base_dir,
            split=split,
            patch_size=self.base_resolution,
            transform=transform,
            use_augmentations=use_augmentations,
            verbose=verbose
        )

        if verbose:
            print(f"\nMultiResolutionDDRDataset initialized:")
            print(f"  Resolution levels: {self.resolutions}")
            print(f"  Augmentation strategy: {'augment-then-downsample' if augment_before_pyramid else 'downsample-then-augment'}")
            print(f"  Base resolution: {self.base_resolution}")
            print(f"  Number of scales: {len(self.resolutions)}")

    def _generate_pyramid(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Generate multi-resolution pyramid for image and mask.

        Args:
            image: RGB image (H, W, 3) at base resolution
            mask: Multi-channel mask (H, W, 4) at base resolution

        Returns:
            Tuple of (image_pyramid, mask_pyramid) where each is a list
            of resolutions from largest to smallest
        """
        image_pyramid = [image]  # Start with base resolution
        mask_pyramid = [mask]

        # Generate downsampled versions
        for target_res in self.resolutions[1:]:
            # Downsample image (bilinear for smooth interpolation)
            downsampled_image = cv2.resize(
                image,
                (target_res, target_res),
                interpolation=cv2.INTER_LINEAR
            )
            image_pyramid.append(downsampled_image)

            # Downsample mask (nearest neighbor to preserve binary values)
            downsampled_mask = cv2.resize(
                mask,
                (target_res, target_res),
                interpolation=cv2.INTER_NEAREST
            )
            mask_pyramid.append(downsampled_mask)

        return image_pyramid, mask_pyramid

    def _apply_synchronized_augmentation(
        self,
        image_pyramid: List[np.ndarray],
        mask_pyramid: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply the same augmentation parameters to all resolution levels.

        This is the "downsample-then-augment" approach for comparison.

        Args:
            image_pyramid: List of images at different resolutions
            mask_pyramid: List of masks at different resolutions

        Returns:
            Augmented (image_pyramid, mask_pyramid)
        """
        if not self.use_augmentations:
            return image_pyramid, mask_pyramid

        from albumentations import ReplayCompose

        # Create replay compose with same augmentations as parent
        replay_transform = ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.CLAHE(clip_limit=1.5, tile_grid_size=(8, 8), p=1.0),
            ], p=0.3),
        ])

        # Apply to base resolution and save replay
        augmented = replay_transform(image=image_pyramid[0], mask=mask_pyramid[0])
        aug_image_pyramid = [augmented['image']]
        aug_mask_pyramid = [augmented['mask']]

        # Replay same augmentation on other resolutions
        for i, (img, msk) in enumerate(zip(image_pyramid[1:], mask_pyramid[1:]), start=1):
            augmented = ReplayCompose.replay(augmented['replay'], image=img, mask=msk)

            # Resize back to ensure consistent dimensions after ShiftScaleRotate
            target_res = self.resolutions[i]
            aug_img = cv2.resize(augmented['image'], (target_res, target_res), interpolation=cv2.INTER_LINEAR)
            aug_msk = cv2.resize(augmented['mask'], (target_res, target_res), interpolation=cv2.INTER_NEAREST)

            aug_image_pyramid.append(aug_img)
            aug_mask_pyramid.append(aug_msk)

        return aug_image_pyramid, aug_mask_pyramid

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get multi-resolution image pyramid and masks.

        Returns:
            Dictionary containing:
            - 'images': Dict mapping resolution -> tensor (3, H, W)
            - 'masks': Dict mapping resolution -> tensor (4, H, W)
            - 'image_id': str (base filename)
        """
        # Get base resolution image and mask from parent class
        base_image_np, base_mask_np, image_id = super().get_raw_item(idx)

        # --- Store the untransformed base image if requested ---
        untransformed_display_image = None
        if self.return_untransformed_highest_res:
            untransformed_display_image = base_image_np.copy()

        if self.augment_before_pyramid:
            # APPROACH 1: Augment-then-downsample (better quality, preserves details)
            if self.use_augmentations:
                from ddr_dataset import patch_aug
                try:
                    augmented = patch_aug(image=base_image_np, mask=base_mask_np)
                    base_image_np = augmented['image']
                    base_mask_np = augmented['mask']
                except Exception as e:
                    print(f"Error during augmentation for {image_id}: {e}")

            # Generate pyramid from the augmented base image
            image_pyramid, mask_pyramid = self._generate_pyramid(base_image_np, base_mask_np)

        else:
            # APPROACH 2: Downsample-then-augment (for comparison/ablation)
            # Generate pyramid first, then apply synchronized augmentation
            image_pyramid, mask_pyramid = self._generate_pyramid(base_image_np, base_mask_np)
            image_pyramid, mask_pyramid = self._apply_synchronized_augmentation(
                image_pyramid, mask_pyramid
            )

        # Convert pyramids to tensors
        images_dict = {}
        masks_dict = {}

        for res, img, msk in zip(self.resolutions, image_pyramid, mask_pyramid):
            # Apply transforms to image
            if self.transform:
                img_tensor = self.transform(img)
            else:
                img_tensor = transforms.ToTensor()(img)

            # Convert mask to tensor: (H, W, 4) -> (4, H, W)
            msk_tensor = torch.from_numpy(msk.transpose(2, 0, 1)).float()

            images_dict[res] = img_tensor
            masks_dict[res] = msk_tensor

        output = {
            'images': images_dict,
            'masks': masks_dict,
            'image_id': image_id
        }

        if self.return_untransformed_highest_res:
            output['display_image'] = untransformed_display_image
        
        return output


# Example usage and testing
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    print("Testing MultiResolutionDDRDataset...")

    # Define standard transforms
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Test dataset
    try:
        dataset = MultiResolutionDDRDataset(
            split='train',
            resolutions=[1024, 512, 256],
            transform=image_transform,
            use_augmentations=True,
            verbose=True
        )

        print(f"\nDataset size: {len(dataset)}")

        # Test single sample
        print("\nTesting single sample retrieval...")
        sample = dataset[0]

        print(f"\nSample structure:")
        print(f"  Image ID: {sample['image_id']}")
        print(f"  Image resolutions: {list(sample['images'].keys())}")
        print(f"  Mask resolutions: {list(sample['masks'].keys())}")

        for res in [1024, 512, 256]:
            img_shape = sample['images'][res].shape
            msk_shape = sample['masks'][res].shape
            print(f"\n  Resolution {res}x{res}:")
            print(f"    Image shape: {img_shape} (C, H, W)")
            print(f"    Mask shape: {msk_shape} (C, H, W)")
            print(f"    Mask unique values: {torch.unique(sample['masks'][res])}")

        # Test dataloader
        print("\nTesting DataLoader...")
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(loader))

        print(f"\nBatch structure:")
        print(f"  Batch size: {len(batch['image_id'])}")
        for res in [1024, 512, 256]:
            print(f"  Resolution {res}: images {batch['images'][res].shape}, masks {batch['masks'][res].shape}")

        print("\n✓ MultiResolutionDDRDataset test PASSED")

    except Exception as e:
        print(f"\n✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
