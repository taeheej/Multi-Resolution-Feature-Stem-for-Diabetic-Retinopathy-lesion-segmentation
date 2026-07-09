"""
Integration Guide for Multi-Resolution UNet++ with train.py

This file documents how to integrate multi-resolution support into the existing
train.py without duplicating code. All existing features (wandb, experiment DB,
metrics, loss functions) are preserved.

INTEGRATION APPROACH:
--------------------
Add multi-resolution support as a new model_type in train.py, similar to how
'lesion_unetpp', 'vit', etc. are handled.

REQUIRED CHANGES TO train.py:
------------------------------

1. ADD IMPORT (after line 54):
   from thesis_implementation.multi_resolution.multi_res_unetpp import MultiResUNetPlusPlus
   from thesis_implementation.multi_resolution.multi_res_dataset import MultiResolutionDDRDataset

2. ADD MODEL TYPE in argparse (line 919):
   parser.add_argument('--model_type', type=str, default='unet',
                      choices=['unet', 'unetplusplus', 'vit', 'swin_transformer',
                               'deeplabv3plus', 'lesion_unetpp', 'multi_res_unetpp'],
                      help='Type of model architecture')

3. ADD MULTI-RES ARGUMENTS in argparse (after line 936):
   # Multi-Resolution UNet++ Hyperparameters
   parser.add_argument('--resolutions', type=int, nargs='+', default=[1024, 512, 256],
                      help='Resolution levels for multi-res pyramid (e.g., 1024 512 256)')
   parser.add_argument('--pyramid_channels', type=int, default=64,
                      help='Feature pyramid output channels')
   parser.add_argument('--fusion_type', type=str, default='concat',
                      choices=['concat', 'attention', 'progressive'],
                      help='Multi-scale feature fusion strategy')
   parser.add_argument('--use_adaptive_pyramid', action='store_true',
                      help='Use adaptive pyramid with resolution-specific layers')

4. MODIFY DATASET LOADING (around line 384-397):
   Replace the DDRPatchDataset instantiation with conditional logic:

   if model_type == 'multi_res_unetpp':
       # Use multi-resolution dataset
       full_train_dataset = MultiResolutionDDRDataset(
           base_dir=args.dataset_base_dir,
           split='train',
           resolutions=args.resolutions,
           transform=image_patch_transform,
           use_augmentations=args.use_augmentation,
           verbose=True
       )
       full_val_dataset = MultiResolutionDDRDataset(
           base_dir=args.dataset_base_dir,
           split='valid',
           resolutions=args.resolutions,
           transform=image_patch_transform,
           use_augmentations=False,
           verbose=True
       )
   else:
       # Use standard patch dataset
       full_train_dataset = DDRPatchDataset(
           base_dir=args.dataset_base_dir,
           split='train',
           patch_size=args.patch_size,
           transform=image_patch_transform,
           use_augmentations=args.use_augmentation
       )
       full_val_dataset = DDRPatchDataset(
           base_dir=args.dataset_base_dir,
           split='valid',
           patch_size=args.patch_size,
           transform=image_patch_transform,
           use_augmentations=False
       )

5. ADD MODEL CREATION (after line 558, before elif model_type == 'vit'):
   elif model_type == 'multi_res_unetpp':
       print(f"  Creating Multi-Resolution UNet++ with:")
       print(f"    - Encoder: {args.encoder_name}")
       print(f"    - Resolutions: {args.resolutions}")
       print(f"    - Pyramid channels: {args.pyramid_channels}")
       print(f"    - Fusion type: {args.fusion_type}")

       model = MultiResUNetPlusPlus(
           encoder_name=args.encoder_name,
           encoder_weights='imagenet' if args.pretrained else None,
           resolutions=args.resolutions,
           pyramid_channels=args.pyramid_channels,
           fusion_type=args.fusion_type,
           use_adaptive_pyramid=args.use_adaptive_pyramid,
           n_classes=n_classes,
           freeze_encoder_epochs=0  # Can be extended to support freezing
       )
       print(f"  ✓ Multi-Resolution UNet++ created")

6. MODIFY train_one_epoch() TO HANDLE MULTI-RES INPUT (line 84):
   Add conditional logic to handle dict input from MultiResolutionDDRDataset:

   for i, batch in enumerate(progress_bar):
       # Handle both standard and multi-resolution datasets
       if isinstance(batch, dict) and 'images' in batch:
           # Multi-resolution dataset returns dict
           images = batch['images']  # Dict of {resolution: tensor}
           masks = batch['masks']    # Dict of {resolution: tensor}
           # Use highest resolution mask as target
           target_masks = masks[max(masks.keys())].to(device=device, dtype=torch.float32)
       else:
           # Standard dataset returns tuple
           images, target_masks = batch
           images = images.to(device=device, dtype=torch.float32)
           target_masks = target_masks.to(device=device, dtype=torch.float32)

       # Forward pass (model handles dict or tensor input automatically)
       outputs = model(images)
       # ... rest of training logic remains same

7. MODIFY evaluate() SIMILARLY (line 227):
   Same conditional logic as train_one_epoch for handling multi-res inputs.

USAGE EXAMPLE:
--------------
# Single experiment
python train.py \\
    --model_type multi_res_unetpp \\
    --encoder_name resnet34 \\
    --pretrained \\
    --resolutions 1024 512 256 \\
    --pyramid_channels 64 \\
    --fusion_type concat \\
    --batch_size 2 \\
    --epochs 50 \\
    --loss_type dice_focal \\
    --checkpoint_metric ap \\
    --run_name multi_res_concat_r34

# Via experiments config (configs/experiments_multi_res.yaml):
base_config:
  dataset_base_dir: '../DDR-dataset/lesion_segmentation'
  epochs: 50
  batch_size: 2
  model_type: 'multi_res_unetpp'
  pretrained: true
  use_augmentation: true
  checkpoint_metric: 'ap'

experiments:
  - name: 'multi_res_concat_r34'
    params:
      encoder_name: 'resnet34'
      resolutions: [1024, 512, 256]
      fusion_type: 'concat'
      pyramid_channels: 64

  - name: 'multi_res_attention_r34'
    params:
      encoder_name: 'resnet34'
      resolutions: [1024, 512, 256]
      fusion_type: 'attention'
      pyramid_channels: 64
      use_adaptive_pyramid: true

Then run:
python run_experiments.py --config configs/experiments_multi_res.yaml

BENEFITS:
---------
✓ Reuses all existing infrastructure (wandb, DB, metrics, losses)
✓ Compatible with run_experiments.py batch processing
✓ Minimal code changes to train.py
✓ All AP/IoU metrics work automatically
✓ Supports all existing loss functions
✓ Works with encoder freezing, weighted sampling, etc.

METRICS TRACKED (automatic):
----------------------------
- val_mean_ap, val_mean_iou (primary)
- val_ap_EX, val_ap_HE, val_ap_MA, val_ap_SE (per-class AP)
- val_iou_EX, val_iou_HE, val_iou_MA, val_iou_SE (per-class IoU)
- val_dice, spatial_coherence, fragmentation
- All logged to wandb and experiments.db

TESTING:
--------
# Quick test (2 epochs, 1 batch)
python train.py \\
    --model_type multi_res_unetpp \\
    --encoder_name resnet34 \\
    --epochs 2 \\
    --overfit_test_batches 1 \\
    --run_name multi_res_test
"""

# This file is documentation only - no code execution needed
# Apply the changes above to train.py manually or via automated patch

if __name__ == '__main__':
    print(__doc__)
