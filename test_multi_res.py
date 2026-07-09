"""
Integration Tests for Multi-Resolution UNet++

Tests all components individually and end-to-end integration.
"""

import torch
import sys
import os

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from thesis_implementation.multi_resolution.multi_res_dataset import MultiResolutionDDRDataset
from thesis_implementation.multi_resolution.feature_pyramid import FeaturePyramidModule
from thesis_implementation.multi_resolution.feature_fusion import FeatureFusion
from thesis_implementation.multi_resolution.multi_res_unetpp import MultiResUNetPlusPlus
from torch.utils.data import DataLoader
from torchvision import transforms


def test_dataset():
    """Test multi-resolution dataset."""
    print("\n" + "="*60)
    print("TEST 1: Multi-Resolution Dataset")
    print("="*60)

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = MultiResolutionDDRDataset(
            split='train',
            resolutions=[1024, 512, 256],
            transform=image_transform,
            use_augmentations=True,
            verbose=False
        )

        sample = dataset[0]
        assert 'images' in sample, "Sample missing 'images' key"
        assert 'masks' in sample, "Sample missing 'masks' key"
        assert 1024 in sample['images'], "Sample missing 1024 resolution"
        assert sample['images'][1024].shape == (3, 1024, 1024), f"Wrong image shape: {sample['images'][1024].shape}"
        assert sample['masks'][1024].shape == (4, 1024, 1024), f"Wrong mask shape: {sample['masks'][1024].shape}"

        print("  ✓ Dataset loads samples correctly")
        img_shapes = ', '.join([f"{res}: {sample['images'][res].shape}" for res in [1024, 512, 256]])
        mask_shapes = ', '.join([f"{res}: {sample['masks'][res].shape}" for res in [1024, 512, 256]])
        print(f"  ✓ Image shapes: {img_shapes}")
        print(f"  ✓ Mask shapes: {mask_shapes}")
        print("✓ Dataset test PASSED")
        return True
    except Exception as e:
        print(f"✗ Dataset test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pyramid():
    """Test feature pyramid module."""
    print("\n" + "="*60)
    print("TEST 2: Feature Pyramid Module")
    print("="*60)

    try:
        pyramid = FeaturePyramidModule(
            in_channels=3,
            pyramid_channels=64,
            resolutions=[1024, 512, 256],
            num_conv_blocks=2
        )

        images = {
            1024: torch.randn(2, 3, 1024, 1024),
            512: torch.randn(2, 3, 512, 512),
            256: torch.randn(2, 3, 256, 256)
        }

        features = pyramid(images)

        assert 1024 in features, "Missing 1024 resolution in output"
        assert features[1024].shape == (2, 64, 1024, 1024), f"Wrong feature shape: {features[1024].shape}"

        num_params = pyramid.get_num_parameters()
        print(f"  ✓ Pyramid processes all resolutions")
        feat_shapes = ', '.join([f"{res}: {features[res].shape}" for res in [1024, 512, 256]])
        print(f"  ✓ Output shapes: {feat_shapes}")
        print(f"  ✓ Parameters: {num_params:,}")
        print("✓ Pyramid test PASSED")
        return True
    except Exception as e:
        print(f"✗ Pyramid test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fusion():
    """Test feature fusion module."""
    print("\n" + "="*60)
    print("TEST 3: Feature Fusion Module (all strategies)")
    print("="*60)

    try:
        fusion_types = ['concat', 'attention', 'progressive']

        for fusion_type in fusion_types:
            # Build kwargs based on fusion type
            kwargs = {
                'pyramid_channels': 64,
                'output_channels': 64,
                'num_resolutions': 3,
                'target_resolution': 1024,
                'fusion_type': fusion_type
            }
            # Only progressive fusion needs resolutions list
            if fusion_type == 'progressive':
                kwargs['resolutions'] = [1024, 512, 256]

            fusion = FeatureFusion(**kwargs)

            features = {
                1024: torch.randn(2, 64, 1024, 1024),
                512: torch.randn(2, 64, 512, 512),
                256: torch.randn(2, 64, 256, 256)
            }

            fused = fusion(features)
            assert fused.shape == (2, 64, 1024, 1024), f"{fusion_type}: Wrong fused shape: {fused.shape}"

            num_params = fusion.get_num_parameters()
            print(f"  ✓ {fusion_type.upper()}: output shape {fused.shape}, params: {num_params:,}")

        print("✓ Fusion test PASSED")
        return True
    except Exception as e:
        print(f"✗ Fusion test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test complete multi-resolution UNet++ model."""
    print("\n" + "="*60)
    print("TEST 4: Multi-Resolution UNet++ Model")
    print("="*60)

    try:
        model = MultiResUNetPlusPlus(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            resolutions=[1024, 512, 256],
            pyramid_channels=64,
            fusion_type='concat',
            n_classes=4
        )

        images = {
            1024: torch.randn(2, 3, 1024, 1024),
            512: torch.randn(2, 3, 512, 512),
            256: torch.randn(2, 3, 256, 256)
        }

        # Test forward pass
        output = model(images, return_intermediate=True)

        assert 'segmentation' in output, "Output missing 'segmentation' key"
        assert output['segmentation'].shape == (2, 4, 1024, 1024), f"Wrong output shape: {output['segmentation'].shape}"
        assert 'pyramid_features' in output, "Missing intermediate pyramid features"
        assert 'fused_features' in output, "Missing intermediate fused features"

        # Test parameter counts
        param_counts = model.get_num_parameters(trainable_only=True)

        print(f"  ✓ Model forward pass successful")
        print(f"  ✓ Output shape: {output['segmentation'].shape}")
        print(f"  ✓ Parameter breakdown:")
        for component, count in param_counts.items():
            print(f"      {component}: {count:,}")

        print("✓ Model test PASSED")
        return True
    except Exception as e:
        print(f"✗ Model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test gradient flow through model."""
    print("\n" + "="*60)
    print("TEST 5: Gradient Flow")
    print("="*60)

    try:
        model = MultiResUNetPlusPlus(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            resolutions=[1024, 512, 256],
            pyramid_channels=64,
            fusion_type='concat',
            n_classes=4
        )

        images = {
            1024: torch.randn(2, 3, 1024, 1024),
            512: torch.randn(2, 3, 512, 512),
            256: torch.randn(2, 3, 256, 256)
        }

        output = model(images)
        loss = output['segmentation'].mean()
        loss.backward()

        # Check gradients in each component
        pyramid_has_grad = any(p.grad is not None for p in model.pyramid.parameters())
        fusion_has_grad = any(p.grad is not None for p in model.fusion.parameters())
        unetpp_has_grad = any(p.grad is not None for p in model.unetpp.parameters())

        assert pyramid_has_grad, "Pyramid has no gradients"
        assert fusion_has_grad, "Fusion has no gradients"
        assert unetpp_has_grad, "UNet++ has no gradients"

        print(f"  ✓ Pyramid: gradients flowing")
        print(f"  ✓ Fusion: gradients flowing")
        print(f"  ✓ UNet++: gradients flowing")
        print("✓ Gradient flow test PASSED")
        return True
    except Exception as e:
        print(f"✗ Gradient flow test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_integration():
    """Test integration with PyTorch DataLoader."""
    print("\n" + "="*60)
    print("TEST 6: DataLoader Integration")
    print("="*60)

    try:
        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = MultiResolutionDDRDataset(
            split='train',
            resolutions=[1024, 512, 256],
            transform=image_transform,
            use_augmentations=False,
            verbose=False
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
        batch = next(iter(loader))

        assert 'images' in batch, "Batch missing 'images' key"
        assert 'masks' in batch, "Batch missing 'masks' key"
        assert batch['images'][1024].shape[0] == 2, f"Wrong batch size: {batch['images'][1024].shape[0]}"

        print(f"  ✓ DataLoader batches correctly")
        batch_shapes = ', '.join([f"{res}: {batch['images'][res].shape}" for res in [1024, 512, 256]])
        print(f"  ✓ Batch image shapes: {batch_shapes}")
        print("✓ DataLoader integration test PASSED")
        return True
    except Exception as e:
        print(f"✗ DataLoader integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_train_integration():
    """Test that model works with train.py style input handling."""
    print("\n" + "="*60)
    print("TEST 7: train.py Integration")
    print("="*60)

    try:
        model = MultiResUNetPlusPlus(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            resolutions=[1024, 512, 256],
            pyramid_channels=64,
            fusion_type='concat',
            n_classes=4
        )

        # Simulate batch from MultiResolutionDDRDataset
        batch = {
            'images': {
                1024: torch.randn(2, 3, 1024, 1024),
                512: torch.randn(2, 3, 512, 512),
                256: torch.randn(2, 3, 256, 256)
            },
            'masks': {
                1024: torch.randint(0, 2, (2, 4, 1024, 1024)).float(),
                512: torch.randint(0, 2, (2, 4, 512, 512)).float(),
                256: torch.randint(0, 2, (2, 4, 256, 256)).float()
            },
            'image_id': ['img1', 'img2']
        }

        # Simulate train.py handling
        images = batch['images']
        masks_dict = batch['masks']
        target = masks_dict[max(masks_dict.keys())]  # Use highest res as target

        # Forward pass
        output = model(images)

        # Check output format
        assert isinstance(output, dict), "Output should be dict"
        assert 'segmentation' in output, "Output missing 'segmentation'"
        assert output['segmentation'].shape == target.shape, f"Output shape {output['segmentation'].shape} != target {target.shape}"

        # Test loss computation
        import torch.nn as nn
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(output['segmentation'], target)
        loss.backward()

        print(f"  ✓ Batch format compatible with train.py")
        print(f"  ✓ Output shape matches target: {output['segmentation'].shape}")
        print(f"  ✓ Loss computation successful: {loss.item():.4f}")
        print("✓ train.py integration test PASSED")
        return True
    except Exception as e:
        print(f"✗ train.py integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*70)
    print("MULTI-RESOLUTION UNet++ INTEGRATION TESTS")
    print("="*70)

    tests = [
        ("Dataset", test_dataset),
        ("Feature Pyramid", test_pyramid),
        ("Feature Fusion", test_fusion),
        ("Complete Model", test_model),
        ("Gradient Flow", test_gradient_flow),
        ("DataLoader Integration", test_dataloader_integration),
        ("train.py Integration", test_train_integration)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed ({passed*100//total}%)")

    if passed == total:
        print("\n🎉 All tests PASSED!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
