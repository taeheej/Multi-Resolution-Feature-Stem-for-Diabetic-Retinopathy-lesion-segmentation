"""
A generic 'meta-model' that combines a multi-resolution stem with any
standard segmentation model from the segmentation_models_pytorch library.
"""
import torch
import torch.nn as nn
from typing import Dict
import segmentation_models_pytorch as smp

# Add parent directories to path for standalone execution/testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multi_resolution.multi_resolution_stem import MultiResolutionStem

class GenericMultiResModel(nn.Module):
    """
    A wrapper model that combines a `MultiResolutionStem` with a standard
    segmentation backbone (e.g., Unet, DeepLabV3+).

    This allows for a flexible, plug-and-play approach to applying
    multi-resolution processing to various architectures.
    """
    def __init__(self, stem: MultiResolutionStem, backbone: nn.Module):
        super().__init__()
        self.stem = stem
        self.backbone = backbone

    def forward(self, images: Dict[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Passes the multi-resolution images through the stem, then the backbone.
        """
        fused_features = self.stem(images)
        segmentation_output = self.backbone(fused_features)
        
        # We wrap the output in a dictionary to maintain consistency
        # with the multi-output format of our other models.
        return {'segmentation': segmentation_output}

def adapt_encoder(encoder, in_channels, encoder_name):
    """
    Generic function to replace the first convolutional layer of a standard
    segmentation model's encoder.
    """
    # This logic is critical for connecting the stem to the backbone.
    # It assumes encoder_weights have been loaded before this function is called.
    
    if 'resnet' in encoder_name or 'resnext' in encoder_name:
        old_conv = encoder.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                               stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)
        
        # Intelligent weight initialization
        if in_channels % 3 == 0 and old_conv.in_channels == 3:
            with torch.no_grad():
                repeat_factor = in_channels // 3
                repeated_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)
                new_conv.weight.copy_(repeated_weights / repeat_factor)
                print(f"  ✓ Initialized first conv with repeated pretrained weights ({in_channels} channels)")
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            print(f"  ✓ Initialized first conv with Kaiming initialization ({in_channels} channels)")

        encoder.conv1 = new_conv
        
    elif 'efficientnet' in encoder_name:
        old_conv = encoder.conv_stem
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                               stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)
        
        if in_channels % 3 == 0 and old_conv.in_channels == 3:
            with torch.no_grad():
                repeat_factor = in_channels // 3
                repeated_weights = old_conv.weight.repeat(1, repeat_factor, 1, 1)
                new_conv.weight.copy_(repeated_weights / repeat_factor)
                print(f"  ✓ Initialized first conv with repeated pretrained weights ({in_channels} channels)")
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            print(f"  ✓ Initialized first conv with Kaiming initialization ({in_channels} channels)")
            
        encoder.conv_stem = new_conv
    else:
        raise NotImplementedError(f"Encoder adaptation not implemented for {encoder_name}")
    
    print(f"  ✓ Adapted backbone encoder input: 3 -> {in_channels} channels.")
    return encoder

def create_generic_multi_res_model(model_type, encoder_name, pretrained, n_classes, **kwargs):
    """
    Factory function to create a generic multi-resolution model.
    This encapsulates all the complex model-building logic, including the
    special case for Pascal VOC pretrained DeepLabV3+.
    """
    print(f"  Creating Generic Multi-Resolution Model with:")
    print(f"    - Stem Fusion: {kwargs.get('fusion_type')}")
    print(f"    - Backbone: {model_type.split('_')[-1]}")
    print(f"    - Encoder: {encoder_name}")

    # 1. Create the Multi-Resolution Stem
    stem = MultiResolutionStem(
        resolutions=kwargs.get('resolutions', [1024, 512, 256]),
        pyramid_channels=kwargs.get('pyramid_channels', 64),
        fusion_type=kwargs.get('fusion_type', 'concat'),
        fusion_output_channels=kwargs.get('fusion_output_channels'),
        expansion_channels=kwargs.get('expansion_channels'),
        use_adaptive_pyramid=kwargs.get('use_adaptive_pyramid', False)
    )

    # 2. Create the chosen SMP backbone (with special handling for VOC)
    backbone_class_name = None
    if 'unet' in model_type:
        backbone_class_name = 'unet'
    elif 'deeplab' in model_type:
        backbone_class_name = 'deeplab'
    elif 'fpn' in model_type:
        backbone_class_name = 'fpn'
    
    if backbone_class_name is None:
        raise ValueError(f"Could not determine backbone from model_type: {model_type}")

    if 'deeplab' in backbone_class_name and kwargs.get('pretrained_source') == 'pascal_voc':
        # --- Special Case: DeepLabV3+ with Pascal VOC weights ---
        if encoder_name != 'resnet101':
            raise ValueError("Pascal VOC weights are only available for 'resnet101' encoder.")
        
        print(f"  Loading DeepLabV3+ with Pascal VOC pretrained weights...")
        from torchvision.models.segmentation import deeplabv3_resnet101
        from torchvision.models.segmentation.deeplabv3 import DeepLabV3_ResNet101_Weights

        pretrained_torchvision_model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        
        backbone = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=n_classes)
        
        # Manually transfer the encoder weights
        backbone.encoder.load_state_dict(pretrained_torchvision_model.backbone.state_dict())
        print("  ✓ Transferred Pascal VOC encoder weights.")

    else:
        # --- Standard SMP Backbone Creation ---
        encoder_weights = "imagenet" if pretrained else None
        if backbone_class_name == 'unet':
            backbone = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=n_classes)
        elif backbone_class_name == 'deeplab':
            backbone = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=n_classes)
        elif backbone_class_name == 'fpn':
            backbone = smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=n_classes)
        else:
            raise ValueError(f"Unsupported multi-res backbone: {backbone_class_name}")

    # 3. Adapt the backbone's encoder to accept the stem's output
    backbone.encoder = adapt_encoder(backbone.encoder, stem.output_channels, encoder_name)
    
    # 4. Combine them in the generic wrapper
    model = GenericMultiResModel(stem=stem, backbone=backbone)
    print(f"  ✓ Generic Multi-Resolution model created.")
    
    return model
