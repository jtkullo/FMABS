"""
骨干网络模块 - 提供特征提取功能
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List, Tuple, Optional


def get_backbone(name: str = 'wide_resnet50_2', pretrained: bool = True, weights=None):
    """
    获取预训练的骨干网络
    
    Args:
        name: 网络名称
        pretrained: 是否使用预训练权重（已弃用，保留用于兼容性）
        weights: 权重枚举，如果为None且pretrained=True则使用默认权重
    
    Returns:
        骨干网络模型
    """
    # 处理权重参数：优先使用weights，如果未提供则根据pretrained决定
    try:
        # 尝试使用新的weights参数（PyTorch 0.13+）
        if weights is None:
            # 根据pretrained参数决定使用默认权重还是None
            if pretrained:
                weights = 'DEFAULT'
            else:
                weights = None
        
        if name == 'wide_resnet50_2':
            if weights == 'DEFAULT':
                model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
            elif weights is None:
                model = models.wide_resnet50_2(weights=None)
            else:
                model = models.wide_resnet50_2(weights=weights)
        elif name == 'resnet18':
            if weights == 'DEFAULT':
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            elif weights is None:
                model = models.resnet18(weights=None)
            else:
                model = models.resnet18(weights=weights)
        elif name == 'resnet50':
            if weights == 'DEFAULT':
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            elif weights is None:
                model = models.resnet50(weights=None)
            else:
                model = models.resnet50(weights=weights)
        elif name == 'efficientnet_b0':
            if weights == 'DEFAULT':
                model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            elif weights is None:
                model = models.efficientnet_b0(weights=None)
            else:
                model = models.efficientnet_b0(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {name}")
    except (AttributeError, TypeError):
        # 如果新API不可用，回退到旧API（兼容旧版本PyTorch）
        if name == 'wide_resnet50_2':
            model = models.wide_resnet50_2(pretrained=pretrained)
        elif name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        elif name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {name}")
    
    return model


class FeatureExtractor(nn.Module):
    """
    特征提取器 - 从骨干网络中提取多层特征
    """
    def __init__(self, backbone_name: str = 'wide_resnet50_2', 
                 feature_layers: List[str] = None,
                 pretrained: bool = True):
        super(FeatureExtractor, self).__init__()
        
        self.backbone_name = backbone_name
        self.backbone = get_backbone(backbone_name, pretrained)
        self.feature_layers = feature_layers or ['layer1', 'layer2', 'layer3']
        
        # 根据不同的backbone设置特征层
        if 'resnet' in backbone_name:
            self._setup_resnet_layers()
        elif 'efficientnet' in backbone_name:
            self._setup_efficientnet_layers()
    
    def _setup_resnet_layers(self):
        """设置ResNet的特征层"""
        self.layer1 = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1
        )
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
    
    def _setup_efficientnet_layers(self):
        """设置EfficientNet的特征层"""
        features = list(self.backbone.features.children())
        self.layer1 = nn.Sequential(*features[:3])
        self.layer2 = nn.Sequential(*features[3:5])
        self.layer3 = nn.Sequential(*features[5:7])
        self.layer4 = nn.Sequential(*features[7:])
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播，提取多层特征
        
        Args:
            x: 输入张量 [B, C, H, W]
        
        Returns:
            包含各层特征的字典
        """
        features = {}
        
        if 'resnet' in self.backbone_name:
            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            
            if 'layer1' in self.feature_layers:
                x = self.backbone.layer1(x)
                features['layer1'] = x
            if 'layer2' in self.feature_layers:
                x = self.backbone.layer2(x)
                features['layer2'] = x
            if 'layer3' in self.feature_layers:
                x = self.backbone.layer3(x)
                features['layer3'] = x
            if 'layer4' in self.feature_layers:
                x = self.backbone.layer4(x)
                features['layer4'] = x
        
        elif 'efficientnet' in self.backbone_name:
            features_list = []
            for i, layer in enumerate(self.backbone.features):
                x = layer(x)
                if i in [2, 4, 6, 15]:  # 关键特征层
                    features_list.append(x)
            
            if 'layer1' in self.feature_layers:
                features['layer1'] = features_list[0]
            if 'layer2' in self.feature_layers:
                features['layer2'] = features_list[1]
            if 'layer3' in self.feature_layers:
                features['layer3'] = features_list[2]
            if 'layer4' in self.feature_layers:
                features['layer4'] = features_list[3]
        
        return features
