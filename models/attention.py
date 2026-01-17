"""
基于特征不一致性的注意力模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class FeatureInconsistencyAttention(nn.Module):
    """
    特征不一致性引导的注意力模块
    
    该模块接收教师-学生网络的特征不一致矩阵作为输入，
    生成注意力权重来引导重建过程关注异常区域。
    """
    def __init__(self, feature_dim: int, attention_dim: int = 256):
        """
        Args:
            feature_dim: 特征维度
            attention_dim: 注意力中间维度
        """
        super(FeatureInconsistencyAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # 特征不一致性编码器
        self.inconsistency_encoder = nn.Sequential(
            nn.Conv2d(feature_dim, attention_dim, kernel_size=1),
            nn.BatchNorm2d(attention_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim, attention_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(attention_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力权重生成器
        self.attention_generator = nn.Sequential(
            nn.Conv2d(attention_dim, attention_dim // 2, kernel_size=1),
            nn.BatchNorm2d(attention_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(attention_dim // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 特征细化模块
        self.feature_refiner = nn.Sequential(
            nn.Conv2d(feature_dim + attention_dim, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, student_features: torch.Tensor, 
                inconsistency_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            student_features: 学生网络特征 [B, C, H, W]
            inconsistency_matrix: 特征不一致矩阵 [B, C, H, W]
        
        Returns:
            包含细化特征和注意力权重的字典
        """
        # 编码特征不一致性
        inconsistency_encoded = self.inconsistency_encoder(inconsistency_matrix)
        
        # 生成注意力权重
        attention_weights = self.attention_generator(inconsistency_encoded)
        
        # 应用注意力到不一致性编码
        attended_inconsistency = inconsistency_encoded * attention_weights
        
        # 拼接学生特征和注意力引导的不一致性特征
        concatenated = torch.cat([student_features, attended_inconsistency], dim=1)
        
        # 细化特征
        refined_features = self.feature_refiner(concatenated)
        
        return {
            'refined_features': refined_features,
            'attention_weights': attention_weights,
            'inconsistency_encoded': inconsistency_encoded
        }


class MultiScaleAttention(nn.Module):
    """
    多尺度注意力模块 - 处理不同尺度的特征
    """
    def __init__(self, feature_dims: list, attention_dim: int = 256):
        """
        Args:
            feature_dims: 各层特征的维度列表
            attention_dim: 注意力维度
        """
        super(MultiScaleAttention, self).__init__()
        
        self.attention_modules = nn.ModuleList([
            FeatureInconsistencyAttention(feat_dim, attention_dim)
            for feat_dim in feature_dims
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(sum(feature_dims), attention_dim, kernel_size=1),
            nn.BatchNorm2d(attention_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, student_features_list: list, 
                inconsistency_matrices: list) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            student_features_list: 学生网络的多层特征列表
            inconsistency_matrices: 多层特征不一致矩阵列表
        
        Returns:
            融合后的特征和注意力信息
        """
        refined_features_list = []
        attention_weights_list = []
        
        # 对每一层应用注意力
        for i, (student_feat, inconsistency) in enumerate(
            zip(student_features_list, inconsistency_matrices)
        ):
            output = self.attention_modules[i](student_feat, inconsistency)
            refined_features_list.append(output['refined_features'])
            attention_weights_list.append(output['attention_weights'])
        
        # 上采样到相同尺寸并融合
        target_size = refined_features_list[-1].shape[2:]
        upsampled_features = []
        
        for feat in refined_features_list:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            upsampled_features.append(feat)
        
        # 拼接所有特征
        fused_features = torch.cat(upsampled_features, dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        return {
            'fused_features': fused_features,
            'attention_weights': attention_weights_list,
            'refined_features': refined_features_list
        }
