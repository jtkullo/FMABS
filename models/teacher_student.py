"""
教师-学生网络架构
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from .backbone import FeatureExtractor
from .attention import FeatureInconsistencyAttention, MultiScaleAttention
from .decoder import Decoder


class TeacherNetwork(nn.Module):
    """
    教师网络 - 预训练的网络，用于提取正常样本的特征表示
    """
    def __init__(self, backbone_name: str = 'wide_resnet50_2',
                 feature_layers: List[str] = None,
                 pretrained: bool = True):
        super(TeacherNetwork, self).__init__()
        
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone_name,
            feature_layers=feature_layers,
            pretrained=pretrained
        )
        
        # 冻结教师网络参数（在训练过程中不更新）
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            多层特征字典
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features


class StudentNetwork(nn.Module):
    """
    学生网络 - 学习重建任务和特征模仿任务
    """
    def __init__(self, backbone_name: str = 'wide_resnet50_2',
                 feature_layers: List[str] = None,
                 embedding_dim: int = 512,
                 use_attention: bool = True,
                 attention_dim: int = 256,
                 pretrained: bool = True):
        super(StudentNetwork, self).__init__()
        
        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone_name,
            feature_layers=feature_layers,
            pretrained=pretrained
        )
        
        self.feature_layers = feature_layers or ['layer1', 'layer2', 'layer3']
        self.use_attention = use_attention
        
        # 获取特征维度
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            dummy_features = self.feature_extractor(dummy_input)
            feature_dims = [dummy_features[layer].shape[1] for layer in self.feature_layers]
        
        # 注意力模块
        if use_attention:
            if len(feature_dims) > 1:
                self.attention_module = MultiScaleAttention(feature_dims, attention_dim)
                # 多尺度注意力输出维度是attention_dim
                decoder_input_dim = attention_dim
            else:
                self.attention_module = FeatureInconsistencyAttention(
                    feature_dims[0], attention_dim
                )
                # 单层注意力时，refined_features的维度仍然是feature_dims[0]
                # 因为FeatureInconsistencyAttention的refiner输出维度是feature_dim
                decoder_input_dim = feature_dims[0]
        else:
            # 不使用注意力时，使用最后一层特征的维度
            decoder_input_dim = feature_dims[-1]
        
        # 重建解码器
        self.decoder = Decoder(
            feature_dims=feature_dims,
            output_channels=3,
            embedding_dim=embedding_dim,
            input_dim=decoder_input_dim  # 传递实际输入维度
        )
    
    def forward(self, x: torch.Tensor, 
                teacher_features: Optional[Dict[str, torch.Tensor]] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            teacher_features: 教师网络特征（用于计算不一致性）
            return_features: 是否返回中间特征
        
        Returns:
            包含重建图像和特征的字典
        """
        # 提取学生网络特征
        student_features_dict = self.feature_extractor(x)
        student_features_list = [student_features_dict[layer] for layer in self.feature_layers]
        
        # 计算特征不一致性（如果提供了教师特征）
        inconsistency_matrices = None
        if teacher_features is not None and self.use_attention:
            inconsistency_matrices = []
            for layer in self.feature_layers:
                teacher_feat = teacher_features[layer]
                student_feat = student_features_dict[layer]
                
                # 确保尺寸匹配
                if teacher_feat.shape != student_feat.shape:
                    teacher_feat = F.interpolate(
                        teacher_feat, 
                        size=student_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 计算特征不一致性（L2距离）
                inconsistency = torch.abs(teacher_feat - student_feat)
                inconsistency_matrices.append(inconsistency)
        
        # 应用注意力模块
        if self.use_attention and inconsistency_matrices is not None:
            if len(self.feature_layers) > 1:
                attention_output = self.attention_module(
                    student_features_list, inconsistency_matrices
                )
                refined_features = attention_output['fused_features']
            else:
                attention_output = self.attention_module(
                    student_features_list[0], inconsistency_matrices[0]
                )
                refined_features = attention_output['refined_features']
        else:
            # 不使用注意力时，直接使用最后一层特征
            refined_features = student_features_list[-1]
            attention_output = None
        
        # 重建图像（确保输出尺寸与输入匹配）
        target_size = x.shape[2:]  # 获取输入图像尺寸
        reconstructed = self.decoder(refined_features, student_features_list, target_size=target_size)
        
        output = {
            'reconstructed': reconstructed,
            'student_features': student_features_dict,
            'refined_features': refined_features
        }
        
        if attention_output is not None:
            output['attention_weights'] = attention_output.get('attention_weights')
            output['inconsistency_matrices'] = inconsistency_matrices
        
        return output
