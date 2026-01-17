"""
损失函数模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class ReconstructionLoss(nn.Module):
    """
    重建损失 - L1和L2损失的组合
    """
    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 1.0):
        super(ReconstructionLoss, self).__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
    
    def forward(self, reconstructed: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """
        计算重建损失
        
        Args:
            reconstructed: 重建图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
        
        Returns:
            损失值
        """
        l1 = self.l1_loss(reconstructed, target)
        l2 = self.l2_loss(reconstructed, target)
        
        return self.l1_weight * l1 + self.l2_weight * l2


class FeatureMimickingLoss(nn.Module):
    """
    特征模仿损失 - 让学生网络模仿教师网络的特征
    """
    def __init__(self, reduction: str = 'mean'):
        super(FeatureMimickingLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算特征模仿损失
        
        Args:
            student_features: 学生网络特征字典
            teacher_features: 教师网络特征字典
        
        Returns:
            损失值
        """
        total_loss = 0.0
        num_layers = 0
        
        for layer_name in student_features.keys():
            if layer_name in teacher_features:
                student_feat = student_features[layer_name]
                teacher_feat = teacher_features[layer_name]
                
                # 确保尺寸匹配
                if student_feat.shape != teacher_feat.shape:
                    teacher_feat = F.interpolate(
                        teacher_feat,
                        size=student_feat.shape[2:],
                        mode='bilinear',
                        align_corners=False
                    )
                
                # 计算L2距离
                layer_loss = F.mse_loss(student_feat, teacher_feat, reduction=self.reduction)
                total_loss += layer_loss
                num_layers += 1
        
        if num_layers > 0:
            total_loss = total_loss / num_layers
        
        return total_loss


class AttentionLoss(nn.Module):
    """
    注意力损失 - 鼓励注意力关注异常区域
    """
    def __init__(self):
        super(AttentionLoss, self).__init__()
    
    def forward(self, attention_weights: torch.Tensor,
                inconsistency_matrix: torch.Tensor) -> torch.Tensor:
        """
        计算注意力损失
        
        Args:
            attention_weights: 注意力权重 [B, 1, H, W]
            inconsistency_matrix: 特征不一致矩阵 [B, C, H, W]
        
        Returns:
            损失值
        """
        # 将不一致矩阵聚合为单通道
        inconsistency = inconsistency_matrix.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 归一化
        inconsistency = (inconsistency - inconsistency.min()) / (inconsistency.max() - inconsistency.min() + 1e-8)
        
        # 计算注意力与不一致性的相关性（负相关，因为希望注意力关注不一致性高的区域）
        # 使用交叉熵或MSE
        loss = F.mse_loss(attention_weights, inconsistency)
        
        return loss


class TotalLoss(nn.Module):
    """
    总损失函数 - 组合所有损失
    """
    def __init__(self, recon_weight: float = 1.0,
                 feature_mimic_weight: float = 0.5,
                 attention_weight: float = 0.3):
        super(TotalLoss, self).__init__()
        
        self.recon_weight = recon_weight
        self.feature_mimic_weight = feature_mimic_weight
        self.attention_weight = attention_weight
        
        self.recon_loss = ReconstructionLoss()
        self.feature_mimic_loss = FeatureMimickingLoss()
        self.attention_loss = AttentionLoss()
    
    def forward(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """
        计算总损失
        
        Args:
            outputs: 模型输出字典
            targets: 目标字典
        
        Returns:
            损失字典
        """
        losses = {}
        
        # 重建损失
        recon_loss = self.recon_loss(
            outputs['reconstructed'],
            targets['image']
        )
        losses['recon_loss'] = recon_loss
        
        # 特征模仿损失
        if 'student_features' in outputs and 'teacher_features' in targets:
            feature_mimic_loss = self.feature_mimic_loss(
                outputs['student_features'],
                targets['teacher_features']
            )
            losses['feature_mimic_loss'] = feature_mimic_loss
        else:
            losses['feature_mimic_loss'] = torch.tensor(0.0, device=recon_loss.device)
        
        # 注意力损失（可选）
        if 'attention_weights' in outputs and 'inconsistency_matrices' in outputs:
            if len(outputs['inconsistency_matrices']) > 0:
                # 使用第一个不一致矩阵
                inconsistency = outputs['inconsistency_matrices'][0]
                attention_weights = outputs['attention_weights']
                
                if isinstance(attention_weights, list):
                    attention_weights = attention_weights[0]
                
                if attention_weights is not None:
                    attn_loss = self.attention_loss(attention_weights, inconsistency)
                    losses['attention_loss'] = attn_loss
                else:
                    losses['attention_loss'] = torch.tensor(0.0, device=recon_loss.device)
            else:
                losses['attention_loss'] = torch.tensor(0.0, device=recon_loss.device)
        else:
            losses['attention_loss'] = torch.tensor(0.0, device=recon_loss.device)
        
        # 总损失
        total_loss = (
            self.recon_weight * losses['recon_loss'] +
            self.feature_mimic_weight * losses['feature_mimic_loss'] +
            self.attention_weight * losses['attention_loss']
        )
        losses['total_loss'] = total_loss
        
        return losses
