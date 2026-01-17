"""
评估指标模块
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List, Tuple, Optional


def compute_auroc(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    """
    计算AUROC（Area Under ROC Curve）
    
    Args:
        anomaly_scores: 异常分数数组
        labels: 标签数组（0=正常, 1=异常）
    
    Returns:
        AUROC分数
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    try:
        auroc = roc_auc_score(labels, anomaly_scores)
        return float(auroc)
    except Exception as e:
        print(f"Error computing AUROC: {e}")
        return 0.0


def compute_ap(anomaly_scores: np.ndarray, labels: np.ndarray) -> float:
    """
    计算AP（Average Precision）
    
    Args:
        anomaly_scores: 异常分数数组
        labels: 标签数组（0=正常, 1=异常）
    
    Returns:
        AP分数
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    try:
        ap = average_precision_score(labels, anomaly_scores)
        return float(ap)
    except Exception as e:
        print(f"Error computing AP: {e}")
        return 0.0


def compute_pixel_auroc(anomaly_maps: np.ndarray, 
                   ground_truth_maps: np.ndarray) -> float:
    """
    计算像素级AUROC（用于异常定位）
    
    Args:
        anomaly_maps: 异常图数组 [N, H, W]
        ground_truth_maps: 真实标签图数组 [N, H, W]
    
    Returns:
        像素级AUROC分数
    """
    # 展平所有像素
    anomaly_scores = anomaly_maps.flatten()
    labels = ground_truth_maps.flatten()
    
    return compute_auroc(anomaly_scores, labels)


def compute_anomaly_score(reconstruction_error: torch.Tensor,
                         feature_inconsistency: Optional[torch.Tensor] = None,
                         attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    计算异常分数
    
    Args:
        reconstruction_error: 重建误差 [B, C, H, W] 或 [B, H, W]
        feature_inconsistency: 特征不一致性 [B, C, H, W]（可选）
        attention_weights: 注意力权重 [B, 1, H, W]（可选）
    
    Returns:
        异常分数 [B, H, W]
    """
    # 如果重建误差是多通道的，计算平均值
    if reconstruction_error.dim() == 4:
        reconstruction_error = reconstruction_error.mean(dim=1)  # [B, H, W]
    
    anomaly_score = reconstruction_error
    
    # 如果提供了特征不一致性，加权融合
    if feature_inconsistency is not None:
        if feature_inconsistency.dim() == 4:
            feature_inconsistency = feature_inconsistency.mean(dim=1)  # [B, H, W]
        
        # 归一化
        feature_inconsistency = (feature_inconsistency - feature_inconsistency.min()) / \
                               (feature_inconsistency.max() - feature_inconsistency.min() + 1e-8)
        
        anomaly_score = 0.7 * anomaly_score + 0.3 * feature_inconsistency
    
    # 如果提供了注意力权重，应用注意力
    if attention_weights is not None:
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.squeeze(1)  # [B, H, W]
        
        anomaly_score = anomaly_score * (1 + attention_weights)
    
    return anomaly_score
