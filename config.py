"""
配置文件 - Feature Mimicking with Attention for Anomaly Detection
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """主配置类"""
    # 数据集配置
    dataset: str = 'mvtec'  # 'mvtec', 'cuhk_avenue', 'shanghaitech'
    data_root: str = './data'
    num_workers: int = 4
    batch_size: int = 8
    image_size: int = 256
    
    # 模型配置
    backbone: str = 'wide_resnet50_2'  # 或 'resnet18', 'efficientnet_b0'
    feature_layers: list = None  # 用于特征提取的层
    embedding_dim: int = 512
    
    # 训练配置
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    
    # 损失权重
    recon_weight: float = 1.0
    feature_mimic_weight: float = 0.5
    attention_weight: float = 0.3
    
    # 注意力配置
    attention_dim: int = 256
    use_attention: bool = True
    
    # 推理配置
    test_batch_size: int = 1
    threshold_percentile: float = 95.0
    
    # 路径配置
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    result_dir: str = './results'
    
    # 其他
    seed: int = 42
    device: str = 'cuda'  # 'cuda' 或 'cpu'
    resume: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.feature_layers is None:
            # 默认使用Wide ResNet50的特征层
            self.feature_layers = ['layer1', 'layer2', 'layer3']
        
        # 创建必要的目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 根据数据集调整配置
        if self.dataset == 'cuhk_avenue' or self.dataset == 'shanghaitech':
            # 视频数据集配置
            self.sequence_length = 16
            self.temporal_stride = 1
        else:
            # 图像数据集配置
            self.sequence_length = 1
            self.temporal_stride = 1
