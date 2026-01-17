"""
重建解码器模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class Decoder(nn.Module):
    """
    重建解码器 - 将特征解码回原始图像空间
    """
    def __init__(self, feature_dims: List[int], output_channels: int = 3,
                 embedding_dim: int = 512, input_dim: Optional[int] = None):
        """
        Args:
            feature_dims: 各层特征的通道数列表
            output_channels: 输出图像通道数
            embedding_dim: 嵌入维度
            input_dim: 输入特征的通道数（如果为None，则使用feature_dims[-1]）
        """
        super(Decoder, self).__init__()
        
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim if input_dim is not None else feature_dims[-1]
        
        # 特征投影层 - 使用实际的输入维度
        self.feature_projector = nn.Sequential(
            nn.Conv2d(self.input_dim, embedding_dim, kernel_size=1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # 解码器主体 - 使用转置卷积逐步上采样
        self.decoder_layers = nn.ModuleList([
            # 第一层：从embedding_dim到256
            nn.Sequential(
                nn.ConvTranspose2d(embedding_dim, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # 第二层：从256到128
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # 第三层：从128到64
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # 第四层：从64到32
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            # 最后一层：从32到输出通道
            nn.Sequential(
                nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
                nn.Tanh()  # 输出范围[-1, 1]，需要后续归一化到[0, 1]
            )
        ])
        
        # 跳跃连接处理（可选）
        # 只为除了最后一层之外的特征层创建跳跃连接
        # 注意：跳跃连接的输出通道数需要与解码器对应层的通道数匹配
        # 解码器层的通道数序列：[embedding_dim, 256, 128, 64, 32, output_channels]
        # 跳跃连接应该输出到对应解码器层的输出通道数
        # 第i个跳跃连接应该输出到第i个解码器层后的通道数
        skip_output_channels = [256, 128, 64, 32]  # 对应解码器前4层的输出通道数
        self.skip_connections = nn.ModuleList([
            nn.Conv2d(dim, skip_output_channels[i], kernel_size=1) 
            for i, dim in enumerate(feature_dims[:-1])
            if i < len(skip_output_channels)
        ]) if len(feature_dims) > 1 else nn.ModuleList()
    
    def forward(self, main_features: torch.Tensor,
                multi_scale_features: Optional[List[torch.Tensor]] = None,
                target_size: Optional[tuple] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            main_features: 主要特征 [B, C, H, W]
            multi_scale_features: 多尺度特征列表（用于跳跃连接）
            target_size: 目标输出尺寸 (H, W)，如果为None则自动计算
        
        Returns:
            重建的图像 [B, 3, H_out, W_out]
        """
        # 投影特征
        x = self.feature_projector(main_features)
        
        # 通过解码器层
        # 解码器各层的输出通道数：[256, 128, 64, 32, 3]
        decoder_output_channels = [256, 128, 64, 32]
        
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)
            
            # 添加跳跃连接（如果有多尺度特征）
            # 只在前面几层添加跳跃连接（不包括最后一层输出层）
            if multi_scale_features is not None and len(self.skip_connections) > 0 and i < len(decoder_output_channels):
                # 计算跳跃连接的索引
                # 跳跃连接应该匹配到对应的解码器层输出通道数
                # skip_connections[i] 应该输出到 decoder_output_channels[i] 通道
                # 但是我们需要从 multi_scale_features 中选择合适的特征层
                # 通常是从后往前匹配：最后一个特征层对应第一个解码器层
                skip_idx = len(multi_scale_features) - 1 - i
                
                # 确保索引在有效范围内
                if skip_idx >= 0 and skip_idx < len(multi_scale_features) and skip_idx < len(self.skip_connections):
                    skip_feat = multi_scale_features[skip_idx]
                    # 上采样到当前尺寸
                    if skip_feat.shape[2:] != x.shape[2:]:
                        skip_feat = F.interpolate(
                            skip_feat, 
                            size=x.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # 使用第i个跳跃连接层（不是skip_idx），因为跳跃连接的输出通道数已经设置为匹配第i层
                    if i < len(self.skip_connections):
                        skip_feat = self.skip_connections[i](skip_feat)
                        
                        # 确保通道数匹配
                        if skip_feat.shape[1] != x.shape[1]:
                            # 如果通道数不匹配，使用零填充或截断
                            if skip_feat.shape[1] < x.shape[1]:
                                padding = torch.zeros(
                                    skip_feat.shape[0], 
                                    x.shape[1] - skip_feat.shape[1],
                                    skip_feat.shape[2],
                                    skip_feat.shape[3],
                                    device=skip_feat.device,
                                    dtype=skip_feat.dtype
                                )
                                skip_feat = torch.cat([skip_feat, padding], dim=1)
                            else:
                                skip_feat = skip_feat[:, :x.shape[1], :, :]
                        
                        x = x + skip_feat
        
        # 如果指定了目标尺寸，调整输出尺寸
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # 将输出从[-1, 1]归一化到[0, 1]
        x = (x + 1) / 2
        
        return x
