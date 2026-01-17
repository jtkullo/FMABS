"""
工具函数
"""
import torch
import random
import numpy as np
import os
from typing import Dict, Optional


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(state: Dict, filepath: str):
    """保存检查点"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, device: str = 'cuda') -> Dict:
    """加载检查点"""
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
    else:
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")


def denormalize_image(tensor: torch.Tensor, mean=(0.485, 0.456, 0.406), 
                     std=(0.229, 0.224, 0.225)) -> torch.Tensor:
    """反归一化图像"""
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean
