"""
MVTec AD数据集加载器
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import Tuple, Optional


class MVTecDataset(Dataset):
    """
    MVTec AD数据集 - 工业异常检测数据集
    """
    def __init__(self, root: str, category: str = 'bottle', 
                 split: str = 'train', image_size: int = 256,
                 is_train: bool = True):
        """
        Args:
            root: 数据集根目录
            category: 类别名称（如 'bottle', 'cable', 'capsule'等）
            split: 'train' 或 'test'
            image_size: 图像尺寸
            is_train: 是否为训练集（训练集只包含正常样本）
        """
        self.root = root
        self.category = category
        self.split = split
        self.image_size = image_size
        self.is_train = is_train
        
        # 构建路径 - MVTec数据集在root下有mvtec子目录
        mvtec_root = os.path.join(root, 'mvtec')
        if is_train:
            self.data_dir = os.path.join(mvtec_root, category, 'train', 'good')
            self.labels = None  # 训练集没有标签
        else:
            self.data_dir = os.path.join(mvtec_root, category, 'test')
            # 测试集包含正常和异常样本
            self.labels = []
            self.image_paths = []
            
            # 检查目录是否存在
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(
                    f"测试数据目录不存在: {self.data_dir}\n"
                    f"请确保MVTec数据集已正确下载并解压到 {mvtec_root} 目录下"
                )
            
            # 遍历测试目录
            for subdir in os.listdir(self.data_dir):
                subdir_path = os.path.join(self.data_dir, subdir)
                if os.path.isdir(subdir_path):
                    label = 0 if subdir == 'good' else 1  # 0=正常, 1=异常
                    for img_name in os.listdir(subdir_path):
                        if img_name.endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(os.path.join(subdir_path, img_name))
                            self.labels.append(label)
        
        # 如果是训练集，获取所有图像路径
        if is_train:
            # 检查目录是否存在
            if not os.path.exists(self.data_dir):
                raise FileNotFoundError(
                    f"训练数据目录不存在: {self.data_dir}\n"
                    f"请确保MVTec数据集已正确下载并解压到 {mvtec_root} 目录下\n"
                    f"预期路径结构: {mvtec_root}/{category}/train/good/"
                )
            
            self.image_paths = [
                os.path.join(self.data_dir, f) 
                for f in os.listdir(self.data_dir) 
                if f.endswith(('.png', '.jpg', '.jpeg'))
            ]
            
            if len(self.image_paths) == 0:
                raise ValueError(
                    f"在 {self.data_dir} 目录下未找到任何图像文件\n"
                    f"请检查数据集是否正确下载和解压"
                )
        
        # 数据增强
        # 注意：对于异常检测，训练集和验证集都只包含正常样本
        # 区别在于训练时使用数据增强，验证时不使用
        if is_train:
            # 训练时使用数据增强
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        else:
            # 测试/验证时不使用数据增强
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[int]]:
        """
        获取数据项
        
        Returns:
            image: 图像张量 [C, H, W]
            label: 标签（训练集为None，测试集为0或1）
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        label = self.labels[idx] if self.labels is not None else None
        
        return image, label
