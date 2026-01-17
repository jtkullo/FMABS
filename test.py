"""
测试脚本
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Any

from config import Config
from models import TeacherNetwork, StudentNetwork
from datasets import MVTecDataset, VideoAnomalyDataset
from utils import load_checkpoint, set_seed
from utils.metrics import compute_anomaly_score, compute_auroc, compute_ap


def collate_fn(batch):
    """
    自定义collate函数，处理None标签和视频序列
    """
    items = []
    labels = []
    
    for item in batch:
        if len(item) == 2:
            data, label = item
            items.append(data)
            labels.append(label)
        else:
            items.append(item[0])
            labels.append(None)
    
    # 检查第一个元素是图像还是序列
    first_item = items[0]
    if first_item.dim() == 3:  # [C, H, W] - 单张图像
        # 堆叠图像
        images = torch.stack(items)
        return images, labels
    elif first_item.dim() == 4:  # [T, C, H, W] - 视频序列
        # 堆叠序列
        sequences = torch.stack(items)
        return sequences, labels
    else:
        # 其他情况，直接堆叠
        try:
            stacked = torch.stack(items)
            return stacked, labels
        except:
            # 如果无法堆叠，返回列表
            return items, labels


def get_dataloader(config: Config, is_train: bool = False):
    """获取数据加载器"""
    if config.dataset == 'mvtec':
        dataset = MVTecDataset(
            root=config.data_root,
            category='bottle',  # 可以根据需要修改
            split='train' if is_train else 'test',
            image_size=config.image_size,
            is_train=is_train
        )
    elif config.dataset in ['cuhk_avenue', 'shanghaitech']:
        dataset = VideoAnomalyDataset(
            root=config.data_root,
            dataset_name=config.dataset,
            split='train' if is_train else 'test',
            sequence_length=config.sequence_length,
            image_size=config.image_size,
            temporal_stride=config.temporal_stride
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn  # 使用自定义collate函数处理None标签
    )
    
    return dataloader


def compute_threshold(train_loader, model, teacher_model, device, config, percentile=95.0):
    """在训练集上计算阈值"""
    print("Computing threshold on training set...")
    model.eval()
    teacher_model.eval()
    
    all_scores = []
    
    with torch.no_grad():
        for batch in tqdm(train_loader):
            if config.dataset == 'mvtec':
                images, _ = batch
                images = images.to(device)
            else:
                sequences, _ = batch
                images = sequences[:, sequences.shape[1] // 2].to(device)
            
            # 前向传播
            teacher_features = teacher_model(images)
            outputs = model(images, teacher_features=teacher_features)
            
            # 计算重建误差
            recon_error = torch.abs(outputs['reconstructed'] - images)
            
            # 计算异常分数
            inconsistency = None
            if 'inconsistency_matrices' in outputs and len(outputs['inconsistency_matrices']) > 0:
                inconsistency = outputs['inconsistency_matrices'][0]
            
            attention = None
            if 'attention_weights' in outputs:
                attention = outputs['attention_weights']
                if isinstance(attention, list):
                    attention = attention[0]
            
            anomaly_scores = compute_anomaly_score(
                recon_error, inconsistency, attention
            )
            
            # 使用最大分数作为图像级分数
            image_scores = anomaly_scores.view(anomaly_scores.shape[0], -1).max(dim=1)[0]
            all_scores.extend(image_scores.cpu().numpy())
    
    threshold = np.percentile(all_scores, percentile)
    print(f"Threshold (percentile {percentile}): {threshold:.4f}")
    return threshold


def test(model, teacher_model, test_loader, device, config, threshold=None):
    """测试模型"""
    model.eval()
    teacher_model.eval()
    
    all_scores = []
    all_labels = []
    all_pixel_scores = []
    all_pixel_labels = []
    
    print("Testing...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            if config.dataset == 'mvtec':
                images, labels = batch
                images = images.to(device)
                labels = labels.numpy() if labels is not None else None
            else:
                sequences, labels = batch
                images = sequences[:, sequences.shape[1] // 2].to(device)
                if labels is not None:
                    labels = labels[:, labels.shape[1] // 2].numpy()  # 使用中间帧的标签
            
            # 前向传播
            teacher_features = teacher_model(images)
            outputs = model(images, teacher_features=teacher_features)
            
            # 计算重建误差
            recon_error = torch.abs(outputs['reconstructed'] - images)
            
            # 计算异常分数
            inconsistency = None
            if 'inconsistency_matrices' in outputs and len(outputs['inconsistency_matrices']) > 0:
                inconsistency = outputs['inconsistency_matrices'][0]
            
            attention = None
            if 'attention_weights' in outputs:
                attention = outputs['attention_weights']
                if isinstance(attention, list):
                    attention = attention[0]
            
            anomaly_scores = compute_anomaly_score(
                recon_error, inconsistency, attention
            )
            
            # 图像级分数（使用最大值）
            image_scores = anomaly_scores.view(anomaly_scores.shape[0], -1).max(dim=1)[0]
            all_scores.extend(image_scores.cpu().numpy())
            
            # 像素级分数（用于定位）
            pixel_scores = anomaly_scores.cpu().numpy()
            all_pixel_scores.append(pixel_scores)
            
            # 标签
            if labels is not None:
                all_labels.extend(labels)
                # 对于像素级标签，需要ground truth mask（这里简化处理）
                # 实际应用中需要加载真实的像素级标签
    
    # 转换为numpy数组
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels) if len(all_labels) > 0 else None
    
    # 计算指标
    results = {}
    
    if all_labels is not None and len(np.unique(all_labels)) > 1:
        # 图像级AUROC
        auroc = compute_auroc(all_scores, all_labels)
        results['Image-level AUROC'] = auroc
        
        # AP
        ap = compute_ap(all_scores, all_labels)
        results['Image-level AP'] = ap
        
        print(f"Image-level AUROC: {auroc:.4f}")
        print(f"Image-level AP: {ap:.4f}")
        
        # 使用阈值进行分类
        if threshold is not None:
            predictions = (all_scores > threshold).astype(int)
            accuracy = (predictions == all_labels).mean()
            results['Accuracy (with threshold)'] = accuracy
            print(f"Accuracy (threshold={threshold:.4f}): {accuracy:.4f}")
    
    return results, all_scores, all_labels


def main():
    parser = argparse.ArgumentParser(description='Test Feature Mimicking with Attention')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--dataset', type=str, default='mvtec',
                       choices=['mvtec', 'cuhk_avenue', 'shanghaitech'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--threshold_percentile', type=float, default=95.0)
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    config.dataset = args.dataset
    config.data_root = args.data_root
    config.threshold_percentile = args.threshold_percentile
    
    # 从检查点加载配置
    checkpoint = load_checkpoint(args.checkpoint)
    if 'config' in checkpoint:
        checkpoint_config = checkpoint['config']
        # 合并配置
        for key in ['dataset', 'backbone', 'feature_layers', 'embedding_dim',
                   'use_attention', 'attention_dim']:
            if hasattr(checkpoint_config, key):
                setattr(config, key, getattr(checkpoint_config, key))
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型
    print("Creating models...")
    teacher_model = TeacherNetwork(
        backbone_name=config.backbone,
        feature_layers=config.feature_layers,
        pretrained=True
    ).to(device)
    
    student_model = StudentNetwork(
        backbone_name=config.backbone,
        feature_layers=config.feature_layers,
        embedding_dim=config.embedding_dim,
        use_attention=config.use_attention,
        attention_dim=config.attention_dim,
        pretrained=False
    ).to(device)
    
    # 加载权重
    student_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # 数据加载器
    print("Loading datasets...")
    train_loader = get_dataloader(config, is_train=True)
    test_loader = get_dataloader(config, is_train=False)
    
    # 计算阈值
    threshold = compute_threshold(
        train_loader, student_model, teacher_model, device, config,
        percentile=config.threshold_percentile
    )
    
    # 测试
    results, scores, labels = test(
        student_model, teacher_model, test_loader, device, config, threshold
    )
    
    # 保存结果
    result_file = os.path.join(config.result_dir, 'test_results.txt')
    with open(result_file, 'w') as f:
        f.write("Test Results\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")
    
    print(f"\nResults saved to {result_file}")
    print("Testing completed!")


if __name__ == '__main__':
    main()
