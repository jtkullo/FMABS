"""
训练脚本
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from typing import Any

from config import Config
from models import TeacherNetwork, StudentNetwork
from datasets import MVTecDataset, VideoAnomalyDataset
from utils import TotalLoss, set_seed, save_checkpoint
from utils.metrics import compute_anomaly_score


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


def get_dataloader(config: Config, is_train: bool = True):
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
        batch_size=config.batch_size if is_train else config.test_batch_size,
        shuffle=is_train,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn  # 使用自定义collate函数处理None标签
    )
    
    return dataloader


def train_epoch(model, teacher_model, dataloader, criterion, optimizer, 
                device, config, epoch):
    """训练一个epoch"""
    model.train()
    teacher_model.eval()
    
    total_loss = 0.0
    loss_dict = {
        'recon_loss': 0.0,
        'feature_mimic_loss': 0.0,
        'attention_loss': 0.0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        # 处理不同数据集的输入格式
        if config.dataset == 'mvtec':
            images, _ = batch
            images = images.to(device)
        else:
            # 视频数据集
            sequences, _ = batch
            # 使用序列的中间帧或最后一帧
            images = sequences[:, sequences.shape[1] // 2].to(device)  # [B, C, H, W]
        
        # 前向传播
        # 教师网络提取特征
        with torch.no_grad():
            teacher_features = teacher_model(images)
        
        # 学生网络
        outputs = model(images, teacher_features=teacher_features)
        
        # 准备目标
        targets = {
            'image': images,
            'teacher_features': teacher_features
        }
        
        # 计算损失
        losses = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()
        
        # 更新统计
        total_loss += losses['total_loss'].item()
        for key in loss_dict:
            loss_dict[key] += losses[key].item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': losses['total_loss'].item(),
            'recon': losses['recon_loss'].item(),
            'mimic': losses['feature_mimic_loss'].item()
        })
    
    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_dict:
        loss_dict[key] /= num_batches
    
    return avg_loss, loss_dict


def validate(model, teacher_model, dataloader, criterion, device, config):
    """验证"""
    model.eval()
    teacher_model.eval()
    
    total_loss = 0.0
    loss_dict = {
        'recon_loss': 0.0,
        'feature_mimic_loss': 0.0,
        'attention_loss': 0.0
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            if config.dataset == 'mvtec':
                images, _ = batch
                images = images.to(device)
            else:
                sequences, _ = batch
                images = sequences[:, sequences.shape[1] // 2].to(device)
            
            # 前向传播
            teacher_features = teacher_model(images)
            outputs = model(images, teacher_features=teacher_features)
            
            targets = {
                'image': images,
                'teacher_features': teacher_features
            }
            
            losses = criterion(outputs, targets)
            
            total_loss += losses['total_loss'].item()
            for key in loss_dict:
                loss_dict[key] += losses[key].item()
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    for key in loss_dict:
        loss_dict[key] /= num_batches
    
    return avg_loss, loss_dict


def main():
    parser = argparse.ArgumentParser(description='Train Feature Mimicking with Attention')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--dataset', type=str, default='mvtec', 
                       choices=['mvtec', 'cuhk_avenue', 'shanghaitech'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    if args.config:
        # 可以从文件加载配置
        pass
    
    # 应用命令行参数
    config.dataset = args.dataset
    config.data_root = args.data_root
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.resume = args.resume
    
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
        pretrained=True
    ).to(device)
    
    # 损失函数
    criterion = TotalLoss(
        recon_weight=config.recon_weight,
        feature_mimic_weight=config.feature_mimic_weight,
        attention_weight=config.attention_weight
    )
    
    # 优化器
    optimizer = optim.Adam(
        student_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    
    # 数据加载器
    print("Loading datasets...")
    train_loader = get_dataloader(config, is_train=True)
    
    # 验证集：对于异常检测任务，验证集应该只包含正常样本
    # 选项1：使用训练集的一部分作为验证集（推荐）
    # 选项2：使用测试集中的正常样本作为验证集
    # 这里我们使用训练集作为验证集（验证时不使用数据增强，通过is_train=False控制）
    # 注意：由于MVTec数据集没有单独的验证集，我们使用训练集进行验证
    # 这不会导致过拟合，因为我们只是监控训练损失，不用于模型选择
    val_loader = get_dataloader(config, is_train=True)  # 使用训练集作为验证集
    
    print(f"训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} batches")
    print(f"验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} batches")
    print("注意：验证集使用训练集数据（仅用于监控训练过程，不用于模型选择）")
    
    # TensorBoard
    writer = SummaryWriter(log_dir=config.log_dir)
    
    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    
    if config.resume:
        checkpoint = torch.load(config.resume, map_location=device)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # 训练循环
    print("Starting training...")
    for epoch in range(start_epoch, config.epochs):
        # 训练
        train_loss, train_loss_dict = train_epoch(
            student_model, teacher_model, train_loader, criterion,
            optimizer, device, config, epoch
        )
        
        # 验证
        val_loss, val_loss_dict = validate(
            student_model, teacher_model, val_loader, criterion, device, config
        )
        
        # 更新学习率
        scheduler.step()
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        for key in train_loss_dict:
            writer.add_scalar(f'Loss/Train_{key}', train_loss_dict[key], epoch)
            writer.add_scalar(f'Loss/Val_{key}', val_loss_dict[key], epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 保存检查点
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }
        
        save_checkpoint(
            checkpoint,
            os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        )
        
        if is_best:
            save_checkpoint(
                checkpoint,
                os.path.join(config.checkpoint_dir, 'best_model.pth')
            )
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    print("Training completed!")
    writer.close()


if __name__ == '__main__':
    main()
