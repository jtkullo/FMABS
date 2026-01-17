"""
视频异常检测数据集加载器（CUHK Avenue, ShanghaiTech）
"""
import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from torchvision import transforms
from typing import Tuple, Optional, List
from PIL import Image
try:
    import scipy.io as sio
except ImportError:
    sio = None


class VideoAnomalyDataset(Dataset):
    """
    视频异常检测数据集
    """
    def __init__(self, root: str, dataset_name: str = 'cuhk_avenue',
                 split: str = 'train', sequence_length: int = 16,
                 image_size: int = 256, temporal_stride: int = 1):
        """
        Args:
            root: 数据集根目录
            dataset_name: 'cuhk_avenue' 或 'shanghaitech'
            split: 'train' 或 'test'
            sequence_length: 视频序列长度
            image_size: 图像尺寸
            temporal_stride: 时间步长
        """
        self.root = root
        self.dataset_name = dataset_name
        self.split = split
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.temporal_stride = temporal_stride
        self.is_train = (split == 'train')
        
        # 构建数据路径
        # 注意：CUHK Avenue和ShanghaiTech的实际目录结构可能不同
        # 这里使用root作为基础路径
        self.data_dir = os.path.join(root, dataset_name, split) if split in ['train', 'test'] else os.path.join(root, dataset_name)
        
        # 加载视频序列
        self.sequences = self._load_sequences()
        
        # 数据增强
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
    
    def _load_sequences(self) -> List[dict]:
        """加载视频序列信息"""
        sequences = []
        
        if self.dataset_name == 'cuhk_avenue':
            sequences = self._load_cuhk_avenue()
        elif self.dataset_name == 'shanghaitech':
            sequences = self._load_shanghaitech()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        return sequences
    
    def _load_cuhk_avenue(self) -> List[dict]:
        """加载CUHK Avenue数据集"""
        sequences = []
        
        # CUHK Avenue数据集结构：training_videos/ 和 testing_videos/ 包含.avi文件
        if self.is_train:
            video_dir = os.path.join(self.root, 'cuhk_avenue', 'training_videos')
        else:
            video_dir = os.path.join(self.root, 'cuhk_avenue', 'testing_videos')
        
        if os.path.exists(video_dir):
            # 检查是否有已提取的帧目录
            frames_dir = os.path.join(self.data_dir, 'frames')
            if os.path.exists(frames_dir):
                # 使用已提取的帧
                videos = sorted([f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))])
                for video in videos:
                    video_path = os.path.join(frames_dir, video)
                    frames = sorted([f for f in os.listdir(video_path) 
                                   if f.endswith(('.jpg', '.png'))])
                    
                    if len(frames) >= self.sequence_length:
                        # 加载标签（如果存在）
                        label_path = os.path.join(self.data_dir, 'annotations', f'{video}.txt')
                        labels = None
                        if os.path.exists(label_path) and not self.is_train:
                            labels = self._load_labels(label_path, len(frames))
                        
                        sequences.append({
                            'video_path': video_path,
                            'frames': frames,
                            'labels': labels,
                            'video_file': None  # 使用帧而不是视频文件
                        })
            else:
                # 使用视频文件，需要从视频中提取帧
                video_files = sorted([f for f in os.listdir(video_dir) 
                                     if f.endswith(('.avi', '.mp4', '.mov'))])
                for video_file in video_files:
                    video_path = os.path.join(video_dir, video_file)
                    # 获取视频信息
                    cap = cv2.VideoCapture(video_path)
                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    if num_frames >= self.sequence_length:
                        # 加载标签（如果存在）
                        video_name = os.path.splitext(video_file)[0]
                        # CUHK Avenue标签在testing_vol中
                        if not self.is_train:
                            vol_path = os.path.join(self.root, 'cuhk_avenue', 'testing_vol', f'vol{video_name}.mat')
                            labels = self._load_cuhk_labels_from_mat(vol_path, num_frames)
                        else:
                            labels = None
                        
                        sequences.append({
                            'video_path': video_path,
                            'frames': None,  # 需要从视频提取
                            'labels': labels,
                            'video_file': video_file
                        })
        
        return sequences
    
    def _load_shanghaitech(self) -> List[dict]:
        """加载ShanghaiTech数据集"""
        sequences = []
        
        # ShanghaiTech数据集结构：part_A_final 和 part_B_final
        # 每个部分包含 train_data/images/ 和 test_data/images/
        parts = ['part_A_final', 'part_B_final']
        
        for part in parts:
            part_dir = os.path.join(self.root, 'shanghaitech', part)
            if not os.path.exists(part_dir):
                continue
            
            # 确定使用训练还是测试数据
            if self.is_train:
                data_type = 'train_data'
            else:
                data_type = 'test_data'
            
            images_dir = os.path.join(part_dir, data_type, 'images')
            ground_truth_dir = os.path.join(part_dir, data_type, 'ground_truth')
            
            if os.path.exists(images_dir):
                # 获取所有图像文件
                image_files = sorted([f for f in os.listdir(images_dir) 
                                    if f.endswith(('.jpg', '.png', '.jpeg'))])
                
                # 将图像组织成序列（每个图像作为一个序列，或者按视频分组）
                # ShanghaiTech数据集通常每个图像是一个独立的帧
                # 我们可以将连续的图像组成序列
                if len(image_files) >= self.sequence_length:
                    # 方法1：将图像按顺序分组为序列
                    num_sequences = len(image_files) // self.sequence_length
                    for i in range(num_sequences):
                        start_idx = i * self.sequence_length
                        end_idx = start_idx + self.sequence_length
                        sequence_frames = image_files[start_idx:end_idx]
                        
                        # 加载标签（如果存在）
                        labels = None
                        if not self.is_train and os.path.exists(ground_truth_dir):
                            # 尝试加载对应的标签文件
                            # ShanghaiTech的标签通常在ground_truth目录中，文件名与图像对应
                            sequence_labels = []
                            for frame_file in sequence_frames:
                                label_file = os.path.splitext(frame_file)[0] + '.mat'
                                label_path = os.path.join(ground_truth_dir, label_file)
                                if os.path.exists(label_path):
                                    label_data = self._load_shanghaitech_label(label_path)
                                    sequence_labels.append(label_data)
                            
                            if len(sequence_labels) > 0:
                                # 将标签合并（如果有多个帧的标签）
                                labels = np.array(sequence_labels)
                        
                        sequences.append({
                            'video_path': images_dir,
                            'frames': sequence_frames,
                            'labels': labels,
                            'part': part,
                            'video_file': None
                        })
                    
                    # 如果还有剩余图像，创建一个序列
                    remaining = len(image_files) % self.sequence_length
                    if remaining > 0:
                        start_idx = num_sequences * self.sequence_length
                        sequence_frames = image_files[start_idx:]
                        # 填充到sequence_length
                        while len(sequence_frames) < self.sequence_length:
                            sequence_frames.append(sequence_frames[-1])
                        
                        labels = None
                        if not self.is_train and os.path.exists(ground_truth_dir):
                            sequence_labels = []
                            for frame_file in sequence_frames[:remaining]:
                                label_file = os.path.splitext(frame_file)[0] + '.mat'
                                label_path = os.path.join(ground_truth_dir, label_file)
                                if os.path.exists(label_path):
                                    label_data = self._load_shanghaitech_label(label_path)
                                    sequence_labels.append(label_data)
                            
                            if len(sequence_labels) > 0:
                                labels = np.array(sequence_labels)
                        
                        sequences.append({
                            'video_path': images_dir,
                            'frames': sequence_frames,
                            'labels': labels,
                            'part': part,
                            'video_file': None
                        })
        
        return sequences
    
    def _load_shanghaitech_label(self, label_path: str) -> Optional[np.ndarray]:
        """加载ShanghaiTech标签文件"""
        if sio is None or not os.path.exists(label_path):
            return None
        
        try:
            mat_data = sio.loadmat(label_path)
            # ShanghaiTech的标签格式：通常包含'volLabel'或类似字段
            # 这里需要根据实际格式调整
            for key in ['volLabel', 'label', 'mask', 'gt']:
                if key in mat_data:
                    label = mat_data[key]
                    if isinstance(label, np.ndarray):
                        # 如果是2D数组，转换为1D（帧级标签）
                        if label.ndim == 2:
                            # 如果标签是像素级的，返回是否有异常（最大值）
                            return np.array([label.max() > 0], dtype=np.int32)
                        elif label.ndim == 1:
                            return label.astype(np.int32)
                        else:
                            return np.array([label.max() > 0], dtype=np.int32)
        except Exception as e:
            print(f"Warning: Could not load label from {label_path}: {e}")
        
        return None
    
    def _load_labels(self, label_path: str, num_frames: int) -> np.ndarray:
        """加载标签文件"""
        if self.dataset_name == 'cuhk_avenue':
            # CUHK Avenue使用文本格式的标签
            labels = np.zeros(num_frames, dtype=np.int32)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                start, end = int(parts[0]), int(parts[1])
                                labels[start:end+1] = 1
            return labels
        else:
            return None
    
    def _load_cuhk_labels_from_mat(self, mat_path: str, num_frames: int) -> Optional[np.ndarray]:
        """从.mat文件加载CUHK Avenue标签"""
        if sio is None or not os.path.exists(mat_path):
            return None
        
        try:
            mat_data = sio.loadmat(mat_path)
            # CUHK Avenue的标签格式可能不同，需要根据实际格式调整
            # 这里假设标签在某个字段中
            if 'volLabel' in mat_data:
                labels = mat_data['volLabel'].flatten()
                # 确保长度匹配
                if len(labels) > num_frames:
                    labels = labels[:num_frames]
                elif len(labels) < num_frames:
                    labels = np.pad(labels, (0, num_frames - len(labels)), 'constant')
                return labels.astype(np.int32)
        except Exception as e:
            print(f"Warning: Could not load labels from {mat_path}: {e}")
        
        return None
    
    def __len__(self) -> int:
        total = 0
        for seq in self.sequences:
            # 对于ShanghaiTech，每个序列已经是一个完整的sequence_length长度的序列
            if self.dataset_name == 'shanghaitech':
                total += 1
            else:
                # 对于其他数据集，可能需要滑动窗口
                num_frames = len(seq['frames']) if seq['frames'] is not None else 1000  # 默认值
                num_clips = max(1, (num_frames - self.sequence_length) // self.temporal_stride + 1)
                total += num_clips
        return total
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        获取数据项
        
        Returns:
            sequence: 视频序列 [T, C, H, W]
            label: 标签（训练集为None，测试集为异常帧标签）
        """
        # 找到对应的序列
        current_idx = 0
        for seq in self.sequences:
            # 对于ShanghaiTech，每个序列已经是一个完整的sequence_length长度的序列
            if self.dataset_name == 'shanghaitech':
                if current_idx == idx:
                    return self._load_sequence(seq, 0)
                current_idx += 1
            else:
                # 对于其他数据集，可能需要滑动窗口
                num_frames = len(seq['frames']) if seq['frames'] is not None else 1000  # 默认值
                num_clips = max(1, (num_frames - self.sequence_length) // self.temporal_stride + 1)
                if current_idx + num_clips > idx:
                    clip_idx = idx - current_idx
                    start_frame = clip_idx * self.temporal_stride
                    return self._load_sequence(seq, start_frame)
                current_idx += num_clips
        
        # 如果索引超出范围，返回最后一个序列
        if len(self.sequences) > 0:
            seq = self.sequences[-1]
            return self._load_sequence(seq, 0)
        else:
            # 如果没有序列，返回一个空序列
            dummy_sequence = torch.zeros(self.sequence_length, 3, self.image_size, self.image_size)
            return dummy_sequence, None
    
    def _load_sequence(self, seq: dict, start_frame: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """加载视频序列"""
        frames = []
        frame_indices = range(start_frame, start_frame + self.sequence_length)
        
        # 如果使用视频文件，需要从视频中提取帧
        if seq.get('video_file') is not None:
            cap = cv2.VideoCapture(seq['video_path'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for idx in frame_indices:
                ret, frame_bgr = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame_rgb)
                    frame = self.transform(frame)
                    frames.append(frame)
                else:
                    # 如果读取失败，使用最后一帧
                    if len(frames) > 0:
                        frames.append(frames[-1])
                    break
            
            cap.release()
        else:
            # 使用已提取的帧
            for idx in frame_indices:
                if seq['frames'] is not None and idx < len(seq['frames']):
                    frame_path = os.path.join(seq['video_path'], seq['frames'][idx])
                    frame = Image.open(frame_path).convert('RGB')
                    frame = self.transform(frame)
                    frames.append(frame)
        
        # 如果帧数不足，重复最后一帧
        while len(frames) < self.sequence_length:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                # 如果完全没有帧，创建一个黑色图像
                dummy_frame = torch.zeros(3, self.image_size, self.image_size)
                frames.append(dummy_frame)
        
        sequence = torch.stack(frames)  # [T, C, H, W]
        
        # 加载标签
        label = None
        if seq['labels'] is not None and not self.is_train:
            label_indices = list(frame_indices)
            if len(seq['labels']) > 0:
                valid_indices = [i for i in label_indices if i < len(seq['labels'])]
                if valid_indices:
                    label = torch.tensor(seq['labels'][valid_indices], dtype=torch.float32)
        
        return sequence, label
