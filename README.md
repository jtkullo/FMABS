# Feature Mimicking with Attention: A Boosting Strategy for Visual Anomaly Detection

Complete implementation of the paper "Feature Mimicking with Attention: A Boosting Strategy for Visual Anomaly Detection".

## Project Overview

This project implements a visual anomaly detection method based on feature mimicking and attention mechanism. The method improves anomaly detection performance through a teacher-student network architecture, combining feature mimicking tasks and feature-inconsistency-based attention mechanism.

### Key Features

- **Feature Mimicking Task**: The student network learns to mimic the feature representations of the teacher network
- **Attention Mechanism**: Feature-inconsistency-based attention module that guides the model to focus on anomalous regions
- **Multi-Dataset Support**: Supports MVTec AD (image), CUHK Avenue, and ShanghaiTech (video) datasets
- **End-to-End Training**: Complete training and testing pipeline

## Bset Checkpoints
You can find our best checkpoints in:

- **MVTec AD Dataset**: https://drive.google.com/file/d/1OEjqbglKq_zFEeMAB0lIhTyUxfQpOIng/view?usp=drive_link
- **CUHK Avenue Dataset**: https://drive.google.com/file/d/1c-POB1FLiC_My0YFIYyDZ7haYc3Q0ly8/view?usp=drive_link
- **ShanghaiTech Dataset**: https://drive.google.com/file/d/1qF3Q7d1yW8lYbeVduUf4k-d7eq9_TIX8/view?usp=drive_link

## Project Structure

```
FMABS/
├── config.py              # Configuration file
├── train.py               # Training script
├── test.py                # Testing script
├── requirements.txt       # Dependencies
├── models/                # Model definitions
│   ├── __init__.py
│   ├── backbone.py        # Backbone networks
│   ├── teacher_student.py # Teacher-student networks
│   ├── attention.py       # Attention modules
│   └── decoder.py         # Reconstruction decoder
├── datasets/              # Dataset loaders
│   ├── __init__.py
│   ├── mvtec.py           # MVTec AD dataset
│   └── video_dataset.py   # Video datasets
└── utils/                 # Utility functions
    ├── __init__.py
    ├── losses.py          # Loss functions
    ├── metrics.py         # Evaluation metrics
    └── utils.py           # Utility functions
```

## Installation

### 1. Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (recommended for GPU acceleration)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Data Preparation

### MVTec AD Dataset

1. Download MVTec AD dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Extract to `./data/mvtec/` directory
3. Directory structure should be:
```
data/
└── mvtec/
    ├── bottle/
    │   ├── train/
    │   │   └── good/          # Training set (normal samples only)
    │   ├── test/
    │   │   ├── good/          # Test set normal samples
    │   │   ├── broken_large/  # Anomaly type 1
    │   │   ├── broken_small/  # Anomaly type 2
    │   │   └── contamination/ # Anomaly type 3
    │   └── ground_truth/      # Pixel-level labels (for evaluation)
    │       ├── broken_large/
    │       ├── broken_small/
    │       └── contamination/
    ├── cable/
    ├── capsule/
    └── ... (15 categories in total)
```

**Note**: MVTec AD dataset contains 15 categories: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

### CUHK Avenue Dataset

1. Download CUHK Avenue dataset
2. Extract to `./data/cuhk_avenue/` directory
3. Directory structure should be:
```
data/
└── cuhk_avenue/
    ├── training_videos/       # Training videos (16 .avi files)
    │   ├── 01.avi
    │   ├── 02.avi
    │   └── ...
    ├── testing_videos/        # Testing videos (21 .avi files)
    │   ├── 01.avi
    │   ├── 02.avi
    │   └── ...
    ├── training_vol/          # Training labels (16 .mat files)
    │   ├── vol01.mat
    │   └── ...
    └── testing_vol/           # Testing labels (21 .mat files)
        ├── vol01.mat
        └── ...
```

**Note**: 
- The code supports reading frames directly from .avi video files, and also supports using pre-extracted frames (if `frames/` directory exists)
- Label files are in .mat format, containing annotations for anomalous frames

### ShanghaiTech Dataset

1. Download ShanghaiTech dataset
2. Extract to `./data/shanghaitech/` directory
3. Directory structure should be:
```
data/
└── shanghaitech/
    ├── part_A_final/
    │   ├── train_data/
    │   │   ├── images/        # Training images (300 .jpg files)
    │   │   └── ground_truth/  # Training labels (300 .mat files)
    │   └── test_data/
    │       ├── images/        # Test images (182 .jpg files)
    │       └── ground_truth/  # Test labels (182 .mat files)
    └── part_B_final/
        ├── train_data/
        │   ├── images/        # Training images (400 .jpg files)
        │   └── ground_truth/  # Training labels (400 .mat files)
        └── test_data/
            ├── images/        # Test images (316 .jpg files)
            └── ground_truth/  # Test labels (316 .mat files)
```

**Note**: 
- ShanghaiTech dataset is divided into part_A and part_B
- The code automatically processes data from both parts
- Label files are in .mat format, containing pixel-level annotations for anomalous regions

## Usage

### Training

#### MVTec AD Dataset

```bash
python train.py --dataset mvtec --data_root ./data --epochs 100 --batch_size 8 --lr 1e-4
```

#### CUHK Avenue Dataset

```bash
python train.py --dataset cuhk_avenue --data_root ./data --epochs 100 --batch_size 4 --lr 1e-4
```

#### ShanghaiTech Dataset

```bash
python train.py --dataset shanghaitech --data_root ./data --epochs 100 --batch_size 4 --lr 1e-4
```

### Testing

```bash
python test.py \
    --checkpoint ./checkpoints/best_model.pth \
    --dataset mvtec \
    --data_root ./data \
    --threshold_percentile 95.0
```

### Resume Training

```bash
python train.py \
    --dataset mvtec \
    --data_root ./data \
    --resume ./checkpoints/checkpoint_epoch_50.pth
```

## Configuration

Main configuration parameters are defined in `config.py`:

- `dataset`: Dataset name ('mvtec', 'cuhk_avenue', 'shanghaitech')
- `backbone`: Backbone network ('wide_resnet50_2', 'resnet18', 'resnet50')
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `epochs`: Number of training epochs
- `recon_weight`: Reconstruction loss weight
- `feature_mimic_weight`: Feature mimicking loss weight
- `attention_weight`: Attention loss weight

## Model Architecture

### Teacher Network
- Uses pre-trained backbone network (e.g., Wide ResNet-50)
- Parameters are frozen, used only for feature extraction

### Student Network
- Same backbone structure as teacher network
- Adds reconstruction decoder
- Integrates attention module

### Loss Functions
- **Reconstruction Loss**: L1 + L2 loss, measures reconstruction quality
- **Feature Mimicking Loss**: MSE loss, makes student network mimic teacher network features
- **Attention Loss**: Guides attention to focus on anomalous regions

## Evaluation Metrics

- **AUROC**: Area Under ROC Curve (image-level)
- **AP**: Average Precision (image-level)
- **Pixel-level AUROC**: For anomaly localization

## Experimental Results

Results reported in the paper:
- **MVTec AD**: Image-level AUROC > 95%
- **CUHK Avenue**: Video-level AUROC > 85%
- **ShanghaiTech**: Video-level AUROC > 80%

### Dataset Statistics

- **MVTec AD**: 
  - 15 categories (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper)
  - Approximately 200-400 training samples per category (normal samples only)
  - Approximately 50-150 test samples per category (includes normal and multiple anomaly types)
  - Provides pixel-level ground truth for anomaly localization evaluation

- **CUHK Avenue**: 
  - 16 training videos (.avi format)
  - 21 testing videos (.avi format)
  - Label files in .mat format, containing anomalous frame annotations

- **ShanghaiTech**: 
  - Part A: 300 training images + 182 test images
  - Part B: 400 training images + 316 test images
  - Each image has a corresponding .mat format ground truth file

## Notes

1. **GPU Memory**: If you encounter GPU memory issues, reduce `batch_size` or `image_size`
2. **Data Paths**: Ensure data paths are correct and dataset directory structure meets requirements
3. **Pre-trained Models**: Pre-trained ImageNet weights will be automatically downloaded on first run
4. **Training Time**: Complete training may take hours to days depending on dataset size and hardware

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{zheng2024boosting,
  title={A Boosting Strategy Based on Feature Mimicking with Attention for Visual Anomaly Detection},
  author={Zheng, Boyuan and Gan, Yi and Wang, Lianggang and Cong, Xunchao and Hu, Chao},
  journal={...},
  year={2024}
}
```

## License

This project is for academic research use only.

## Contact

For questions or suggestions, please submit an Issue.
