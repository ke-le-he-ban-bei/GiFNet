# GIFNet: Gender-Independent Kinship Verification Network via Fuzzy Disentangling and Multi-metric Inference

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)

Official PyTorch implementation of **"Gender-independent Kinship Verification Network Via Fuzzy Disentangling and Multi-metric Inference"**.

## 📖 Abstract
This repository contains the implementation of GIFNet, a novel kinship verification framework that achieves gender-independent performance through fuzzy feature disentanglement and multi-metric inference mechanisms. The model effectively separates gender-related features from kinship features, enabling robust verification across all gender combinations.

## ✨ Features
- **Gender-Independent Architecture**: Robust kinship verification across all gender combinations
- **Fuzzy Disentangling Module**: Separates gender-related features from kinship features using fuzzy logic
- **Multi-Metric Inference**: Combines multiple similarity metrics for enhanced accuracy
- **End-to-End Pipeline**: Complete training, validation, and inference workflow
- **Swin Transformer Backbone**: Leverages hierarchical vision transformer architecture
- **Contrastive Learning**: Uses contrastive loss for better feature discriminability

## 🚀 Quick Start

### Prerequisites
See [`requirements.txt`](requirements.txt) for the complete dependency list.

### Installation
```bash
# Clone repository
git clone https://github.com/ke-le-he-ban-bei/GiFNet.git
cd GIFNet

# Install dependencies
pip install -r requirements.txt

# For GPU support (CUDA 11.3 example)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

## ⚙️ Configuration

Before running the model, **you must update the following configuration paths** in three different files:

### 1. **Main Training Script** → `train_infer.py`
**Line to modify: 195** - Update the sample list file path
```python
# Original (line 195):
# sample_list_path = "your/default/path/here"

# Modified Example:
sample_list_path = "/your/actual/path/to/datasets/kinface/sample_list.txt"
```

### 2. **Training Dataset Configuration** → `utl/dataset_train.py`
**Lines to modify:67-68-** - Update training data directory and label file
```python
# Original (lines 68-69):
parent_path = line[2].replace('D:/xx/xx/kinshipdatabase/', 'D:/xx/')
child_path = line[3].replace('\n', '').replace('D:/xxx/xx/kinshipdatabase/', 'D:/xx/')
```

### 3. **Testing Dataset Configuration** → `utl/dataset_test.py`
**Lines to modify: 40-41** - Update testing data directory and label file
```python
# Original (lines 40-41):
  parent_path = line[2].replace('./xx/data/kinshipdatabase/', './xx/')
child_path = line[3].replace('\n', '').replace('./xx/data/kinshipdatabase/', './xx/')
```
🔧 Pretraining Options
Masked Pretraining
For masked image modeling pretraining, you can refer to the SimMIM implementation:
https://github.com/microsoft/SimMIM
