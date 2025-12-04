
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
**Lines to modify: 37-38** - Update training data directory and label file
```python
# Original (lines 37-38):
# self.data_dir = "default_train_data_path"
# self.label_file = "default_train_labels.csv"

# Modified Example:
self.data_dir = "/your/actual/path/to/datasets/kinface/train/images/"
self.label_file = "/your/actual/path/to/datasets/kinface/train/labels.csv"
```

### 3. **Testing Dataset Configuration** → `utl/dataset_test.py`
**Lines to modify: 40-41** - Update testing data directory and label file
```python
# Original (lines 40-41):
# self.data_dir = "default_test_data_path"
# self.label_file = "default_test_labels.csv"

# Modified Example:
self.data_dir = "/your/actual/path/to/datasets/kinface/test/images/"
self.label_file = "/your/actual/path/to/datasets/kinface/test/labels.csv"
```
