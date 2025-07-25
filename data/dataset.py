"""
数据集和数据加载模块
"""
import os
import random
import numpy as np
from collections import Counter
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class OptimizedGardenDataset(Dataset):
    """优化的园林数据集类
    
    支持预训练和微调两种模式
    """
    def __init__(self, image_paths, labels=None, transform=None, is_pretraining=False):
        """
        Args:
            image_paths: 图像路径列表
            labels: 标签列表（预训练时为None）
            transform: 数据变换
            is_pretraining: 是否是预训练模式
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_pretraining = is_pretraining
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            # 加载图像
            image = Image.open(self.image_paths[idx]).convert('RGB')
            
            # 应用数据变换
            if self.transform:
                image = self.transform(image)
            
            # 根据模式返回不同的数据
            if self.is_pretraining:
                return image
            else:
                label = self.labels[idx]
                return image, label
                
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # 返回一个默认图像
            default_image = Image.new('RGB', (224, 224), (255, 255, 255))
            if self.transform:
                default_image = self.transform(default_image)
            
            if self.is_pretraining:
                return default_image
            else:
                return default_image, self.labels[idx] if self.labels else 0


def load_dataset(data_path):
    """加载数据集
    
    Args:
        data_path: 数据集根目录路径
        
    Returns:
        image_paths: 图像路径列表
        labels: 标签列表
        class_names: 类别名称列表
    """
    image_paths = []
    labels = []
    class_names = []
    
    print("📂 正在加载数据集...")
    
    # 支持的图像格式
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # 遍历数据目录
    for class_folder in sorted(os.listdir(data_path)):
        class_path = os.path.join(data_path, class_folder)
        if os.path.isdir(class_path):
            class_names.append(class_folder)
            
            # 遍历类别文件夹中的所有图像
            for root, dirs, files in os.walk(class_path):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        image_path = os.path.join(root, file)
                        image_paths.append(image_path)
                        labels.append(class_folder)
    
    print(f"✅ 找到 {len(class_names)} 个类别")
    print(f"✅ 总共 {len(image_paths)} 张图像")
    
    # 打印类别分布
    label_counts = Counter(labels)
    print("\n📊 类别分布:")
    for class_name, count in sorted(label_counts.items()):
        print(f"  {class_name}: {count} 张图像")
    
    return image_paths, labels, class_names


def split_dataset(image_paths, labels, test_size=0.2, val_size=0.1, random_state=42):
    """分割数据集为训练集、验证集和测试集
    
    Args:
        image_paths: 图像路径列表
        labels: 标签列表
        test_size: 测试集比例
        val_size: 验证集比例（相对于总数据）
        random_state: 随机种子
        
    Returns:
        (X_train, y_train): 训练集
        (X_val, y_val): 验证集
        (X_test, y_test): 测试集
        label_encoder: 标签编码器
    """
    # 标签编码
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # 先分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, encoded_labels, 
        test_size=test_size, 
        stratify=encoded_labels, 
        random_state=random_state
    )
    
    # 再从剩余数据中分出验证集
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=val_size_adjusted, 
        stratify=y_temp, 
        random_state=random_state
    )
    
    print(f"\n📊 数据集分割完成:")
    print(f"  训练集: {len(X_train)} 张图像")
    print(f"  验证集: {len(X_val)} 张图像")  
    print(f"  测试集: {len(X_test)} 张图像")
    
    # 检查类别平衡
    print("\n📈 各数据集类别分布:")
    for name, y in [("训练集", y_train), ("验证集", y_val), ("测试集", y_test)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n{name}:")
        for cls_idx, count in zip(unique, counts):
            cls_name = label_encoder.inverse_transform([cls_idx])[0]
            print(f"  {cls_name}: {count} 张")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder


def create_balanced_dataloader(dataset, labels, batch_size=32, num_workers=2):
    """创建平衡的数据加载器
    
    使用加权随机采样来平衡类别
    
    Args:
        dataset: 数据集实例
        labels: 标签列表
        batch_size: 批次大小
        num_workers: 工作进程数
        
    Returns:
        dataloader: 平衡的数据加载器
    """
    # 计算类别权重
    class_counts = Counter(labels)
    total_samples = len(labels)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    
    # 创建样本权重
    sample_weights = [class_weights[label] for label in labels]
    
    # 创建加权采样器
    sampler = WeightedRandomSampler(
        sample_weights, 
        len(sample_weights), 
        replacement=True
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=2):
    """创建标准数据加载器
    
    Args:
        dataset: 数据集实例
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        
    Returns:
        dataloader: 数据加载器
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader


def create_progressive_mask_strategy(batch_size, n_patches, epoch, max_epochs, 
                                   min_ratio=0.6, max_ratio=0.8):
    """渐进式掩码策略 - 随训练进程增加难度
    
    Args:
        epoch: 当前轮次
        max_epochs: 总轮数
        min_ratio: 最小掩码比例
        max_ratio: 最大掩码比例
    """
    # 动态调整掩码比例
    progress = epoch / max_epochs
    mask_ratio = min_ratio + (max_ratio - min_ratio) * progress
    
    context_patches_list = []
    target_patches_list = []
    
    mask_count = int(n_patches * mask_ratio)
    grid_size = int(n_patches ** 0.5)
    
    for _ in range(batch_size):
        masked_patches = set()
        
        # 策略1: 块状掩码 (70%概率)
        if random.random() < 0.7:
            num_centers = random.randint(2, min(5, int(3 + progress * 2)))
            centers = random.sample(range(n_patches), num_centers)
            
            for center in centers:
                row, col = center // grid_size, center % grid_size
                # 动态块大小
                block_size = random.randint(1, min(3, int(2 + progress)))
                
                for dr in range(-block_size, block_size + 1):
                    for dc in range(-block_size, block_size + 1):
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                            patch_idx = new_row * grid_size + new_col
                            masked_patches.add(patch_idx)
        
        # 策略2: 随机掩码 (30%概率) - 增加难度
        else:
            masked_patches = set(random.sample(range(n_patches), mask_count))
        
        # 确保掩码数量
        masked_patches = list(masked_patches)[:mask_count]
        if len(masked_patches) < mask_count:
            remaining = [i for i in range(n_patches) if i not in masked_patches]
            additional = random.sample(remaining, mask_count - len(masked_patches))
            masked_patches.extend(additional)
        
        context_patches = [i for i in range(n_patches) if i not in masked_patches]
        context_patches_list.append(sorted(context_patches))
        
        # 选择多个目标patches进行预测
        num_targets = min(4, len(masked_patches))
        target_patches_list.append(random.sample(masked_patches, num_targets))
    
    return context_patches_list, target_patches_list