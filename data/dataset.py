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


def create_optimized_context_and_target_patches(batch_size, n_patches, mask_ratio=0.75):
    """优化的掩码策略 - 使用块状掩码
    
    生成用于I-JEPA预训练的上下文和目标patches
    
    Args:
        batch_size: 批次大小
        n_patches: 总patch数量
        mask_ratio: 掩码比例
        
    Returns:
        context_patches_list: 每个样本的可见patch索引
        target_patches_list: 每个样本的目标patch索引
    """
    context_patches_list = []
    target_patches_list = []
    
    mask_count = int(n_patches * mask_ratio)
    grid_size = int(n_patches ** 0.5)
    
    for _ in range(batch_size):
        # 创建连续的掩码块
        masked_patches = set()
        
        # 选择2-4个掩码中心
        num_centers = random.randint(2, 4)
        centers = random.sample(range(n_patches), min(num_centers, n_patches))
        
        for center in centers:
            row, col = center // grid_size, center % grid_size
            # 创建3x3或2x2的掩码块
            block_size = random.randint(1, 2)
            for dr in range(-block_size, block_size + 1):
                for dc in range(-block_size, block_size + 1):
                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                        patch_idx = new_row * grid_size + new_col
                        masked_patches.add(patch_idx)
        
        # 调整到目标掩码数量
        masked_patches = list(masked_patches)
        if len(masked_patches) > mask_count:
            masked_patches = random.sample(masked_patches, mask_count)
        elif len(masked_patches) < mask_count:
            # 随机添加更多掩码
            remaining = [i for i in range(n_patches) if i not in masked_patches]
            additional = random.sample(
                remaining, 
                min(mask_count - len(masked_patches), len(remaining))
            )
            masked_patches.extend(additional)
        
        # 生成上下文patches（可见的）
        context_patches = [i for i in range(n_patches) if i not in masked_patches]
        
        context_patches_list.append(sorted(context_patches))
        # 随机选择一个掩码patch作为预测目标
        target_patches_list.append(
            random.choice(masked_patches) if masked_patches else 0
        )
    
    return context_patches_list, target_patches_list