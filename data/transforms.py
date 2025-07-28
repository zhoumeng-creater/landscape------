"""
数据增强模块 - 包含各种数据变换策略
"""
from torchvision import transforms
from config import Config
import numpy as np
import torch
import random
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


def get_optimized_transforms():
    """获取优化的数据变换
    
    Returns:
        pretrain_transform: 预训练时的强增强
        train_transform: 微调时的适度增强
        val_test_transform: 验证/测试时的标准变换
    """
    # 预训练时的强增强 - 用于自监督学习
    pretrain_transform = transforms.Compose([
        transforms.Resize(Config.PRETRAIN_AUGMENTATION['resize']),
        transforms.RandomResizedCrop(
            Config.IMG_SIZE, 
            scale=Config.PRETRAIN_AUGMENTATION['crop_scale'], 
            ratio=Config.PRETRAIN_AUGMENTATION['crop_ratio']
        ),
        RandAugment(n=2, m=10),  # 添加RandAugment
        transforms.RandomHorizontalFlip(p=Config.PRETRAIN_AUGMENTATION['horizontal_flip_p']),
        transforms.RandomVerticalFlip(p=Config.PRETRAIN_AUGMENTATION['vertical_flip_p']),
        transforms.RandomRotation(Config.PRETRAIN_AUGMENTATION['rotation_degrees']),
        transforms.ColorJitter(
            brightness=Config.PRETRAIN_AUGMENTATION['color_jitter']['brightness'],
            contrast=Config.PRETRAIN_AUGMENTATION['color_jitter']['contrast'],
            saturation=Config.PRETRAIN_AUGMENTATION['color_jitter']['saturation'],
            hue=Config.PRETRAIN_AUGMENTATION['color_jitter']['hue']
        ),
        transforms.RandomGrayscale(p=Config.PRETRAIN_AUGMENTATION['grayscale_p']),
        transforms.GaussianBlur(
            kernel_size=Config.PRETRAIN_AUGMENTATION['gaussian_blur']['kernel_size'],
            sigma=Config.PRETRAIN_AUGMENTATION['gaussian_blur']['sigma']
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    # 微调时的适度增强 - 用于监督学习
    train_transform = transforms.Compose([
        transforms.Resize(Config.FINETUNE_AUGMENTATION['resize']),
        transforms.RandomCrop(Config.FINETUNE_AUGMENTATION['crop_size']),
        transforms.RandomHorizontalFlip(p=Config.FINETUNE_AUGMENTATION['horizontal_flip_p']),
        transforms.RandomRotation(Config.FINETUNE_AUGMENTATION['rotation_degrees']),
        transforms.ColorJitter(
            brightness=Config.FINETUNE_AUGMENTATION['color_jitter']['brightness'],
            contrast=Config.FINETUNE_AUGMENTATION['color_jitter']['contrast'],
            saturation=Config.FINETUNE_AUGMENTATION['color_jitter']['saturation'],
            hue=Config.FINETUNE_AUGMENTATION['color_jitter']['hue']
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    # 验证/测试时的标准变换 - 无数据增强
    val_test_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    return pretrain_transform, train_transform, val_test_transform


def get_custom_transform(resize_size=256, crop_size=224, augmentation_level='medium'):
    """获取自定义数据变换
    
    Args:
        resize_size: 调整大小
        crop_size: 裁剪大小
        augmentation_level: 增强级别 ('none', 'light', 'medium', 'heavy')
        
    Returns:
        transform: 数据变换
    """
    transform_list = [transforms.Resize((resize_size, resize_size))]
    
    if augmentation_level != 'none':
        transform_list.append(transforms.RandomCrop(crop_size))
        
        if augmentation_level in ['light', 'medium', 'heavy']:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
        if augmentation_level in ['medium', 'heavy']:
            transform_list.extend([
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            ])
            
        if augmentation_level == 'heavy':
            transform_list.extend([
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ])
    else:
        transform_list.append(transforms.CenterCrop(crop_size))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    return transforms.Compose(transform_list)


class RandomMixup:
    """Mixup数据增强
    
    将两个样本及其标签进行线性混合
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, images, labels):
        """
        Args:
            images: 批次图像 [B, C, H, W]
            labels: 批次标签 [B]
            
        Returns:
            mixed_images: 混合后的图像
            labels_a, labels_b: 原始标签
            lam: 混合系数
        """
        batch_size = images.size(0)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        index = torch.randperm(batch_size).to(images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, labels, labels[index], lam


class RandomCutmix:
    """Cutmix数据增强
    
    随机裁剪一个样本的部分区域并粘贴到另一个样本上
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, images, labels):
        """
        Args:
            images: 批次图像 [B, C, H, W]
            labels: 批次标签 [B]
            
        Returns:
            mixed_images: 混合后的图像
            labels_a, labels_b: 原始标签
            lam: 混合系数
        """
        batch_size = images.size(0)
        W = images.size(2)
        H = images.size(3)
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        index = torch.randperm(batch_size).to(images.device)
        
        # 计算裁剪区域
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        
        # 随机选择裁剪中心
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # 计算裁剪边界
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # 应用Cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # 调整lambda值
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return mixed_images, labels, labels[index], lam
    
class RandAugment:
    """RandAugment数据增强"""
    def __init__(self, n=2, m=10):
        self.n = n
        self.m = m
        self.augment_list = [
            (T.AutoContrast(), 0, 1),
            (T.Equalize(), 0, 1),
            (T.Rotate(0), -30, 30),
            (T.Solarize(0), 0, 256),
            (T.ColorJitter(brightness=0), 0.1, 1.9),
            (T.ColorJitter(contrast=0), 0.1, 1.9),
            (T.ColorJitter(saturation=0), 0.1, 1.9),
            (T.GaussianBlur(3), 0.1, 2.0),
        ]
    
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = (self.m / 30) * (max_val - min_val) + min_val
            if isinstance(op, (T.Rotate, T.Solarize)):
                img = op(img, val)
            elif isinstance(op, T.ColorJitter):
                # 动态设置ColorJitter参数
                img = op(img)
            else:
                img = op(img)
        return img

