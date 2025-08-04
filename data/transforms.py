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
from PIL import Image, ImageOps, ImageEnhance, ImageFilter


def get_optimized_transforms():
    """获取优化的数据变换（原版本）
    
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


def get_advanced_transforms():
    """获取高级数据变换（强数据增强版本）
    
    Returns:
        train_transform: 训练时的强增强
        val_transform: 验证时的标准变换
        test_transform: 测试时的标准变换（支持TTA）
    """
    # 训练时的强增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomChoice([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
            transforms.Compose([
                transforms.RandomCrop(224),
                transforms.RandomRotation(15)
            ])
        ]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        
        # RandAugment
        RandAugment(
            num_ops=Config.STRONG_AUGMENTATION['randaugment_n'],
            magnitude=Config.STRONG_AUGMENTATION['randaugment_m']
        ),
        
        # 额外的强增强
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
        ], p=0.8),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.RandomGrayscale(p=0.1),
        
        # Cutout
        transforms.ToTensor(),
        Cutout(
            n_holes=Config.STRONG_AUGMENTATION['cutout_n_holes'],
            length=Config.STRONG_AUGMENTATION['cutout_length']
        ),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    # 验证时的标准变换
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    # 测试时的变换（支持TTA）
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ])
    
    return train_transform, val_transform, test_transform


def get_tta_transforms():
    """获取测试时增强(TTA)的变换列表
    
    Returns:
        tta_transforms: TTA变换列表
    """
    tta_transforms = []
    
    # 原始图像
    tta_transforms.append(transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ]))
    
    # 水平翻转
    tta_transforms.append(transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
    ]))
    
    # 不同的裁剪
    for scale in [0.85, 0.95, 1.05]:
        crop_size = int(224 * scale)
        tta_transforms.append(transforms.Compose([
            transforms.Resize((crop_size + 32, crop_size + 32)),
            transforms.CenterCrop(crop_size),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ]))
    
    # 轻微旋转
    for angle in [-10, 10]:
        tta_transforms.append(transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation((angle, angle)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD)
        ]))
    
    return tta_transforms


class RandAugment:
    """RandAugment数据增强
    
    参考: https://arxiv.org/abs/1909.13719
    """
    def __init__(self, num_ops=2, magnitude=10):
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.augment_list = [
            (self.auto_contrast, 0, 1),
            (self.equalize, 0, 1),
            (self.rotate, -30, 30),
            (self.solarize, 0, 256),
            (self.color, 0.1, 1.9),
            (self.contrast, 0.1, 1.9),
            (self.brightness, 0.1, 1.9),
            (self.sharpness, 0.1, 1.9),
            (self.shear_x, -0.3, 0.3),
            (self.shear_y, -0.3, 0.3),
            (self.translate_x, -0.45, 0.45),
            (self.translate_y, -0.45, 0.45),
        ]
    
    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.num_ops)
        for op, min_val, max_val in ops:
            val = (self.magnitude / 30) * (max_val - min_val) + min_val
            img = op(img, val)
        return img
    
    def auto_contrast(self, img, _):
        return ImageOps.autocontrast(img)
    
    def equalize(self, img, _):
        return ImageOps.equalize(img)
    
    def rotate(self, img, degrees):
        return img.rotate(degrees)
    
    def solarize(self, img, threshold):
        return ImageOps.solarize(img, threshold)
    
    def color(self, img, factor):
        return ImageEnhance.Color(img).enhance(factor)
    
    def contrast(self, img, factor):
        return ImageEnhance.Contrast(img).enhance(factor)
    
    def brightness(self, img, factor):
        return ImageEnhance.Brightness(img).enhance(factor)
    
    def sharpness(self, img, factor):
        return ImageEnhance.Sharpness(img).enhance(factor)
    
    def shear_x(self, img, shear):
        return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0))
    
    def shear_y(self, img, shear):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0))
    
    def translate_x(self, img, translate):
        return img.transform(img.size, Image.AFFINE, (1, 0, translate * img.size[0], 0, 1, 0))
    
    def translate_y(self, img, translate):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, translate * img.size[1]))


class Cutout:
    """Cutout数据增强
    
    在图像中随机裁剪出方形区域并填充0
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask).unsqueeze(0)
        img = img * mask
        
        return img


class RandomMixup:
    """Mixup数据增强（保留原版本）"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, images, labels):
        batch_size = images.size(0)
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        index = torch.randperm(batch_size).to(images.device)
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, labels, labels[index], lam


class RandomCutmix:
    """Cutmix数据增强（保留原版本）"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, images, labels):
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


def get_custom_transform(resize_size=256, crop_size=224, augmentation_level='medium'):
    """获取自定义数据变换（保留原版本）"""
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