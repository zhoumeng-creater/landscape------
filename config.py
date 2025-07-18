"""
配置文件 - 包含所有训练和模型相关的配置参数
"""
import torch
import os

class Config:
    """配置类 - 包含所有超参数和路径配置"""
    
    # ============================== 基础配置 ==============================
    # 随机种子
    SEED = 42
    
    # 设备配置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 数据路径
    DATA_PATH = '/kaggle/input/yuanlin2/yuanlin_dataset'
    
    # 保存路径
    CHECKPOINT_DIR = './checkpoints'
    PRETRAIN_MODEL_PATH = 'optimized_ijepa_best.pth'
    FINETUNE_MODEL_PATH = 'optimized_best_classifier.pth'
    
    # ============================== 数据配置 ==============================
    # 图像配置
    IMG_SIZE = 224
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    
    # 数据集分割
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # 数据加载
    BATCH_SIZE_PRETRAIN = 20
    BATCH_SIZE_FINETUNE = 28
    BATCH_SIZE_EVAL = 32
    NUM_WORKERS = 2
    
    # ============================== 模型配置 ==============================
    # Transformer配置
    EMBED_DIM = 768
    DEPTH = 12
    N_HEADS = 12
    MLP_RATIO = 4
    
    # Patch Embedding配置
    OVERLAP_RATIO = 0.25
    
    # 注意力配置
    ATTENTION_DROPOUT = 0.1
    MAX_RELATIVE_POSITION = 14
    
    # Predictor配置
    PREDICTOR_DEPTH = 6
    
    # 分类器配置
    CLASSIFIER_DROPOUT = 0.5
    
    # ============================== 训练配置 ==============================
    # 预训练配置
    PRETRAIN_EPOCHS = 60
    PRETRAIN_LR = 3e-5
    PRETRAIN_WEIGHT_DECAY = 0.05
    PRETRAIN_BETAS = (0.9, 0.95)
    WARMUP_EPOCHS = 10
    EMA_MOMENTUM = 0.996
    EMA_MOMENTUM_END = 0.9995
    
    # 微调配置
    FINETUNE_EPOCHS = 50
    FINETUNE_LR = 1e-3
    FINETUNE_WEIGHT_DECAY = 0.01
    FINETUNE_STAGE2_EPOCH = 15  # 开始微调预训练模型的epoch
    FINETUNE_LR_RATIO = 10  # 预训练模型学习率降低的比例
    
    # 损失函数配置
    LABEL_SMOOTHING = 0.1
    DIVERSITY_WEIGHT = 0.15
    L2_WEIGHT = 0.1
    CONTRASTIVE_WEIGHT = 0.05
    
    # 正则化配置
    DROP_PATH_RATE = 0.1
    LAYER_SCALE_INIT = 1e-4
    GRADIENT_CLIP = 1.0
    
    # 掩码策略
    MASK_RATIO = 0.75
    MIN_MASK_CENTERS = 2
    MAX_MASK_CENTERS = 4
    MIN_MASK_BLOCK_SIZE = 1
    MAX_MASK_BLOCK_SIZE = 2
    
    # ============================== 评估配置 ==============================
    # 早停配置
    PATIENCE_PRETRAIN = 20
    PATIENCE_FINETUNE = 15
    
    # 日志配置
    LOG_INTERVAL = 100
    SAVE_BEST_ONLY = True
    
    # ============================== 数据增强配置 ==============================
    # 预训练数据增强
    PRETRAIN_AUGMENTATION = {
        'resize': (256, 256),
        'crop_scale': (0.2, 1.0),
        'crop_ratio': (0.75, 1.33),
        'horizontal_flip_p': 0.5,
        'vertical_flip_p': 0.3,
        'rotation_degrees': 30,
        'color_jitter': {
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.4,
            'hue': 0.2
        },
        'grayscale_p': 0.1,
        'gaussian_blur': {
            'kernel_size': 3,
            'sigma': (0.1, 2.0)
        }
    }
    
    # 微调数据增强
    FINETUNE_AUGMENTATION = {
        'resize': (256, 256),
        'crop_size': 224,
        'horizontal_flip_p': 0.5,
        'rotation_degrees': 15,
        'color_jitter': {
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.1
        }
    }
    
    # 图像归一化参数
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    @classmethod
    def get_config_dict(cls):
        """获取所有配置参数的字典形式"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def print_config(cls):
        """打印所有配置参数"""
        print("="*60)
        print("配置参数:")
        print("="*60)
        config_dict = cls.get_config_dict()
        for key, value in sorted(config_dict.items()):
            print(f"{key}: {value}")
        print("="*60)
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
