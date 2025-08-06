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
    DATA_PATH = '/kaggle/input/yuanlin-dataset/yuanlin_dataset'
    
    # 保存路径
    CHECKPOINT_DIR = './checkpoints'
    PRETRAIN_MODEL_PATH = 'optimized_ijepa_best.pth'  # 保留兼容性
    FINETUNE_MODEL_PATH = 'optimized_best_classifier.pth'
    PRETRAINED_MODEL_PATH = 'pretrained_best_model.pth'  # 新增
    
    # ============================== 模型选择 ==============================
    # 模型类型选择
    MODEL_TYPE = 'hybrid_ijepa'  # 可选: 'ijepa', 'hybrid_ijepa', 'pretrained', 'ensemble', 'lightweight'
    
    # 预训练模型配置
    PRETRAINED_MODEL_NAME = 'vit_base_patch16_224.dino'  # 推荐选项:
    # - 'vit_base_patch16_224.dino' (最佳特征质量)
    # - 'vit_base_patch16_clip_224.openai' (语义理解强)
    # - 'convnext_base.fb_in22k_ft_in1k' (细粒度分类好)
    # - 'swin_base_patch4_window7_224' (层次化特征)
    # - 'efficientnet_b3' (轻量级，适合P100)
    
    # 冻结层数（预训练模型）
    FREEZE_LAYERS = 6  # ViT有12层，冻结前6层
    
    # 分类器配置
    CLASSIFIER_DROPOUT = 0.5

    # ============================== 数据配置 ==============================
    # 图像配置
    IMG_SIZE = 224
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    
    # 数据集分割
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # 数据加载
    BATCH_SIZE_PRETRAIN = 20  # 保留原设置
    BATCH_SIZE_FINETUNE = 16  # 减小以适应更大的模型
    BATCH_SIZE_EVAL = 32
    NUM_WORKERS = 2
    
    # ============================== 原I-JEPA模型配置（保留兼容） ==============================
    # Transformer配置
    EMBED_DIM = 768
    DEPTH = 12
    N_HEADS = 12
    MLP_RATIO = 4
    
    # Predictor配置
    PREDICTOR_DEPTH = 6
    
    # ============================== 训练配置 ==============================
    # 训练模式
    USE_PRETRAINED_MODEL = True  # 是否使用预训练模型
    SKIP_PRETRAIN = False  # 跳过I-JEPA预训练阶段
    
    # 预训练配置（如果使用）
    PRETRAIN_EPOCHS = 30
    PRETRAIN_LR = 3e-5
    PRETRAIN_WEIGHT_DECAY = 0.05
    PRETRAIN_BETAS = (0.9, 0.95)  # AdamW的beta参数 - 添加这一行！
    
    # I-JEPA特定配置 - 也需要添加这些
    EMA_MOMENTUM = 0.996  # 目标编码器的EMA动量
    WARMUP_EPOCHS = 5     # 学习率预热轮数
    LOG_INTERVAL = 50     # 打印日志的间隔
    PATIENCE_PRETRAIN = 10  # 预训练早停耐心值
    
    # 微调配置
    FINETUNE_EPOCHS = 30  # 使用预训练模型时可以减少
    FINETUNE_LR = 5e-5  # 使用预训练模型时要用更小的学习率
    FINETUNE_WEIGHT_DECAY = 0.01
    FINETUNE_STAGE2_EPOCH = 10  # 开始解冻更多层
    FINETUNE_LR_RATIO = 10
    
    # 优化器配置
    OPTIMIZER_TYPE = 'adamw'  # 可选: 'adamw', 'lamb', 'lion'
    USE_SAM = False  # 禁用SAM优化器
    SAM_RHO = 0.05
    
    # 混合精度训练
    USE_AMP = True  # 使用自动混合精度训练
    
    # 梯度累积
    GRADIENT_ACCUMULATION_STEPS = 2  # 模拟更大的batch size
    
    # 损失函数配置
    LABEL_SMOOTHING = 0.1
    USE_AUXILIARY_LOSS = True  # 使用辅助损失
    AUX_LOSS_WEIGHT = 0.3
    
    # 正则化配置
    DROP_PATH_RATE = 0.1
    LAYER_SCALE_INIT = 1e-4
    GRADIENT_CLIP = 1.0
    
    # ============================== 评估配置 ==============================
    # 早停配置
    PATIENCE_PRETRAIN = 20
    PATIENCE_FINETUNE = 10  # 使用预训练模型收敛更快
    
    # 测试时增强
    USE_TTA = True  # Test Time Augmentation
    TTA_TIMES = 5  # TTA次数
    
    # 日志配置
    LOG_INTERVAL = 100
    SAVE_BEST_ONLY = True

    # ============================== 数据增强配置 ==============================
    # 增强级别
    AUGMENTATION_LEVEL = 'strong'  # 可选: 'basic', 'medium', 'strong'
    
    # 高级增强
    USE_CUTMIX = True
    CUTMIX_ALPHA = 1.0
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2
    
    # 强数据增强配置
    STRONG_AUGMENTATION = {
        'randaugment_n': 2,
        'randaugment_m': 10,
        'cutout_n_holes': 1,
        'cutout_length': 16,
        'auto_augment_policy': 'imagenet',
    }
    
    # 基础数据增强（保留原配置）
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
    
    # ============================== 集成学习配置 ==============================
    USE_ENSEMBLE = False  # 是否使用模型集成
    ENSEMBLE_MODELS = [
        'vit_base_patch16_224.dino',
        'convnext_base.fb_in22k_ft_in1k',
        'swin_base_patch4_window7_224'
    ]
    
    # ============================== 知识蒸馏配置 ==============================
    USE_KNOWLEDGE_DISTILLATION = False  # 是否使用知识蒸馏
    TEACHER_MODEL = 'vit_large_patch16_224'  # 教师模型
    DISTILLATION_TEMPERATURE = 4.0
    DISTILLATION_ALPHA = 0.7  # 蒸馏损失权重
    
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
        print(f"🔧 模型类型: {cls.MODEL_TYPE}")
        print(f"🔧 预训练模型: {cls.PRETRAINED_MODEL_NAME if cls.USE_PRETRAINED_MODEL else '无'}")
        print(f"🔧 使用混合精度: {cls.USE_AMP}")
        print(f"🔧 使用SAM优化器: {cls.USE_SAM}")
        print(f"🔧 数据增强级别: {cls.AUGMENTATION_LEVEL}")
        print("="*60)
        
        config_dict = cls.get_config_dict()
        for key, value in sorted(config_dict.items()):
            if not isinstance(value, dict):  # 跳过字典类型的配置
                print(f"{key}: {value}")
        print("="*60)
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)