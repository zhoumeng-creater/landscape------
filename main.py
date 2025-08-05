"""
主程序入口 - 整合所有模块完成训练和评估
支持原I-JEPA和新的预训练模型
"""
import os
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

# 导入配置
from config import Config

# 导入原模型（保留兼容性）
from models.ijepa import OptimizedIJEPAModel
from models.classifier import AdvancedClassifier

# 导入新模型
from models.pretrained_model import PretrainedVisionModel, MultiModelEnsemble, LightweightModel

# 导入数据相关
from data.dataset import (
    load_dataset, split_dataset, OptimizedGardenDataset,
    create_balanced_dataloader, create_dataloader
)
from data.transforms import get_optimized_transforms, get_advanced_transforms

# 导入训练函数
from training.pretrain import optimized_pretrain_ijepa
from training.finetune import optimized_finetune_classifier
from training.advanced_train import train_pretrained_model

# 导入评估和可视化
from utils.evaluation import evaluate_optimized_model, evaluate_pretrained_model
from utils.visualization import (
    plot_enhanced_learning_curves, plot_pretrain_curves,
    plot_confusion_matrix, plot_class_distribution
)

# 混合精度训练
from torch.cuda.amp import GradScaler


def set_random_seed(seed=42):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(num_classes, model_type='pretrained'):
    """创建模型
    
    Args:
        num_classes: 类别数
        model_type: 模型类型
        
    Returns:
        model: 模型实例
        use_ijepa: 是否使用I-JEPA架构
    """
    if model_type == 'ijepa':
        # 原I-JEPA模型
        ijepa_model = OptimizedIJEPAModel(
            img_size=Config.IMG_SIZE,
            patch_size=Config.PATCH_SIZE,
            embed_dim=Config.EMBED_DIM,
            depth=Config.DEPTH,
            n_heads=Config.N_HEADS,
            predictor_depth=Config.PREDICTOR_DEPTH
        )
        classifier = AdvancedClassifier(
            embed_dim=Config.EMBED_DIM,
            num_classes=num_classes,
            dropout=Config.CLASSIFIER_DROPOUT
        )
        return (ijepa_model, classifier), True
    
    elif model_type == 'hybrid_ijepa':
        # 新增：混合I-JEPA模型
        from models.hybrid_ijepa import create_hybrid_ijepa
        ijepa_model, classifier = create_hybrid_ijepa(
            pretrained_model_name=Config.PRETRAINED_MODEL_NAME,
            num_classes=num_classes
        )
        return (ijepa_model, classifier), True

    elif model_type == 'pretrained':
        # 预训练模型
        model = PretrainedVisionModel(
            model_name=Config.PRETRAINED_MODEL_NAME,
            num_classes=num_classes,
            feature_dim=Config.EMBED_DIM,
            dropout=Config.CLASSIFIER_DROPOUT,
            freeze_layers=Config.FREEZE_LAYERS
        )
        return model, False
    
    elif model_type == 'ensemble':
        # 模型集成
        model = MultiModelEnsemble(num_classes=num_classes)
        return model, False
    
    elif model_type == 'lightweight':
        # 轻量级模型
        model = LightweightModel(num_classes=num_classes)
        return model, False
    
    else:
        raise ValueError(f"未知的模型类型: {model_type}")


def main():
    """主函数 - 执行完整的训练和评估流程"""
    print("🌸 ===== 园林分类系统 V2.0 =====")
    print(f"🎯 使用模型: {Config.MODEL_TYPE}")
    print(f"🎯 预期性能: 85-90%准确率")
    print("="*50)
    
    # 设置随机种子
    set_random_seed(Config.SEED)
    
    # 打印配置
    Config.print_config()
    
    # 创建必要的目录
    Config.create_directories()
    
    # 打印设备信息
    print(f'\n🚀 使用设备: {Config.DEVICE}')
    if Config.DEVICE.type == 'cuda':
        print(f'  GPU型号: {torch.cuda.get_device_name(0)}')
        print(f'  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
        print(f'  使用混合精度训练: {Config.USE_AMP}')
    
    # ===== 数据准备 =====
    print("\n" + "📂"*20)
    print("步骤1: 数据准备")
    print("📂"*20)
    
    # 检查数据路径
    if not os.path.exists(Config.DATA_PATH):
        print(f"❌ 错误: 数据路径 {Config.DATA_PATH} 不存在")
        return
    
    # 加载数据集
    image_paths, labels, class_names = load_dataset(Config.DATA_PATH)
    
    if len(image_paths) == 0:
        print("❌ 错误: 没有找到图像文件")
        return
    
    # 分割数据集
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = split_dataset(
        image_paths, labels, Config.TEST_SIZE, Config.VAL_SIZE
    )
    
    # 显示类别分布
    plot_class_distribution(labels, class_names, save_path='class_distribution.png')
    
    # 获取数据变换
    if Config.AUGMENTATION_LEVEL == 'strong':
        train_transform, val_transform, test_transform = get_advanced_transforms()
        print("📈 使用强数据增强策略")
    else:
        pretrain_transform, train_transform, val_test_transform = get_optimized_transforms()
        val_transform = test_transform = val_test_transform
        print("📈 使用标准数据增强策略")
    
    # ===== 模型创建 =====
    print("\n" + "🧠"*20)
    print("步骤2: 模型创建")
    print("🧠"*20)
    
    num_classes = len(class_names)
    model_or_tuple, use_ijepa = create_model(num_classes, Config.MODEL_TYPE)
    
    if use_ijepa:
        ijepa_model, classifier = model_or_tuple
        print(f"✅ 创建I-JEPA模型和分类器")
    else:
        model = model_or_tuple
        print(f"✅ 创建{Config.MODEL_TYPE}模型")
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params/1e6:.2f}M")
        print(f"  可训练参数量: {trainable_params/1e6:.2f}M")
        print(f"  冻结参数量: {(total_params-trainable_params)/1e6:.2f}M")
    
    # ===== 训练流程 =====
    if use_ijepa and not Config.SKIP_PRETRAIN:
        # 原I-JEPA流程
        print("\n" + "🚀"*20)
        print("阶段1: I-JEPA自监督预训练")
        print("🚀"*20)
        
        # 预训练数据集和加载器
        pretrain_dataset = OptimizedGardenDataset(
            X_train, transform=pretrain_transform, is_pretraining=True
        )
        pretrain_loader = create_dataloader(
            pretrain_dataset, 
            batch_size=Config.BATCH_SIZE_PRETRAIN,
            shuffle=True,
            num_workers=Config.NUM_WORKERS
        )
        
        # 执行预训练
        ijepa_model, pretrain_history = optimized_pretrain_ijepa(
            ijepa_model, pretrain_loader,
            num_epochs=Config.PRETRAIN_EPOCHS,
            learning_rate=Config.PRETRAIN_LR
        )
        
        # 绘制预训练曲线
        plot_pretrain_curves(pretrain_history, save_path='pretrain_curves.png')
        
        # 微调阶段
        print("\n" + "🎯"*20)
        print("阶段2: 分类器微调")
        print("🎯"*20)
        
        # 准备数据
        train_dataset = OptimizedGardenDataset(X_train, y_train, train_transform)
        val_dataset = OptimizedGardenDataset(X_val, y_val, val_test_transform)
        test_dataset = OptimizedGardenDataset(X_test, y_test, val_test_transform)
        
        # 创建数据加载器
        train_loader = create_balanced_dataloader(
            train_dataset, y_train, 
            batch_size=Config.BATCH_SIZE_FINETUNE,
            num_workers=Config.NUM_WORKERS
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=Config.BATCH_SIZE_EVAL,
            shuffle=False,
            num_workers=Config.NUM_WORKERS
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=Config.BATCH_SIZE_EVAL,
            shuffle=False,
            num_workers=Config.NUM_WORKERS
        )
        
        # 执行微调
        classifier, finetune_history = optimized_finetune_classifier(
            ijepa_model, classifier,
            train_loader, val_loader,
            num_epochs=Config.FINETUNE_EPOCHS,
            learning_rate=Config.FINETUNE_LR
        )
        
        # 评估
        results, predictions, targets = evaluate_optimized_model(
            ijepa_model, classifier, test_loader, label_encoder
        )
        
    else:
        # 新的预训练模型流程
        print("\n" + "🎯"*20)
        print("训练预训练模型")
        print("🎯"*20)
        
        # 准备数据
        train_dataset = OptimizedGardenDataset(X_train, y_train, train_transform)
        val_dataset = OptimizedGardenDataset(X_val, y_val, val_transform)
        test_dataset = OptimizedGardenDataset(X_test, y_test, test_transform)
        
        # 创建数据加载器
        # 使用梯度累积，所以可以用更小的batch size
        actual_batch_size = Config.BATCH_SIZE_FINETUNE // Config.GRADIENT_ACCUMULATION_STEPS
        
        train_loader = create_balanced_dataloader(
            train_dataset, y_train, 
            batch_size=actual_batch_size,
            num_workers=Config.NUM_WORKERS
        )
        val_loader = create_dataloader(
            val_dataset,
            batch_size=Config.BATCH_SIZE_EVAL,
            shuffle=False,
            num_workers=Config.NUM_WORKERS
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=Config.BATCH_SIZE_EVAL,
            shuffle=False,
            num_workers=Config.NUM_WORKERS
        )
        
        # 训练模型
        model, finetune_history = train_pretrained_model(
            model, train_loader, val_loader,
            num_epochs=Config.FINETUNE_EPOCHS,
            num_classes=num_classes
        )
        
        # 评估
        results, predictions, targets = evaluate_pretrained_model(
            model, test_loader, label_encoder, use_tta=Config.USE_TTA
        )
    
    # 绘制学习曲线
    plot_enhanced_learning_curves(finetune_history, save_path='learning_curves.png')
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        targets, predictions, class_names,
        save_path='confusion_matrix.png'
    )
    
    # ===== 最终总结 =====
    print("\n" + "🎉"*30)
    print("🏆 园林分类系统训练完成!")
    print("🎉"*30)
    
    test_acc = results['accuracy']
    f1_macro = results['f1_macro']
    
    print(f"\n📊 最终测试准确率: {test_acc*100:.2f}%")
    print(f"📊 宏平均F1分数: {f1_macro:.4f}")
    
    if test_acc > 0.85:
        print("🌟 太棒了！达到了85%以上的目标准确率！")
    elif test_acc > 0.80:
        print("✅ 优秀！性能显著提升！")
    elif test_acc > 0.75:
        print("👍 不错！比原来的模型有明显改进！")
    else:
        print("💪 还有提升空间，可以尝试更多优化策略。")
    
    print("\n🔧 进一步提升性能的建议：")
    if not Config.USE_ENSEMBLE:
        print("  1. 启用模型集成 (设置 USE_ENSEMBLE = True)")
    if not Config.USE_TTA:
        print("  2. 启用测试时增强 (设置 USE_TTA = True)")
    if not Config.USE_SAM:
        print("  3. 使用SAM优化器 (设置 USE_SAM = True)")
    if Config.AUGMENTATION_LEVEL != 'strong':
        print("  4. 使用更强的数据增强 (设置 AUGMENTATION_LEVEL = 'strong')")
    print("  5. 尝试不同的预训练模型")
    print("  6. 收集更多训练数据或使用伪标签")
    
    print("\n✅ 所有结果已保存到当前目录")
    print(f"  - 模型检查点: {Config.CHECKPOINT_DIR}/")
    print("  - 学习曲线: learning_curves.png")
    print("  - 混淆矩阵: confusion_matrix.png")
    print("  - 类别分布: class_distribution.png")


if __name__ == "__main__":
    main()