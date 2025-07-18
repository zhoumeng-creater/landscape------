"""
主程序入口 - 整合所有模块完成训练和评估
"""
import os
import torch
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

# 导入配置
from config import Config

# 导入模型
from models.ijepa import OptimizedIJEPAModel
from models.classifier import AdvancedClassifier

# 导入数据相关
from data.dataset import (
    load_dataset, split_dataset, OptimizedGardenDataset,
    create_balanced_dataloader, create_dataloader
)
from data.transforms import get_optimized_transforms

# 导入训练函数
from training.pretrain import optimized_pretrain_ijepa
from training.finetune import optimized_finetune_classifier

# 导入评估和可视化
from utils.evaluation import evaluate_optimized_model, print_evaluation_results
from utils.visualization import (
    plot_enhanced_learning_curves, plot_pretrain_curves,
    plot_confusion_matrix, plot_class_distribution
)


def set_random_seed(seed=42):
    """设置随机种子以确保可重复性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """主函数 - 执行完整的训练和评估流程"""
    print("🌸 ===== 优化版I-JEPA园林分类系统 =====")
    print("🎯 预期性能提升：从40%提升至70-80%")
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
    
    # ===== 数据准备 =====
    print("\n" + "📂"*20)
    print("步骤1: 数据准备")
    print("📂"*20)
    
    # 检查数据路径
    if not os.path.exists(Config.DATA_PATH):
        print(f"❌ 错误: 数据路径 {Config.DATA_PATH} 不存在")
        print("请修改 config.py 中的 DATA_PATH 为正确的数据集路径")
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
    pretrain_transform, train_transform, val_test_transform = get_optimized_transforms()
    
    # ===== 模型创建 =====
    print("\n" + "🧠"*20)
    print("步骤2: 模型创建")
    print("🧠"*20)
    
    # 创建I-JEPA模型
    ijepa_model = OptimizedIJEPAModel(
        img_size=Config.IMG_SIZE,
        patch_size=Config.PATCH_SIZE,
        embed_dim=Config.EMBED_DIM,
        depth=Config.DEPTH,
        n_heads=Config.N_HEADS,
        predictor_depth=Config.PREDICTOR_DEPTH
    )
    
    # 创建分类器
    num_classes = len(class_names)
    classifier = AdvancedClassifier(
        embed_dim=Config.EMBED_DIM,
        num_classes=num_classes,
        dropout=Config.CLASSIFIER_DROPOUT
    )
    
    # 打印模型信息
    print_evaluation_results(ijepa_model, classifier)
    
    # ===== 阶段1: I-JEPA预训练 =====
    print("\n" + "🚀"*20)
    print("阶段1: 优化版I-JEPA自监督预训练")
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
    
    # ===== 阶段2: 分类器微调 =====
    print("\n" + "🎯"*20)
    print("阶段2: 优化版分类器微调")
    print("🎯"*20)
    
    # 微调数据集
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
    
    # 绘制学习曲线
    plot_enhanced_learning_curves(finetune_history, save_path='learning_curves.png')
    
    # ===== 阶段3: 模型评估 =====
    print("\n" + "📊"*20)
    print("阶段3: 模型评估")
    print("📊"*20)
    
    # 评估模型
    results, predictions, targets = evaluate_optimized_model(
        ijepa_model, classifier, test_loader, label_encoder
    )
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        targets, predictions, class_names,
        save_path='confusion_matrix.png'
    )
    
    # ===== 最终总结 =====
    print("\n" + "🎉"*30)
    print("🏆 优化版I-JEPA园林分类系统训练完成!")
    print("🎉"*30)
    
    test_acc = results['accuracy']
    f1_macro = results['f1_macro']
    
    print(f"\n📊 最终测试准确率: {test_acc*100:.2f}%")
    print(f"📊 宏平均F1分数: {f1_macro:.4f}")
    
    if test_acc > 0.70:
        print("🌟 恭喜！达到了70%以上的目标准确率！")
    elif test_acc > 0.60:
        print("✅ 不错！性能有显著提升！")
    else:
        print("💪 还有提升空间，可以尝试更多优化策略。")
    
    print("\n🔧 如果需要进一步提升性能，可以尝试：")
    print("  1. 增加预训练轮数到100+")
    print("  2. 使用更大的模型（depth=16, embed_dim=1024）")
    print("  3. 收集更多训练数据")
    print("  4. 尝试不同的掩码比例和策略")
    print("  5. 使用更复杂的数据增强")
    print("  6. 调整学习率和优化器参数")
    print("  7. 使用集成学习方法")
    
    print("\n✅ 所有结果已保存到当前目录")
    print("  - 预训练曲线: pretrain_curves.png")
    print("  - 学习曲线: learning_curves.png")
    print("  - 混淆矩阵: confusion_matrix.png")
    print("  - 类别分布: class_distribution.png")
    print(f"  - 模型检查点: {Config.CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()