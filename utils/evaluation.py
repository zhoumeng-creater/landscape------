"""
评估函数模块 - 模型评估相关函数
包含原版本和新增的预训练模型评估函数
"""
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score
)
from config import Config
from data.transforms import get_tta_transforms
from torch.cuda.amp import autocast


def evaluate_optimized_model(ijepa_model, classifier, test_loader, label_encoder):
    """评估优化模型（原版本）
    
    Args:
        ijepa_model: I-JEPA模型
        classifier: 分类器
        test_loader: 测试数据加载器
        label_encoder: 标签编码器
        
    Returns:
        results: 评估结果字典
        predictions: 所有预测
        targets: 所有真实标签
    """
    device = Config.DEVICE
    ijepa_model.eval()
    classifier.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("🔍 正在评估优化模型...")
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 提取特征
            features = ijepa_model.encode(data)
            
            # 分类预测
            output = classifier(features)
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # 计算各种指标
    results = calculate_metrics(all_targets, all_predictions, all_probabilities)
    
    # 打印结果
    print_evaluation_results(results, label_encoder)
    
    return results, all_predictions, all_targets


def evaluate_pretrained_model(model, test_loader, label_encoder, use_tta=False):
    """评估预训练模型（新增）
    
    Args:
        model: 预训练模型
        test_loader: 测试数据加载器
        label_encoder: 标签编码器
        use_tta: 是否使用测试时增强
        
    Returns:
        results: 评估结果字典
        predictions: 所有预测
        targets: 所有真实标签
    """
    device = Config.DEVICE
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    print("🔍 正在评估预训练模型...")
    if use_tta:
        print("  使用测试时增强(TTA)...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            if use_tta:
                # 测试时增强
                probabilities = perform_tta(model, data)
            else:
                # 标准预测
                with autocast(enabled=Config.USE_AMP):
                    output = model(data)
                    if isinstance(output, tuple):
                        logits, _ = output
                    else:
                        logits = output
                    probabilities = F.softmax(logits, dim=1)
            
            _, predicted = torch.max(probabilities, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # 打印进度
            if batch_idx % 10 == 0:
                print(f"  处理批次 {batch_idx}/{len(test_loader)}")
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # 计算各种指标
    results = calculate_metrics(all_targets, all_predictions, all_probabilities)
    
    # 打印结果
    print_evaluation_results(results, label_encoder)
    
    # 生成详细报告
    generate_detailed_report(all_targets, all_predictions, all_probabilities, label_encoder)
    
    return results, all_predictions, all_targets


def perform_tta(model, data):
    """执行测试时增强
    
    Args:
        model: 模型
        data: 输入数据（已经是tensor）
        
    Returns:
        averaged_probs: 平均概率
    """
    # 获取TTA变换
    tta_transforms = get_tta_transforms()
    
    # 存储所有预测
    all_probs = []
    
    # 原始预测
    with autocast(enabled=Config.USE_AMP):
        output = model(data)
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)
    
    # TTA预测（简化版本，因为输入已经是tensor）
    # 水平翻转
    flipped_data = torch.flip(data, dims=[3])  # 水平翻转
    with autocast(enabled=Config.USE_AMP):
        output = model(flipped_data)
        if isinstance(output, tuple):
            logits, _ = output
        else:
            logits = output
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)
    
    # 小角度旋转（使用torch的旋转）
    for angle in [-5, 5]:
        angle_rad = angle * np.pi / 180
        theta = torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0]
        ], dtype=torch.float32).unsqueeze(0).repeat(data.size(0), 1, 1).to(data.device)
        
        grid = F.affine_grid(theta, data.size(), align_corners=False)
        rotated_data = F.grid_sample(data, grid, align_corners=False)
        
        with autocast(enabled=Config.USE_AMP):
            output = model(rotated_data)
            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
    
    # 平均所有预测
    averaged_probs = torch.stack(all_probs).mean(dim=0)
    
    return averaged_probs


def calculate_metrics(y_true, y_pred, y_prob=None):
    """计算各种评估指标（保留原版本）"""
    metrics = {}
    
    # 基础指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    
    # F1分数（多种平均方式）
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    
    # 精确率和召回率
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    
    # 如果有概率预测，计算AUC
    if y_prob is not None:
        try:
            # 多分类AUC
            num_classes = y_prob.shape[1]
            if num_classes > 2:
                metrics['auc_macro'] = roc_auc_score(y_true, y_prob, 
                                                    multi_class='ovr', average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_prob, 
                                                       multi_class='ovr', average='weighted')
        except:
            metrics['auc_macro'] = None
            metrics['auc_weighted'] = None
    
    # 每个类别的指标
    metrics['per_class_f1'] = f1_score(y_true, y_pred, average=None)
    metrics['per_class_precision'] = precision_score(y_true, y_pred, average=None)
    metrics['per_class_recall'] = recall_score(y_true, y_pred, average=None)
    
    return metrics


def print_evaluation_results(results, label_encoder):
    """打印评估结果（增强版本）"""
    print("\n" + "="*60)
    print("📊 模型测试结果:")
    print("="*60)
    
    # 打印总体指标
    print(f"\n🎯 总体性能指标:")
    print(f"  准确率 (Accuracy): {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"  平衡准确率 (Balanced Accuracy): {results['balanced_accuracy']:.4f}")
    print(f"  Cohen's Kappa: {results['cohen_kappa']:.4f}")
    
    print(f"\n📈 F1分数:")
    print(f"  宏平均 (Macro): {results['f1_macro']:.4f}")
    print(f"  微平均 (Micro): {results['f1_micro']:.4f}")
    print(f"  加权平均 (Weighted): {results['f1_weighted']:.4f}")
    
    print(f"\n🎪 精确率和召回率:")
    print(f"  精确率 - 宏平均: {results['precision_macro']:.4f}")
    print(f"  精确率 - 加权平均: {results['precision_weighted']:.4f}")
    print(f"  召回率 - 宏平均: {results['recall_macro']:.4f}")
    print(f"  召回率 - 加权平均: {results['recall_weighted']:.4f}")
    
    if results.get('auc_macro') is not None:
        print(f"\n📊 AUC分数:")
        print(f"  宏平均: {results['auc_macro']:.4f}")
        print(f"  加权平均: {results['auc_weighted']:.4f}")
    
    # 打印每个类别的性能
    print(f"\n🏷️ 每个类别的性能:")
    class_names = label_encoder.classes_
    
    # 找出表现最好和最差的类别
    f1_scores = results['per_class_f1']
    best_class_idx = np.argmax(f1_scores)
    worst_class_idx = np.argmin(f1_scores)
    
    for i, class_name in enumerate(class_names):
        if i < len(results['per_class_f1']):
            marker = ""
            if i == best_class_idx:
                marker = " ⭐ (最佳)"
            elif i == worst_class_idx:
                marker = " ⚠️  (最差)"
            
            print(f"  {class_name}{marker}:")
            print(f"    F1: {results['per_class_f1'][i]:.3f} | "
                  f"精确率: {results['per_class_precision'][i]:.3f} | "
                  f"召回率: {results['per_class_recall'][i]:.3f}")
    
    print("="*60)


def generate_detailed_report(y_true, y_pred, y_prob, label_encoder):
    """生成详细的分类报告（新增）"""
    print("\n" + "="*60)
    print("📋 详细分类报告:")
    print("="*60)
    
    # 生成分类报告
    report = classification_report(
        y_true, y_pred, 
        target_names=label_encoder.classes_,
        digits=3
    )
    print(report)
    
    # 计算并显示置信度统计
    if y_prob is not None:
        max_probs = np.max(y_prob, axis=1)
        correct_mask = y_pred == y_true
        
        print("\n📊 预测置信度分析:")
        print(f"  平均置信度: {np.mean(max_probs):.3f}")
        print(f"  正确预测的平均置信度: {np.mean(max_probs[correct_mask]):.3f}")
        print(f"  错误预测的平均置信度: {np.mean(max_probs[~correct_mask]):.3f}")
        
        # 高置信度但错误的预测
        high_conf_wrong = np.sum((max_probs > 0.9) & (~correct_mask))
        low_conf_correct = np.sum((max_probs < 0.5) & correct_mask)
        
        print(f"\n⚠️  潜在问题:")
        print(f"  高置信度(>0.9)但错误的预测: {high_conf_wrong} 个")
        print(f"  低置信度(<0.5)但正确的预测: {low_conf_correct} 个")
    
    print("="*60)