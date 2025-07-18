"""
评估函数模块 - 模型评估相关函数
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


def evaluate_optimized_model(ijepa_model, classifier, test_loader, label_encoder):
    """评估优化模型
    
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
    
    # 生成详细报告
    # generate_classification_report(all_targets, all_predictions, label_encoder)
    
    return results, all_predictions, all_targets


def calculate_metrics(y_true, y_pred, y_prob=None):
    """计算各种评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（可选）
        
    Returns:
        metrics: 指标字典
    """
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
    """打印评估结果
    
    Args:
        results: 评估结果字典
        label_encoder: 标签编码器
    """
    print("\n" + "="*60)
    print("📊 优化模型测试结果:")
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
    for i, class_name in enumerate(class_names):
        if i < len(results['per_class_f1']):
            print(f"  {class_name}:")
            print(f"    F1: {results['per_class_f1'][i]:.3f} | "
                  f"精确率: {results['per_class_precision'][i]:.3f} | "
                  f"召回率: {results['per_class_recall'][i]:.3f}")
    
    print("="*60)