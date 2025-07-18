"""
可视化工具模块 - 用于绘制训练曲线、混淆矩阵等
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import torch


def plot_enhanced_learning_curves(history, save_path=None):
    """绘制增强的学习曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['train_loss'], label='训练损失', color='blue', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='验证损失', color='red', linewidth=2)
    axes[0, 0].set_title('训练和验证损失', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[0, 1].plot(history['train_acc'], label='训练准确率', color='blue', linewidth=2)
    axes[0, 1].plot(history['val_acc'], label='验证准确率', color='red', linewidth=2)
    axes[0, 1].set_title('训练和验证准确率', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('准确率 (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 损失平滑趋势
    if len(history['train_loss']) > 5:
        window_size = 5
        train_smooth = np.convolve(history['train_loss'], np.ones(window_size)/window_size, mode='valid')
        val_smooth = np.convolve(history['val_loss'], np.ones(window_size)/window_size, mode='valid')
        
        axes[1, 0].plot(range(window_size-1, len(train_smooth) + window_size-1), train_smooth, 
                       label='训练损失(平滑)', color='blue', linewidth=2)
        axes[1, 0].plot(range(window_size-1, len(val_smooth) + window_size-1), val_smooth, 
                       label='验证损失(平滑)', color='red', linewidth=2)
        axes[1, 0].set_title('损失平滑趋势', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('损失')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 性能总结
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    
    summary_text = f"""🏆 性能总结
    
最佳验证准确率: {best_val_acc:.2f}%
最佳轮次: {best_epoch}
最终训练准确率: {final_train_acc:.2f}%
最终验证准确率: {final_val_acc:.2f}%
准确率差距: {final_train_acc - final_val_acc:.2f}%

模型状态: """
    
    acc_gap = final_train_acc - final_val_acc
    if acc_gap > 15:
        status = "⚠️  可能过拟合"
        color = 'red'
    elif final_val_acc < 60:
        status = "⚠️  可能欠拟合"  
        color = 'orange'
    else:
        status = "✅ 性能良好"
        color = 'green'
    
    summary_text += status
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                   transform=axes[1, 1].transAxes, verticalalignment='center')
    axes[1, 1].text(0.1, 0.15, status, fontsize=14, fontweight='bold',
                   transform=axes[1, 1].transAxes, color=color)
    axes[1, 1].axis('off')
    
    plt.suptitle('优化版I-JEPA园林分类系统 - 学习曲线分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_pretrain_curves(history, save_path=None):
    """绘制预训练曲线
    
    Args:
        history: 预训练历史
        save_path: 保存路径（可选）
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    axes[0, 0].plot(history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_title('预训练损失曲线', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 余弦相似度曲线
    axes[0, 1].plot(history['cosine_similarity'], 'g-', linewidth=2)
    axes[0, 1].set_title('余弦相似度趋势', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('余弦相似度')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 特征多样性曲线
    axes[1, 0].plot(history['feature_diversity'], 'r-', linewidth=2)
    axes[1, 0].set_title('特征多样性', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('特征标准差')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1, 1].plot(history['learning_rate'], 'm-', linewidth=2)
    axes[1, 1].set_title('学习率变化', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('学习率')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('I-JEPA预训练监控', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径（可选）
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': '样本数'})
    
    plt.title('优化版I-JEPA - 混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 添加准确率信息
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(0.5, -0.15, f'总体准确率: {accuracy:.2%}', 
             transform=plt.gca().transAxes, ha='center', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_distribution(labels, class_names, save_path=None):
    """绘制类别分布图
    
    Args:
        labels: 标签列表（可以是字符串或整数）
        class_names: 类别名称列表
        save_path: 保存路径（可选）
    """
    # 统计类别分布
    unique, counts = np.unique(labels, return_counts=True)
    
    # 创建条形图
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(unique)), counts)
    
    # 设置颜色
    colors = plt.cm.viridis(counts / max(counts))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # 添加数值标签
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts)*0.01, str(count), 
                ha='center', va='bottom', fontsize=10)
    
    plt.title('各园林类别图像数量分布', fontsize=14, fontweight='bold')
    plt.xlabel('园林类别', fontsize=12)
    plt.ylabel('图像数量', fontsize=12)
    
    # 处理x轴标签
    # 如果labels是字符串，unique也是字符串，直接使用
    # 如果labels是整数（编码后的），需要用class_names解码
    if isinstance(unique[0], (int, np.integer)):
        # labels是编码后的整数
        x_labels = [class_names[i] for i in unique]
    else:
        # labels是原始字符串
        x_labels = unique
    
    plt.xticks(range(len(unique)), x_labels, rotation=45, ha='right')
    
    # 添加平均线
    mean_count = np.mean(counts)
    plt.axhline(y=mean_count, color='r', linestyle='--', alpha=0.7, 
                label=f'平均数量: {mean_count:.0f}')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_attention_weights(attention_weights, save_path=None):
    """可视化注意力权重
    
    Args:
        attention_weights: 注意力权重矩阵
        save_path: 保存路径（可选）
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='hot', cbar=True)
    plt.title('注意力权重可视化', fontsize=14, fontweight='bold')
    plt.xlabel('Key位置')
    plt.ylabel('Query位置')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()