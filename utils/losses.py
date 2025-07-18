"""
损失函数模块 - 包含各种损失函数实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


def compute_enhanced_loss(predictions, targets):
    """增强的损失函数 - 多任务学习
    
    结合多种损失函数来提高预训练效果
    
    Args:
        predictions: 预测的表示 [B, D]
        targets: 目标表示 [B, D]
        
    Returns:
        total_loss: 总损失
        cosine_sim: 平均余弦相似度（用于监控）
    """
    # 归一化
    predictions_norm = F.normalize(predictions, dim=-1)
    targets_norm = F.normalize(targets, dim=-1)
    
    # 1. 余弦相似度损失
    cosine_sim = F.cosine_similarity(predictions_norm, targets_norm)
    cosine_loss = -cosine_sim.mean()
    
    # 2. 特征多样性损失（防止模式坍塌）
    pred_std = torch.std(predictions_norm, dim=0).mean()
    target_std = torch.std(targets_norm, dim=0).mean()
    diversity_loss = -torch.log(pred_std + 1e-6) - torch.log(target_std + 1e-6)
    
    # 3. L2距离损失（特征对齐）
    l2_loss = F.mse_loss(predictions_norm, targets_norm)
    
    # 4. 对比学习损失（如果batch_size > 1）
    contrastive_loss = 0
    if predictions.size(0) > 1:
        try:
            # 计算batch内的相似度矩阵
            sim_matrix = torch.mm(predictions_norm, predictions_norm.t())
            target_sim_matrix = torch.mm(targets_norm, targets_norm.t())
            # 对齐两个相似度矩阵
            contrastive_loss = F.mse_loss(sim_matrix, target_sim_matrix)
        except:
            contrastive_loss = 0
    
    # 组合损失（使用配置中的权重）
    total_loss = (cosine_loss + 
                 Config.DIVERSITY_WEIGHT * diversity_loss + 
                 Config.L2_WEIGHT * l2_loss + 
                 Config.CONTRASTIVE_WEIGHT * contrastive_loss)
    
    return total_loss, cosine_sim.mean()


class FocalLoss(nn.Module):
    """Focal Loss - 用于处理类别不平衡
    
    通过降低易分类样本的权重，让模型更关注难分类样本
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失
    
    通过软化标签来提高模型的泛化能力
    """
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # 创建平滑的标签分布
        smooth_labels = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        smooth_labels = smooth_labels * (1 - self.epsilon) + self.epsilon / n_classes
        
        # 计算损失
        loss = -(smooth_labels * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MixupCrossEntropy(nn.Module):
    """Mixup交叉熵损失
    
    用于Mixup数据增强时的损失计算
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, targets_a, targets_b, lam):
        """
        Args:
            inputs: 模型输出
            targets_a: 第一个样本的标签
            targets_b: 第二个样本的标签
            lam: 混合系数
        """
        loss_a = F.cross_entropy(inputs, targets_a)
        loss_b = F.cross_entropy(inputs, targets_b)
        return lam * loss_a + (1 - lam) * loss_b


class ContrastiveLoss(nn.Module):
    """对比学习损失
    
    用于自监督学习中的特征学习
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels=None):
        """
        Args:
            features: 特征向量 [B, D]
            labels: 标签（可选）[B]
        """
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建mask（排除对角线）
        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool).to(features.device)
        
        if labels is not None:
            # 监督对比学习
            labels = labels.unsqueeze(0) == labels.unsqueeze(1)
            mask = mask | ~labels
        
        # 计算损失
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim.masked_fill(mask, 0)
        
        pos_sim = exp_sim.diag()
        neg_sim = exp_sim.sum(dim=1) - pos_sim
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        
        return loss.mean()


def get_loss_function(loss_type='cross_entropy', **kwargs):
    """获取损失函数
    
    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数
        
    Returns:
        loss_fn: 损失函数实例
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_type == 'label_smoothing':
        epsilon = kwargs.get('epsilon', Config.LABEL_SMOOTHING)
        return LabelSmoothingCrossEntropy(epsilon=epsilon)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'contrastive':
        return ContrastiveLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")