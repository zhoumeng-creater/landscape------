"""
I-JEPA主模型 - Image-based Joint-Embedding Predictive Architecture
"""
import torch
import torch.nn as nn
from .encoders import AdvancedContextEncoder, TargetEncoder, Predictor


class OptimizedIJEPAModel(nn.Module):
    """优化的I-JEPA模型
    
    实现了基于图像的联合嵌入预测架构，通过预测被遮挡的图像块来学习表示
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 depth=12, n_heads=12, predictor_depth=6):
        super().__init__()
        
        # 上下文编码器 - 编码可见的图像块
        self.context_encoder = AdvancedContextEncoder(
            img_size, patch_size, embed_dim, depth, n_heads
        )
        
        # 目标编码器 - 编码目标图像块（EMA更新）
        self.target_encoder = TargetEncoder(self.context_encoder)
        
        # 预测器 - 基于上下文预测目标
        self.predictor = Predictor(embed_dim, predictor_depth, n_heads)
        
        self.patch_size = patch_size
        self.n_patches = self.context_encoder.n_patches
        
def forward(self, x, context_patches, target_patches):
    """前向传播
    
    Args:
        x: 输入图像 [B, C, H, W]
        context_patches: 每个样本的可见patch索引列表
        target_patches: 每个样本的目标patch索引列表
        
    Returns:
        predictions: 预测的目标表示
        targets: 真实的目标表示
    """
    # 上下文编码 - 只编码可见的patches
    context_repr = self.context_encoder(x, context_patches)
    
    # 目标编码 - 使用完整图像
    with torch.no_grad():
        target_repr = self.target_encoder(x)
    
    # 预测目标表示
    predictions = self.predictor(context_repr, target_patches)
    
    # 获取对应的目标表示 - 修改这部分
    batch_size = target_repr.size(0)
    targets = []
    
    for i in range(batch_size):
        if i < len(target_patches) and target_patches[i] is not None and len(target_patches[i]) > 0:
            # 获取多个目标patches
            batch_targets = []
            for target_idx in target_patches[i]:
                # 考虑CLS token，所以索引要+1
                if target_idx + 1 < target_repr.size(1):
                    batch_targets.append(target_repr[i, target_idx + 1])
                else:
                    batch_targets.append(target_repr[i, 1])  # 默认使用第一个patch
            targets.append(torch.stack(batch_targets))
        else:
            # 如果没有有效的目标patches，创建与predictions相同数量的默认目标
            num_targets = predictions.size(1) if predictions.dim() == 3 else 1
            default_targets = [target_repr[i, 1] for _ in range(num_targets)]
            targets.append(torch.stack(default_targets))
    
    targets = torch.stack(targets)
    
    # 确保predictions和targets的形状匹配
    if predictions.dim() == 3 and targets.dim() == 3:
        # 如果都是3D张量，展平为2D进行损失计算
        predictions = predictions.reshape(-1, predictions.size(-1))
        targets = targets.reshape(-1, targets.size(-1))
    
    return predictions, targets
    
    def encode(self, x):
        """提取特征表示用于下游任务
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            features: 编码后的特征表示 [B, n_patches+1, embed_dim]
        """
        with torch.no_grad():
            features = self.context_encoder(x)
        return features
    
    def get_params_groups(self, weight_decay=0.05):
        """获取参数组，用于设置不同的权重衰减
        
        Args:
            weight_decay: 权重衰减系数
            
        Returns:
            param_groups: 参数组列表
        """
        # 不对偏置和归一化层参数使用权重衰减
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return param_groups
    
    def get_num_params(self, trainable_only=True):
        """获取模型参数数量
        
        Args:
            trainable_only: 是否只计算可训练参数
            
        Returns:
            num_params: 参数数量
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())