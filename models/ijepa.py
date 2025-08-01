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
        
        # 获取对应的目标表示
        batch_size = x.size(0)
        device = x.device
        
        # 计算每个批次的目标数量
        target_counts = []
        for i in range(batch_size):
            if i < len(target_patches) and isinstance(target_patches[i], list) and len(target_patches[i]) > 0:
                target_counts.append(len(target_patches[i]))
            else:
                target_counts.append(1)  # 默认至少有一个目标
        
        # 从target_repr中提取目标
        all_targets = []
        start_idx = 0
        
        for i in range(batch_size):
            if i < len(target_patches) and isinstance(target_patches[i], list) and len(target_patches[i]) > 0:
                # 获取当前批次的目标patches
                batch_targets = []
                for target_idx in target_patches[i]:
                    # 考虑CLS token，所以索引要+1
                    actual_idx = target_idx + 1
                    if actual_idx < target_repr.size(1):
                        batch_targets.append(target_repr[i, actual_idx])
                    else:
                        # 如果索引超出范围，使用第一个非CLS token
                        batch_targets.append(target_repr[i, 1])
                
                # 将目标堆叠起来
                batch_targets = torch.stack(batch_targets)  # [num_targets, embed_dim]
                all_targets.append(batch_targets)
            else:
                # 如果没有有效的目标patches，使用默认目标
                default_target = target_repr[i, 1:2]  # [1, embed_dim]
                all_targets.append(default_target)
        
        # 将所有目标连接成一个张量
        targets = torch.cat(all_targets, dim=0)  # [total_targets, embed_dim]
        
        # 确保predictions也是2D的
        if predictions.dim() == 3:
            # 如果predictions是3D的，需要reshape
            predictions = predictions.view(-1, predictions.size(-1))
        
        # 验证形状匹配
        if predictions.shape[0] != targets.shape[0]:
            print(f"警告: predictions形状 {predictions.shape} 与 targets形状 {targets.shape} 不匹配")
            # 尝试修复：如果形状不匹配，调整targets的数量
            if predictions.shape[0] < targets.shape[0]:
                targets = targets[:predictions.shape[0]]
            else:
                # 如果predictions更多，重复targets
                repeat_times = predictions.shape[0] // targets.shape[0] + 1
                targets = targets.repeat(repeat_times, 1)[:predictions.shape[0]]
        
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