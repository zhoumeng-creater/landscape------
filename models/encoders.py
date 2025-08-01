"""
编码器模块 - 包含上下文编码器、目标编码器和预测器
"""
import torch
import torch.nn as nn
from copy import deepcopy
from .layers import ImprovedPatchEmbedding, ImprovedTransformerBlock


class AdvancedContextEncoder(nn.Module):
    """高级上下文编码器
    
    用于编码输入图像的上下文信息，是I-JEPA的核心组件
    """
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, 
                 depth=12, n_heads=12, mlp_ratio=4):
        super().__init__()
        self.patch_embed = ImprovedPatchEmbedding(
            img_size, patch_size, embed_dim=embed_dim
        )
        self.n_patches = self.patch_embed.n_patches
        
        # 位置嵌入
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer层（使用随机深度）
        drop_path_rates = [x.item() for x in torch.linspace(0, 0.1, depth)]
        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(
                embed_dim, n_heads, mlp_ratio, drop_path=drop_path_rates[i]
            ) for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
        # 特征增强层
        self.feature_enhancer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 添加特征融合层
        self.fusion_layers = nn.ModuleList([
            nn.Linear(embed_dim * 2, embed_dim) 
            for _ in range(depth // 3)  # 每3层一个融合
        ])
        
        # 保存中间特征
        self.intermediate_features = []

    def forward(self, x, context_patches=None):
        B = x.shape[0]
        
        # 分块嵌入
        x = self.patch_embed(x)
        
        # 处理上下文patches
        if context_patches is not None:
            context_x = []
            context_pos = []
            max_len = 0
            
            for i in range(B):
                if len(context_patches[i]) > 0 and max(context_patches[i]) < x.shape[1]:
                    patches = x[i, context_patches[i]]
                    pos_embed = self.pos_embed[0, context_patches[i]]
                    context_x.append(patches)
                    context_pos.append(pos_embed)
                    max_len = max(max_len, len(patches))
                else:
                    # 如果上下文patches无效，使用所有patches
                    context_x.append(x[i])
                    context_pos.append(self.pos_embed[0])
                    max_len = max(max_len, x.shape[1])
            
            # 填充到相同长度
            padded_x = []
            padded_pos = []
            
            for ctx, pos in zip(context_x, context_pos):
                if len(ctx) < max_len:
                    padding_len = max_len - len(ctx)
                    padding = torch.zeros(padding_len, ctx.shape[-1], device=ctx.device)
                    ctx = torch.cat([ctx, padding], dim=0)
                    pos = torch.cat([pos, padding], dim=0)
                padded_x.append(ctx)
                padded_pos.append(pos)
            
            x = torch.stack(padded_x)
            pos_embed = torch.stack(padded_pos)
            x = x + pos_embed
        else:
            x = x + self.pos_embed
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.dropout(x)
        
        # 通过Transformer层
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 特征增强
        cls_features = self.feature_enhancer(x[:, 0])
        x = torch.cat([cls_features.unsqueeze(1), x[:, 1:]], dim=1)
        
        return x


class TargetEncoder(nn.Module):
    """I-JEPA目标编码器（EMA更新）
    
    使用指数移动平均(EMA)更新的编码器，用于生成预测目标
    """
    def __init__(self, context_encoder):
        super().__init__()
        # 深拷贝上下文编码器
        self.encoder = deepcopy(context_encoder)
        
        # 冻结参数，不参与梯度计算
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.encoder(x)


class Predictor(nn.Module):
    """简化版预测器 - 更稳定和可预测"""
    def __init__(self, embed_dim=768, predictor_depth=6, n_heads=12):
        super().__init__()
        
        # 使用简单的Transformer层堆叠
        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, n_heads)
            for _ in range(predictor_depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, context_features, target_positions):
        """
        Args:
            context_features: 上下文特征 [B, seq_len, embed_dim]
            target_positions: 目标位置列表 (每个batch的目标位置)
            
        Returns:
            predictions: 预测的目标特征 [total_targets, embed_dim]
        """
        batch_size = context_features.size(0)
        device = context_features.device
        
        # 通过Transformer层处理上下文
        x = context_features
        for block in self.blocks:
            x = block(x)
        
        # 归一化
        x = self.norm(x)
        
        # 收集目标位置的预测
        all_predictions = []
        
        for i in range(batch_size):
            if i < len(target_positions) and isinstance(target_positions[i], list) and len(target_positions[i]) > 0:
                # 获取指定位置的特征
                for pos in target_positions[i]:
                    # 考虑CLS token，位置需要+1
                    actual_pos = pos + 1
                    if actual_pos < x.size(1):
                        all_predictions.append(x[i, actual_pos])
                    else:
                        # 位置超出范围，使用第一个非CLS位置
                        all_predictions.append(x[i, 1])
            else:
                # 没有指定目标位置，使用第一个非CLS位置
                all_predictions.append(x[i, 1])
        
        # 将所有预测堆叠成一个张量
        predictions = torch.stack(all_predictions, dim=0)  # [total_targets, embed_dim]
        
        return predictions


def update_target_encoder(context_encoder, target_encoder, momentum=0.996):
    """EMA更新目标编码器
    
    Args:
        context_encoder: 上下文编码器（参与梯度更新）
        target_encoder: 目标编码器（EMA更新）
        momentum: EMA动量系数
    """
    with torch.no_grad():
        for param_c, param_t in zip(
            context_encoder.parameters(), 
            target_encoder.encoder.parameters()
        ):
            param_t.data.mul_(momentum).add_(param_c.data, alpha=1.0 - momentum)