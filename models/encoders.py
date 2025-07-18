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
    """I-JEPA预测器
    
    基于上下文信息预测目标patch的表示
    """
    def __init__(self, embed_dim=768, predictor_depth=6, n_heads=12):
        super().__init__()
        
        # 使用较浅的Transformer结构
        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, n_heads)
            for _ in range(predictor_depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, target_patches):
        # 通过Transformer层
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # 使用CLS token作为全局上下文进行预测
        batch_size = x.size(0)
        predictions = []
        
        for i in range(batch_size):
            cls_token = x[i, 0]
            predictions.append(cls_token)
        
        return torch.stack(predictions)


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