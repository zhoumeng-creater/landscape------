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
    """改进的预测器 - 使用交叉注意力和空间感知"""
    def __init__(self, embed_dim=768, predictor_depth=6, n_heads=12):
        super().__init__()
        
        # 使用编码器-解码器架构
        self.encoder_blocks = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, n_heads)
            for _ in range(predictor_depth // 2)
        ])
        
        # 交叉注意力层
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
            for _ in range(predictor_depth // 2)
        ])
        
        self.decoder_blocks = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, n_heads)
            for _ in range(predictor_depth // 2)
        ])
        
        # 空间位置编码
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 196, embed_dim))
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, context_features, target_positions):
        batch_size = context_features.size(0)
        
        # 编码上下文
        x = context_features
        for block in self.encoder_blocks:
            x = block(x)
        
        # 为目标位置创建查询
        num_targets = len(target_positions[0]) if isinstance(target_positions[0], list) else 1
        target_queries = self.spatial_pos_embed[:, target_positions[0], :].expand(batch_size, -1, -1)
        
        # 交叉注意力预测
        for cross_attn, decoder in zip(self.cross_attention, self.decoder_blocks):
            # 交叉注意力：目标查询关注上下文
            target_queries, _ = cross_attn(target_queries, x, x)
            # 自注意力refinement
            target_queries = decoder(target_queries)
        
        predictions = self.norm(target_queries)
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