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
        device = context_features.device
        
        # 编码上下文
        x = context_features
        for block in self.encoder_blocks:
            x = block(x)
        
        # 收集所有预测
        all_predictions = []
        
        for i in range(batch_size):
            if i < len(target_positions) and isinstance(target_positions[i], list) and len(target_positions[i]) > 0:
                # 为当前样本的目标位置创建查询
                curr_target_positions = target_positions[i]
                num_targets = len(curr_target_positions)
                
                # 获取目标位置的空间编码
                target_queries = []
                for pos in curr_target_positions:
                    if pos < self.spatial_pos_embed.size(1):
                        target_queries.append(self.spatial_pos_embed[0, pos, :])
                    else:
                        # 如果位置超出范围，使用默认编码
                        target_queries.append(self.spatial_pos_embed[0, 0, :])
                
                target_queries = torch.stack(target_queries).unsqueeze(0)  # [1, num_targets, embed_dim]
                
                # 对当前样本进行交叉注意力预测
                curr_context = x[i:i+1]  # [1, seq_len, embed_dim]
                
                for cross_attn, decoder in zip(self.cross_attention, self.decoder_blocks):
                    # 交叉注意力：目标查询关注上下文
                    target_queries, _ = cross_attn(target_queries, curr_context, curr_context)
                    # 自注意力refinement
                    target_queries = decoder(target_queries)
                
                predictions = self.norm(target_queries)  # [1, num_targets, embed_dim]
                all_predictions.append(predictions[0])  # [num_targets, embed_dim]
            else:
                # 如果没有有效的目标位置，返回一个默认预测
                default_pred = self.norm(x[i, 0:1, :])  # 使用第一个位置作为默认
                all_predictions.append(default_pred[0])
        
        # 将所有预测拼接成一个大张量
        # 由于每个样本的目标数量可能不同，我们需要将它们展平
        predictions = torch.cat(all_predictions, dim=0)  # [total_targets, embed_dim]
        
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