"""
基础层模块 - 包含所有基础的神经网络层
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DropPath(nn.Module):
    """随机深度(Stochastic Depth) - 提高模型泛化能力
    
    在训练时随机丢弃整个残差分支，测试时使用所有分支
    """
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ImprovedPatchEmbedding(nn.Module):
    """改进的图像分块嵌入层 - 多尺度特征提取
    
    使用重叠的patch和多尺度投影来提取更丰富的特征
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 embed_dim=768, overlap_ratio=0.25):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 重叠patch的步长
        self.stride = int(patch_size * (1 - overlap_ratio))
        self.n_patches = ((img_size - patch_size) // self.stride + 1) ** 2
        
        # 主要投影层
        self.projection = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=self.stride, padding=0
        )
        
        # 多尺度辅助投影
        self.aux_projections = nn.ModuleList([
            nn.Conv2d(in_channels, embed_dim // 4, 
                     kernel_size=patch_size//2, stride=self.stride, padding=0),
            nn.Conv2d(in_channels, embed_dim // 4, 
                     kernel_size=patch_size*2, stride=self.stride*2, padding=0)
        ])
        
        # 特征融合层
        self.feature_fusion = nn.Linear(embed_dim + embed_dim//2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 主要特征
        main_features = self.projection(x)
        main_features = main_features.flatten(2).transpose(1, 2)
        
        # 多尺度辅助特征
        aux_features = []
        for proj in self.aux_projections:
            try:
                aux_feat = proj(x)
                # 调整大小以匹配主特征的patch数量
                target_size = int(main_features.shape[1] ** 0.5)
                aux_feat = F.adaptive_avg_pool2d(aux_feat, target_size)
                aux_feat = aux_feat.flatten(2).transpose(1, 2)
                aux_features.append(aux_feat)
            except:
                # 如果尺寸不匹配，跳过该尺度
                continue
        
        # 特征融合
        if aux_features:
            aux_features = torch.cat(aux_features, dim=-1)
            # 确保patch数量匹配
            if aux_features.shape[1] != main_features.shape[1]:
                aux_features = F.interpolate(
                    aux_features.transpose(1, 2), 
                    size=main_features.shape[1], 
                    mode='linear', align_corners=False
                ).transpose(1, 2)
            
            combined_features = torch.cat([main_features, aux_features], dim=-1)
            features = self.feature_fusion(combined_features)
        else:
            features = main_features
        
        features = self.norm(features)
        return features


class EnhancedMultiHeadAttention(nn.Module):
    """增强的多头注意力机制 - 添加相对位置编码
    
    在标准多头注意力基础上加入相对位置信息
    """
    def __init__(self, embed_dim, n_heads, dropout=0.1, max_relative_position=14):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.max_relative_position = max_relative_position
        
        assert self.head_dim * n_heads == embed_dim, "embed_dim必须被n_heads整除"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        # 相对位置编码
        self.relative_position_embeddings = nn.Embedding(
            2 * max_relative_position + 1, self.head_dim
        )
        
    def forward(self, q, k=None, v=None, return_attention=False):
        # 如果k和v为None，则使用q（自注意力）
        if k is None:
            k = q
        if v is None:
            v = q
            
        batch_size, seq_len = q.size(0), q.size(1)
        
        # 线性变换并分头
        Q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 添加相对位置编码
        if seq_len <= self.max_relative_position * 2 + 1:
            try:
                relative_positions = self._get_relative_positions(seq_len)
                relative_position_embeddings = self.relative_position_embeddings(relative_positions)
                relative_scores = torch.matmul(Q, relative_position_embeddings.transpose(-2, -1))
                scores = scores + relative_scores
            except:
                pass  # 如果相对位置编码失败，跳过
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # 应用注意力
        context = torch.matmul(attention, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        output = self.out(context)
        
        if return_attention:
            return output, attention
        return output
    
    def _get_relative_positions(self, seq_len):
        """生成相对位置矩阵"""
        device = next(self.parameters()).device
        positions = torch.arange(seq_len, device=device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = torch.clamp(
            relative_positions + self.max_relative_position,
            0, 2 * self.max_relative_position
        )
        return relative_positions


class ImprovedTransformerBlock(nn.Module):
    """改进的Transformer块 - 添加层尺度和随机深度
    
    包含注意力层、前馈网络、层归一化、残差连接等
    """
    def __init__(self, embed_dim, n_heads, mlp_ratio=4, dropout=0.1, 
                 drop_path=0.0, layer_scale=1e-4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EnhancedMultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 层尺度 - 用于稳定深层网络训练
        self.layer_scale_1 = nn.Parameter(torch.ones(embed_dim) * layer_scale)
        self.layer_scale_2 = nn.Parameter(torch.ones(embed_dim) * layer_scale)
        
        # 随机深度
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
    def forward(self, x):
        # 自注意力 + 层尺度 + 随机深度
        normalized_x = self.norm1(x)
        x = x + self.drop_path(self.layer_scale_1 * self.attn(normalized_x))
        
        # MLP + 层尺度 + 随机深度
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x
    
class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 (RoPE)"""
    def __init__(self, dim, max_seq_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('sin', freqs.sin())
        self.register_buffer('cos', freqs.cos())
    
    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        seq_len = x.shape[1]
        sin = self.sin[:seq_len, :].unsqueeze(0)
        cos = self.cos[:seq_len, :].unsqueeze(0)
        
        # 旋转操作
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
        
        return x * cos + x_rot * sin

class Position2DEmbedding(nn.Module):
    """2D位置编码 - 考虑patch的行列位置"""
    def __init__(self, embed_dim, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.row_embed = nn.Parameter(torch.randn(grid_size, embed_dim // 2))
        self.col_embed = nn.Parameter(torch.randn(grid_size, embed_dim // 2))
        
    def forward(self, x):
        # x: [batch_size, num_patches, embed_dim]
        batch_size, num_patches, _ = x.shape
        
        # 生成行列位置编码
        pos_embed = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos_embed.append(torch.cat([self.row_embed[i], self.col_embed[j]]))
        
        pos_embed = torch.stack(pos_embed).unsqueeze(0).expand(batch_size, -1, -1)
        return x + pos_embed