"""
分类器模块 - 用于下游分类任务
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvancedClassifier(nn.Module):
    """高级分类器 - 多层感知机 + 注意力池化
    
    用于将I-JEPA编码的特征映射到类别标签
    """
    def __init__(self, embed_dim=768, num_classes=35, dropout=0.5):
        super().__init__()
        
        # 特征预处理
        self.feature_norm = nn.LayerNorm(embed_dim)
        
        # 注意力池化层 - 聚合序列特征
        self.attention_pool = nn.MultiheadAttention(
            embed_dim, num_heads=8, batch_first=True
        )
        
        # 多层MLP with 残差连接
        self.layers = nn.ModuleList([
            # 第一层：特征变换
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # 第二层：特征提取
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.BatchNorm1d(embed_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            ),
            # 第三层：特征压缩
            nn.Sequential(
                nn.Linear(embed_dim // 2, embed_dim // 4),
                nn.BatchNorm1d(embed_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout // 2)
            )
        ])
        
        # 分类层
        self.classifier = nn.Linear(embed_dim // 4, num_classes)
        
        # 残差连接的投影层
        self.skip_connections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Linear(embed_dim // 2, embed_dim // 4)
        ])
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入特征 
               - 如果是3D: [B, seq_len, embed_dim] (来自I-JEPA编码器)
               - 如果是2D: [B, embed_dim] (已经池化的特征)
               
        Returns:
            logits: 分类logits [B, num_classes]
        """
        # 处理序列输入（如果是3D）
        if x.dim() == 3:
            # 使用注意力池化聚合序列特征
            pooled, _ = self.attention_pool(x, x, x)
            # 平均池化作为最终特征
            x = pooled.mean(dim=1)
        
        # 特征归一化
        x = self.feature_norm(x)
        
        # 逐层处理 + 残差连接
        features = x
        for i, (layer, skip) in enumerate(zip(self.layers, self.skip_connections)):
            # 计算残差
            residual = skip(features)
            # 通过主层
            features = layer(features)
            
            # 残差连接（维度匹配时）
            if features.shape == residual.shape:
                features = features + residual
        
        # 最终分类
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x):
        """获取中间特征表示（用于可视化或其他目的）
        
        Args:
            x: 输入特征
            
        Returns:
            features: 分类前的特征表示
        """
        # 处理序列输入
        if x.dim() == 3:
            pooled, attention_weights = self.attention_pool(x, x, x)
            x = pooled.mean(dim=1)
        
        x = self.feature_norm(x)
        
        features = x
        for i, (layer, skip) in enumerate(zip(self.layers, self.skip_connections)):
            residual = skip(features)
            features = layer(features)
            if features.shape == residual.shape:
                features = features + residual
        
        return features
    
    def get_attention_weights(self, x):
        """获取注意力权重（用于可视化）
        
        Args:
            x: 输入特征 [B, seq_len, embed_dim]
            
        Returns:
            attention_weights: 注意力权重
        """
        if x.dim() == 3:
            _, attention_weights = self.attention_pool(x, x, x, need_weights=True)
            return attention_weights
        else:
            return None


class SimpleClassifier(nn.Module):
    """简单分类器 - 用于快速实验
    
    只包含一个线性层的简单分类器
    """
    def __init__(self, embed_dim=768, num_classes=35):
        super().__init__()
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # 如果输入是序列，取CLS token或平均池化
        if x.dim() == 3:
            x = x[:, 0]  # 使用CLS token
        return self.classifier(x)