"""
混合I-JEPA架构 - 结合预训练模型和I-JEPA的优势
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.encoders import Predictor
from models.layers import ImprovedTransformerBlock

class HybridIJEPAModel(nn.Module):
    """混合I-JEPA模型 - 使用预训练模型作为backbone"""
    
    def __init__(self, pretrained_model_name='vit_base_patch16_224', 
                 embed_dim=768, predictor_depth=6, n_heads=12):
        super().__init__()
        
        # 1. 使用预训练模型作为基础编码器
        self.context_encoder = timm.create_model(
            pretrained_model_name,
            pretrained=True,
            num_classes=0,  # 移除分类头
            global_pool=''   # 保留序列输出
        )
        
        # 冻结预训练模型的前几层
        if hasattr(self.context_encoder, 'blocks'):
            for i, block in enumerate(self.context_encoder.blocks):
                if i < 6:  # 冻结前6层
                    for param in block.parameters():
                        param.requires_grad = False
        
        # 2. 目标编码器（EMA更新）
        self.target_encoder = timm.create_model(
            pretrained_model_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        # 冻结目标编码器
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # 3. I-JEPA预测器
        self.predictor = Predictor(embed_dim, predictor_depth, n_heads)
        
        # 4. 特征适配层（如果预训练模型的维度不同）
        self.adapter = nn.Identity()  # 默认不需要适配
        
        # 获取patch数量
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.context_encoder(dummy_input)
            self.n_patches = dummy_output.shape[1] - 1  # 减去CLS token
            
    def forward(self, x, context_patches=None, target_patches=None):
        """前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            context_patches: 可见patch索引（预训练时使用）
            target_patches: 目标patch索引（预训练时使用）
            
        Returns:
            如果是预训练：返回predictions, targets
            如果是推理：返回特征
        """
        if context_patches is not None and target_patches is not None:
            # 预训练模式
            return self.pretrain_forward(x, context_patches, target_patches)
        else:
            # 特征提取模式
            return self.encode(x)
    
    def pretrain_forward(self, x, context_patches, target_patches):
        """预训练前向传播"""
        batch_size = x.size(0)
        
        # 1. 获取所有patch的特征
        all_features = self.context_encoder(x)  # [B, N+1, D]
        
        # 2. 创建mask并提取context特征
        context_features = []
        for i in range(batch_size):
            # 获取CLS token
            cls_token = all_features[i:i+1, 0]
            
            # 获取context patches
            if len(context_patches[i]) > 0:
                ctx_patches = all_features[i, [p+1 for p in context_patches[i]]]
                # 组合CLS token和context patches
                ctx_feat = torch.cat([cls_token, ctx_patches], dim=0)
            else:
                ctx_feat = cls_token
                
            context_features.append(ctx_feat)
        
        # Pad到相同长度
        max_len = max(f.size(0) for f in context_features)
        padded_features = []
        for feat in context_features:
            if feat.size(0) < max_len:
                padding = torch.zeros(max_len - feat.size(0), feat.size(1), 
                                    device=feat.device)
                feat = torch.cat([feat, padding], dim=0)
            padded_features.append(feat)
        
        context_features = torch.stack(padded_features)  # [B, L, D]
        
        # 3. 使用目标编码器获取目标特征（无梯度）
        with torch.no_grad():
            target_features = self.target_encoder(x)  # [B, N+1, D]
        
        # 4. 使用预测器预测目标特征
        predictions = self.predictor(context_features, target_patches)
        
        # 5. 提取对应的目标特征
        targets = []
        for i in range(batch_size):
            if len(target_patches[i]) > 0:
                tgt_feats = target_features[i, [p+1 for p in target_patches[i]]]
                targets.append(tgt_feats)
            else:
                # 如果没有目标，使用第一个非CLS patch
                targets.append(target_features[i, 1:2])
        
        targets = torch.cat(targets, dim=0)  # [total_targets, D]
        
        return predictions, targets
    
    def encode(self, x):
        """编码图像获取特征（用于下游任务）"""
        with torch.no_grad():
            features = self.context_encoder(x)
        return features
    
    def update_target_encoder(self, momentum=0.996):
        """EMA更新目标编码器"""
        with torch.no_grad():
            for param_c, param_t in zip(
                self.context_encoder.parameters(),
                self.target_encoder.parameters()
            ):
                param_t.data.mul_(momentum).add_(param_c.data, alpha=1-momentum)


class PretrainedIJEPAClassifier(nn.Module):
    """使用预训练I-JEPA的分类器 - 修复版"""
    
    def __init__(self, ijepa_model=None, num_classes=35, feature_dim=768, dropout=0.5):
        super().__init__()
        self.ijepa = ijepa_model  # 可选的编码器
        
        # 注意力池化
        self.attention_pool = nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: 输入，可以是：
               - 图像 [B, C, H, W]
               - 特征 [B, N+1, D]
        
        Returns:
            logits: 分类结果 [B, num_classes]
        """
        # 判断输入类型
        if x.dim() == 4:  # 图像输入 [B, C, H, W]
            if self.ijepa is None:
                raise ValueError("需要ijepa_model来编码图像")
            features = self.ijepa.encode(x)
        elif x.dim() == 3:  # 特征输入 [B, N+1, D]
            features = x
        else:
            raise ValueError(f"不支持的输入维度: {x.dim()}")
        
        # 注意力池化
        pooled, _ = self.attention_pool(features, features, features)
        cls_features = pooled[:, 0]  # 使用CLS token
        
        # 分类
        logits = self.classifier(cls_features)
        return logits


def create_hybrid_ijepa(pretrained_model_name='vit_base_patch16_224', 
                       num_classes=35):
    """创建混合I-JEPA模型的便捷函数"""
    
    # 创建I-JEPA模型
    ijepa_model = HybridIJEPAModel(
        pretrained_model_name=pretrained_model_name,
        embed_dim=768,
        predictor_depth=6,
        n_heads=12
    )
    
    # 创建分类器
    classifier = PretrainedIJEPAClassifier(
        ijepa_model=ijepa_model,
        num_classes=num_classes,
        feature_dim=768,
        dropout=0.5
    )
    
    return ijepa_model, classifier


# 使用示例
if __name__ == "__main__":
    # 创建模型
    ijepa, classifier = create_hybrid_ijepa(
        pretrained_model_name='vit_base_patch16_224',
        num_classes=35
    )
    
    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # 测试预训练模式
    context_patches = [[0, 1, 2], [3, 4, 5]]
    target_patches = [[10, 11], [12, 13]]
    
    predictions, targets = ijepa(dummy_input, context_patches, target_patches)
    print(f"预训练模式 - Predictions shape: {predictions.shape}")
    print(f"预训练模式 - Targets shape: {targets.shape}")
    
    # 测试分类模式
    logits = classifier(dummy_input)
    print(f"分类模式 - Logits shape: {logits.shape}")
