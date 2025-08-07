"""
预训练模型架构 - 替代原始的I-JEPA
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from config import Config


class PretrainedVisionModel(nn.Module):
    """使用预训练模型的视觉分类器
    
    这个模型将替代原来的I-JEPA + Classifier组合
    """
    def __init__(self, model_name='vit_base_patch16_224', num_classes=35, 
                 feature_dim=768, dropout=0.5, freeze_layers=6):
        super().__init__()
        
        # 加载预训练模型
        self.backbone = timm.create_model(
            model_name, 
            pretrained=True,
            num_classes=0,  # 移除原始分类头
            global_pool=''  # 我们自己处理池化
        )
        
        # 获取实际的特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            if dummy_features.dim() == 3:  # [B, N, D]
                self.feature_dim = dummy_features.shape[-1]
            else:  # [B, D]
                self.feature_dim = dummy_features.shape[1]
        
        # 冻结前几层（节省计算资源）
        if hasattr(self.backbone, 'blocks'):  # Vision Transformer
            for i, block in enumerate(self.backbone.blocks):
                if i < freeze_layers:
                    for param in block.parameters():
                        param.requires_grad = False
        elif hasattr(self.backbone, 'features'):  # CNN
            # 冻结前面的卷积层
            for i, layer in enumerate(self.backbone.features):
                if i < len(self.backbone.features) // 2:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # 特征增强层（类似原来的feature_enhancer）
        self.feature_enhancer = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 注意力池化（处理序列输出）
        self.attention_pool = nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True
        )
        
        # 分类器（类似原来的AdvancedClassifier）
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.BatchNorm1d(feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_dim // 4, num_classes)
        )
        
        # 辅助分类器（用于深监督）
        self.aux_classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, x, return_features=False):
        # 提取特征
        features = self.backbone(x)
        
        # 处理不同维度的输出
        if features.dim() == 3:  # [B, N, D] - Transformer输出
            # 使用注意力池化
            pooled_features, _ = self.attention_pool(features, features, features)
            # 取CLS token或平均池化
            if hasattr(self.backbone, 'cls_token'):
                main_features = pooled_features[:, 0]  # CLS token
            else:
                main_features = pooled_features.mean(dim=1)  # 平均池化
        else:  # [B, D] - CNN输出
            main_features = features
        
        # 特征增强
        enhanced_features = self.feature_enhancer(main_features)
        
        # 主分类
        logits = self.classifier(enhanced_features)
        
        # 辅助分类（训练时使用）
        aux_logits = None
        if self.training:
            aux_logits = self.aux_classifier(enhanced_features)
        
        if return_features:
            return logits, aux_logits, enhanced_features
        return logits, aux_logits
    
    def encode(self, x):
        """兼容原来的encode接口"""
        with torch.no_grad():
            features = self.backbone(x)
            if features.dim() == 2:
                features = features.unsqueeze(1)  # [B, D] -> [B, 1, D]
        return features


class MultiModelEnsemble(nn.Module):
    """多模型集成 - 使用多个不同的预训练模型"""
    def __init__(self, num_classes=35):
        super().__init__()
        
        # 使用不同的预训练模型
        self.models = nn.ModuleDict({
            'dino': PretrainedVisionModel('vit_base_patch16_224', num_classes),
            'clip': PretrainedVisionModel('vit_base_patch16_clip_224.openai', num_classes),
            'convnext': PretrainedVisionModel('convnext_base.fb_in22k_ft_in1k', num_classes),
        })
        
        # 学习每个模型的权重
        self.model_weights = nn.Parameter(torch.ones(len(self.models)))
        
    def forward(self, x):
        outputs = []
        aux_outputs = []
        
        for name, model in self.models.items():
            if model.training:
                logits, aux_logits = model(x)
                aux_outputs.append(aux_logits)
            else:
                logits, _ = model(x)
            outputs.append(logits)
        
        # 加权平均
        weights = F.softmax(self.model_weights, dim=0)
        weighted_outputs = []
        for i, output in enumerate(outputs):
            weighted_outputs.append(output * weights[i])
        
        final_output = torch.stack(weighted_outputs).sum(dim=0)
        
        if self.training and aux_outputs:
            aux_output = torch.stack(aux_outputs).mean(dim=0)
            return final_output, aux_output
        
        return final_output, None


class LightweightModel(nn.Module):
    """轻量级模型 - 适合P100资源限制"""
    def __init__(self, num_classes=35):
        super().__init__()
        
        # 使用EfficientNet作为backbone
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=True,
            num_classes=0,
            drop_rate=0.3,
            drop_path_rate=0.2
        )
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        
        # GeM池化（更好的特征聚合）
        self.gem = GeM(p=3.0)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # 特征提取
        features = self.backbone(x)
        
        # 池化（如果需要）
        if features.dim() == 4:
            features = self.gem(features).squeeze(-1).squeeze(-1)
        
        # 分类
        logits = self.classifier(features)
        
        return logits, None
    
    def encode(self, x):
        """兼容接口"""
        with torch.no_grad():
            features = self.backbone(x)
            if features.dim() == 4:
                features = self.gem(features).squeeze(-1).squeeze(-1)
            return features.unsqueeze(1)


class GeM(nn.Module):
    """Generalized Mean Pooling"""
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean((-2, -1)).pow(1. / self.p)