"""
预训练模块 - I-JEPA自监督预训练
"""
import torch
import torch.optim as optim
import numpy as np
import time
import math
import os
from config import Config
from models.encoders import update_target_encoder
from data.dataset import create_optimized_context_and_target_patches
from utils.losses import compute_enhanced_loss


def create_optimizer(model, learning_rate, weight_decay, betas):
    """创建优化器
    
    Args:
        model: 模型
        learning_rate: 学习率
        weight_decay: 权重衰减
        betas: Adam的beta参数
        
    Returns:
        optimizer: 优化器
    """
    # 分离需要和不需要权重衰减的参数
    param_groups = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'bias' not in n and p.requires_grad], 
            'weight_decay': weight_decay
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if 'bias' in n and p.requires_grad], 
            'weight_decay': 0.0
        }
    ]
    
    optimizer = optim.AdamW(param_groups, lr=learning_rate, betas=betas)
    return optimizer


def create_scheduler(optimizer, num_epochs, warmup_epochs):
    """创建学习率调度器
    
    Args:
        optimizer: 优化器
        num_epochs: 总训练轮数
        warmup_epochs: 预热轮数
        
    Returns:
        scheduler: 学习率调度器
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(model, train_loader, optimizer, device, epoch, log_interval=100):
    """训练一个epoch
    
    Args:
        model: I-JEPA模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        log_interval: 日志打印间隔
        
    Returns:
        avg_loss: 平均损失
        avg_cosine_sim: 平均余弦相似度
        avg_feature_std: 平均特征标准差
    """
    model.train()
    epoch_loss = 0.0
    cosine_similarities = []
    feature_stds = []
    
    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        batch_size = images.size(0)
        
        # 生成掩码策略
        context_patches, target_patches = create_optimized_context_and_target_patches(
            batch_size, model.n_patches, mask_ratio=Config.MASK_RATIO
        )
        
        optimizer.zero_grad()
        
        try:
            # 前向传播
            predictions, targets = model(images, context_patches, target_patches)
            
            # 计算损失
            loss, cosine_sim = compute_enhanced_loss(predictions, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRADIENT_CLIP)
            
            # 优化器步骤
            optimizer.step()
            
            # 更新目标编码器（EMA）
            momentum = Config.EMA_MOMENTUM + (Config.EMA_MOMENTUM_END - Config.EMA_MOMENTUM) * epoch / Config.PRETRAIN_EPOCHS
            update_target_encoder(model.context_encoder, model.target_encoder, momentum=momentum)
            
            # 记录指标
            epoch_loss += loss.item()
            cosine_similarities.append(cosine_sim.item())
            
            # 计算特征多样性
            import torch.nn.functional as F
            pred_std = torch.std(F.normalize(predictions, dim=-1), dim=0).mean()
            feature_stds.append(pred_std.item())
            
            # 打印日志
            if batch_idx % log_interval == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, Cosine Sim: {cosine_sim.item():.4f}, '
                      f'Feature Std: {pred_std.item():.4f}')
                
        except Exception as e:
            print(f"训练步骤出错: {e}")
            continue
    
    # 计算平均值
    avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0
    avg_cosine_sim = np.mean(cosine_similarities) if cosine_similarities else 0
    avg_feature_std = np.mean(feature_stds) if feature_stds else 0
    
    return avg_loss, avg_cosine_sim, avg_feature_std


def optimized_pretrain_ijepa(model, train_loader, num_epochs=None, learning_rate=None):
    """优化的I-JEPA预训练
    
    Args:
        model: I-JEPA模型
        train_loader: 训练数据加载器
        num_epochs: 训练轮数（默认使用配置）
        learning_rate: 学习率（默认使用配置）
        
    Returns:
        model: 训练后的模型
        history: 训练历史
    """
    # 使用配置中的默认值
    if num_epochs is None:
        num_epochs = Config.PRETRAIN_EPOCHS
    if learning_rate is None:
        learning_rate = Config.PRETRAIN_LR
        
    device = Config.DEVICE
    model = model.to(device)
    
    # 创建优化器和调度器
    optimizer = create_optimizer(
        model, learning_rate, 
        Config.PRETRAIN_WEIGHT_DECAY, 
        Config.PRETRAIN_BETAS
    )
    scheduler = create_scheduler(optimizer, num_epochs, Config.WARMUP_EPOCHS)
    
    print("🚀 开始优化版I-JEPA预训练...")
    
    # 训练历史
    history = {
        'loss': [],
        'cosine_similarity': [],
        'feature_diversity': [],
        'learning_rate': []
    }
    
    best_loss = float('inf')
    best_cosine_sim = 0.0
    patience = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 训练一个epoch
        avg_loss, avg_cosine_sim, avg_feature_std = train_one_epoch(
            model, train_loader, optimizer, device, epoch, Config.LOG_INTERVAL
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['loss'].append(avg_loss)
        history['cosine_similarity'].append(avg_cosine_sim)
        history['feature_diversity'].append(avg_feature_std)
        history['learning_rate'].append(current_lr)
        
        # 打印epoch总结
        print(f'📊 Epoch {epoch+1}/{num_epochs}:')
        print(f'  平均损失: {avg_loss:.4f}')
        print(f'  余弦相似度: {avg_cosine_sim:.4f}')
        print(f'  特征多样性: {avg_feature_std:.4f}')
        print(f'  学习率: {current_lr:.2e}')
        print('-' * 60)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_cosine_sim = avg_cosine_sim
            patience = 0
            
            # 保存检查点
            save_pretrain_checkpoint(
                model, optimizer, epoch, avg_loss, 
                avg_cosine_sim, avg_feature_std, time.time() - start_time
            )
            print(f"✅ 保存最佳模型，损失: {avg_loss:.4f}, 余弦相似度: {avg_cosine_sim:.4f}")
        else:
            patience += 1
            
        # 早停
        if patience >= Config.PATIENCE_PRETRAIN:
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    total_training_time = time.time() - start_time
    
    # 加载最佳模型
    model = load_pretrain_checkpoint(model)
    
    print(f"🎉 优化版I-JEPA预训练完成! 总耗时: {total_training_time:.1f}秒")
    
    return model, history


def save_pretrain_checkpoint(model, optimizer, epoch, loss, cosine_sim, feature_std, training_time):
    """保存预训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 损失值
        cosine_sim: 余弦相似度
        feature_std: 特征标准差
        training_time: 训练时间
    """
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'cosine_similarity': cosine_sim,
        'feature_diversity': feature_std,
        'training_time': training_time
    }
    
    # 确保目录存在
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # 保存检查点
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, Config.PRETRAIN_MODEL_PATH)
    torch.save(checkpoint, checkpoint_path)


def load_pretrain_checkpoint(model, checkpoint_path=None):
    """加载预训练检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径（默认使用配置）
        
    Returns:
        model: 加载了权重的模型
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, Config.PRETRAIN_MODEL_PATH)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载预训练模型，损失: {checkpoint['loss']:.4f}")
        return model
    except Exception as e:
        print(f"❌ 加载预训练模型失败: {e}")
        return model