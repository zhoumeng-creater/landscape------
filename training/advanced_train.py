"""
高级训练模块 - 支持预训练模型、SAM、混合精度等
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import time
from config import Config
from utils.losses import FocalLoss, MixupCrossEntropy


class SAMOptimizer:
    """SAM (Sharpness Aware Minimization) 优化器包装器"""
    def __init__(self, base_optimizer, rho=0.05):
        self.base_optimizer = base_optimizer
        self.rho = rho
        
    def first_step(self):
        """计算梯度并在参数空间中移动"""
        self.backup = {}
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.backup[p] = p.data.clone()
                grad_norm = p.grad.data.norm(2)
                epsilon = self.rho / (grad_norm + 1e-12)
                p.data.add_(p.grad.data, alpha=epsilon)
    
    def second_step(self):
        """恢复参数并使用SAM梯度更新"""
        for group in self.base_optimizer.param_groups:
            for p in group['params']:
                if p in self.backup:
                    p.data = self.backup[p]
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()


def create_optimizer(model, learning_rate):
    """创建优化器
    
    Args:
        model: 模型
        learning_rate: 学习率
        
    Returns:
        optimizer: 优化器
        use_sam: 是否使用SAM
    """
    # 分离不同部分的参数
    backbone_params = []
    head_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'backbone' in name:
            backbone_params.append(param)
        elif 'classifier' in name or 'aux_classifier' in name:
            head_params.append(param)
        else:
            other_params.append(param)
    
    # 不同部分使用不同的学习率
    param_groups = [
        {'params': backbone_params, 'lr': learning_rate * 0.1, 'weight_decay': 0.01},
        {'params': head_params, 'lr': learning_rate, 'weight_decay': 0.001},
        {'params': other_params, 'lr': learning_rate * 0.5, 'weight_decay': 0.01}
    ]
    
    # 选择优化器
    if Config.OPTIMIZER_TYPE == 'adamw':
        base_optimizer = optim.AdamW(param_groups)
    elif Config.OPTIMIZER_TYPE == 'lamb':
        # 需要安装: pip install torch-optimizer
        from torch_optimizer import Lamb
        base_optimizer = Lamb(param_groups)
    else:
        base_optimizer = optim.Adam(param_groups)
    
    # 是否使用SAM
    if Config.USE_SAM:
        return SAMOptimizer(base_optimizer, Config.SAM_RHO), True
    else:
        return base_optimizer, False


def mixup_data(x, y, alpha=1.0):
    """Mixup数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, epoch, 
                   use_sam=False, accumulation_steps=1):
    """训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scaler: 混合精度缩放器
        epoch: 当前epoch
        use_sam: 是否使用SAM
        accumulation_steps: 梯度累积步数
        
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Mixup损失函数
    mixup_criterion = MixupCrossEntropy() if Config.USE_MIXUP else None
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
        
        # Mixup数据增强
        if Config.USE_MIXUP and np.random.random() > 0.5:
            data, target_a, target_b, lam = mixup_data(data, target, Config.MIXUP_ALPHA)
            use_mixup = True
        else:
            use_mixup = False
        
        # 混合精度训练
        with autocast(enabled=Config.USE_AMP):
            output = model(data)
            
            # 处理不同的输出格式
            if isinstance(output, tuple):
                logits, aux_logits = output
            else:
                logits, aux_logits = output, None
            
            # 计算损失
            if use_mixup:
                loss = mixup_criterion(logits, target_a, target_b, lam)
            else:
                loss = criterion(logits, target)
            
            # 添加辅助损失
            if aux_logits is not None and Config.USE_AUXILIARY_LOSS:
                if use_mixup:
                    aux_loss = mixup_criterion(aux_logits, target_a, target_b, lam)
                else:
                    aux_loss = criterion(aux_logits, target)
                loss = loss + Config.AUX_LOSS_WEIGHT * aux_loss
            
            # 梯度累积
            loss = loss / accumulation_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积步骤
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪
            scaler.unscale_(optimizer if not use_sam else optimizer.base_optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
            
            if use_sam:
                # SAM第一步
                scaler.step(optimizer.base_optimizer)
                scaler.update()
                optimizer.first_step()
                
                # SAM第二步：重新计算梯度
                optimizer.base_optimizer.zero_grad()
                with autocast(enabled=Config.USE_AMP):
                    output2 = model(data)
                    if isinstance(output2, tuple):
                        logits2, aux_logits2 = output2
                    else:
                        logits2 = output2
                    
                    if use_mixup:
                        loss2 = mixup_criterion(logits2, target_a, target_b, lam)
                    else:
                        loss2 = criterion(logits2, target)
                    loss2 = loss2 / accumulation_steps
                
                scaler.scale(loss2).backward()
                scaler.unscale_(optimizer.base_optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP)
                optimizer.second_step()
            else:
                # 普通优化器步骤
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        
        # 统计
        total_loss += loss.item() * accumulation_steps
        if not use_mixup:
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # 打印进度
        if batch_idx % Config.LOG_INTERVAL == 0:
            if use_mixup:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item()*accumulation_steps:.4f} (Mixup)')
            else:
                acc = 100 * correct / total if total > 0 else 0
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item()*accumulation_steps:.4f}, Acc: {acc:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion):
    """验证模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(Config.DEVICE), target.to(Config.DEVICE)
            
            with autocast(enabled=Config.USE_AMP):
                output = model(data)
                
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output
                
                loss = criterion(logits, target)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def train_pretrained_model(model, train_loader, val_loader, num_epochs, num_classes):
    """训练预训练模型
    
    Args:
        model: 预训练模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        num_classes: 类别数
        
    Returns:
        model: 训练后的模型
        history: 训练历史
    """
    device = Config.DEVICE
    model = model.to(device)
    
    # 损失函数
    if Config.LABEL_SMOOTHING > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer, use_sam = create_optimizer(model, Config.FINETUNE_LR)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer if not use_sam else optimizer.base_optimizer,
        max_lr=Config.FINETUNE_LR,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader) // Config.GRADIENT_ACCUMULATION_STEPS,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # 混合精度训练
    scaler = GradScaler(enabled=Config.USE_AMP)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    start_time = time.time()
    
    print("🚀 开始训练预训练模型...")
    print(f"  使用SAM: {use_sam}")
    print(f"  使用混合精度: {Config.USE_AMP}")
    print(f"  梯度累积步数: {Config.GRADIENT_ACCUMULATION_STEPS}")
    
    for epoch in range(num_epochs):
        # 动态解冻层（渐进式解冻）
        if epoch == Config.FINETUNE_STAGE2_EPOCH:
            print(f"\n🔄 Epoch {epoch}: 解冻更多层")
            unfreeze_layers(model, stage=2)
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, 
            epoch, use_sam, Config.GRADIENT_ACCUMULATION_STEPS
        )
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        current_lr = optimizer.param_groups[0]['lr'] if not use_sam else optimizer.base_optimizer.param_groups[0]['lr']
        print(f'\n📊 Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Learning Rate: {current_lr:.2e}')
        print('-' * 60)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_acc, val_loss)
            print(f"✅ 保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= Config.PATIENCE_FINETUNE:
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    # 加载最佳模型
    load_checkpoint(model)
    
    total_time = time.time() - start_time
    print(f"\n🎉 训练完成! 总耗时: {total_time/60:.1f}分钟")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    return model, history


def unfreeze_layers(model, stage=2):
    """渐进式解冻层"""
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'blocks'):  # Vision Transformer
            total_blocks = len(model.backbone.blocks)
            if stage == 2:
                # 解冻后半部分
                for i in range(total_blocks // 2, total_blocks):
                    for param in model.backbone.blocks[i].parameters():
                        param.requires_grad = True
            elif stage == 3:
                # 解冻所有层
                for param in model.backbone.parameters():
                    param.requires_grad = True


def save_checkpoint(model, optimizer, epoch, val_acc, val_loss):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else optimizer.base_optimizer.state_dict(),
        'val_accuracy': val_acc,
        'val_loss': val_loss,
    }
    
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, Config.PRETRAINED_MODEL_PATH)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(model, checkpoint_path=None):
    """加载检查点"""
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, Config.PRETRAINED_MODEL_PATH)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 加载模型检查点，验证准确率: {checkpoint['val_accuracy']:.2f}%")
    
    return model