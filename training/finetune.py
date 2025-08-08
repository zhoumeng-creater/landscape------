"""
微调模块 - 分类器微调训练（修复版）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
from config import Config


def create_finetune_optimizer(ijepa_model, classifier, stage, learning_rate):
    """创建微调优化器 - 修复版
    
    Args:
        ijepa_model: I-JEPA模型
        classifier: 分类器
        stage: 训练阶段（1或2）
        learning_rate: 基础学习率
        
    Returns:
        optimizer: 优化器
    """
    if stage == 1:
        # 阶段1：只训练分类器
        param_groups = [
            {'params': classifier.parameters(), 'lr': learning_rate}
        ]
    else:
        # 阶段2：微调预训练模型的最后几层 + 分类器
        # 需要区分不同类型的模型架构
        
        # 收集所有参数ID，避免重复
        param_ids = set()
        ijepa_params = []
        classifier_only_params = []
        
        # 收集ijepa_model的可训练参数
        for name, param in ijepa_model.named_parameters():
            if param.requires_grad:
                ijepa_params.append(param)
                param_ids.add(id(param))
        
        # 收集classifier独有的参数（排除已经在ijepa_params中的参数）
        for name, param in classifier.named_parameters():
            if param.requires_grad:
                if id(param) not in param_ids:
                    classifier_only_params.append(param)
                    param_ids.add(id(param))
        
        # 创建参数组
        param_groups = []
        
        # 添加ijepa_model的参数（使用较小的学习率）
        if ijepa_params:
            param_groups.append({
                'params': ijepa_params, 
                'lr': learning_rate / Config.FINETUNE_LR_RATIO,
                'name': 'ijepa_params'  # 添加名称以便调试
            })
        
        # 添加classifier独有的参数（使用标准学习率）
        if classifier_only_params:
            param_groups.append({
                'params': classifier_only_params, 
                'lr': learning_rate,
                'name': 'classifier_params'  # 添加名称以便调试
            })
        
        # 如果没有找到任何独立参数（可能classifier完全依赖ijepa）
        # 则使用所有classifier参数
        if not param_groups:
            # 这种情况下，可能是原始I-JEPA架构
            param_groups = [
                {
                    'params': [p for n, p in ijepa_model.named_parameters() if p.requires_grad], 
                    'lr': learning_rate / Config.FINETUNE_LR_RATIO,
                    'name': 'ijepa_all'
                },
                {
                    'params': classifier.parameters(), 
                    'lr': learning_rate,
                    'name': 'classifier_all'
                }
            ]
            
            # 再次检查是否有重复参数
            ijepa_param_ids = {id(p) for n, p in ijepa_model.named_parameters() if p.requires_grad}
            classifier_param_ids = {id(p) for p in classifier.parameters()}
            
            # 如果有重复，只使用classifier参数
            if ijepa_param_ids & classifier_param_ids:
                print("⚠️ 检测到参数重复，只使用classifier参数")
                param_groups = [
                    {'params': classifier.parameters(), 'lr': learning_rate}
                ]
        
        # 打印参数组信息以便调试
        print(f"📊 参数组信息:")
        for i, group in enumerate(param_groups):
            param_count = len(list(group['params']))
            group_name = group.get('name', f'group_{i}')
            print(f"  {group_name}: {param_count} 个参数, lr={group['lr']:.2e}")
    
    optimizer = optim.AdamW(param_groups, weight_decay=Config.FINETUNE_WEIGHT_DECAY)
    return optimizer


def train_epoch(ijepa_model, classifier, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch
    
    Args:
        ijepa_model: I-JEPA模型
        classifier: 分类器
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    # 根据epoch决定是否训练I-JEPA模型
    if epoch >= Config.FINETUNE_STAGE2_EPOCH:
        ijepa_model.train()
    else:
        ijepa_model.eval()
    
    classifier.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 提取特征
        if epoch >= Config.FINETUNE_STAGE2_EPOCH:
            features = ijepa_model.encode(data)
        else:
            with torch.no_grad():
                features = ijepa_model.encode(data)
        
        # 分类
        output = classifier(features)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪 - 需要获取所有参与训练的参数
        params_to_clip = []
        
        # 收集所有需要裁剪的参数
        if epoch >= Config.FINETUNE_STAGE2_EPOCH:
            # 阶段2：包括ijepa和classifier的参数
            for param_group in optimizer.param_groups:
                params_to_clip.extend(param_group['params'])
        else:
            # 阶段1：只有classifier的参数
            params_to_clip = list(classifier.parameters())
        
        # 应用梯度裁剪
        if params_to_clip:
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=Config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # 打印进度
        if batch_idx % Config.LOG_INTERVAL == 0:
            print(f'Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(ijepa_model, classifier, val_loader, criterion, device):
    """验证模型
    
    Args:
        ijepa_model: I-JEPA模型
        classifier: 分类器
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    ijepa_model.eval()
    classifier.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # 提取特征
            features = ijepa_model.encode(data)
            
            # 分类
            output = classifier(features)
            loss = criterion(output, target)
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def optimized_finetune_classifier(ijepa_model, classifier, train_loader, val_loader, 
                                 num_epochs=None, learning_rate=None):
    """优化的分类器微调（修复版）
    
    Args:
        ijepa_model: I-JEPA模型
        classifier: 分类器
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        
    Returns:
        classifier: 训练后的分类器
        history: 训练历史
    """
    # 使用默认配置
    if num_epochs is None:
        num_epochs = Config.FINETUNE_EPOCHS
    if learning_rate is None:
        learning_rate = Config.FINETUNE_LR
    
    device = Config.DEVICE
    ijepa_model = ijepa_model.to(device)
    classifier = classifier.to(device)
    
    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    
    # 初始优化器（阶段1）
    optimizer = create_finetune_optimizer(ijepa_model, classifier, stage=1, learning_rate=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, 
        epochs=num_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("🚀 开始优化版分类器微调...")
    
    for epoch in range(num_epochs):
        # 阶段切换
        if epoch == Config.FINETUNE_STAGE2_EPOCH:
            print(f"🔄 切换到阶段2：微调预训练模型的最后几层")
            
            # 解冻最后几层
            for name, param in ijepa_model.named_parameters():
                # 根据模型类型决定解冻策略
                if hasattr(ijepa_model, 'context_encoder'):
                    # 原始I-JEPA或HybridIJEPA
                    if 'blocks.10' in name or 'blocks.11' in name or 'feature_enhancer' in name:
                        param.requires_grad = True
                        print(f"  解冻层: {name}")
                else:
                    # 其他模型类型
                    if 'layer4' in name or 'head' in name:
                        param.requires_grad = True
                        print(f"  解冻层: {name}")
            
            # 重新创建优化器（修复版）
            optimizer = create_finetune_optimizer(
                ijepa_model, classifier, stage=2, learning_rate=learning_rate
            )
            
            # 重新创建调度器
            remaining_epochs = num_epochs - Config.FINETUNE_STAGE2_EPOCH
            
            # 获取所有学习率
            max_lrs = []
            for group in optimizer.param_groups:
                max_lrs.append(group['lr'])
            
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=max_lrs if len(max_lrs) > 1 else max_lrs[0], 
                epochs=remaining_epochs, 
                steps_per_epoch=len(train_loader)
            )
        
        # 训练
        train_loss, train_acc = train_epoch(
            ijepa_model, classifier, train_loader, 
            criterion, optimizer, device, epoch
        )
        
        # 更新学习率
        scheduler.step()
        
        # 验证
        val_loss, val_acc = validate(
            ijepa_model, classifier, val_loader, criterion, device
        )
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # 打印结果
        print(f'📊 Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 60)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            save_finetune_checkpoint(
                ijepa_model, classifier, epoch, 
                val_acc, val_loss, train_acc, train_loss
            )
            print(f"✅ 保存最佳模型，验证准确率: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= Config.PATIENCE_FINETUNE:
            print(f"早停触发，在第 {epoch+1} 轮停止训练")
            break
    
    # 加载最佳模型
    ijepa_model, classifier = load_finetune_checkpoint(ijepa_model, classifier)
    
    print(f"🎉 优化版微调完成! 最佳验证准确率: {best_val_acc:.2f}%")
    
    return classifier, history


def save_finetune_checkpoint(ijepa_model, classifier, epoch, val_acc, val_loss, train_acc, train_loss):
    """保存微调检查点
    
    Args:
        ijepa_model: I-JEPA模型
        classifier: 分类器
        epoch: 当前epoch
        val_acc: 验证准确率
        val_loss: 验证损失
        train_acc: 训练准确率
        train_loss: 训练损失
    """
    checkpoint = {
        'epoch': epoch + 1,
        'ijepa_state': ijepa_model.state_dict(),
        'classifier_state': classifier.state_dict(),
        'val_accuracy': val_acc,
        'val_loss': val_loss,
        'train_accuracy': train_acc,
        'train_loss': train_loss
    }
    
    # 确保目录存在
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    
    # 保存检查点
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, Config.FINETUNE_MODEL_PATH)
    torch.save(checkpoint, checkpoint_path)


def load_finetune_checkpoint(ijepa_model, classifier, checkpoint_path=None):
    """加载微调检查点
    
    Args:
        ijepa_model: I-JEPA模型
        classifier: 分类器
        checkpoint_path: 检查点路径
        
    Returns:
        ijepa_model: 加载了权重的I-JEPA模型
        classifier: 加载了权重的分类器
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, Config.FINETUNE_MODEL_PATH)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
        ijepa_model.load_state_dict(checkpoint['ijepa_state'])
        classifier.load_state_dict(checkpoint['classifier_state'])
        print(f"✅ 加载最佳模型，验证准确率: {checkpoint['val_accuracy']:.2f}%")
        return ijepa_model, classifier
    except Exception as e:
        print(f"❌ 加载微调模型失败: {e}")
        return ijepa_model, classifier