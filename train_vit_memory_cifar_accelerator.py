import os
import shutil
import random
import numpy as np
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
# from memory_vit_v1_model import MemoryViT
from VitMemoryv3 import SimpleViT

# -----------------------------------------------------------------------------
# Custom FLOP counter for neural memory operations
# -----------------------------------------------------------------------------

def count_flops(model, input_size=(1, 3, 32, 32)):
    """
    Custom FLOP counter that considers neural memory operations.
    Returns two values: standard FLOPs and memory-aware FLOPs.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        
        # Standard FLOP count
        input_tensor = torch.randn(input_size)
        standard_flops = FlopCountAnalysis(model, input_tensor)
        standard_total = standard_flops.total()
        
        # Memory-aware FLOP count (estimate additional cost)
        memory_flops = standard_total
        
        # Add estimated cost for memory operations
        for module in model.modules():
            if hasattr(module, 'memory_chunk_size'):
                # Estimate memory operations cost (simplified)
                chunk_size = module.memory_chunk_size
                dim = module.dim if hasattr(module, 'dim') else 192
                memory_cost = chunk_size * dim * 2  # Rough estimate
                memory_flops += memory_cost
        
        return standard_total, memory_flops
        
    except ImportError:
        print("Warning: fvcore not available. Using simplified FLOP estimation.")
        # Fallback: simple parameter-based estimation
        params = sum(p.numel() for p in model.parameters())
        return params * 2, params * 4  # Rough estimates

# -----------------------------------------------------------------------------
# Mixup implementation (optional)
# -----------------------------------------------------------------------------

class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

def update_memory_lr(optimizer, epoch, base_lr, warmup_epochs, progressive_epochs):
    """Progressively enable memory learning by ramping up its LR."""
    # This function assumes memory params are in the second param group
    if len(optimizer.param_groups) < 2:
        return

    if epoch >= warmup_epochs:
        if epoch < warmup_epochs + progressive_epochs:
            # Ramp up LR for memory param group
            ratio = (epoch - warmup_epochs + 1) / progressive_epochs
            optimizer.param_groups[1]['lr'] = base_lr * ratio
        else:
            # Ensure LR is set to base_lr after ramp-up
            optimizer.param_groups[1]['lr'] = base_lr

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@click.command()
@click.option('--baseline', is_flag=True, help='Use Baseline ViT (no memory)')
@click.option('--batch_size', default=128, help='Batch size per GPU')  # 128 x 2 GPUs = 256 total
@click.option('--epochs', default=150, help='Number of epochs')
@click.option('--lr', default=1e-3, help='Learning rate')
@click.option('--dim', default=192, help='Model dimension')
@click.option('--wandb_project', default='memory-vit-cifar10', help='WandB Project Name')
@click.option('--seed', default=42, help='Random seed')
@click.option('--use_mixup', is_flag=True, help='Use mixup augmentation (alpha=0.2)')
@click.option('--warmup_memory_epochs', default=5, help='Epochs to train with memory frozen (LR=0).')
@click.option('--progressive_memory', is_flag=True, help='Progressively enable memory by ramping up its LR.')
def train(baseline, batch_size, epochs, lr, dim, wandb_project, seed, use_mixup, warmup_memory_epochs, progressive_memory):
    set_seed(seed)
    
    # Setup accelerator for multi-GPU training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
    
    if accelerator.is_main_process:
        print(f"Training on {accelerator.device} with {accelerator.num_processes} GPUs")
        print(f"Total batch size: {batch_size * accelerator.num_processes}")

    # Hyperparameters configuration
    config = {
        "model_name": "baseline" if baseline else "MemoryViT",
        "optimizer": "AdamW",
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size_per_gpu": batch_size,
        "total_batch_size": batch_size * accelerator.num_processes,
        "dim": dim,
        "weight_decay": 0.05,
        "warmup_epochs": 10,
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2 if use_mixup else 0.0,
        "architecture": "SimpleViT",
        "seed": seed,
        "warmup_memory_epochs": warmup_memory_epochs,
        "progressive_memory": progressive_memory,
    }
    
    # Set run name based on configuration
    run_name = "baseline" if baseline else "memory_vit"

    # Initialize wandb tracking
    accelerator.init_trackers(
        project_name=wandb_project, 
        config=config,
        init_kwargs={"wandb": {"name": run_name}}
    )
    
    # Data augmentation - stronger than before
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download dataset on main process only
    if accelerator.is_main_process:
        datasets.CIFAR10(root='./data', train=True, download=True)
        datasets.CIFAR10(root='./data', train=False, download=True)
    accelerator.wait_for_everyone()

    # Create datasets
    full_trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    
    # Split into train/val
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(seed))
    
    # Create dataloaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Model initialization
    model = SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=dim,
        depth=6,
        heads=3,
        mlp_dim=dim * 4,
        use_memory=False if baseline else True,
        memory_chunk_size=64,
        gated=False
    )
    
    # Count parameters and FLOPs (main process only)
    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {params:,}")
        
        # Calculate FLOPs using custom counter
        try:
            standard_flops, memory_flops = count_flops(model)
            print(f"Standard FLOPs: {standard_flops / 1e9:.4f} GFLOPs")
            print(f"Memory-aware FLOPs: {memory_flops / 1e9:.4f} GFLOPs")
            
            # Log FLOP metrics
            accelerator.log({
                "standard_flops": standard_flops,
                "memory_flops": memory_flops
            })
        except Exception as e:
            print(f"FLOP counting failed: {e}")
            print("Proceeding without FLOP metrics...")
    
    # Optimizer with weight decay 0.05. For MemoryViT, we can freeze memory initially.
    if not baseline:
        if accelerator.is_main_process:
            print("Configuring optimizer for MemoryViT with separate memory param group.")
        memory_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'neural_memory' in name or 'memory_model' in name:
                memory_params.append(param)
            else:
                other_params.append(param)
        
        # Freeze memory params by setting their LR to 0 initially
        param_groups = [
            {'params': other_params},
            {'params': memory_params, 'lr': 0.0}
        ]
        optimizer = optim.AdamW(param_groups, lr=lr, weight_decay=0.05)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    # Criterion with label smoothing 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Mixup - optional
    mixup_fn = Mixup(alpha=0.2) if use_mixup else None
    
    # Prepare model and dataloaders for distributed training
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    
    # Scheduler setup - warmup for 10 epochs then cosine decay
    warmup_epochs = 10
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = epochs * len(train_loader)
    
    # Linear warmup scheduler
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1e-6, 
        end_factor=1.0, 
        total_iters=warmup_steps
    )
    
    # Cosine decay scheduler
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=1e-6
    )
    
    # Combined sequential scheduler
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    # Setup for checkpointing
    best_val_acc = 0.0
    checkpoint_dir = "best_checkpoint"
    
    
    # Training loop
    for epoch in range(epochs):
        # Handle memory learning rate schedule
        if not baseline:
            if progressive_memory:
                # Ramp up over 10% of total epochs
                progressive_epochs = max(1, int(epochs * 0.1))
                # We need to access the underlying optimizer's param_groups
                update_memory_lr(optimizer.optimizer, epoch, lr, warmup_memory_epochs, progressive_epochs)
            elif epoch == warmup_memory_epochs:
                # One-shot unfreezing if not progressive
                if accelerator.is_main_process:
                    print(f"Unfreezing memory modules. Setting LR for memory params to {lr}.")
                # Access underlying optimizer wrapped by accelerator
                optimizer.optimizer.param_groups[1]['lr'] = lr

        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", 
                               disable=not accelerator.is_main_process):
            
            optimizer.zero_grad()
            
            # Apply mixup if enabled
            if use_mixup and epoch < epochs * 0.8:  # Only apply mixup for first 80% of training
                imgs, labels_a, labels_b, lam = mixup_fn(imgs, labels)
                logits, _ = model(imgs)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                
                # Mixup accuracy calculation
                _, predicted = logits.max(1)
                correct += (lam * predicted.eq(labels_a).sum().item() + 
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
            
            total += labels.size(0)
            total_loss += loss.item()
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping for stability
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()  # Step scheduler every batch
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        
        # Log training metrics
        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {current_lr:.2e}")
            
            accelerator.log({
                "train_loss": avg_loss,
                "train_acc": acc,
                "lr": current_lr,
                "epoch": epoch + 1
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation", 
                                    disable=not accelerator.is_main_process):
                logits, _ = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Gather predictions across GPUs for accurate metrics
                predictions = logits.argmax(dim=-1)
                predictions, labels = accelerator.gather_for_metrics((predictions, labels))
                
                total += labels.size(0)
                correct += predictions.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        # Log validation metrics
        if accelerator.is_main_process:
            print(f"--> Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            accelerator.log({
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "epoch": epoch + 1
            })
            
            # Save best model checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                accelerator.save_state(checkpoint_dir)
                print(f"New best validation accuracy: {best_val_acc:.2f}%. Saved checkpoint.")
            
    # Final test evaluation using best model
    accelerator.wait_for_everyone()
    if os.path.exists(checkpoint_dir):
        accelerator.load_state(checkpoint_dir)
        if accelerator.is_main_process:
            print("Loaded best model for final testing.")
            
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Final Test Evaluation", 
                                disable=not accelerator.is_main_process):
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            predictions, labels = accelerator.gather_for_metrics((predictions, labels))
            
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()
            
    test_acc = 100. * correct / total
    
    # Log final test results
    if accelerator.is_main_process:
        print(f"--> Final Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {test_acc:.2f}%")
        accelerator.log({
            "test_loss": test_loss/len(test_loader),
            "test_acc": test_acc,
            "best_val_acc": best_val_acc
        })
        
        # Cleanup checkpoint
        # if os.path.exists(checkpoint_dir):
        #     shutil.rmtree(checkpoint_dir)
            
    # End training and close trackers
    accelerator.end_training()

if __name__ == '__main__':
    train()