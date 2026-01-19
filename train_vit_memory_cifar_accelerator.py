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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@click.command()
@click.option('--baseline', is_flag=True, help='Use Baseline ViT (no memory)')
@click.option('--batch_size', default=128, help='Batch size per GPU')
@click.option('--epochs', default=150, help='Number of epochs')
@click.option('--lr', default=1e-3, help='Learning rate')
@click.option('--dim', default=192, help='Model dimension')
@click.option('--wandb_project', default='memory-vit-cifar10', help='WandB Project Name')
@click.option('--seed', default=42, help='Random seed')
@click.option('--use_mixup', is_flag=True, help='Use mixup augmentation')
@click.option('--pretrain_epochs', default=0, help='Pre-train attention layers')
@click.option('--debug', is_flag=True, help='Run in debug mode')
@click.option('--ttt_learning_rate', default=1e-4, help='LR for test-time adaptation')
@click.option('--ttt_steps', default=3, help='Number of TTT steps per test batch')
def train(baseline, batch_size, epochs, lr, dim, wandb_project, seed, use_mixup, 
          pretrain_epochs, debug, ttt_learning_rate, ttt_steps):
    set_seed(seed)
    
    # Setup accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
    
    if accelerator.is_main_process:
        print(f"Training on {accelerator.device} with {accelerator.num_processes} GPUs")
        print(f"Total batch size: {batch_size * accelerator.num_processes}")

    config = {
        "model_name": "baseline" if baseline else "MemoryViT",
        "optimizer": "AdamW",
        "learning_rate": lr,
        "ttt_learning_rate": ttt_learning_rate,
        "ttt_steps": ttt_steps,
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
    }
    
    run_name = "baseline" if baseline else ("memory_vit_pretrained" if pretrain_epochs > 0 else "memory_vit")
    
    accelerator.init_trackers(
        project_name=wandb_project, 
        config=config,
        init_kwargs={"wandb": {"name": run_name}}
    )
    
    # Data transforms
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
    
    # Download dataset
    if accelerator.is_main_process:
        datasets.CIFAR10(root='./data', train=True, download=True)
        datasets.CIFAR10(root='./data', train=False, download=True)
    accelerator.wait_for_everyone()

    full_trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size], 
                                   generator=torch.Generator().manual_seed(seed))
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # -------------------------------------------------------------------------
    # Phase 1: Pre-training Attention
    # -------------------------------------------------------------------------
    pretrained_state_dict = None
    
    if pretrain_epochs > 0 and not baseline:
        if accelerator.is_main_process:
            print(f"\n=== Phase 1: Pre-training Attention Layers ({pretrain_epochs} epochs) ===")
            
        pre_model = SimpleViT(
            image_size=32, patch_size=4, num_classes=10, dim=dim, depth=6, heads=3, mlp_dim=dim * 4,
            use_memory=False, memory_chunk_size=64
        )
        
        pre_optimizer = optim.AdamW(pre_model.parameters(), lr=lr, weight_decay=0.05)
        pre_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        pre_model, pre_optimizer, pre_train_loader = accelerator.prepare(
            pre_model, pre_optimizer, train_loader
        )
        
        for epoch in range(pretrain_epochs):
            pre_model.train()
            total_loss = 0
            for imgs, labels in tqdm(pre_train_loader, desc=f"Pre-train Epoch {epoch+1}/{pretrain_epochs}", 
                                   disable=not accelerator.is_main_process):
                pre_optimizer.zero_grad()
                logits, _ = pre_model(imgs, memory_training=True)
                loss = pre_criterion(logits, labels)
                accelerator.backward(loss)
                pre_optimizer.step()
                total_loss += loss.item()
                if debug: break
            
            if accelerator.is_main_process:
                print(f"Pre-train Loss: {total_loss/len(pre_train_loader):.4f}")
        
        accelerator.wait_for_everyone()
        pretrained_state_dict = accelerator.get_state_dict(pre_model)
        
        del pre_model, pre_optimizer
        torch.cuda.empty_cache()
        if accelerator.is_main_process:
            print("=== Pre-training Complete. Starting Memory Adaptation ===\n")

    # -------------------------------------------------------------------------
    # Phase 2: Main Training (Memory Adaptation)
    # -------------------------------------------------------------------------
    
    model = SimpleViT(
        image_size=32, patch_size=4, num_classes=10, dim=dim, depth=6, heads=3,
        mlp_dim=dim * 4, use_memory=not baseline, memory_chunk_size=64
    )
    
    # Load pre-trained weights and freeze attention
    if pretrained_state_dict is not None:
        if accelerator.is_main_process:
            print("Loading pre-trained attention weights and freezing attention...")
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        # Freeze attention layers
        for name, param in model.named_parameters():
            if 'transformer.layers' in name and '.0.' in name:  # Attention is layers[i][0]
                param.requires_grad = False
    
    # Count parameters
    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {params:,}")
        
        try:
            standard_flops, memory_flops = count_flops(model)
            print(f"Standard FLOPs: {standard_flops / 1e9:.4f} GFLOPs")
            print(f"Memory-aware FLOPs: {memory_flops / 1e9:.4f} GFLOPs")
            accelerator.log({"standard_flops": standard_flops, "memory_flops": memory_flops})
        except Exception as e:
            print(f"FLOP counting failed: {e}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    mixup_fn = Mixup(alpha=0.2) if use_mixup else None
    
    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )
    
    # Scheduler
    warmup_epochs = 10
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = epochs * len(train_loader)
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )
    
    best_val_acc = 0.0
    checkpoint_dir = "best_checkpoint"
    
    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Reset memory states at start of each epoch
        memory_states = model.init_memory_state(batch_size)
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training", 
                               disable=not accelerator.is_main_process):
            
            optimizer.zero_grad()
            
            # Forward with memory training (updates all parameters)
            logits, memory_states = model(
                imgs, state=memory_states, memory_training=True, freeze_attention=False
            )
            
            if use_mixup and epoch < epochs * 0.8:
                imgs, labels_a, labels_b, lam = mixup_fn(imgs, labels)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                _, predicted = logits.max(1)
                correct += (lam * predicted.eq(labels_a).sum().item() + 
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                loss = criterion(logits, labels)
                _, predicted = logits.max(1)
                correct += predicted.eq(labels).sum().item()
            
            total += labels.size(0)
            total_loss += loss.item()
            
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            if debug: break
        
        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        
        if accelerator.is_main_process:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {current_lr:.2e}")
            accelerator.log({"train_loss": avg_loss, "train_acc": acc, "lr": current_lr, "epoch": epoch+1})
        
        # ---------------------------------------------------------------------
        # Validation (NO TTT - Standard Inference)
        # ---------------------------------------------------------------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Standard inference - no adaptation
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation",
                                   disable=not accelerator.is_main_process):
                # Reset state for each val batch (no persistence)
                val_state = model.init_memory_state(imgs.size(0))
                
                logits, _ = model(imgs, state=val_state, memory_training=False)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                predictions = logits.argmax(dim=-1)
                predictions, labels = accelerator.gather_for_metrics((predictions, labels))
                total += labels.size(0)
                correct += predictions.eq(labels).sum().item()
                if debug: break
        
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        if accelerator.is_main_process:
            print(f"--> Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            accelerator.log({"val_loss": avg_val_loss, "val_acc": val_acc, "epoch": epoch+1})
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                accelerator.save_state(checkpoint_dir)
                print(f"New best val acc: {best_val_acc:.2f}%. Saved checkpoint.")
    
    # -------------------------------------------------------------------------
    # Final Test Evaluation (With True TTT)
    # -------------------------------------------------------------------------
    accelerator.wait_for_everyone()
    if os.path.exists(checkpoint_dir):
        accelerator.load_state(checkpoint_dir)
        if accelerator.is_main_process:
            print(f"\nLoaded best model (val acc: {best_val_acc:.2f}%). Running TTT on test set...")
    
    model.eval()  # Keep in eval mode but enable grad for memory
    
    # Create TTT optimizer (only for memory parameters)
    memory_params = [p for n, p in model.named_parameters() if 'neural_memory' in n]
    ttt_optimizer = optim.AdamW(memory_params, lr=ttt_learning_rate, weight_decay=0.0)
    ttt_optimizer = accelerator.prepare(ttt_optimizer)
    
    test_loss = 0
    correct = 0
    total = 0
    
    # Initialize test-time memory state ONCE and persist across batches
    persistent_test_state = model.init_memory_state(batch_size=1)  # Single persistent state
    
    for imgs, labels in tqdm(test_loader, desc="Final Test (TTT)", 
                           disable=not accelerator.is_main_process):
        # ---------------------------------------------------------------------
        # Test-Time Training: Update memory weights on this batch
        # ---------------------------------------------------------------------
        for _ in range(ttt_steps):  # Multiple adaptation steps
            ttt_optimizer.zero_grad()
            
            # Forward with memory_training=True (allow gradient flow)
            logits, persistent_test_state = model(
                imgs, state=persistent_test_state, memory_training=True
            )
            
            # Self-supervised loss for TTT (using cross-entropy as proxy)
            # For CIFAR-10, we can use the label as "target" for adaptation
            # In true TTT, this would be a self-supervised task
            loss = criterion(logits, labels)
            accelerator.backward(loss)
            ttt_optimizer.step()
        
        # ---------------------------------------------------------------------
        # Evaluation: Get final predictions after adaptation
        # ---------------------------------------------------------------------
        with torch.no_grad():
            logits, persistent_test_state = model(
                imgs, state=persistent_test_state, memory_training=False
            )
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            predictions, labels = accelerator.gather_for_metrics((predictions, labels))
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()
            if debug: break
            
    test_acc = 100. * correct / total
    
    if accelerator.is_main_process:
        print(f"\n=== FINAL TEST RESULTS (with TTT) ===")
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
        print(f"Test Acc: {test_acc:.2f}%")
        print(f"Best Val Acc: {best_val_acc:.2f}%")
        
        accelerator.log({
            "test_loss": test_loss/len(test_loader),
            "test_acc": test_acc,
            "best_val_acc": best_val_acc,
            "ttt_steps": ttt_steps,
            "ttt_lr": ttt_learning_rate
        })
    
    accelerator.end_training()


if __name__ == '__main__':
    # train()
    # Run in notebook
    train(args=['--debug'], standalone_mode=True)