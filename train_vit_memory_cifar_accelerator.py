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
from memory_vit_v1_model import MemoryViT
# from VitMemoryv3 import MemoryViT

# accele

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
@click.option('--batch_size', default=64, help='Batch size')
@click.option('--epochs', default=50, help='Number of epochs')
@click.option('--lr', default=3e-4, help='Learning rate')
@click.option('--dim', default=192, help='Model dimension')
@click.option('--wandb_project', default='memory-vit-cifar10', help='WandB Project Name')
@click.option('--seed', default=42, help='Random seed')
def train(batch_size, epochs, lr, dim, wandb_project, seed):
    set_seed(seed)
    
    # 1. Using accelerator to ddp with world size = 2 and find unused parameters = True
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])
    
    if accelerator.is_main_process:
        print(f"Training on {accelerator.device}")
    
    # 5. Create a variable to store hparams and logged it later on with config instead
    config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "dim": dim,
        "architecture": "MemoryViT-V1",
        "seed": seed
    }
    
    # 3. Logging with accelator log and wandb
    accelerator.init_trackers(project_name=wandb_project, config=config)
    
    # Augmentation is key for CIFAR
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
    
    # 4. Download cifar10 dataset in main process to avoid race condition
    if accelerator.is_main_process:
        datasets.CIFAR10(root='./data', train=True, download=True)
        datasets.CIFAR10(root='./data', train=False, download=True)
    accelerator.wait_for_everyone()
    
    # 6. Split random cifar to train val
    full_trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    
    train_size = int(0.9 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)
    
    # Model
    model = MemoryViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=dim,
        depth=6,
        heads=3,
        memory_chunk_size=64 # 8x8 patches + 1 CLS = 65, so chunk 64 is perfect
    )
    
    # Count parameters
    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # 2. Setup model and dataloader using accelerator.prepare
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler
    )
    
    best_val_acc = 0.0
    checkpoint_dir = "best_checkpoint"
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", disable=not accelerator.is_main_process):
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            accelerator.backward(loss)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        
        # Gather metrics for logging
        avg_loss = total_loss / len(train_loader)
        acc = 100. * correct / total
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
            
            accelerator.log({
                "train_loss": avg_loss,
                "train_acc": acc,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1
            })
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", disable=not accelerator.is_main_process):
                logits = model(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                # Gather predictions for accurate metrics
                predictions = logits.argmax(dim=-1)
                predictions, labels = accelerator.gather_for_metrics((predictions, labels))
                
                total += labels.size(0)
                correct += predictions.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        if accelerator.is_main_process:
            print(f"--> Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            accelerator.log({
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "epoch": epoch + 1
            })
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                accelerator.save_state(checkpoint_dir)
                print(f"New best validation accuracy: {best_val_acc:.2f}%. Saved checkpoint.")
            
    # 6. Run on test once on final using best model
    accelerator.wait_for_everyone()
    if os.path.exists(checkpoint_dir):
        accelerator.load_state(checkpoint_dir)
        if accelerator.is_main_process:
            print("Loaded best model for testing.")
            
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Final Test", disable=not accelerator.is_main_process):
            logits = model(imgs)
            loss = criterion(logits, labels)
            test_loss += loss.item()
            
            predictions = logits.argmax(dim=-1)
            predictions, labels = accelerator.gather_for_metrics((predictions, labels))
            
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()
            
    test_acc = 100. * correct / total
    if accelerator.is_main_process:
        print(f"--> Final Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {test_acc:.2f}%")
        accelerator.log({
            "test_loss": test_loss/len(test_loader),
            "test_acc": test_acc
        })
        
        # Cleanup
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            
    accelerator.end_training()

if __name__ == '__main__':
    train()