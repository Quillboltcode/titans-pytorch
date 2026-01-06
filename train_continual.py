"""
Continual Learning Experiment: Split CIFAR-10 with DDP

This script compares Standard ViT vs Memory ViT on class-incremental learning.
- Task A: Classes 0-4 (airplane, automobile, bird, cat, deer)
- Task B: Classes 5-9 (dog, frog, horse, ship, truck)

Uses DistributedDataParallel (DDP) for multi-GPU training with proper memory state.
"""

import os
import math
import json
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import datasets, transforms
from einops import rearrange, repeat
from titans_pytorch.neural_memory import NeuralMemory
from tqdm import tqdm
import wandb
from types import SimpleNamespace

def ddp_setup(rank, world_size):
    """Initialize DDP process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# -----------------------------------------------------------------------------
# Helper: DropPath (Stochastic Depth)
# -----------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, p=0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor

# -----------------------------------------------------------------------------
# Standard ViT Block (Baseline without NeuralMemory)
# -----------------------------------------------------------------------------

class StandardTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., drop_path=0.):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        
        self.norm_mlp = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        attn_residual = x
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = attn_residual + self.drop_path(attn_out)
        
        mlp_residual = x
        x_norm = self.norm_mlp(x)
        mlp_out = self.mlp(x_norm)
        x = mlp_residual + self.drop_path(mlp_out)
        
        return x

# -----------------------------------------------------------------------------
# Standard ViT Model (Baseline)
# -----------------------------------------------------------------------------

class StandardViT(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=192,
        depth=6,
        heads=3,
        mlp_ratio=4.,
        drop_path_rate=0.
    ):
        super().__init__()
        self.patch_size = patch_size
        assert image_size % patch_size == 0
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            self.layers.append(
                StandardTransformerBlock(
                    dim=dim, 
                    heads=heads, 
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i]
                )
            )
            
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_classes)

    def forward(self, img):
        patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
                           p1=self.patch_size, p2=self.patch_size)
        x = self.to_patch_embedding(patches)
        
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((x, cls_tokens), dim=1)
        
        for layer in self.layers:
            x = layer(x)
        
        cls_token_out = x[:, -1]
        return self.to_logits(self.norm(cls_token_out))

# -----------------------------------------------------------------------------
# Memory Transformer Block (with NeuralMemory)
# -----------------------------------------------------------------------------

class MemoryFFNTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        memory_chunk_size=64,
        drop_path=0.
    ):
        super().__init__()
        
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        
        self.norm_mem = nn.LayerNorm(dim)
        self.neural_memory = NeuralMemory(
            dim=dim,
            chunk_size=memory_chunk_size,
            qkv_receives_diff_views=False
        )
        
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, memory_state=None):
        attn_residual = x
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = attn_residual + self.drop_path(attn_out)
        
        mem_residual = x
        x_norm = self.norm_mem(x)
        
        mem_out, next_memory_state = self.neural_memory(
            x_norm, 
            state=memory_state
        )
        
        x = mem_residual + self.drop_path(self.to_out(mem_out))
        
        return x, next_memory_state

# -----------------------------------------------------------------------------
# Memory ViT Model
# -----------------------------------------------------------------------------

class MemoryViT(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=192,
        depth=6,
        heads=3,
        memory_chunk_size=64,
        drop_path_rate=0.
    ):
        super().__init__()
        self.patch_size = patch_size
        assert image_size % patch_size == 0
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.layers = nn.ModuleList([])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            self.layers.append(
                MemoryFFNTransformerBlock(
                    dim=dim, 
                    heads=heads, 
                    memory_chunk_size=memory_chunk_size,
                    drop_path=dpr[i]
                )
            )
            
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_classes)

    def forward(self, img):
        patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', 
                           p1=self.patch_size, p2=self.patch_size)
        x = self.to_patch_embedding(patches)
        
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((x, cls_tokens), dim=1)
        
        memory_states = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            x, memory_states[i] = layer(x, memory_state=memory_states[i])
            
        cls_token_out = x[:, -1]
        return self.to_logits(self.norm(cls_token_out))

    def freeze_attention(self):
        """Freeze all attention layers, only allow memory and head to update."""
        for layer in self.layers:
            for param in layer.attn.parameters():
                param.requires_grad = False
            layer.norm_attn.weight.requires_grad = False
            layer.norm_attn.bias.requires_grad = False
        
        self.pos_embedding.requires_grad = False
        self.cls_token.requires_grad = False
        
        for param in self.to_patch_embedding.parameters():
            param.requires_grad = False
        
        if torch.distributed.get_rank() == 0:
            print("Attention layers frozen. Memory modules and classification head remain trainable.")

# -----------------------------------------------------------------------------
# Split CIFAR-10 Dataset
# -----------------------------------------------------------------------------

class SplitCIFAR10:
    TASK_A_CLASSES = [0, 1, 2, 3, 4]
    TASK_B_CLASSES = [5, 6, 7, 8, 9]
    
    def __init__(self, task='A', train=True, transform=None, download=False):
        self.task = task.upper()
        self.target_classes = self.TASK_A_CLASSES if self.task == 'A' else self.TASK_B_CLASSES
        
        full_dataset = datasets.CIFAR10(
            root='./data', 
            train=train, 
            download=download, 
            transform=transform
        )
        
        indices = [i for i, (_, label) in enumerate(full_dataset) if label in self.target_classes]
        self.dataset = Subset(full_dataset, indices)
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        self.target_class_names = [self.classes[i] for i in self.target_classes]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def get_data_loaders(batch_size, task=None, rank=0, world_size=1):
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
    
    if task is None:
        task_a = SplitCIFAR10(task='A', train=True, transform=transform_train, download=True)
        task_a_test = SplitCIFAR10(task='A', train=False, transform=transform_test, download=True)
        task_b = SplitCIFAR10(task='B', train=True, transform=transform_train, download=True)
        task_b_test = SplitCIFAR10(task='B', train=False, transform=transform_test, download=True)
        
        return {
            'train_A': DataLoader(task_a, batch_size=batch_size, shuffle=False, 
                                  sampler=DistributedSampler(task_a, rank=rank, shuffle=True)),
            'test_A': DataLoader(task_a_test, batch_size=batch_size, shuffle=False,
                                 sampler=DistributedSampler(task_a_test, rank=rank)),
            'train_B': DataLoader(task_b, batch_size=batch_size, shuffle=False,
                                  sampler=DistributedSampler(task_b, rank=rank, shuffle=True)),
            'test_B': DataLoader(task_b_test, batch_size=batch_size, shuffle=False,
                                 sampler=DistributedSampler(task_b_test, rank=rank)),
        }
    else:
        task_dataset = SplitCIFAR10(task=task, train=True, transform=transform_train, download=True)
        task_test = SplitCIFAR10(task=task, train=False, transform=transform_test, download=True)
        
        return {
            'train': DataLoader(task_dataset, batch_size=batch_size, shuffle=False,
                               sampler=DistributedSampler(task_dataset, rank=rank, shuffle=True)),
            'test': DataLoader(task_test, batch_size=batch_size, shuffle=False,
                              sampler=DistributedSampler(task_test, rank=rank)),
        }

# -----------------------------------------------------------------------------
# Model Factory
# -----------------------------------------------------------------------------

def create_model(model_type, num_classes, image_size=32, patch_size=4, dim=192, depth=6, 
                heads=3, memory_chunk_size=64, drop_path_rate=0.1):
    if model_type == 'standard':
        return StandardViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            drop_path_rate=drop_path_rate
        )
    elif model_type == 'memory':
        return MemoryViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            memory_chunk_size=memory_chunk_size,
            drop_path_rate=drop_path_rate
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# -----------------------------------------------------------------------------
# Training Functions
# -----------------------------------------------------------------------------

def train_epoch(model, train_loader, optimizer, criterion, device, rank, epoch, world_size):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    train_loader.sampler.set_epoch(epoch)  # For DDP
    
    for imgs, labels in tqdm(train_loader, desc="Training", leave=False, disable=rank != 0):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Sync metrics across processes
    metrics = torch.tensor([total_loss / len(train_loader), 100. * correct / total]).to(device)
    torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
    metrics /= world_size
    
    return metrics[0].item(), metrics[1].item()

def evaluate(model, test_loader, device, rank, world_size):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    test_loader.sampler.set_epoch(0)  # For DDP
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating", leave=False, disable=rank != 0):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Sync metrics
    metrics = torch.tensor([total_loss / len(test_loader), 100. * correct / total]).to(device)
    torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
    metrics /= world_size
    
    return metrics[0].item(), metrics[1].item()

def train_phase(rank, world_size, model, train_loader, test_loader, task_name, epochs, lr, device):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, rank, epoch, world_size)
        scheduler.step()
        
        if (epoch + 1) % 10 == 0 and rank == 0:
            val_loss, val_acc = evaluate(model, test_loader, device, rank, world_size)
            if val_acc > best_acc:
                best_acc = val_acc
            print(f"  Epoch {epoch+1}/{epochs} | {task_name} Loss: {val_loss:.4f} | {task_name} Acc: {val_acc:.2f}%")
            
            wandb.log({
                f'{task_name}/train_loss': train_loss,
                f'{task_name}/train_acc': train_acc,
                f'{task_name}/val_loss': val_loss,
                f'{task_name}/val_acc': val_acc,
            })
    
    return best_acc

# -----------------------------------------------------------------------------
# Main Training Function (run on each process)
# -----------------------------------------------------------------------------

def main_worker(rank, world_size, args):
    # Setup DDP
    ddp_setup(rank, world_size)
    device = rank
    
    # Initialize wandb only on rank 0
    if rank == 0:
        wandb.init(project='continual-learning-splits', name=f'{args.model_type}_split_cifar10')
        wandb.config.update({
            'model_type': args.model_type,
            'epochs_task_a': args.epochs_task_a,
            'epochs_task_b': args.epochs_task_b,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'dim': args.dim,
            'drop_path_rate': args.drop_path_rate,
            'memory_chunk_size': args.memory_chunk_size,
            'num_gpus': world_size,
        })
    
    # Results tracking
    results = {
        'task_A_acc_phase1': None,
        'task_A_acc_phase2': None,
        'task_B_acc': None,
        'forgetting': None,
    }
    
    # =========================================================================
    # Phase 1: Train on Task A (Classes 0-4)
    # =========================================================================
    if args.phase in ['1', 'both']:
        if rank == 0:
            print("\n" + "="*60)
            print("Phase 1: Training on Task A (Classes 0-4)")
            print("="*60)
        
        num_classes_A = 5
        model = create_model(
            model_type=args.model_type,
            num_classes=num_classes_A,
            dim=args.dim,
            drop_path_rate=args.drop_path_rate,
            memory_chunk_size=args.memory_chunk_size
        ).to(device)
        
        # Wrap with DDP
        model = DDP(model, device_ids=[rank], output_device=rank)
        
        if rank == 0:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total Parameters: {params:,}")
        
        # Get data loaders for Task A
        loaders_A = get_data_loaders(args.batch_size, task='A', rank=rank, world_size=world_size)
        
        # Train on Task A
        best_acc_A = train_phase(
            rank, world_size, model, loaders_A['train'], loaders_A['test'], 
            'task_A_phase1', args.epochs_task_a, args.lr, device
        )
        
        results['task_A_acc_phase1'] = best_acc_A
        if rank == 0:
            print(f"\nTask A Best Validation Accuracy: {best_acc_A:.2f}%")
            
            # Save checkpoint
            os.makedirs(args.save_path, exist_ok=True)
            checkpoint_A_path = os.path.join(args.save_path, f'{args.model_type}_phase1_taskA.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes_A,
                'results': results,
            }, checkpoint_A_path)
            print(f"Checkpoint saved to {checkpoint_A_path}")
    
    # =========================================================================
    # Phase 2: Train on Task B (Classes 5-9)
    # =========================================================================
    if args.phase in ['2', 'both']:
        if rank == 0:
            print("\n" + "="*60)
            print("Phase 2: Training on Task B (Classes 5-9)")
            print("="*60)
        
        # For Memory ViT: Freeze attention layers
        if args.model_type == 'memory':
            model.module.freeze_attention()
        
        # Replace classification head for Task B
        num_classes_B = 5
        model.module.to_logits = nn.Linear(args.dim, num_classes_B).to(device)
        
        # Get data loaders for Task B
        loaders_B = get_data_loaders(args.batch_size, task='B', rank=rank, world_size=world_size)
        loaders_A_test = get_data_loaders(args.batch_size, task='A', rank=rank, world_size=world_size)
        
        if rank == 0:
            print("\nTraining on Task B...")
        
        train_phase(
            rank, world_size, model, loaders_B['train'], loaders_B['test'],
            'task_B_phase2', args.epochs_task_b, args.lr, device
        )
        
        # Evaluate on Task B
        if rank == 0:
            print("\nEvaluating on Task B...")
        _, task_B_acc = evaluate(model, loaders_B['test'], device, rank, world_size)
        results['task_B_acc'] = task_B_acc
        if rank == 0:
            print(f"Task B Accuracy: {task_B_acc:.2f}%")
        
        # Evaluate on Task A (measuring forgetting)
        if rank == 0:
            print("\nEvaluating on Task A (measuring forgetting)...")
        _, task_A_acc_phase2 = evaluate(model, loaders_A_test['test'], device, rank, world_size)
        results['task_A_acc_phase2'] = task_A_acc_phase2
        
        forgetting = results['task_A_acc_phase1'] - results['task_A_acc_phase2']
        results['forgetting'] = forgetting
        
        if rank == 0:
            print("\n" + "="*60)
            print("RESULTS SUMMARY")
            print("="*60)
            print(f"Task A Accuracy (Phase 1): {results['task_A_acc_phase1']:.2f}%")
            print(f"Task A Accuracy (Phase 2): {results['task_A_acc_phase2']:.2f}%")
            print(f"Task B Accuracy:           {results['task_B_acc']:.2f}%")
            print(f"Forgetting:                {results['forgetting']:.2f}%")
            print("="*60)
            
            wandb.log({
                'task_A_acc_phase1': results['task_A_acc_phase1'],
                'task_A_acc_phase2': results['task_A_acc_phase2'],
                'task_B_acc': results['task_B_acc'],
                'forgetting': results['forgetting'],
            })
            
            # Save final checkpoint
            checkpoint_final_path = os.path.join(args.save_path, f'{args.model_type}_final.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': num_classes_B,
                'results': results,
            }, checkpoint_final_path)
            print(f"\nFinal checkpoint saved to {checkpoint_final_path}")
    
    if rank == 0:
        wandb.finish()
        print("\nExperiment complete!")
    
    destroy_process_group()

# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

@click.command()
@click.option('--model_type', type=click.Choice(['standard', 'memory']), required=True)
@click.option('--phase', type=click.Choice(['1', '2', 'both']), default='both')
@click.option('--epochs_task_a', default=50)
@click.option('--epochs_task_b', default=50)
@click.option('--batch_size', default=128)
@click.option('--lr', default=6e-4)
@click.option('--dim', default=192)
@click.option('--drop_path_rate', default=0.1)
@click.option('--memory_chunk_size', default=64)
@click.option('--checkpoint_path', default=None)
@click.option('--save_path', default='./checkpoints')
def main(model_type, phase, epochs_task_a, epochs_task_b, batch_size, lr, dim, 
         drop_path_rate, memory_chunk_size, checkpoint_path, save_path):
    
    # Create args object for main_worker using SimpleNamespace
    args = SimpleNamespace(
        model_type=model_type,
        phase=phase,
        epochs_task_a=epochs_task_a,
        epochs_task_b=epochs_task_b,
        batch_size=batch_size,
        lr=lr,
        dim=dim,
        drop_path_rate=drop_path_rate,
        memory_chunk_size=memory_chunk_size,
        checkpoint_path=checkpoint_path,
        save_path=save_path
    )
    
    # Detect number of GPUs
    world_size = torch.cuda.device_count()
    print(f"Starting DDP training with {world_size} GPUs")
    
    # Launch DDP processes
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
