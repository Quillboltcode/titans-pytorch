import os
import math
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange, repeat
from titans_pytorch.neural_memory import NeuralMemory
from tqdm import tqdm
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
import timm

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
        # Attention
        attn_residual = x
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = attn_residual + self.drop_path(attn_out)
        
        # MLP
        mlp_residual = x
        x_norm = self.norm_mlp(x)
        mlp_out = self.mlp(x_norm)
        x = mlp_residual + self.drop_path(mlp_out)
        
        return x

# -----------------------------------------------------------------------------
# Memory Transformer Block (Strategy 2)
# -----------------------------------------------------------------------------

class MemoryFFNTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        memory_chunk_size = 64,
        num_persistent_mem_tokens = 4
    ):
        super().__init__()
        
        # 1. Standard Attention
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        
        # 2. Neural Memory (Replacing FFN)
        self.norm_mem = nn.LayerNorm(dim)
        
        # Initialize Neural Memory
        # qkv_receives_diff_views = False for this standard implementation
        self.neural_memory = NeuralMemory(
            dim = dim,
            chunk_size = memory_chunk_size,
            qkv_receives_diff_views = False
        )
        
        self.to_out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, memory_state = None):
        # Attention Branch
        attn_residual = x
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = attn_residual + attn_out
        
        # Memory Branch (Replacing FFN)
        mem_residual = x
        x_norm = self.norm_mem(x)
        
        # Neural Memory Forward
        mem_out, next_memory_state = self.neural_memory(
            x_norm, 
            state = memory_state
        )
        
        # Combine
        x = mem_residual + self.to_out(mem_out)
        
        return x, next_memory_state

# -----------------------------------------------------------------------------
# Memory ViT for Classification
# -----------------------------------------------------------------------------

class MemoryViT(nn.Module):
    def __init__(
        self,
        image_size = 224,
        patch_size = 16,
        num_classes = 10,
        dim = 192,           # Small dimension for CIFAR
        vit_depth = 4,
        memory_depth = 2,
        heads = 3,
        memory_chunk_size = 196, # Equal to sequence length (14*14)
        drop_path_rate = 0.
    ):
        super().__init__()
        self.patch_size = patch_size
        assert image_size % patch_size == 0
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        memory_chunk_size = min(memory_chunk_size, num_patches)
        
        # 1. Patch Embedding
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # 2. Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # +1 for CLS token
        
        # 3. CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 4. Transformer Layers
        self.layers = nn.ModuleList([])
        total_depth = vit_depth + memory_depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]  # stochastic depth decay rule
        
        # Standard ViT Layers
        for i in range(vit_depth):
            self.layers.append(
                StandardTransformerBlock(
                    dim=dim,
                    heads=heads,
                    drop_path=dpr[i]
                )
            )
        # Memory ViT Layers
        for i in range(vit_depth, total_depth):
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
        device = img.device
        
        # 1. Patchify
        # img: (B, 3, 32, 32) -> (B, 64, 48)
        patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.to_patch_embedding(patches)
        
        b, n, _ = x.shape
        
        # 2. Add Positional Embedding
        x += self.pos_embedding[:, :n]
        
        # 3. Append CLS Token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        cls_tokens += self.pos_embedding[:, n:]
        x = torch.cat((x, cls_tokens), dim = 1)
        
        # 4. Pass through Memory Layers
        memory_states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MemoryFFNTransformerBlock):
                x, memory_states[i] = layer(x, memory_state=memory_states[i])
            else:
                x = layer(x)
            
        # 5. Get CLS Token Output
        cls_token_out = x[:, -1]
        
        return self.to_logits(self.norm(cls_token_out))

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

@click.command()
@click.option('--batch_size', default=64, help='Batch size')
@click.option('--epochs', default=100, help='Number of epochs')
@click.option('--lr', default=5e-4, help='Learning rate')
@click.option('--dim', default=192, help='Model dimension')
@click.option('--image_size', default=32, help='Image size (e.g. 32 or 224)')
@click.option('--patch_size', default=4, help='Patch size')
@click.option('--memory_chunk_size', default=196, help='Memory chunk size')
@click.option('--drop_path_rate', default=0.1, help='Stochastic depth rate')
@click.option('--wandb_project', default='memory-vit-cifar10', help='WandB Project Name')
@click.option('--resume', default=None, help='Path to checkpoint to resume training')
@click.option('--gradient_accumulation_steps', default=4, help='Number of steps for gradient accumulation')
@click.option('--simple_aug', is_flag=True, help='Use simple augmentation (only resize and flip)')
@click.option('--vit_depth', default=4, help='Number of standard ViT layers')
@click.option('--memory_depth', default=2, help='Number of memory ViT layers')
def train(batch_size, epochs, lr, dim, image_size, patch_size, memory_chunk_size, drop_path_rate, wandb_project, resume, gradient_accumulation_steps, simple_aug, vit_depth, memory_depth):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    
    if accelerator.is_main_process:
        print(f"Training on {device}")
        wandb.init(project=wandb_project, config={
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "dim": dim,
            "image_size": image_size,
            "patch_size": patch_size,
            "memory_chunk_size": memory_chunk_size,
            "drop_path_rate": drop_path_rate,
            "architecture": "MemoryViT",
            "simple_aug": simple_aug,
            "vit_depth": vit_depth,
            "memory_depth": memory_depth
        })
    
    # CONFIGURE DATASET-SPECIFIC STATS HERE!
    if image_size <= 32:
        MEAN = (0.4914, 0.4822, 0.4465)
        STD = (0.2023, 0.1994, 0.2010)
    else:
        MEAN = (0.485, 0.456, 0.406)
        STD = (0.229, 0.224, 0.225)

    if simple_aug:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        # SMART HYPERPARAMETERS
        policy = 'rand-m5-mstd0.5-inc1' if image_size <= 32 else 'rand-m9-mstd0.5-n2'
        translate_const = max(5, int(image_size * 0.1))  # Safe for all sizes
        
        transform_train = transforms.Compose([
            # CRITICAL ORDER FIX
            timm.data.auto_augment.rand_augment_transform(
                policy,
                hparams={'translate_const': translate_const}  # NO img_mean NEEDED
            ),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    # TEST TRANSFORM FIX
    if image_size <= 32:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        resize_size = int(image_size / 0.875 + 0.5)  # Proper rounding
        transform_test = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    
    if accelerator.is_main_process:
        datasets.CIFAR10(root='./data', train=True, download=True)
        datasets.CIFAR10(root='./data', train=False, download=True)
    accelerator.wait_for_everyone()

    # Note: CIFAR10 images are 32x32
    trainset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Model
    model = MemoryViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=10,
        dim=dim,
        vit_depth=vit_depth,
        memory_depth=memory_depth,
        heads=3,
        memory_chunk_size=memory_chunk_size,
        drop_path_rate=drop_path_rate
    )

    # Count parameters
    if accelerator.is_main_process:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # Warm-up scheduler for the first 5 epochs
    warmup_epochs = 3
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs * len(train_loader))

    # Main scheduler after warm-up
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    # Use CrossEntropyLoss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model, optimizer, train_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, test_loader
    )
    
    # Register schedulers for checkpointing
    accelerator.register_for_checkpointing(warmup_scheduler)
    accelerator.register_for_checkpointing(main_scheduler)

    # Load checkpoint if resuming training
    start_epoch = 0
    if resume:
        if accelerator.is_main_process:
            print(f"Resuming training from {resume}")
        accelerator.load_state(resume)
        try:
            # Correctly parse epoch from directory name
            start_epoch = int(os.path.basename(os.path.normpath(resume)).split('_')[-1])
        except (ValueError, IndexError):
            print("Could not parse epoch from checkpoint path, starting from 0.")
            start_epoch = 0
        
        if accelerator.is_main_process:
            print(f"Resumed training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for step, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} Training", disable=not accelerator.is_main_process)):
            
            logits = model(imgs)
            loss = criterion(logits, labels)
            accelerator.backward(loss)
            
            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        # Handle warm-up and main schedulers
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        if accelerator.is_main_process:
            acc = 100. * correct / total
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
            
            wandb.log({
                "train_loss": total_loss/len(train_loader),
                "train_acc": acc,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1
            })
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or (epoch + 1) == 1:
            checkpoint_path = f"checkpoint_epoch_{epoch+1}"
            accelerator.save_state(checkpoint_path)
            if accelerator.is_main_process:
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in tqdm(test_loader, desc=f"Epoch {epoch+1} Validation", disable=not accelerator.is_main_process):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    test_loss += loss.item()
                    
                    # Gather predictions and labels for accurate metrics across GPUs
                    predictions = logits.argmax(dim=-1)
                    gathered_preds, gathered_labels = accelerator.gather_for_metrics((predictions, labels))
                    
                    total += gathered_labels.size(0)
                    correct += gathered_preds.eq(gathered_labels).sum().item()
            
            if accelerator.is_main_process:
                print(f"--> Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {100.*correct/total:.2f}%")
                wandb.log({
                    "test_loss": test_loss/len(test_loader),
                    "test_acc": 100.*correct/total,
                    "epoch": epoch + 1
                })
            
    if accelerator.is_main_process:
        wandb.finish()

if __name__ == '__main__':
    train()