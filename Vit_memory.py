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
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 192,           # Small dimension for CIFAR
        depth = 6,
        heads = 3,
        memory_chunk_size = 64 # Equal to sequence length (8*8)
    ):
        super().__init__()
        self.patch_size = patch_size
        assert image_size % patch_size == 0
        
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        # 1. Patch Embedding
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # 2. Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        
        # 3. CLS Token (Placed at the END to see full memory context)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 4. Transformer Layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                MemoryFFNTransformerBlock(
                    dim=dim, 
                    heads=heads, 
                    memory_chunk_size=memory_chunk_size
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
        
        # 3. Prepend CLS Token
        # Note: Standard ViT prepends, but we APPEND it to the end for the causal memory
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((x, cls_tokens), dim = 1) 
        
        # 4. Pass through Memory Layers
        memory_states = [None] * len(self.layers)
        
        for i, layer in enumerate(self.layers):
            x, memory_states[i] = layer(x, memory_state=memory_states[i])
            
        # 5. Get CLS Token Output
        # It's at the last position (-1)
        cls_token_out = x[:, -1]
        
        return self.to_logits(self.norm(cls_token_out))

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

@click.command()
@click.option('--batch_size', default=64, help='Batch size')
@click.option('--epochs', default=50, help='Number of epochs')
@click.option('--lr', default=3e-4, help='Learning rate')
@click.option('--dim', default=192, help='Model dimension')
def train(batch_size, epochs, lr, dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
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
    
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Model
    model = MemoryViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=dim,
        depth=6,
        heads=3,
        memory_chunk_size=64 # 8x8 patches + 1 CLS = 65, so chunk 64 is perfect
    ).to(device)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
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
            
        scheduler.step()
        acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%")
        
        # Validation
        if (epoch + 1) % 5 == 0:
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    test_loss += loss.item()
                    _, predicted = logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            print(f"--> Test Loss: {test_loss/len(test_loader):.4f} | Test Acc: {100.*correct/total:.2f}%")

if __name__ == '__main__':
    train()