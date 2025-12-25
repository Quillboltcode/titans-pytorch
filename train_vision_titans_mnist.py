import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from einops import rearrange
from einops.layers.torch import Rearrange
from titans_pytorch import MemoryAsContextTransformer

class TitansVisionModel(Module):
    def __init__(self, dim=512, patch_size=4, depth=6):
        super().__init__()
        self.patch_size = patch_size
        patch_dim = 3 * (patch_size ** 2) # 48 for CIFAR
        
        self.titans = MemoryAsContextTransformer(
            dim=dim,
            depth=depth,
            num_tokens=0, # Not used
            segment_len=16, # 8x8 patches = 64 total; segment of 16 handles 1/4th of image
            num_longterm_mem_tokens=8,
            neural_memory_layers=(2, 4, 6), # Inject memory into specific layers
            token_emb=nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
                nn.Linear(patch_dim, dim)
            )
        )
        self.patch_predictor = nn.Linear(dim, patch_dim)

    def forward(self, img):
        # 1. Create patches for the 'labels' (target)
        # We want the model to predict the pixels of the next patch
        patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                            p1=self.patch_size, p2=self.patch_size)
        
        # 2. Get Transformer outputs
        # Shift patches to create autoregressive targets
        input_patches = patches[:, :-1]
        target_patches = patches[:, 1:]
        
        logits = self.titans(input_patches)
        predictions = self.patch_predictor(logits)
        
        # 3. Continuous Loss (The "Vision Titans" way)
        return F.mse_loss(predictions, target_patches)

class TitansClassifier(nn.Module):
    def __init__(
        self,
        *,
        img_size=224,
        patch_size=16,
        channels=3,
        num_classes=1000,
        dim=512,
        **kwargs
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = channels * (patch_size ** 2)

        # 1. Custom Patch Embedding instead of nn.Embedding
        self.patch_to_emb = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 2. Initialize the core MAC Transformer
        # We set num_tokens to 1 just as a placeholder because we use custom token_emb
        self.transformer = MemoryAsContextTransformer(
            dim=dim,
            num_tokens=1, 
            token_emb=self.patch_to_emb,
            **kwargs
        )

        # 3. Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # Forward through the Titans MAC Transformer
        # x will have shape: [batch, seq_len, dim]
        # Note: The MAC Transformer automatically handles NLM interspersing
        x = self.transformer(img, return_embeddings=True)

        # 4. Global Average Pooling (GAP)
        # We average across the sequence dimension (n)
        x = x.mean(dim=1) 

        # 5. Classify
        return self.mlp_head(x)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from einops import rearrange
import click

# Note: Ensure the TitansClassifier and MemoryAsContextTransformer 
# from the previous steps are defined in your script.

@click.command()
@click.option('--batch_size', default=128, help='Batch size for training')
@click.option('--epochs', default=50, help='Number of training epochs')
@click.option('--learning_rate', default=1e-3, help='Learning rate')
@click.option('--patch_size', default=4, help='Patch size for image tokenization')
@click.option('--dim', default=256, help='Dimension of the model')
@click.option('--depth', default=6, help='Depth of the transformer')
@click.option('--heads', default=8, help='Number of attention heads')
@click.option('--segment_len', default=16, help='Segment length for memory context')
@click.option('--num_longterm_mem_tokens', default=8, help='Number of long-term memory tokens')
def train_cifar10(batch_size, epochs, learning_rate, patch_size, dim, depth, heads, segment_len, num_longterm_mem_tokens):
    # 1. Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Data Augmentation & Loading
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

    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Model Initialization
    # We adapt the kwargs to the MemoryAsContextTransformer logic
    model = TitansClassifier(
        img_size=32,
        patch_size=patch_size,
        channels=3,
        num_classes=10,
        dim=dim,
        depth=depth,
        segment_len=segment_len,             # Processes 1/4 of the image per segment
        num_longterm_mem_tokens=num_longterm_mem_tokens,  # NLM capacity
        heads=heads,
        neural_mem_weight_residual=True,     # Critical for stability/performance per train_mac.py
        neural_memory_qkv_receives_diff_views=True, # Allows flexible grafting of memory
        neural_memory_batch_size=16,         # Updates memory weights ~4 times per image (64 patches)
        neural_memory_layers=(2, 4),
        neural_memory_kwargs=dict(
            dim_head=64,
            heads=4,                         # Multi-head memory is more expressive
            qk_rmsnorm=True,
            momentum=True,
            spectral_norm_surprises=True,
            default_step_transform_max_lr=1e-1,
            per_parameter_lr_modulation=True, # Learned LR per weight matrix
            attn_pool_chunks=True            # Better than AvgPool for memory updates
        )
    ).to(device)

    # 4. Optimizer & Loss
    # AdamW is generally preferred for Transformer-based architectures
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=epochs
    )

    # 5. Training Loop
    print(f"Starting training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping is recommended for Titans/NLM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # 6. Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.3f} | "
              f"Acc: {100.*correct/total:.2f}% | Val Acc: {100.*val_correct/val_total:.2f}%")

    print("Training Complete.")

if __name__ == "__main__":
    train_cifar10()
