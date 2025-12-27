import os
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from einops import rearrange
from einops.layers.torch import Rearrange
from titans_pytorch import MemoryAsContextTransformer

class TitansImageGenerator(nn.Module):
    def __init__(
        self,
        dim=256,
        patch_size=4,
        num_classes=10,
        depth=4,
        heads=8,
        segment_len=16,
        num_longterm_mem_tokens=8,
        channels=1,
        image_size=28
    ):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        self.channels = channels
        self.image_size = image_size
        self.patch_dim = channels * (patch_size ** 2)
        
        # Class conditioning embedding
        self.class_emb = nn.Embedding(num_classes, dim)
        
        # Patch embedding components
        self.patch_emb_layer = nn.Sequential(
            nn.Conv2d(channels, dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim // 2),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            Rearrange('b d h w -> b (h w) d'),
            nn.LayerNorm(dim)
        )
        
        # Core Titans Model
        # We use Identity for token_emb because we manually construct the input sequence
        # by concatenating class embeddings and patch embeddings.
        self.titans = MemoryAsContextTransformer(
            dim=dim,
            depth=depth,
            num_tokens=1, # Placeholder, not used due to manual embedding
            token_emb=nn.Identity(), 
            segment_len=segment_len,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            heads=heads,
            neural_memory_layers=(2, 4) if depth >= 4 else (2,),
            neural_mem_weight_residual=True,
            neural_memory_qkv_receives_diff_views=True,
        )
        
        # Predictor head: projects latent dim back to pixel space
        self.to_pixels = nn.Linear(dim, self.patch_dim)

    def get_patch_embeddings(self, img):
        # img: (b, c, h, w)
        return self.patch_emb_layer(img)

    def forward(self, img, labels):
        # img: (b, 1, 28, 28)
        # labels: (b,)
        
        # 1. Embed inputs
        patch_embs = self.get_patch_embeddings(img) # (b, N, dim)
        class_embs = self.class_emb(labels).unsqueeze(1) # (b, 1, dim)
        
        # 2. Construct Autoregressive Input
        # We want to predict P_i given Class, P_0, ..., P_{i-1}
        # Input sequence: [Class, P_0, ..., P_{N-2}] (Length N)
        input_seq = torch.cat((class_embs, patch_embs[:, :-1]), dim=1)
        
        # 3. Pass through Titans
        out = self.titans(input_seq, return_embeddings=True) # (b, N, dim)
        
        # 4. Predict Pixels
        pred_pixels = self.to_pixels(out) # (b, N, patch_dim)
        
        # 5. Calculate Loss
        # Target sequence: [P_0, P_1, ..., P_{N-1}] (Length N)
        target_patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        return F.mse_loss(pred_pixels, target_patches)

    @torch.no_grad()
    def sample(self, labels, device):
        # labels: (b,)
        b = labels.shape[0]
        h = w = self.image_size // self.patch_size
        seq_len = h * w
        
        # Start with class token
        current_emb = self.class_emb(labels).unsqueeze(1) # (b, 1, dim)
        inputs = current_emb
        
        generated_patches = []
        
        # Autoregressive generation loop
        for i in range(seq_len):
            # Forward pass (re-processing full sequence for simplicity)
            out = self.titans(inputs, return_embeddings=True)
            
            # Get the prediction for the next patch (last token output)
            last_out = out[:, -1:]
            pred_patch = self.to_pixels(last_out) # (b, 1, patch_dim)
            
            generated_patches.append(pred_patch)
            
            # Prepare input for next step (if not finished)
            if i < seq_len - 1:
                pred_patch_img = rearrange(pred_patch, 'b 1 (p1 p2 c) -> b c p1 p2', p1=self.patch_size, p2=self.patch_size, c=self.channels)
                pred_patch_emb = self.patch_emb_layer(pred_patch_img)
                inputs = torch.cat((inputs, pred_patch_emb), dim=1)
                
        # Reassemble image
        full_patches = torch.cat(generated_patches, dim=1) # (b, N, patch_dim)
        img = rearrange(full_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=h, w=w, p1=self.patch_size, p2=self.patch_size, c=self.channels)
        return img.clamp(0, 1)

@click.command()
@click.option('--batch_size', default=64, help='Batch size')
@click.option('--epochs', default=20, help='Number of epochs')
@click.option('--lr', default=1e-3, help='Learning rate')
@click.option('--patch_size', default=4, help='Patch size')
@click.option('--dim', default=256, help='Model dimension')
@click.option('--depth', default=4, help='Transformer depth')
@click.option('--dataset', default='mnist', help='Dataset to use: mnist or cifar10')
def train(batch_size, epochs, lr, patch_size, dim, depth, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    os.makedirs('./data', exist_ok=True)
    
    if dataset == 'mnist':
        ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        channels = 1
        image_size = 28
    elif dataset == 'cifar10':
        ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        channels = 3
        image_size = 32
        
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    
    # Model
    model = TitansImageGenerator(dim=dim, patch_size=patch_size, depth=depth, num_classes=10, channels=channels, image_size=image_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(loader), epochs=epochs)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            loss = model(imgs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
        
        # Generate samples
        if (epoch + 1) % 1 == 0:
            model.eval()
            sample_labels = torch.arange(10).to(device).repeat(2) # 2 rows of 0-9
            samples = model.sample(sample_labels, device)
            os.makedirs('results', exist_ok=True)
            save_image(samples, f'results/epoch_{epoch+1}.png', nrow=10)

if __name__ == '__main__':
    train()