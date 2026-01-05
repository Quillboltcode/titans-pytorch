import os
import math
import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from einops import rearrange
from PIL import Image
from titans_pytorch import MemoryAsContextTransformer

# -----------------------------------------------------------------------------
# VQ-VAE Components (Stage 1)
# -----------------------------------------------------------------------------

class KaggleCelebA(torch.utils.data.Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.transform = transform
        import pandas as pd
        
        # Load CSVs
        attr_df = pd.read_csv(os.path.join(root, 'list_attr_celeba.csv'))
        part_df = pd.read_csv(os.path.join(root, 'list_eval_partition.csv'))
        
        self.df = pd.merge(attr_df, part_df, on='image_id')
        
        # Filter split
        # 0: Train, 1: Valid, 2: Test
        split_map = {'train': 0, 'valid': 1, 'test': 2}
        self.df = self.df[self.df['partition'] == split_map.get(split, 0)].reset_index(drop=True)
        
        # Image directory
        self.img_dir = os.path.join(root, 'img_align_celeba/img_align_celeba')
        if not os.path.exists(self.img_dir):
            self.img_dir = os.path.join(root, 'img_align_celeba')
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Target: Male (1) or Female (0)
        # CelebA attributes: 1 for present, -1 for absent
        attr = row['Male']
        label = 1 if attr == 1 else 0
        
        return image, torch.tensor(label).long()

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs: (B, Dim, H, W) -> (B, H, W, Dim)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        # (B, Dim, H, W)
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices.view(input_shape[0], input_shape[1], input_shape[2])

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1)
        )
    def forward(self, x):
        return x + self.block(x)

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, embedding_dim, 1)
        )
        
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, 1, 1),
            ResBlock(hidden_dim),
            ResBlock(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim // 2, in_channels, 4, 2, 1),
            nn.Sigmoid() # Assuming input is 0-1
        )

    def forward(self, x):
        z = self.encoder(x)
        loss, quantized, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return loss, x_recon

    @torch.no_grad()
    def get_indices(self, x):
        z = self.encoder(x)
        _, _, indices = self.vq(z)
        return indices

    @torch.no_grad()
    def decode_indices(self, indices):
        # indices: (B, H, W)
        quantized = F.embedding(indices, self.vq.embedding.weight)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return self.decoder(quantized)

# -----------------------------------------------------------------------------
# Titans Latent Model (Stage 2)
# -----------------------------------------------------------------------------

class TitansLatentImageGenerator(nn.Module):
    def __init__(
        self,
        vqvae,
        dim=512,
        num_classes=10,
        depth=12,
        heads=8,
        segment_len=32,
        num_longterm_mem_tokens=0, # MAG do not use longterm memory by default
        max_seq_len=1024
    ):
        super().__init__()
        self.vqvae = vqvae
        # Freeze VQVAE
        for p in self.vqvae.parameters():
            p.requires_grad = False
            
        self.num_tokens = vqvae.num_embeddings
        
        # Embeddings
        self.class_emb = nn.Embedding(num_classes, dim)
        self.token_emb = nn.Embedding(self.num_tokens, dim)
        side_len = int(math.sqrt(max_seq_len))
        self.pos_emb = nn.Parameter(torch.randn(1, side_len, side_len, dim))
        
        # Titans
        # Distribute memory layers across depth
        mem_layers = []
        if depth >= 4:
            mem_layers = [2, depth // 2, depth - 2]
        else:
            mem_layers = [depth - 1]
            
        self.titans = MemoryAsContextTransformer(
            dim=dim,
            depth=depth,
            num_tokens=self.num_tokens, # Not used directly due to manual embedding
            token_emb=nn.Identity(), 
            segment_len=segment_len,
            # Memory as gating parameters
            neural_mem_gate_attn_output=True,
            num_longterm_mem_tokens=num_longterm_mem_tokens,
            num_persist_mem_tokens=4,
            heads=heads,
            neural_memory_layers=tuple(mem_layers),
            neural_mem_weight_residual=True,
            neural_memory_qkv_receives_diff_views=True,
        )
        
        self.to_logits = nn.Linear(dim, self.num_tokens)

    def forward(self, img, labels):
        # img: (B, C, H, W)
        # labels: (B,)
        
        # 1. Get discrete tokens from VQVAE
        with torch.no_grad():
            indices = self.vqvae.get_indices(img) # (B, H', W')
        
        b, h, w = indices.shape
            
        # Flatten indices
        indices = rearrange(indices, 'b h w -> b (h w)') # (B, N)
        n = indices.shape[1]
        
        # 2. Embed inputs
        # Class embedding
        c_emb = self.class_emb(labels).unsqueeze(1) # (B, 1, dim)
        
        # Token embeddings + Positional embeddings
        t_emb = self.token_emb(indices) # (B, N, dim)
        
        pos_emb = self.pos_emb[:, :h, :w, :]
        pos_emb = rearrange(pos_emb, 'b h w d -> b (h w) d')
        t_emb = t_emb + pos_emb
        
        # 3. Construct Autoregressive Input
        # Input: [Class, T_0, ..., T_{N-2}]
        input_seq = torch.cat((c_emb, t_emb[:, :-1]), dim=1)
        
        # 4. Pass through Titans
        out = self.titans(input_seq, return_embeddings=True) # (B, N, dim)
        
        # 5. Predict Logits
        logits = self.to_logits(out) # (B, N, num_tokens)
        
        # 6. Calculate Loss (Cross Entropy)
        # Target: [T_0, T_1, ..., T_{N-1}] (which is `indices`)
        return F.cross_entropy(rearrange(logits, 'b n c -> b c n'), indices)

    @torch.no_grad()
    def sample(self, labels, shape, device, top_k=None):
        # labels: (B,)
        # shape: (H_latent, W_latent)
        h, w = shape
        seq_len = h * w
        b = labels.shape[0]
        
        # Start with class token
        current_emb = self.class_emb(labels).unsqueeze(1) # (B, 1, dim)
        inputs = current_emb
        
        generated_indices = []
        
        pos_emb = self.pos_emb[:, :h, :w, :]
        pos_emb = rearrange(pos_emb, 'b h w d -> b (h w) d')
        
        for i in range(seq_len):
            # Forward pass
            out = self.titans(inputs, return_embeddings=True)
            
            # Predict next token from last output
            last_out = out[:, -1:]
            logits = self.to_logits(last_out)
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[..., [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs.squeeze(1), 1) # (B, 1)
            
            generated_indices.append(next_token)
            
            if i < seq_len - 1:
                next_emb = self.token_emb(next_token)
                # Add pos emb
                next_emb = next_emb + pos_emb[:, i:i+1]
                inputs = torch.cat((inputs, next_emb), dim=1)
        
        # Reassemble indices
        indices = torch.cat(generated_indices, dim=1) # (B, N)
        indices = indices.view(b, h, w)
        
        # Decode with VQVAE
        return self.vqvae.decode_indices(indices)

@click.command()
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--kaggle_root', default='/kaggle/input/face-vae', help='Kaggle dataset root')
@click.option('--grad_accum_steps', default=4, help='Gradient accumulation steps')
@click.option('--epochs_vq', default=10, help='Epochs for VQVAE')
@click.option('--epochs_titans', default=50, help='Epochs for Titans')
@click.option('--lr', default=3e-4, help='Learning rate')
@click.option('--dim', default=384, help='Model dimension')
@click.option('--depth', default=8, help='Transformer depth')
@click.option('--dataset', default='cifar10', help='Dataset: cifar10 or mnist or celeba')
def train(batch_size, kaggle_root, grad_accum_steps, epochs_vq, epochs_titans, lr, dim, depth, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)
    
    num_classes = 10

    if dataset == 'cifar10':
        ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        channels = 3
        image_size = 32
    elif dataset == 'mnist':
        ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
        channels = 1
        image_size = 28
    elif dataset == 'celeba':
        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ])
        
        if os.path.exists(kaggle_root):
            print(f"Loading CelebA from Kaggle dataset at {kaggle_root}")
            ds = KaggleCelebA(kaggle_root, split='train', transform=transform)
        else:
            try:
                import gdown
            except ImportError:
                os.system('pip install gdown')
            ds = datasets.CelebA('./data', split='train', download=True, transform=transform, target_type='attr', target_transform=lambda t: t[20].long())
            
        channels = 3
        image_size = 128
        num_classes = 2
        
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    # Initialize VQVAE
    vqvae = VQVAE(in_channels=channels, hidden_dim=256, num_embeddings=512, embedding_dim=128).to(device)
    
    vq_path = f'./checkpoints/vqvae_{dataset}.pt'
    if os.path.exists(vq_path):
        print("Loading VQVAE from checkpoint...")
        vqvae.load_state_dict(torch.load(vq_path))
    else:
        print("Training VQVAE...")
        opt_vq = optim.Adam(vqvae.parameters(), lr=1e-3)
        for epoch in range(epochs_vq):
            vqvae.train()
            total_loss = 0
            for imgs, _ in loader:
                imgs = imgs.to(device)
                loss, recon = vqvae(imgs)
                recon_loss = F.mse_loss(recon, imgs)
                total = loss + recon_loss
                
                opt_vq.zero_grad()
                total.backward()
                opt_vq.step()
                total_loss += total.item()
            
            print(f"VQVAE Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")
            if (epoch+1) % 5 == 0:
                save_image(torch.cat([imgs[:10], recon[:10].detach()], dim=0), f'results/vq_recon_{dataset}_{epoch+1}.png', nrow=10)
        
        torch.save(vqvae.state_dict(), vq_path)
        
        # Cleanup VQVAE optimizer and cache to free memory for Titans
        del opt_vq
        torch.cuda.empty_cache()

    # Initialize Titans
    print("Training Titans...")
    model = TitansLatentImageGenerator(
        vqvae=vqvae,
        dim=dim,
        depth=depth,
        num_classes=num_classes,
        segment_len=32, # Increased segment length
        num_longterm_mem_tokens=0
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(loader) // grad_accum_steps, epochs=epochs_titans)
    scaler = torch.amp.GradScaler('cuda')
    
    latent_h = image_size // 4 # VQVAE downsamples by 4
    
    for epoch in range(epochs_titans):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for idx, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda'):
                loss = model(imgs, labels) / grad_accum_steps
            
            scaler.scale(loss).backward()
            
            if (idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            
        print(f"Titans Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            sample_labels = torch.arange(num_classes).to(device)
            sample_labels = sample_labels.repeat(20 // num_classes + 1)[:20]
            samples = model.sample(sample_labels, (latent_h, latent_h), device)
            save_image(samples, f'results/titans_sample_{dataset}_{epoch+1}.png', nrow=10)

if __name__ == '__main__':
    train()
