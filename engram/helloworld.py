import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import random
import os

# ==============================================================================
# MiniVisionEngram Module
# ==============================================================================

class MiniVisionEngram(nn.Module):
    """
    A Bigram Memory Module for Visual Tokens.
    
    This module learns a static embedding for every possible pair of tokens (bigram).
    During inference, it acts as a read-only lookup table. It does NOT update dynamically
    based on test data, ensuring no data leakage occurs.
    """
    def __init__(self, vocab_size=512, embed_dim=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        
        # The "Memory": A table of embeddings for every possible pair of patches
        # Size: 512 * 512 = ~262k distinct patch-pairs
        self.memory_table = nn.Embedding(vocab_size * vocab_size, embed_dim)
        
        # The Gate: Decides if the memory is useful
        self.gate_proj = nn.Linear(embed_dim, 1)
        
        # Output projection for memory
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_tokens, hidden_state):
        # x_tokens: [Batch, Seq_Len] (The visual word IDs)
        
        # 1. Create Bigrams (The "Key")
        prev_1 = torch.roll(x_tokens, 1, dims=1)
        prev_1[:, 0] = 0 # Mask first token
        
        current = x_tokens
        
        # 2. Hash: Create a unique ID for every pair (t-1, t)
        keys = (prev_1 * self.vocab_size) + current
        
        # 3. Lookup
        memory = self.memory_table(keys) # [Batch, Seq, Dim]
        
        # 4. Gating (Simple Sigmoid)
        gate_score = torch.sigmoid(self.gate_proj(hidden_state))
        
        # 5. Apply output projection to memory
        gated_memory = gate_score * self.output_proj(memory)
        gated_memory = self.dropout(gated_memory)
        
        # 6. Output
        output = hidden_state + gated_memory
        
        return output, gate_score.detach()

# ==============================================================================
# NanoGPT Backbone
# ==============================================================================

class NanoGPT(nn.Module):
    def __init__(self, vocab_size=512, embed_dim=128, depth=4, num_heads=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Token Embedding
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        
        # Positional Embedding
        self.pos_emb = nn.Embedding(64, embed_dim)
        
        # Transformer Layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.ModuleDict({
                'ln1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'ln2': nn.LayerNorm(embed_dim),
                'ff': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
            })
            self.layers.append(layer)
        
        # Final Layer Norm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Language Model Head
        self.head = nn.Linear(embed_dim, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x_tokens, targets=None, return_embeddings=False):
        # x_tokens: [Batch, Seq_Len]
        seq_len = x_tokens.size(1)
        
        # Token + Positional Embedding
        emb = self.token_emb(x_tokens) + self.pos_emb(torch.arange(seq_len, device=x_tokens.device))
        
        # Causal Mask
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x_tokens.device) * float('-inf'), diagonal=1)
        
        # Forward through layers
        x = emb
        for layer in self.layers:
            # Attention Block
            x_ln = layer['ln1'](x)
            attn_out, _ = layer['attn'](x_ln, x_ln, x_ln, attn_mask=attn_mask, need_weights=False)
            x = x + attn_out
            
            # Feed Forward Block
            x_ln = layer['ln2'](x)
            ff_out = layer['ff'](x_ln)
            x = x + ff_out
        
        # Final Norm and Head
        x = self.ln_f(x)
        
        if return_embeddings:
            return x
            
        logits = self.head(x)
        
        # Compute Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), targets)
        
        return logits, loss, None

# ==============================================================================
# Engram-Enhanced Model
# ==============================================================================

class EngramEnhancedGPT(nn.Module):
    def __init__(self, vocab_size=512, embed_dim=128, depth=3, num_heads=4, drop_p=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Token Embedding
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        
        # Positional Embedding
        self.pos_emb = nn.Embedding(64, embed_dim)
        
        # Engram Layer (Added after token embedding)
        self.engram = MiniVisionEngram(vocab_size, embed_dim, dropout=drop_p)
        
        # Transformer Layers (1 layer less than baseline)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.ModuleDict({
                'ln1': nn.LayerNorm(embed_dim),
                'attn': nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                'ln2': nn.LayerNorm(embed_dim),
                'ff': nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Linear(embed_dim * 4, embed_dim)
                )
            })
            self.layers.append(layer)
        
        # Final Layer Norm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # Language Model Head
        self.head = nn.Linear(embed_dim, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, x_tokens, targets=None, return_embeddings=False):
        # x_tokens: [Batch, Seq_Len]
        seq_len = x_tokens.size(1)
        
        # Token + Positional Embedding
        emb = self.token_emb(x_tokens) + self.pos_emb(torch.arange(seq_len, device=x_tokens.device))
        
        # Engram Enhancement
        x, gate_scores = self.engram(x_tokens, emb)
        
        # Causal Mask
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x_tokens.device) * float('-inf'), diagonal=1)
        
        # Forward through layers
        for layer in self.layers:
            # Attention Block
            x_ln = layer['ln1'](x)
            attn_out, _ = layer['attn'](x_ln, x_ln, x_ln, attn_mask=attn_mask, need_weights=False)
            x = x + attn_out
            
            # Feed Forward Block
            x_ln = layer['ln2'](x)
            ff_out = layer['ff'](x_ln)
            x = x + ff_out
        
        # Final Norm and Head
        x = self.ln_f(x)
        
        if return_embeddings:
            return x
            
        logits = self.head(x)
        
        # Compute Loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(rearrange(logits, 'b n c -> b c n'), targets)
        
        return logits, loss, gate_scores

# ==============================================================================
# Classification Wrapper
# ==============================================================================

class GPTClassifier(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)
        
    def forward(self, x, targets=None):
        # x: [Batch, Seq_Len]
        # Get features from backbone (Batch, Seq_Len, Dim)
        features = self.backbone(x, return_embeddings=True)
        # Use last token representation for classification
        last_token_features = features[:, -1, :]
        logits = self.head(last_token_features)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

# ==============================================================================
# Data Preprocessing
# ==============================================================================

class TorchKMeans:
    def __init__(self, n_clusters, device='cuda', max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.device = device
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def _to_tensor(self, X):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        return X.to(self.device)

    def fit(self, X):
        X = self._to_tensor(X)
        
        # Initialize centroids randomly
        indices = torch.randperm(X.size(0))[:self.n_clusters]
        self.centroids = X[indices].clone()
        
        for i in range(self.max_iter):
            dists = torch.cdist(X, self.centroids)
            labels = torch.argmin(dists, dim=1)
            
            new_centroids = torch.zeros_like(self.centroids)
            counts = torch.bincount(labels, minlength=self.n_clusters).float().unsqueeze(1)
            new_centroids.index_add_(0, labels, X)
            new_centroids = new_centroids / counts.clamp(min=1)
            
            shift = torch.norm(new_centroids - self.centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break
        return self

    def predict(self, X):
        X = self._to_tensor(X)
        dists = torch.cdist(X, self.centroids)
        return torch.argmin(dists, dim=1).cpu().numpy()

class QuantizedCIFAR10(Dataset):
    def __init__(self, root, train=True, download=True, patch_size=4, num_clusters=512, kmeans=None):
        self.train = train
        self.patch_size = patch_size
        self.num_clusters = num_clusters
        self.kmeans = kmeans
        
        # Transform to extract patches
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.cifar10 = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=transform
        )
        
        # Quantize the dataset
        self.quantized_data = self._quantize_dataset()
        
    def _quantize_dataset(self):
        """Quantize CIFAR-10 images to patch tokens using K-Means clustering"""
        
        print(f"{'Training' if self.train else 'Loading'} patch quantizer...")
        
        # Collect all patches
        all_patches = []
        for img, _ in tqdm(self.cifar10, desc="Extracting patches"):
            patches = rearrange(img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', 
                              p1=self.patch_size, p2=self.patch_size)
            all_patches.append(patches)
        
        all_patches = torch.cat(all_patches, dim=0).numpy()
        
        # Train K-Means if training dataset
        if self.train:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            kmeans = TorchKMeans(n_clusters=self.num_clusters, device=device)
            
            # Subsample for speed if dataset is large (e.g. >100k patches)
            if len(all_patches) > 100000:
                idx = np.random.choice(len(all_patches), 100000, replace=False)
                kmeans.fit(all_patches[idx])
            else:
                kmeans.fit(all_patches)
            self.kmeans = kmeans
        else:
            # For test set, load from training data
            assert self.kmeans is not None, "Must first quantize training data"
        
        # Quantize patches
        quantized_data = []
        for img, label in tqdm(self.cifar10, desc="Quantizing patches"):
            patches = rearrange(img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', 
                              p1=self.patch_size, p2=self.patch_size)
            patch_tokens = self.kmeans.predict(patches)
            quantized_data.append({
                'tokens': torch.tensor(patch_tokens, dtype=torch.long),
                'label': label
            })
        
        return quantized_data
    
    def __len__(self):
        return len(self.cifar10)
    
    def __getitem__(self, idx):
        return self.quantized_data[idx]['tokens'], self.quantized_data[idx]['label']

def get_quantized_cifar10_loaders(batch_size=32, patch_size=4, num_clusters=512):
    """Get DataLoaders for quantized CIFAR-10"""
    train_dataset = QuantizedCIFAR10(
        root='./data',
        train=True,
        download=True,
        patch_size=patch_size,
        num_clusters=num_clusters
    )
    
    test_dataset = QuantizedCIFAR10(
        root='./data',
        train=False,
        download=True,
        patch_size=patch_size,
        num_clusters=num_clusters,
        kmeans=train_dataset.kmeans
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

# ==============================================================================
# Training and Evaluation
# ==============================================================================

def train_epoch(model, loader, optimizer, device, model_name):
    """Train for one epoch with tqdm and gate monitoring"""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    all_gate_scores = []
    
    for tokens, _ in tqdm(loader, desc=f"{model_name} - Training"):
        tokens = tokens.to(device)
        
        # Prepare autoregressive inputs and targets
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        
        optimizer.zero_grad()
        logits, loss, gate_scores = model(inputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == targets).float().mean().item()
        total_acc += acc
        
        # Collect gate scores if using engram model
        if gate_scores is not None:
            all_gate_scores.append(gate_scores.cpu())
    
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    gate_info = None
    if all_gate_scores:
        gate_values = torch.cat(all_gate_scores, dim=0)
        gate_info = {
            'avg_gate_score': gate_values.mean().item(),
            'gate_distribution': gate_values.numpy(),
            'hit_ratio': (gate_values > 0.5).float().mean().item()
        }
    
    return avg_loss, avg_acc, gate_info

def evaluate_epoch(model, loader, device, model_name):
    """Evaluate loss on dataset with tqdm and gate monitoring"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    all_gate_scores = []
    
    with torch.no_grad():
        for tokens, _ in tqdm(loader, desc=f"{model_name} - Evaluating"):
            tokens = tokens.to(device)
            
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            logits, loss, gate_scores = model(inputs, targets)
            total_loss += loss.item()
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            acc = (preds == targets).float().mean().item()
            total_acc += acc
            
            if gate_scores is not None:
                all_gate_scores.append(gate_scores.cpu())
    
    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    gate_info = None
    if all_gate_scores:
        gate_values = torch.cat(all_gate_scores, dim=0)
        gate_info = {
            'avg_gate_score': gate_values.mean().item(),
            'gate_distribution': gate_values.numpy(),
            'hit_ratio': (gate_values > 0.5).float().mean().item()
        }
    
    return avg_loss, avg_acc, gate_info

def train_classifier_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for tokens, labels in tqdm(loader, desc="Classifier Training"):
        tokens, labels = tokens.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, loss = model(tokens, targets=labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        acc = (logits.argmax(dim=-1) == labels).float().mean().item()
        total_acc += acc
        
    return total_loss / len(loader), total_acc / len(loader)

def evaluate_classifier_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    
    with torch.no_grad():
        for tokens, labels in tqdm(loader, desc="Classifier Eval"):
            tokens, labels = tokens.to(device), labels.to(device)
            logits, loss = model(tokens, targets=labels)
            total_loss += loss.item()
            acc = (logits.argmax(dim=-1) == labels).float().mean().item()
            total_acc += acc
            
    return total_loss / len(loader), total_acc / len(loader)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    # Initialize WandB
    wandb.init(project="vision-engram-experiment", name="texture-memory-test")
    
    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 256
    PRETRAIN_EPOCHS = 15
    CLASSIFICATION_EPOCHS = 5
    VOCAB_SIZE = 512
    EMBED_DIM = 128
    PATCH_SIZE = 4
    DROP_P = 0.1
    
    # Log configuration to WandB
    config = {
        'batch_size': BATCH_SIZE,
        'pretrain_epochs': PRETRAIN_EPOCHS,
        'classification_epochs': CLASSIFICATION_EPOCHS,
        'vocab_size': VOCAB_SIZE,
        'embed_dim': EMBED_DIM,
        'patch_size': PATCH_SIZE,
        'drop_p': DROP_P,
        'device': str(DEVICE)
    }
    wandb.config.update(config)
    
    print(f"Using device: {DEVICE}")
    
    # Load Quantized CIFAR-10
    train_loader, test_loader = get_quantized_cifar10_loaders(
        batch_size=BATCH_SIZE,
        patch_size=PATCH_SIZE,
        num_clusters=VOCAB_SIZE
    )
    
    # Create Models
    baseline_model = NanoGPT(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        depth=4,
        num_heads=4
    ).to(DEVICE)
    
    engram_model = EngramEnhancedGPT(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        depth=3,
        num_heads=4,
        drop_p=DROP_P
    ).to(DEVICE)
    
    # Optimizers
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=1e-4)
    
    # Engram optimizer with higher weight decay for memory module
    engram_mem_params = list(engram_model.engram.parameters())
    mem_param_ids = set(map(id, engram_mem_params))
    rest_params = [p for p in engram_model.parameters() if id(p) not in mem_param_ids]
    
    engram_optimizer = torch.optim.AdamW([
        {'params': rest_params, 'weight_decay': 0.01},
        {'params': engram_mem_params, 'weight_decay': 0.1}
    ], lr=1e-4)
    
    # Learning Rate Schedulers
    baseline_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(baseline_optimizer, T_max=PRETRAIN_EPOCHS)
    engram_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(engram_optimizer, T_max=PRETRAIN_EPOCHS)
    
    # Training History
    baseline_history = {'train': [], 'test': []}
    engram_history = {'train': [], 'test': []}
    baseline_acc_history = {'train': [], 'test': []}
    engram_acc_history = {'train': [], 'test': []}
    engram_gate_history = {'train': [], 'test': []}
    engram_hit_ratio_history = {'train': [], 'test': []}
    
    # Training Loop
    for epoch in range(PRETRAIN_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{PRETRAIN_EPOCHS}")
        
        # Train
        baseline_train_loss, baseline_train_acc, _ = train_epoch(baseline_model, train_loader, baseline_optimizer, DEVICE, "Baseline")
        engram_train_loss, engram_train_acc, engram_train_gate = train_epoch(engram_model, train_loader, engram_optimizer, DEVICE, "Engram")
        
        # Step Schedulers
        baseline_scheduler.step()
        engram_scheduler.step()
        
        # Evaluate
        baseline_test_loss, baseline_test_acc, _ = evaluate_epoch(baseline_model, test_loader, DEVICE, "Baseline")
        engram_test_loss, engram_test_acc, engram_test_gate = evaluate_epoch(engram_model, test_loader, DEVICE, "Engram")
        
        # Track History
        baseline_history['train'].append(baseline_train_loss)
        baseline_history['test'].append(baseline_test_loss)
        engram_history['train'].append(engram_train_loss)
        engram_history['test'].append(engram_test_loss)
        baseline_acc_history['train'].append(baseline_train_acc)
        baseline_acc_history['test'].append(baseline_test_acc)
        engram_acc_history['train'].append(engram_train_acc)
        engram_acc_history['test'].append(engram_test_acc)
        if engram_train_gate:
            engram_gate_history['train'].append(engram_train_gate['avg_gate_score'])
            engram_hit_ratio_history['train'].append(engram_train_gate['hit_ratio'])
        if engram_test_gate:
            engram_gate_history['test'].append(engram_test_gate['avg_gate_score'])
            engram_hit_ratio_history['test'].append(engram_test_gate['hit_ratio'])
        
        # Log to WandB
        wandb.log({
            'epoch': epoch + 1,
            'baseline_train_loss': baseline_train_loss,
            'baseline_test_loss': baseline_test_loss,
            'baseline_train_acc': baseline_train_acc,
            'baseline_test_acc': baseline_test_acc,
            'engram_train_loss': engram_train_loss,
            'engram_test_loss': engram_test_loss,
            'engram_train_acc': engram_train_acc,
            'engram_test_acc': engram_test_acc,
            'baseline_train_perplexity': np.exp(baseline_train_loss),
            'baseline_test_perplexity': np.exp(baseline_test_loss),
            'engram_train_perplexity': np.exp(engram_train_loss),
            'engram_test_perplexity': np.exp(engram_test_loss)
        })
        
        if engram_train_gate and engram_test_gate:
            wandb.log({
                'engram/train/gate/mean_activation': engram_train_gate['avg_gate_score'],
                'engram/train/gate/activation_distribution': wandb.Histogram(engram_train_gate['gate_distribution']),
                'engram/train/engram/hit_ratio': engram_train_gate['hit_ratio'],
                'engram/test/gate/mean_activation': engram_test_gate['avg_gate_score'],
                'engram/test/gate/activation_distribution': wandb.Histogram(engram_test_gate['gate_distribution']),
                'engram/test/engram/hit_ratio': engram_test_gate['hit_ratio']
            })
        
        # Print Results
        print(f"Baseline    - Train Loss: {baseline_train_loss:.4f} | Acc: {baseline_train_acc:.4f} | Test Loss: {baseline_test_loss:.4f} | Acc: {baseline_test_acc:.4f}")
        print(f"Engram      - Train Loss: {engram_train_loss:.4f} | Acc: {engram_train_acc:.4f} | Test Loss: {engram_test_loss:.4f} | Acc: {engram_test_acc:.4f}")
        if engram_train_gate and engram_test_gate:
            print(f"Engram Gate - Train: {engram_train_gate['avg_gate_score']:.3f} | Test: {engram_test_gate['avg_gate_score']:.3f}")
            print(f"Hit Ratio   - Train: {engram_train_gate['hit_ratio']:.3f} | Test: {engram_test_gate['hit_ratio']:.3f}")
    
    # Classification Fine-tuning Demo
    print("\n=== Starting Classification Fine-tuning ===")
    
    # 1. Baseline Classifier
    print("--- Training Baseline Classifier ---")
    baseline_classifier = GPTClassifier(baseline_model, num_classes=10).to(DEVICE)
    baseline_clf_optimizer = torch.optim.AdamW(baseline_classifier.parameters(), lr=1e-4)
    
    baseline_clf_history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(CLASSIFICATION_EPOCHS):
        train_loss, train_acc = train_classifier_epoch(baseline_classifier, train_loader, baseline_clf_optimizer, DEVICE)
        test_loss, test_acc = evaluate_classifier_epoch(baseline_classifier, test_loader, DEVICE)
        
        baseline_clf_history['train_loss'].append(train_loss)
        baseline_clf_history['test_loss'].append(test_loss)
        baseline_clf_history['train_acc'].append(train_acc)
        baseline_clf_history['test_acc'].append(test_acc)
        
        print(f"Baseline Classifier Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

    # 2. Engram Classifier
    print("\n--- Training Engram Classifier ---")
    engram_classifier = GPTClassifier(engram_model, num_classes=10).to(DEVICE)
    engram_clf_optimizer = torch.optim.AdamW(engram_classifier.parameters(), lr=1e-4)
    
    engram_clf_history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(CLASSIFICATION_EPOCHS):
        train_loss, train_acc = train_classifier_epoch(engram_classifier, train_loader, engram_clf_optimizer, DEVICE)
        test_loss, test_acc = evaluate_classifier_epoch(engram_classifier, test_loader, DEVICE)
        
        engram_clf_history['train_loss'].append(train_loss)
        engram_clf_history['test_loss'].append(test_loss)
        engram_clf_history['train_acc'].append(train_acc)
        engram_clf_history['test_acc'].append(test_acc)
        
        print(f"Engram Classifier Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")

    # Plot Results
    plt.figure(figsize=(15, 20))
    
    # Loss Curves
    plt.subplot(4, 2, 1)
    plt.plot(baseline_history['train'], label='Baseline (Train)')
    plt.plot(baseline_history['test'], label='Baseline (Test)')
    plt.plot(engram_history['train'], label='Engram (Train)')
    plt.plot(engram_history['test'], label='Engram (Test)')
    plt.title('Pretraining: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Perplexity
    plt.subplot(4, 2, 2)
    baseline_train_perplexity = [np.exp(loss) for loss in baseline_history['train']]
    baseline_test_perplexity = [np.exp(loss) for loss in baseline_history['test']]
    engram_train_perplexity = [np.exp(loss) for loss in engram_history['train']]
    engram_test_perplexity = [np.exp(loss) for loss in engram_history['test']]
    
    plt.plot(baseline_train_perplexity, label='Baseline (Train)')
    plt.plot(baseline_test_perplexity, label='Baseline (Test)')
    plt.plot(engram_train_perplexity, label='Engram (Train)')
    plt.plot(engram_test_perplexity, label='Engram (Test)')
    plt.title('Pretraining: Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(4, 2, 3)
    plt.plot(baseline_acc_history['train'], label='Baseline (Train)')
    plt.plot(baseline_acc_history['test'], label='Baseline (Test)')
    plt.plot(engram_acc_history['train'], label='Engram (Train)')
    plt.plot(engram_acc_history['test'], label='Engram (Test)')
    plt.title('Pretraining: Next Token Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Gate Scores
    plt.subplot(4, 2, 4)
    plt.plot(engram_gate_history['train'], label='Train Gate Score')
    plt.plot(engram_gate_history['test'], label='Test Gate Score')
    plt.title('Engram Gate Activation')
    plt.xlabel('Epoch')
    plt.ylabel('Average Gate Score')
    plt.legend()
    plt.grid(True)
    
    # Hit Ratios
    plt.subplot(4, 2, 5)
    plt.plot(engram_hit_ratio_history['train'], label='Train Hit Ratio')
    plt.plot(engram_hit_ratio_history['test'], label='Test Hit Ratio')
    plt.title('Engram Hit Ratio (Gate > 0.5)')
    plt.xlabel('Epoch')
    plt.ylabel('Hit Ratio')
    plt.legend()
    plt.grid(True)

    # Classification Loss
    plt.subplot(4, 2, 7)
    plt.plot(baseline_clf_history['train_loss'], label='Baseline (Train)')
    plt.plot(baseline_clf_history['test_loss'], label='Baseline (Test)')
    plt.plot(engram_clf_history['train_loss'], label='Engram (Train)')
    plt.plot(engram_clf_history['test_loss'], label='Engram (Test)')
    plt.title('Classification: Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Classification Accuracy
    plt.subplot(4, 2, 8)
    plt.plot(baseline_clf_history['train_acc'], label='Baseline (Train)')
    plt.plot(baseline_clf_history['test_acc'], label='Baseline (Test)')
    plt.plot(engram_clf_history['train_acc'], label='Engram (Train)')
    plt.plot(engram_clf_history['test_acc'], label='Engram (Test)')
    plt.title('Classification: Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('texture_memory_experiment_results.png')
    wandb.log({'results_plot': wandb.Image('texture_memory_experiment_results.png')})
    plt.show()
    
    # Save Models
    torch.save(baseline_model.state_dict(), 'baseline_nanogpt.pth')
    torch.save(engram_model.state_dict(), 'engram_enhanced_gpt.pth')
    
    print("\nExperiment completed successfully!")
    print("Results saved to 'texture_memory_experiment_results.png'")
    print("Models saved as 'baseline_nanogpt.pth' and 'engram_enhanced_gpt.pth'")
    
    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    main()
