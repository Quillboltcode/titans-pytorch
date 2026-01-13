import torch
import torch.nn as nn
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
        b, _, _, _ = img.shape
        patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=self.patch_size, p2=self.patch_size)
        x = self.to_patch_embedding(patches)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((x, cls_tokens), dim = 1) 
        memory_states = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x, memory_states[i] = layer(x, memory_state=memory_states[i])
        cls_token_out = x[:, -1]
        return self.to_logits(self.norm(cls_token_out))