import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from titans_pytorch.neural_memory import NeuralMemory

# --- Helpers ---

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

# --- Core Hybrid Components ---

class SlidingWindowAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, window_size = 16):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        b, n, d = x.shape
        x = self.norm(x)

        # Padding if sequence length not divisible by window_size
        pad_len = (self.window_size - n % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))

        # Chunk into windows for local softmax attention
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (nw w) (h d) -> (b nw) h w d', w = self.window_size, h = self.heads),
            qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim = -1)

        out = torch.matmul(attn, v)
        out = rearrange(out, '(b nw) h w d -> b (nw w) (h d)', b = b)

        # Remove padding
        if pad_len > 0:
            out = out[:, :n, :]

        return self.to_out(out)

class SuperTitansLayer(nn.Module):
    def __init__(self, dim, heads, dim_head, window_size, memory_chunk_size, num_registers = 4):
        super().__init__()
        self.num_registers = num_registers
        
        # 1. Local Fovea: Sliding Window Softmax
        self.attn = SlidingWindowAttention(dim, heads, dim_head, window_size)
        
        # 2. Global Brain: Neural Memory
        self.memory = NeuralMemory(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            chunk_size = memory_chunk_size,
            default_step_transform_max_lr=1e-2,
            per_head_learned_parameters=False,
            use_accelerated_scan=True,
            spectral_norm_surprises=True,
            max_grad_norm=1.0,
            
        )
        
        # 3. Workspace: Learned Register Tokens
        self.registers = nn.Parameter(torch.randn(1, num_registers, dim))

    def forward(self, x, state = None):
        b, n, d = x.shape
        
        # --- Local Phase ---
        x = self.attn(x) + x
        
        # --- Cognitive Phase (Registers + Memory) ---
        # Inject registers as "scratchpad" tokens for the memory update
        regs = self.registers.expand(b, -1, -1)
        x_full = torch.cat((x, regs), dim = 1)
        
        # Storage & Retrieval
        mem_out, next_state = self.memory(x_full, state = state)
        x_full = mem_out + x_full
        
        # Slice back to original sequence length, discarding processed registers
        return x_full[:, :n, :], next_state

# --- Main Architecture ---

class SuperTitansViT(nn.Module):
    def __init__(
        self, *, 
        image_size, patch_size, num_classes, dim, depth, heads, 
        channels = 3, dim_head = 64, window_size = 14, 
        memory_chunk_size = 64, num_registers = 4
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        patch_dim = channels * patch_height * patch_width

        # Patching & Embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height, w = image_width // patch_width, dim = dim
        ) 

        # Task Guidance: CLS Induction Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Layers
        self.layers = nn.ModuleList([
            SuperTitansLayer(dim, heads, dim_head, window_size, memory_chunk_size, num_registers) 
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, states = None):
        b, _, _, _ = img.shape
        device = img.device

        # 1. Embed and Add Position
        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        # 2. Induce Behavior: Prepend CLS token to guide the memory storage
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)

        # 3. Process Layers with State Carry-over
        new_states = []
        states = states or [None] * len(self.layers)

        for i, layer in enumerate(self.layers):
            x, layer_state = layer(x, state = states[i])
            new_states.append(layer_state)

        # 4. Final Pooling & Head
        x = self.norm(x)
        cls_out = x[:, 0] # Use the CLS token which has been enriched by Neural Memory
        return self.mlp_head(cls_out), new_states

# Flop and parameter analysis
if __name__ == '__main__':
    print("--- SuperTitansViT ---")
    model = SuperTitansViT(image_size=224, patch_size=16, num_classes=1000, dim=768, depth=1
        , heads=12, channels=3, dim_head=64, window_size=14, memory_chunk_size=64, num_registers=4)
    print(f'Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    Flop = sum(p.numel() * 2 for p in model.parameters() if p.requires_grad)
    print(f'Total Flops: {Flop}')