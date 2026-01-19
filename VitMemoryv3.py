import torch
from torch import nn

from einops import rearrange
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from titans_pytorch.neural_memory import NeuralMemory

# helpers

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

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, is_memory = False):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, use_memory=False, memory_chunk_size=64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.use_memory = use_memory
        for _ in range(depth):
            if use_memory:
                ff = NeuralMemory(
                    dim = dim,
                    chunk_size = memory_chunk_size,
                    dim_head = dim_head,
                    heads = heads,
                    # need to set this to cap vram memory
                    per_head_learned_parameters=False,
                    use_accelerated_scan=True,
                    spectral_norm_surprises=True,
                    max_grad_norm=1.0,
                    num_kv_per_token=1,
                    #Minimal model
                    default_model_kwargs=dict(depth=1, expansion_factor=1),
                    
                )
            else:
                ff = FeedForward(dim, mlp_dim)

            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                ff
            ]))
    def forward(self, x, state=None, memory_training=True, freeze_attention=False):
        new_states = []
        state = state or [None] * len(self.layers)
        
        for i, (attn, ff) in enumerate(self.layers):
            # 1. Attention
            if freeze_attention:
                with torch.no_grad():
                    attn_out = attn(x)
            else:
                attn_out = attn(x)
            x = attn_out + x
            
            # 2. Neural Memory (Long-term)
            if self.use_memory:
                if memory_training:
                    with torch.enable_grad():
                        out, layer_state = ff(x, state=state[i])
                else:
                    with torch.no_grad():
                        out, layer_state = ff(x, state=state[i])
                x = out + x
                new_states.append(layer_state)
            else:
                x = ff(x) + x
                new_states.append(None)

        return self.norm(x), new_states

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, use_memory=False, memory_chunk_size=64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h = image_height // patch_height,
            w = image_width // patch_width,
            dim = dim,
        ) 

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, use_memory=use_memory, memory_chunk_size=memory_chunk_size)
        self.use_memory = use_memory

        self.pool = "mean"
        self.to_latent = nn.Identity()

        self.linear_head = nn.Linear(dim, num_classes)

    def set_trainable_parameters(self, memory_training=True, freeze_attention=False):
        """Configure which parameters can learn during TTT"""
        for name, param in self.named_parameters():
            # Default: freeze all parameters
            param.requires_grad = False
            
            # Unfreeze memory-related parameters if needed
            if memory_training and self.use_memory:
                if 'neural_memory' in name or 'memory_model' in name:
                    param.requires_grad = True
            
            # Unfreeze attention if not frozen
            if not freeze_attention:
                if 'attn' in name or 'to_qkv' in name or 'to_out' in name:
                    param.requires_grad = True
    
    def init_memory_state(self, batch_size=1):
        """Initialize memory state for TTT"""
        if not self.use_memory:
            return None
        
        # Create initial state compatible with Transformer's expected format
        return [None] * len(self.transformer.layers)
    
    def forward(self, img, state=None, memory_training=True, freeze_attention=False):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((x, cls_tokens), dim = 1)

        # Pass state through transformer and get updated state
        x, new_state = self.transformer(
            x, 
            state=state,
            memory_training=memory_training, 
            freeze_attention=freeze_attention
        )
        
        x = x.mean(dim=1)
        x = x[:, -1]
        x = self.to_latent(x)
        return self.linear_head(x), new_state


if __name__ == '__main__':
    print("--- SimpleViT ---")
    model = SimpleViT(image_size=32, patch_size=4, num_classes=10, dim=192, depth=6, heads=3, mlp_dim=768)
    print(f'Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')