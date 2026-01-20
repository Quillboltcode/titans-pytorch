import torch
from torch import nn
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
    def __init__(self, dim, heads = 8, dim_head = 64):
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
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, 
                 use_memory=False, memory_chunk_size=64, gated=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.use_memory = use_memory
        self.gated = gated
        
        for _ in range(depth):
            # Core transformer components
            attn = Attention(dim, heads=heads, dim_head=dim_head)
            ff = FeedForward(dim, mlp_dim)
            
            # Optional neural memory component
            memory = None
            if use_memory:
                memory = NeuralMemory(
                    dim=dim,
                    chunk_size=memory_chunk_size,
                    dim_head=dim_head,
                    heads=heads,
                    per_head_learned_parameters=False,
                    use_accelerated_scan=True,
                    spectral_norm_surprises=True,
                    max_grad_norm=1.0,
                    num_kv_per_token=1,
                    default_model_kwargs=dict(depth=1, expansion_factor=1),
                )
            
            # Store components for this layer
            self.layers.append(nn.ModuleList([attn, memory, ff]))
    
    def forward(self, x, state=None, memory_training=True, freeze_attention=False):
        new_states = []
        state = state or [None] * len(self.layers)
        
        for i, (attn, memory, ff) in enumerate(self.layers):
            # Memory as Layer (MAL) - applied before attention
            if self.use_memory and memory is not None and not self.gated:
                with torch.set_grad_enabled(memory_training):
                    mem_out, layer_state = memory(x, state=state[i])
                x = x + mem_out  # Residual connection
                new_states.append(layer_state)
            else:
                new_states.append(None)  # Placeholder for state
            
            # Attention block
            if freeze_attention:
                with torch.no_grad():
                    attn_out = attn(x)
            else:
                attn_out = attn(x)
            
            # Memory as Gate (MAG) - applied on attention output
            if self.use_memory and memory is not None and self.gated:
                with torch.set_grad_enabled(memory_training):
                    mem_out, layer_state = memory(attn_out, state=state[i])
                
                # Generate gating values between 0 and 1
                gate = torch.sigmoid(mem_out)
                
                # Protect CLS token (last token) from aggressive gating
                gate[:, -1] = gate[:, -1] * 0.5 + 0.5
                
                # Apply gated residual connection
                x = x + attn_out * gate
                new_states[i] = layer_state  # Overwrite placeholder
            elif not (self.use_memory and memory is not None and not self.gated):
                # Standard residual connection if not in MAL mode
                x = attn_out + x
            
            # Feedforward block with residual
            x = ff(x) + x
        
        return self.norm(x), new_states

class SimpleViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, 
                 channels=3, dim_head=64, use_memory=False, memory_chunk_size=64, gated=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = posemb_sincos_2d(
            h=image_height // patch_height,
            w=image_width // patch_width,
            dim=dim,
        ) 

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, 
            use_memory=use_memory, 
            memory_chunk_size=memory_chunk_size,
            gated=gated
        )
        
        self.use_memory = use_memory
        self.gated = gated

        self.pool = "cls"  # Changed to reflect actual pooling method
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
        return [None] * len(self.transformer.layers)
    
    def forward(self, img, state=None, memory_training=True, freeze_attention=False):
        device = img.device

        x = self.to_patch_embedding(img)
        x += self.pos_embedding.to(device, dtype=x.dtype)

        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((x, cls_tokens), dim=1)

        # Pass state through transformer and get updated state
        x, new_state = self.transformer(
            x, 
            state=state,
            memory_training=memory_training, 
            freeze_attention=freeze_attention
        )
        
        # Pooling: use CLS token (last token)
        # Note: Fixed original code that had redundant operations
        x = x[:, -1]  # Take CLS token
        x = self.to_latent(x)
        return self.linear_head(x), new_state
    
    @classmethod
    def from_pretrained(cls, pretrained_model, **kwargs):
        """
        Initialize from a pretrained SimpleViT and add memory layers.
        Compatible state dicts will be transferred automatically.
        """
        model = cls(**kwargs)
        
        # Load state dict
        if isinstance(pretrained_model, dict):
            state_dict = pretrained_model
        else:
            state_dict = pretrained_model.state_dict()
        
        # Transfer compatible weights
        model_state = model.state_dict()
        transferred = 0
        
        for name, param in state_dict.items():
            if name in model_state and param.shape == model_state[name].shape:
                model_state[name].copy_(param)
                transferred += 1
        
        model.load_state_dict(model_state, strict=False)
        print(f"Successfully transferred {transferred} weight matrices from pretrained model")
        return model

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = SimpleViT(image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072, channels=3, dim_head=64)
    y = model(x)
    print(y[0].shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))