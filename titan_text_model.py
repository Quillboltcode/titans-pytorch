from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention, 
    MistralForCausalLM,
    MistralMLP, 
    MistralRMSNorm,
)
import torch
import torch.nn as nn
from typing import Optional, Tuple
from titans_pytorch.neural_memory import NeuralMemory

# -----------------------------------------------------------------------------

class SmallTitanConfig(MistralConfig):
    """
    A significantly reduced Mistral configuration optimized for T4 GPU training.
    This config creates a model with ~110M parameters (vs 7B in standard Mistral).
    """
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=384,           # Reduced from 4096
        intermediate_size=1024,    # Reduced from 14336
        num_hidden_layers=12,      # Reduced from 32
        num_attention_heads=6,     # Reduced from 32
        num_key_value_heads=2,     # Reduced from 8
        max_position_embeddings=2048,
        sliding_window=512,        # Smaller window for T4 memory constraints
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            sliding_window=sliding_window,
            **kwargs
        )
        # Keep Mistral-specific defaults
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        # Disabling FlashAttention 2 as requested due to installation issues.
        # "eager" is a safe fallback that works everywhere.
        # For better performance on compatible hardware, consider "sdpa" (requires PyTorch 2.0+).
        self._attn_implementation = "eager"

class SmallTitanModel(nn.Module):
    """
    A compact TitanModel implementation using a small custom config while preserving
    neural memory capabilities (MAL and MAG modes).
    
    Args:
        gated (bool): Whether to use memory as a gate (MAG) or layer (MAL)
        segment_size (int): Size of memory segments (recommend 128-256 for T4)
        neural_memory (bool): Whether to enable neural memory components
    """
    def __init__(self, gated: bool, segment_size: int, neural_memory: bool = True):
        super().__init__()
        self.config = SmallTitanConfig(sliding_window=segment_size)
        self.gated = gated
        self.segment_size = segment_size
        
        # Create base model with small config
        self.model = MistralForCausalLM(self.config)
        
        # Replace decoder layers with memory-enhanced versions
        if neural_memory:
            for idx, layer in enumerate(self.model.model.layers):
                new_layer = TitanDecoderLayer(
                    config=self.config,
                    layer_idx=idx,
                    gated=gated,
                    segment_size=segment_size
                ).to(self.model.device)
                # Transfer weights from original layer where possible
                self._transfer_weights(layer, new_layer)
                self.model.model.layers[idx] = new_layer
                del layer
        else:
            print("Warning: Neural memory disabled. Model will function as standard small transformer.")
            
        # Keep lm_head from base model
        self.lm_head = self.model.lm_head
    
    def _transfer_weights(self, src_layer, tgt_layer):
        """Transfer compatible weights from source layer to target memory layer"""
        state_dict = src_layer.state_dict()
        # Filter out attention-specific keys that might have different shapes
        compatible_keys = {k: v for k, v in state_dict.items() 
                          if k in tgt_layer.state_dict() and v.shape == tgt_layer.state_dict()[k].shape}
        tgt_layer.load_state_dict(compatible_keys, strict=False)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        return outputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, gated=True, segment_size=256):
        """
        Initialize from a pretrained small model (like TinyLlama or Pythia) and add memory layers
        """
        model = cls(gated=gated, segment_size=segment_size)
        
        # Load pretrained weights into the base model architecture
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # Transfer compatible weights
        base_state_dict = pretrained_model.state_dict()
        current_state_dict = model.state_dict()
        
        # Match and transfer weights that have compatible shapes
        transferred_weights = 0
        for name, param in base_state_dict.items():
            if name in current_state_dict and param.shape == current_state_dict[name].shape:
                current_state_dict[name].copy_(param)
                transferred_weights += 1
        
        model.load_state_dict(current_state_dict, strict=False)
        print(f"Successfully transferred {transferred_weights} weight matrices from pretrained model")
        
        # Convert to appropriate dtype
        if torch.cuda.is_available():
            model = model.to(torch.bfloat16)
        
        return model

class TitanDecoderLayer(nn.Module):
    """
    Memory-enhanced decoder layer compatible with the small model configuration.
    Preserves both MAL (Memory As Layer) and MAG (Memory As Gate) operation modes.
    """
    def __init__(self, config: SmallTitanConfig, layer_idx: int, gated: bool, segment_size: int):
        super().__init__()
        self.gated = gated
        self.hidden_size = config.hidden_size
        self.chunk_size = segment_size
        self.layer_idx = layer_idx

        # Configure attention with sliding window
        self.self_attn = MistralAttention(
            config=config,
            layer_idx=layer_idx
        )

        # Memory module - scaled for smaller hidden dimension
        self.memory = NeuralMemory(
            dim=config.hidden_size,
            chunk_size=self.chunk_size
        )
        
        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        
        # ======================
        # MEMORY AS LAYER (MAL)
        # ======================
        if not self.gated:
            # Apply memory as a separate processing layer before attention
            mem_output, _ = self.memory(seq=hidden_states, state=None)
            hidden_states = residual + mem_output  # Residual connection around memory
            residual = hidden_states  # Update residual for attention block

        # Layer norm before attention
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = attn_outputs[0]
        attn_weights = attn_outputs[1] if output_attentions else None
        
        # ======================
        # MEMORY AS GATE (MAG)
        # ======================
        if self.gated:
            # Apply memory as gating mechanism on attention outputs
            mem_output, _ = self.memory(seq=hidden_states, state=None)
            gate = torch.sigmoid(mem_output)  # Sigmoid for gating values between 0-1
            hidden_states = hidden_states * gate
        
        # Residual connection after attention (+ memory if gated)
        hidden_states = residual + hidden_states
        
        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

## Next cell in notebook
############################################################################



from transformers import AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load a small real-world dataset
dataset = load_dataset("databricks/databricks-dolly-15k", split="train[:100]")

# Initialize tokenizer (using Mistral default or fallback)
try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained("gpt2") # Fallback

tokenizer.pad_token = tokenizer.eos_token

# Preprocessing function to tokenize the dataset
def tokenize_function(examples):
    prompts = []
    for instruction, context, response in zip(examples["instruction"], examples["context"], examples["response"]):
        # Format prompt
        if context:
            text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
        prompts.append(text + tokenizer.eos_token)
        
    # Tokenize and create labels (labels = input_ids for causal LM)
    tokenized = tokenizer(prompts, padding="max_length", truncation=True, max_length=256)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Apply tokenization and remove raw columns to fix the ValueError
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# Initialize model (MAG mode example)
model = SmallTitanModel(gated=True, segment_size=256)

# Resize embeddings if needed
if len(tokenizer) != model.config.vocab_size:
    model.model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()


