from __future__ import annotations
from torch import nn
from titans_pytorch.neural_memory import NeuralMemory

def exists(v):
    return v is not None

class MemoryAsGating(nn.Module):
    def __init__(
        self,
        neural_memory: NeuralMemory,
        attention: nn.Module,
        gate_attn_output = True,
        add_mem_residual = False # if not gating, add memory output as residual
    ):
        super().__init__()
        self.neural_memory = neural_memory
        self.attention = attention

        assert gate_attn_output ^ add_mem_residual, 'you can either gate the attention output with memory, or add the memory output as a residual, but not both (or neither)'

        self.gate_attn_output = gate_attn_output
        self.add_mem_residual = add_mem_residual

    def forward(
        self,
        x,
        memory_state = None,
        attn_cache = None,
        neural_mem_kwargs: dict = dict(),
        attn_kwargs: dict = dict()
    ):
        # neural memory

        mem_out, next_mem_state = self.neural_memory(
            x,
            state = memory_state,
            **neural_mem_kwargs
        )

        # gating

        attn_out_gates = None

        if self.gate_attn_output:
            attn_out_gates = mem_out.sigmoid()

        # attention

        attn_out = self.attention(
            x,
            cache = attn_cache,
            **attn_kwargs
        )

        # handle attention output (tensor or tuple)

        if isinstance(attn_out, tuple):
            out, intermediates = attn_out
        else:
            out, intermediates = attn_out, None

        # apply gating

        if exists(attn_out_gates):
            out = out * attn_out_gates

        # maybe add memory output as residual

        if self.add_mem_residual:
            out = out + mem_out

        # return

        if exists(intermediates):
            return out, (next_mem_state, intermediates)

        return out, next_mem_state