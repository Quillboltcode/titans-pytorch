"""
Memory Visualizer Module for Neural Memory Visualization

This module provides utilities for extracting and visualizing the key-value 
representations stored in the NeuralMemory module during inference.

Usage:
    from titans_pytorch.memory_visualizer import MemoryVisualizer
    
    visualizer = MemoryVisualizer(model)
    viz_data = visualizer.extract_visualization_data(seq, state)
    visualizer.plot_memory_evolution(viz_data)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, stack, cat, is_tensor
from torch.utils._pytree import tree_map

import einx
from einops import einsum, rearrange, repeat, reduce, pack, unpack

from titans_pytorch.neural_memory import NeuralMemory, NeuralMemState


# ----------------------------------------------------------------------
# Visualization Data Structures
# ----------------------------------------------------------------------

VisualizationData = namedtuple('VisualizationData', [
    'step_index',
    'queries',
    'keys',
    'values',
    'weights',
    'retrieved',
    'surprises',
    'adaptive_lr',
    'decay_factor',
])


class MemoryVisualizer:
    """
    Helper class for extracting and visualizing neural memory states.
    
    Extracts intermediate representations (keys, values, weights, attention)
    from the NeuralMemory module for visualization purposes.
    """
    
    def __init__(
        self,
        model: NeuralMemory,
        hook_fn: Optional[Callable] = None
    ):
        """
        Initialize the MemoryVisualizer.
        
        Args:
            model: NeuralMemory instance to visualize
            hook_fn: Optional callback function for real-time visualization
        """
        self.model = model
        self.hook_fn = hook_fn
        self._visualization_history: List[VisualizationData] = []
        
        # Register hooks if model supports it
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture intermediate values."""
        # Store references to original methods
        self._original_forward = self.model.forward
        
        # Wrap forward to capture visualization data
        def wrapped_forward(*args, **kwargs):
            # Set visualization hook
            self.model._visualization_hook = self._capture_visualization
            result = self._original_forward(*args, **kwargs)
            # Clear hook
            self.model._visualization_hook = None
            return result
        
        self.model.forward = wrapped_forward
    
    def _capture_visualization(self, data: Dict):
        """Capture visualization data from forward pass."""
        viz_data = VisualizationData(
            step_index=len(self._visualization_history),
            queries=data.get('queries'),
            keys=data.get('keys'),
            values=data.get('values'),
            weights=data.get('weights'),
            retrieved=data.get('retrieved'),
            surprises=data.get('surprises'),
            adaptive_lr=data.get('adaptive_lr'),
            decay_factor=data.get('decay_factor'),
        )
        self._visualization_history.append(viz_data)
        
        if self.hook_fn:
            self.hook_fn(viz_data)
    
    def extract_memory_weights(self, state: NeuralMemState) -> Dict[str, Tensor]:
        """
        Extract memory weights from neural memory state.
        
        Args:
            state: NeuralMemState from forward pass
            
        Returns:
            Dictionary mapping parameter names to weight tensors
        """
        if isinstance(state, tuple):
            weights = state[1]  # weights is the second element
        else:
            weights = state.weights
        
        # Convert TensorDict to regular dict if needed
        if hasattr(weights, 'to_dict'):
            weights = weights.to_dict()
        
        return weights
    
    def extract_keys_values(self, seq: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Extract key-value pairs from input sequence.
        
        Args:
            seq: Input sequence tensor of shape (b, n, d)
            
        Returns:
            Tuple of (keys, values) tensors
        """
        # Apply store norm
        seq_normed = self.model.store_norm(seq)
        
        # Get keys and values
        keys = self.model.to_keys(seq_normed)
        values = self.model.to_values(seq_normed)
        
        # Split into heads if multi-head
        if self.model.heads > 1:
            keys = self.model.split_kv_heads(keys)
            values = self.model.split_kv_heads(values)
        
        return keys, values
    
    def extract_queries(self, seq: Tensor) -> Tensor:
        """
        Extract query representations from input sequence.
        
        Args:
            seq: Input sequence tensor of shape (b, n, d)
            
        Returns:
            Queries tensor
        """
        seq_normed = self.model.retrieve_norm(seq)
        queries = self.model.to_queries(seq_normed)
        
        if self.model.heads > 1:
            queries = self.model.split_heads(queries)
        
        return queries
    
    def compute_query_memory_attention(
        self,
        queries: Tensor,
        weights: Dict[str, Tensor]
    ) -> Tensor:
        """
        Compute attention scores between queries and memory weights.
        
        This computes how much each query attends to each memory weight.
        
        Args:
            queries: Query tensor of shape (b, h, n, d) or (b, n, d)
            weights: Dictionary of memory weights
            
        Returns:
            Attention scores tensor
        """
        # For visualization, we'll compute dot products between queries
        # and a simplified representation of memory weights
        
        # Stack all weight matrices into a single tensor for comparison
        weight_matrices = []
        for name, weight in weights.items():
            if weight.ndim >= 2:
                weight_matrices.append(weight)
        
        if not weight_matrices:
            return None
        
        # Use the first weight matrix as representative
        rep_weight = weight_matrices[0]
        
        # Compute attention as query @ weight.T (simplified)
        if queries.ndim == 3:
            # (b, n, d) -> compute dot product with weight
            queries_flat = rearrange(queries, 'b n d -> (b n) d')
            weight_flat = rep_weight.reshape(-1, rep_weight.shape[-1])
            
            # Compute similarity matrix
            attn = torch.matmul(queries_flat, weight_flat.T)
            attn = rearrange(attn, '(b n) w -> b n w', n=queries.shape[1])
        else:
            # Multi-head case
            b, h, n, d = queries.shape
            attn = torch.matmul(queries, rep_weight.T)
        
        return attn
    
    def compute_weight_similarity(
        self,
        weights1: Dict[str, Tensor],
        weights2: Dict[str, Tensor]
    ) -> Dict[str, float]:
        """
        Compute similarity metrics between two weight states.
        
        Args:
            weights1: First weight dictionary
            weights2: Second weight dictionary
            
        Returns:
            Dictionary of similarity metrics per weight parameter
        """
        similarities = {}
        
        for name in weights1:
            if name in weights2:
                w1 = weights1[name]
                w2 = weights2[name]
                
                # Cosine similarity
                w1_flat = w1.flatten(start_dim=-2)
                w2_flat = w2.flatten(start_dim=-2)
                
                # Normalize
                w1_norm = F.normalize(w1_flat, dim=-1)
                w2_norm = F.normalize(w2_flat, dim=-1)
                
                cosine_sim = (w1_norm * w2_norm).sum(dim=-1).mean().item()
                
                # L2 distance
                l2_dist = (w1 - w2).norm().item()
                
                similarities[name] = {
                    'cosine_similarity': cosine_sim,
                    'l2_distance': l2_dist,
                }
        
        return similarities
    
    def get_persistent_tokens(self, weights: Dict[str, Tensor]) -> Tensor:
        """
        Extract persistent memory tokens from weight state.
        
        The memory tokens are derived from the memory model's weight matrices,
        which encode the accumulated knowledge.
        
        Args:
            weights: Memory weight dictionary
            
        Returns:
            Persistent memory tokens tensor
        """
        # Stack all weight matrices
        weight_tensors = []
        for name, weight in weights.items():
            if weight.ndim >= 2:
                weight_tensors.append(weight.flatten(start_dim=-2))
        
        if not weight_tensors:
            return None
        
        # Concatenate along feature dimension
        tokens = torch.cat(weight_tensors, dim=-1)
        
        return tokens
    
    def compute_token_similarity(
        self,
        tokens1: Tensor,
        tokens2: Tensor
    ) -> float:
        """
        Compute cosine similarity between two token sets.
        
        Args:
            tokens1: First set of tokens
            tokens2: Second set of tokens
            
        Returns:
            Average cosine similarity score
        """
        if tokens1 is None or tokens2 is None:
            return None
        
        t1_norm = F.normalize(tokens1, dim=-1)
        t2_norm = F.normalize(tokens2, dim=-1)
        
        # Compute pairwise similarities
        similarities = torch.matmul(t1_norm, t2_norm.T)
        
        return similarities.mean().item()
    
    def clear_history(self):
        """Clear the visualization history."""
        self._visualization_history.clear()
    
    # ------------------------------------------------------------------
    # Visualization Plotting Methods (using matplotlib)
    # ------------------------------------------------------------------
    
    def plot_memory_weight_heatmap(
        self,
        weights: Dict[str, Tensor],
        title: str = "Memory Weight Heatmap",
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of memory weight matrices.
        
        Args:
            weights: Memory weight dictionary
            title: Plot title
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return
        
        num_weights = len(weights)
        fig, axes = plt.subplots(1, min(num_weights, 4), figsize=figsize)
        if num_weights == 1:
            axes = [axes]
        
        for idx, (name, weight) in enumerate(list(weights.items())[:4]):
            ax = axes[idx]
            
            # Take first two dimensions for visualization
            w = weight.detach().cpu()
            if w.ndim > 2:
                w = w.flatten(start_dim=0, end_dim=w.ndim - 3)
            
            w_2d = w[:min(100, w.shape[0]), :min(100, w.shape[-1])]
            
            im = ax.imshow(w_2d.numpy(), cmap='viridis', aspect='auto')
            ax.set_title(name[:30])
            ax.set_xlabel('Feature')
            ax.set_ylabel('Sample/Head')
            plt.colorbar(im, ax=ax)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_attention_scores(
        self,
        attention: Tensor,
        title: str = "Query-Memory Attention",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot attention scores between queries and memory.
        
        Args:
            attention: Attention scores tensor
            title: Plot title
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return
        
        attn = attention.detach().cpu()
        
        # Average across heads/batch if multi-dimensional
        if attn.ndim > 2:
            attn = attn.mean(dim=tuple(range(attn.ndim - 2)))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(attn.numpy(), cmap='Blues', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Memory Index')
        ax.set_ylabel('Query Index')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_memory_evolution(
        self,
        states: List[NeuralMemState],
        title: str = "Memory Weight Evolution",
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot the evolution of memory weights across multiple forward passes.
        
        Args:
            states: List of NeuralMemStates from sequential forward passes
            title: Plot title
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return
        
        # Extract weights from each state
        weight_norms = []
        for state in states:
            weights = self.extract_memory_weights(state)
            total_norm = sum(w.norm().item() for w in weights.values())
            weight_norms.append(total_norm)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot weight norm evolution
        ax1.plot(weight_norms, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Forward Pass Index')
        ax1.set_ylabel('Total Weight Norm')
        ax1.set_title('Memory Weight Norm Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Plot weight change rate (derivative)
        if len(weight_norms) > 1:
            weight_changes = np.diff(weight_norms)
            ax2.bar(range(len(weight_changes)), weight_changes)
            ax2.set_xlabel('Forward Pass Index')
            ax2.set_ylabel('Weight Change (Δ)')
            ax2.set_title('Memory Weight Change Rate')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_token_similarity_matrix(
        self,
        tokens_list: List[Tensor],
        labels: List[str],
        title: str = "Persistent Memory Token Similarity",
        figsize: Tuple[int, int] = (8, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot a similarity matrix between persistent memory tokens from different states.
        
        Args:
            tokens_list: List of token tensors from different states
            labels: Labels for each state
            title: Plot title
            figsize: Figure size tuple
            save_path: Optional path to save the figure
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return
        
        n = len(tokens_list)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                sim = self.compute_token_similarity(tokens_list[i], tokens_list[j])
                if sim is not None:
                    similarity_matrix[i, j] = sim
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                              ha='center', va='center', color='black')
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def compare_activations(
        self,
        act1: Dict[str, Tensor],
        act2: Dict[str, Tensor],
        title_prefix: str = "Activation Comparison",
        figsize: Tuple[int, int] = (14, 6),
        save_path: Optional[str] = None
    ):
        """
        Compare activations from two different forward passes.
        
        Args:
            act1: First activation dictionary
            act2: Second activation dictionary
            title_prefix: Prefix for plot titles
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            
        Returns:
            Comparison metrics dictionary
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Matplotlib not available. Skipping plot.")
            return None
        
        metrics = {}
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Compare layer-wise activations
        layer_names = list(act1.keys())[:4]
        
        l2_diffs = []
        for i, name in enumerate(layer_names):
            a1 = act1.get(name)
            a2 = act2.get(name)
            
            if a1 is not None and a2 is not None:
                diff = (a1 - a2).norm().item()
                l2_diffs.append(diff)
                metrics[f'{name}_l2_diff'] = diff
        
        axes[0].bar(layer_names, l2_diffs)
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('L2 Distance')
        axes[0].set_title(f'{title_prefix}: Activation L2 Distance')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Cosine similarity per layer
        cos_sims = []
        for name in layer_names:
            a1 = act1.get(name)
            a2 = act2.get(name)
            
            if a1 is not None and a2 is not None:
                a1_flat = a1.flatten(start_dim=-2)
                a2_flat = a2.flatten(start_dim=-2)
                
                cos_sim = F.cosine_similarity(a1_flat, a2_flat, dim=-1).mean().item()
                cos_sims.append(cos_sim)
                metrics[f'{name}_cosine_sim'] = cos_sim
        
        axes[1].bar(layer_names, cos_sims)
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Cosine Similarity')
        axes[1].set_title(f'{title_prefix}: Activation Cosine Similarity')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_ylim(-1, 1)
        
        fig.suptitle(title_prefix)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return metrics


# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------

def extract_kv_from_model(
    model: NeuralMemory,
    seq: Tensor
) -> Dict[str, Tensor]:
    """
    Extract key/value representations from neural memory.
    
    Args:
        model: NeuralMemory instance
        seq: Input sequence tensor
        
    Returns:
        Dictionary containing keys, values, and related tensors
    """
    # Apply store norm
    seq_normed = model.store_norm(seq)
    
    # Get keys and values
    keys = model.to_keys(seq_normed)
    values = model.to_values(seq_normed)
    
    # Split into heads if multi-head
    if model.heads > 1:
        keys = model.split_kv_heads(keys)
        values = model.split_kv_heads(values)
    
    return {
        'keys': keys,
        'values': values,
        'seq_normed': seq_normed,
    }


def compare_memory_states(
    state1: NeuralMemState,
    state2: NeuralMemState
) -> Dict[str, float]:
    """
    Compare two neural memory states.
    
    Args:
        state1: First NeuralMemState
        state2: Second NeuralMemState
        
    Returns:
        Dictionary of comparison metrics
    """
    # Extract weights
    weights1 = state1.weights if isinstance(state1, NeuralMemState) else state1[1]
    weights2 = state2.weights if isinstance(state2, NeuralMemState) else state2[1]
    
    # Convert TensorDict to dict if needed
    if hasattr(weights1, 'to_dict'):
        weights1 = weights1.to_dict()
    if hasattr(weights2, 'to_dict'):
        weights2 = weights2.to_dict()
    
    # Compute total weight norm
    norm1 = sum(w.norm().item() for w in weights1.values())
    norm2 = sum(w.norm().item() for w in weights2.values())
    
    # Compute L2 distance between all weights
    l2_dist = sum(
        (weights1[k] - weights2.get(k, torch.zeros_like(weights1[k]))).norm().item()
        for k in weights1
    )
    
    # Compute average cosine similarity
    cos_sims = []
    for k in weights1:
        if k in weights2:
            w1 = weights1[k].flatten()
            w2 = weights2[k].flatten()
            if w1.shape == w2.shape:
                cos_sim = F.cosine_similarity(w1, w2, dim=0).item()
                cos_sims.append(cos_sim)
    
    avg_cos_sim = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
    
    return {
        'weight_norm_1': norm1,
        'weight_norm_2': norm2,
        'norm_change': norm2 - norm1,
        'l2_distance': l2_dist,
        'avg_cosine_similarity': avg_cos_sim,
    }
