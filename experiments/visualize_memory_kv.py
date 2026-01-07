"""
Dog/Cat Memory Visualization Experiment

This script demonstrates how to visualize the neural memory's key-value storage
by performing forward passes on "Dog" and "Cat" images and comparing the results.

Experiment Steps:
1. Forward pass on Dog image
2. Extract Persistent Memory Tokens (memory weights)
3. Forward pass on Cat image
4. Compare activations and attention weights to persistent memory

Usage:
    python experiments/visualize_memory_kv.py --data_dir ./data --output ./viz_output
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

import numpy as np

from titans_pytorch.neural_memory import NeuralMemory, NeuralMemState
from titans_pytorch.memory_visualizer import (
    MemoryVisualizer,
    extract_kv_from_model,
    compare_memory_states,
)


# ----------------------------------------------------------------------
# Experiment Configuration
# ----------------------------------------------------------------------

def get_default_config():
    """Default configuration for the visualization experiment."""
    return {
        'image_size': 32,
        'patch_size': 4,
        'dim': 192,
        'depth': 6,
        'heads': 3,
        'memory_chunk_size': 64,
        'num_classes': 10,
        'batch_size': 1,  # Single image for visualization
        'output_dir': './viz_output',
    }


# ----------------------------------------------------------------------
# Simple Memory ViT for Visualization
# ----------------------------------------------------------------------

class SimpleMemoryViT(nn.Module):
    """
    Simplified Memory ViT for visualization experiments.
    
    This is a simplified version focused on exposing neural memory states
    for visualization purposes.
    """
    
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        memory_chunk_size: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.dim = dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        patch_dim = 3 * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Neural Memory layers
        self.memory_layers = nn.ModuleList()
        for _ in range(depth):
            self.memory_layers.append(
                NeuralMemory(
                    dim=dim,
                    chunk_size=memory_chunk_size,
                    heads=heads,
                    qkv_receives_diff_views=False,
                )
            )
        
        # Output layers
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, num_classes)
        
        # Store intermediate activations for visualization
        self._intermediate_activations: Dict[str, Tensor] = {}
        self._memory_states: List[NeuralMemState] = []
        
    def forward(
        self,
        img: Tensor,
        state: Optional[Tuple] = None,
        return_activations: bool = True
    ) -> Tuple[Tensor, NeuralMemState, Dict]:
        """
        Forward pass with memory state tracking.
        
        Args:
            img: Input image tensor (B, 3, H, W)
            state: Optional previous memory state
            return_activations: Whether to return intermediate activations
            
        Returns:
            logits: Classification logits
            final_state: Final neural memory state
            activations: Dictionary of intermediate activations
        """
        # Patchify
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(img.shape[0], -1, 3 * self.patch_size * self.patch_size)
        
        b, n, _ = patches.shape
        
        # Embed patches
        x = self.to_patch_embedding(patches)
        x = x + self.pos_embedding[:, :n]
        
        # Add CLS token at the end
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([x, cls_tokens], dim=1)
        
        # Clear storage
        self._intermediate_activations = {}
        self._memory_states = []
        
        # Process through memory layers
        memory_states = [None] * len(self.memory_layers)
        
        for i, memory_layer in enumerate(self.memory_layers):
            # Store pre-activation
            self._intermediate_activations[f'layer_{i}_input'] = x.clone()
            
            # Forward through memory layer
            x, next_state = memory_layer(x, state=memory_states[i])
            
            # Store memory state
            memory_states[i] = next_state
            self._memory_states.append(next_state)
            
            # Store post-activation
            self._intermediate_activations[f'layer_{i}_output'] = x.clone()
            
            # Store memory-specific activations
            self._intermediate_activations[f'layer_{i}_memory_output'] = next_state.weights
        
        # Get CLS token output
        cls_out = x[:, -1]
        
        # Store final memory state
        self._intermediate_activations['final_memory_state'] = memory_states[-1]
        
        logits = self.to_logits(self.norm(cls_out))
        
        activations = self._intermediate_activations if return_activations else {}
        
        return logits, memory_states[-1], activations


# ----------------------------------------------------------------------
# Experiment Functions
# ----------------------------------------------------------------------

def load_sample_images(
    data_dir: str = './data',
    batch_size: int = 1,
) -> Tuple[Tensor, Tensor, str, str]:
    """
    Load sample Dog and Cat images for the experiment.
    
    For simplicity, we use CIFAR-10 classes:
    - Dog (class 5): dog
    - Cat (class 3): cat
    
    Returns:
        dog_img: Dog image tensor
        cat_img: Cat image tensor
        dog_label: Dog label string
        cat_label: Cat label string
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10
    dataset = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    # Find dog (class 5) and cat (class 3) images
    dog_idx = None
    cat_idx = None
    
    for idx, (img, label) in enumerate(dataset):
        if label == 5 and dog_idx is None:
            dog_idx = idx
        elif label == 3 and cat_idx is None:
            cat_idx = idx
        
        if dog_idx is not None and cat_idx is not None:
            break
    
    # Load images
    dog_img, _ = dataset[dog_idx]
    cat_img, _ = dataset[cat_idx]
    
    # Add batch dimension
    dog_img = dog_img.unsqueeze(0)
    cat_img = cat_img.unsqueeze(0)
    
    return dog_img, cat_img, "Dog", "Cat"


def run_dog_cat_experiment(
    model: SimpleMemoryViT,
    dog_img: Tensor,
    cat_img: Tensor,
    output_dir: str = './viz_output',
) -> Dict:
    """
    Run the Dog/Cat memory visualization experiment.
    
    Args:
        model: SimpleMemoryViT model
        dog_img: Dog image tensor
        cat_img: Cat image tensor
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary containing experiment results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    dog_img = dog_img.to(device)
    cat_img = cat_img.to(device)
    
    print("=" * 60)
    print("Dog/Cat Memory Visualization Experiment")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # Phase 1: Forward pass on Dog image
    # ------------------------------------------------------------------
    print("\n[Phase 1] Processing Dog image...")
    
    model.eval()
    with torch.no_grad():
        dog_logits, dog_memory_state, dog_activations = model(dog_img)
    
    dog_pred = dog_logits.argmax(dim=-1).item()
    print(f"  Dog image prediction: {dog_pred}")
    
    # Extract memory weights after Dog
    dog_weights = dog_memory_state.weights
    if hasattr(dog_weights, 'to_dict'):
        dog_weights = dog_weights.to_dict()
    
    dog_memory_tokens = None
    for name, weight in dog_weights.items():
        if weight.ndim >= 2:
            dog_memory_tokens = weight
            break
    
    print(f"  Memory weights shape: {[w.shape for w in dog_weights.values()]}")
    
    # ------------------------------------------------------------------
    # Phase 2: Forward pass on Cat image (with Dog's memory state)
    # ------------------------------------------------------------------
    print("\n[Phase 2] Processing Cat image with Dog's memory state...")
    
    # Re-initialize model for fresh start with Dog memory
    model2 = SimpleMemoryViT(
        image_size=32,
        patch_size=4,
        dim=192,
        depth=6,
        heads=3,
        memory_chunk_size=64,
        num_classes=10,
    ).to(device)
    
    model2.eval()
    with torch.no_grad():
        cat_logits, cat_memory_state, cat_activations = model2(cat_img, state=dog_memory_state)
    
    cat_pred = cat_logits.argmax(dim=-1).item()
    print(f"  Cat image prediction: {cat_pred}")
    
    # Extract memory weights after Cat
    cat_weights = cat_memory_state.weights
    if hasattr(cat_weights, 'to_dict'):
        cat_weights = cat_weights.to_dict()
    
    cat_memory_tokens = None
    for name, weight in cat_weights.items():
        if weight.ndim >= 2:
            cat_memory_tokens = weight
            break
    
    print(f"  Memory weights shape: {[w.shape for w in cat_weights.values()]}")
    
    # ------------------------------------------------------------------
    # Phase 3: Compare memory states
    # ------------------------------------------------------------------
    print("\n[Phase 3] Comparing memory states...")
    
    # Compare memory weights
    comparison = compare_memory_states(dog_memory_state, cat_memory_state)
    
    print(f"  Weight norm after Dog: {comparison['weight_norm_1']:.4f}")
    print(f"  Weight norm after Cat: {comparison['weight_norm_2']:.4f}")
    print(f"  Norm change: {comparison['norm_change']:.4f}")
    print(f"  L2 distance: {comparison['l2_distance']:.4f}")
    print(f"  Avg cosine similarity: {comparison['avg_cosine_similarity']:.4f}")
    
    # ------------------------------------------------------------------
    # Phase 4: Generate visualizations
    # ------------------------------------------------------------------
    print("\n[Phase 4] Generating visualizations...")
    
    results = {
        'dog_prediction': dog_pred,
        'cat_prediction': cat_pred,
        'comparison': comparison,
        'dog_weights': dog_weights,
        'cat_weights': cat_weights,
    }
    
    # Create visualizer
    visualizer = MemoryVisualizer(model2)
    
    # Plot 1: Memory weight heatmaps
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = visualizer.plot_memory_weight_heatmap(
            dog_weights,
            title="Memory Weights After Dog Image",
            save_path=os.path.join(output_dir, 'dog_memory_heatmap.png')
        )
        plt.close(fig)
        
        fig = visualizer.plot_memory_weight_heatmap(
            cat_weights,
            title="Memory Weights After Cat Image (with Dog context)",
            save_path=os.path.join(output_dir, 'cat_memory_heatmap.png')
        )
        plt.close(fig)
        
        print("  Saved: dog_memory_heatmap.png")
        print("  Saved: cat_memory_heatmap.png")
    except Exception as e:
        print(f"  Warning: Could not generate heatmaps: {e}")
    
    # Plot 2: Memory evolution
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig = visualizer.plot_memory_evolution(
            [dog_memory_state, cat_memory_state],
            title="Memory Weight Evolution: Dog → Cat",
            save_path=os.path.join(output_dir, 'memory_evolution.png')
        )
        plt.close(fig)
        print("  Saved: memory_evolution.png")
    except Exception as e:
        print(f"  Warning: Could not generate evolution plot: {e}")
    
    # Plot 3: Token similarity
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if dog_memory_tokens is not None and cat_memory_tokens is not None:
            # Flatten tokens for comparison
            dog_tokens = dog_memory_tokens.flatten(start_dim=-2)
            cat_tokens = cat_memory_tokens.flatten(start_dim=-2)
            
            # Create simple similarity visualization
            dog_tokens_np = dog_tokens.cpu().numpy()
            cat_tokens_np = cat_tokens.cpu().numpy()
            
            # Compute similarity matrix
            sim_matrix = np.dot(dog_tokens_np, cat_tokens_np.T)
            norms = np.linalg.norm(dog_tokens_np, axis=1, keepdims=True) * np.linalg.norm(cat_tokens_np, axis=1)
            sim_matrix = sim_matrix / (norms + 1e-8)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
            ax.set_title("Token Similarity: Dog Memory vs Cat Memory")
            ax.set_xlabel("Cat Memory Tokens")
            ax.set_ylabel("Dog Memory Tokens")
            plt.colorbar(im, ax=ax, label='Cosine Similarity')
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, 'token_similarity.png'), dpi=150, bbox_inches='tight')
            plt.close(fig)
            print("  Saved: token_similarity.png")
    except Exception as e:
        print(f"  Warning: Could not generate token similarity plot: {e}")
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Dog image prediction: {dog_pred} (class {'Dog' if dog_pred == 5 else 'Unknown'})")
    print(f"Cat image prediction: {cat_pred} (class {'Cat' if cat_pred == 3 else 'Unknown'})")
    print(f"Memory weight change: {comparison['norm_change']:.4f}")
    print(f"Memory L2 distance: {comparison['l2_distance']:.4f}")
    print(f"Memory cosine similarity: {comparison['avg_cosine_similarity']:.4f}")
    print(f"\nVisualizations saved to: {output_dir}")
    
    # Save results to file
    results_file = os.path.join(output_dir, 'experiment_results.txt')
    with open(results_file, 'w') as f:
        f.write("Dog/Cat Memory Visualization Experiment Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dog prediction: {dog_pred}\n")
        f.write(f"Cat prediction: {cat_pred}\n\n")
        f.write("Memory Comparison:\n")
        for key, value in comparison.items():
            f.write(f"  {key}: {value:.4f}\n")
    print(f"  Saved: experiment_results.txt")
    
    return results


# ----------------------------------------------------------------------
# Main Entry Point
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Dog/Cat Memory Visualization Experiment')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./viz_output', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to pretrained checkpoint (optional)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create model
    model = SimpleMemoryViT(
        image_size=32,
        patch_size=4,
        dim=192,
        depth=6,
        heads=3,
        memory_chunk_size=64,
        num_classes=10,
    ).to(args.device)
    
    # Load pretrained checkpoint if provided
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            print(f"Loading pretrained checkpoint from {args.checkpoint}")
            state_dict = torch.load(args.checkpoint, map_location=args.device)
            model.load_state_dict(state_dict)
            print("Pretrained model loaded successfully!")
        else:
            print(f"Warning: Checkpoint file not found at {args.checkpoint}")
            print("Using random initialization instead.")
    else:
        print("Using random initialization (no pretrained weights)")
    
    print(f"Model created on {args.device}")
    
    # Load sample images
    dog_img, cat_img, dog_label, cat_label = load_sample_images(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    
    print(f"Loaded {dog_label} and {cat_label} images")
    
    # Run experiment
    results = run_dog_cat_experiment(
        model=model,
        dog_img=dog_img,
        cat_img=cat_img,
        output_dir=args.output_dir,
    )
    
    print("\nExperiment completed successfully!")


if __name__ == '__main__':
    main()
