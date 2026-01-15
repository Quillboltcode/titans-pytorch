#!/usr/bin/env python3
"""
Implementation of neural memory visualization functions from note/qwen_rep.md
This module provides comprehensive visualization tools for analyzing neural memory systems.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.spatial.distance import cosine
import torch
import torch.nn as nn

def visualize_memory_trajectories(state_history, sample_idx=0):
    """
    Visualize how memory weights evolve across timesteps for a single sample
    
    Args:
        state_history: List of NeuralMemState objects collected during inference
        sample_idx: Index of sample to visualize
    """
    # Extract weight tensors for a specific sample
    all_weights = []
    timesteps = []
    
    for timestep, state in enumerate(state_history):
        if hasattr(state, 'weights') and state.weights is not None:
            # Get weights for specific sample
            if isinstance(state.weights, dict):
                sample_weights = {k: v[sample_idx].detach().cpu().numpy() for k, v in state.weights.items()}
            else:
                sample_weights = {f'weight_{i}': state.weights[i][sample_idx].detach().cpu().numpy()
                                for i in range(len(state.weights))}
            
            # Only proceed if we have weights to concatenate
            if sample_weights:
                flat_weights = np.concatenate([w.flatten() for w in sample_weights.values()])
                all_weights.append(flat_weights)
                timesteps.append(timestep)
    
    # Reduce dimensionality for visualization
    if len(all_weights) > 1:
        pca = PCA(n_components=2)
        weights_2d = pca.fit_transform(np.array(all_weights))
        
        plt.figure(figsize=(10, 8))
        plt.scatter(weights_2d[:, 0], weights_2d[:, 1], c=timesteps, cmap='viridis')
        plt.colorbar(label='Timestep')
        
        # Connect points to show trajectory
        for i in range(len(weights_2d)-1):
            plt.plot(weights_2d[i:i+2, 0], weights_2d[i:i+2, 1], 'k-', alpha=0.3)
        
        plt.title(f'Memory Weight Trajectory (Sample {sample_idx})')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.savefig('memory_trajectory.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return plt.gcf()
    else:
        print("Not enough weight data for visualization")
        return None

def visualize_surprises_and_lrs(surprises, adaptive_lrs, token_labels=None):
    """
    Visualize surprise signals and adaptive learning rates across sequence positions
    
    Args:
        surprises: Tensor of surprise values (batch, heads, seq_len)
        adaptive_lrs: Tensor of learning rates (batch, heads, seq_len)
        token_labels: Optional list of token labels for x-axis
    """
    batch_idx, head_idx = 0, 0  # Visualize first sample and head
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Surprise heatmap
    if isinstance(surprises, torch.Tensor):
        surprises_np = surprises[batch_idx, head_idx].unsqueeze(0).cpu().detach().numpy()
    else:
        surprises_np = np.array(surprises)[batch_idx, head_idx].reshape(1, -1)
    
    sns.heatmap(surprises_np, ax=ax1, cmap='coolwarm', center=0)
    ax1.set_title('Surprise Signals (Gradient Magnitude)')
    ax1.set_ylabel('Head')
    
    # Learning rate heatmap
    if isinstance(adaptive_lrs, torch.Tensor):
        adaptive_lrs_np = adaptive_lrs[batch_idx, head_idx].unsqueeze(0).cpu().detach().numpy()
    else:
        adaptive_lrs_np = np.array(adaptive_lrs)[batch_idx, head_idx].reshape(1, -1)
    
    sns.heatmap(adaptive_lrs_np, ax=ax2, cmap='YlOrRd')
    ax2.set_title('Adaptive Learning Rates')
    
    if token_labels:
        ax2.set_xticklabels(token_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('surprises_and_lrs.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def visualize_memory_state_space(model, dataloader, device):
    """
    Project high-dimensional memory states into 2D space to reveal structure
    
    Args:
        model: Neural network containing NeuralMemory module
        dataloader: DataLoader with representative samples
        device: PyTorch device
    """
    # Find NeuralMemory module
    memory_module = None
    for module in model.modules():
        if hasattr(module, '_visualization_hook'):
            memory_module = module
            break
    
    if memory_module is None:
        print("No NeuralMemory module found in model")
        return None
    
    # Register hook to capture memory states
    memory_states = []
    token_contexts = []
    
    def capture_state_hook(viz_data):
        if 'weights' in viz_data:
            weights_dict = {}
            for k, v in viz_data['weights'].items():
                if isinstance(v, torch.Tensor):
                    weights_dict[k] = v[0].detach().cpu().flatten().numpy()
                else:
                    weights_dict[k] = np.array(v[0]).flatten()
            memory_states.append(weights_dict)
        
        # Store context information for coloring
        if 'retrieve_seq' in viz_data:
            if isinstance(viz_data['retrieve_seq'], torch.Tensor):
                token_contexts.append(viz_data['retrieve_seq'][0].detach().cpu().numpy())
            else:
                token_contexts.append(np.array(viz_data['retrieve_seq'][0]))
    
    # Temporarily attach hook
    original_hook = getattr(memory_module, '_visualization_hook', None)
    memory_module._visualization_hook = capture_state_hook
    
    # Process batch
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            _ = model(inputs)
            break  # Just process one batch for visualization
    
    # Restore original hook
    memory_module._visualization_hook = original_hook
    
    if not memory_states:
        print("No memory states captured")
        return None
    
    # Convert to numpy array
    state_array = np.array([np.concatenate(list(state.values())) for state in memory_states])
    
    # Dimensionality reduction
    if len(state_array) > 1:
        tsne = TSNE(n_components=2, random_state=42)
        state_2d = tsne.fit_transform(state_array)
    elif len(state_array) == 1:
        # Fallback to PCA if only one sample
        pca = PCA(n_components=min(2, state_array.shape[1]))
        state_2d = pca.fit_transform(state_array)
    else:
        print("No state data available for visualization")
        return None
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Color by token identity or position
    colors = np.arange(len(state_2d))
    scatter = plt.scatter(state_2d[:, 0], state_2d[:, 1], c=colors, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Sequence Position')
    
    # Add trajectories
    for i in range(len(state_2d)-1):
        plt.plot(state_2d[i:i+2, 0], state_2d[i:i+2, 1], 'k-', alpha=0.2)
    
    plt.title('Memory State Space Trajectory')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig('memory_state_space.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return plt.gcf()

def visualize_memory_access_patterns(model, input_sequence, device):
    """
    Visualize memory access patterns during read/write operations
    
    Args:
        model: Model containing NeuralMemory module
        input_sequence: Input sequence tensor
        device: PyTorch device
    """
    # Find NeuralMemory module
    memory_module = None
    for module in model.modules():
        if hasattr(module, '_visualization_hook'):
            memory_module = module
            break
    
    if memory_module is None:
        print("No NeuralMemory module found in model")
        return None
    
    # Register hook to capture intermediate values
    capture_data = {}
    
    def viz_hook(viz_data):
        capture_data.update(viz_data)
    
    # Temporarily attach hook
    original_hook = getattr(memory_module, '_visualization_hook', None)
    memory_module._visualization_hook = viz_hook
    
    # Forward pass
    with torch.no_grad():
        output, state = memory_module(input_sequence.to(device))
    
    # Restore hook
    memory_module._visualization_hook = original_hook
    
    # Extract weights for visualization
    if 'weights' not in capture_data:
        print("No weights captured in visualization data")
        return None
    
    weights = capture_data['weights']
    queries = capture_data.get('queries', None)
    
    # Get first sample and head
    sample_idx, head_idx = 0, 0
    
    # Visualize weight matrices (for first few layers)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    weight_names = list(weights.keys())[:3]  # First three weight matrices
    
    for i, name in enumerate(weight_names):
        w = weights[name]
        if isinstance(w, torch.Tensor):
            w = w[sample_idx, head_idx].detach().cpu().numpy()
        else:
            w = np.array(w)[sample_idx, head_idx]
        
        # Only visualize if matrix is not too large
        if w.size < 10000:
            sns.heatmap(w, ax=axes[i], cmap='coolwarm', center=0)
            axes[i].set_title(f'Weight Matrix: {name}')
    
    plt.tight_layout()
    plt.savefig('weight_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize query-key attention patterns
    if queries is not None:
        queries = queries[sample_idx, head_idx].detach().cpu().numpy() if isinstance(queries, torch.Tensor) else np.array(queries)[sample_idx, head_idx]
        keys = capture_data.get('keys', None)
        
        if keys is not None:
            keys = keys[sample_idx, head_idx].detach().cpu().numpy() if isinstance(keys, torch.Tensor) else np.array(keys)[sample_idx, head_idx]
            attn = queries @ keys.T
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(attn, cmap='viridis')
            plt.title('Query-Key Attention Pattern')
            plt.xlabel('Key Position')
            plt.ylabel('Query Position')
            plt.savefig('attention_pattern.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    return fig

def analyze_memory_capacity(model, base_input, perturbation_positions, device):
    """
    Analyze how perturbations at different positions affect memory retention
    
    Args:
        model: Model with NeuralMemory
        base_input: Base input sequence
        perturbation_positions: List of positions to perturb
        device: PyTorch device
    """
    # Find NeuralMemory module
    memory_module = None
    for module in model.modules():
        if hasattr(module, '_visualization_hook'):
            memory_module = module
            break
    
    if memory_module is None:
        print("No NeuralMemory module found in model")
        return None
    
    model.eval()
    original_state = None
    perturbation_effects = []
    
    # Get baseline memory state
    with torch.no_grad():
        _, original_state = memory_module(base_input.to(device))
    
    # Apply perturbations at different positions
    for pos in perturbation_positions:
        perturbed_input = base_input.clone()
        # Add noise at specific position
        perturbed_input[:, pos] += torch.randn_like(perturbed_input[:, pos]) * 0.1
        
        with torch.no_grad():
            _, perturbed_state = memory_module(perturbed_input.to(device))
        
        # Compare memory states
        weight_diffs = {}
        for name in original_state.weights.keys():
            orig_w = original_state.weights[name][0]
            pert_w = perturbed_state.weights[name][0]
            diff = torch.norm(orig_w - pert_w).item()
            weight_diffs[name] = diff
        
        perturbation_effects.append((pos, weight_diffs))
    
    # Plot results
    positions = [p for p, _ in perturbation_effects]
    diffs_by_layer = {name: [diffs[name] for _, diffs in perturbation_effects] 
                     for name in perturbation_effects[0][1].keys()}
    
    plt.figure(figsize=(12, 6))
    for name, diffs in diffs_by_layer.items():
        plt.plot(positions, diffs, label=name)
    
    plt.title('Memory Sensitivity to Input Perturbations')
    plt.xlabel('Perturbation Position')
    plt.ylabel('Weight Change Magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig('memory_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return plt.gcf()

def visualize_memory_recall_quality(model, query_sequences, target_values, device):
    """
    Visualize how well the memory recalls stored values given queries
    
    Args:
        model: Model with NeuralMemory
        query_sequences: Query sequences
        target_values: Expected values to retrieve
        device: PyTorch device
    """
    # Find NeuralMemory module
    memory_module = None
    for module in model.modules():
        if hasattr(module, '_visualization_hook'):
            memory_module = module
            break
    
    if memory_module is None:
        print("No NeuralMemory module found in model")
        return None
    
    model.eval()
    all_similarities = []
    all_positions = []
    
    # Process in batches
    for i in range(len(query_sequences)):
        query = query_sequences[i:i+1].to(device)
        target = target_values[i].cpu().numpy()
        
        with torch.no_grad():
            retrieved, _ = memory_module(query)
        
        retrieved = retrieved[0].cpu().numpy()
        
        # Calculate similarity at each position
        for pos in range(retrieved.shape[0]):
            sim = 1 - cosine(retrieved[pos], target[pos])
            all_similarities.append(sim)
            all_positions.append(pos)
    
    # Plot recall quality by position
    plt.figure(figsize=(12, 6))
    plt.plot(all_positions, all_similarities, 'o-', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('Memory Recall Quality by Position')
    plt.xlabel('Sequence Position')
    plt.ylabel('Cosine Similarity to Target')
    plt.grid(True)
    plt.savefig('recall_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return plt.gcf()

def create_memory_animation(state_history, output_path='memory_evolution.mp4'):
    """
    Create animation showing memory evolution over time
    
    Args:
        state_history: List of memory states over time
        output_path: Path to save animation
    """
    try:
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        def update(frame):
            ax.clear()
            state = state_history[frame]
            
            # Visualize state at this frame
            if hasattr(state, 'weights') and state.weights:
                # Extract weights for first sample
                if isinstance(state.weights, dict):
                    weights = list(state.weights.values())[0][0].detach().cpu().numpy()
                else:
                    weights = state.weights[0][0].detach().cpu().numpy()
                
                # Flatten and reduce dimensionality
                flat_weights = weights.flatten().reshape(1, -1)
                pca = PCA(n_components=2)
                weights_2d = pca.fit_transform(flat_weights)
                
                ax.scatter(weights_2d[:, 0], weights_2d[:, 1], c='blue', s=100)
                ax.set_title(f'Memory State at Timestep {frame}')
                ax.set_xlabel('PCA Component 1')
                ax.set_ylabel('PCA Component 2')
        
        ani = animation.FuncAnimation(fig, update, frames=len(state_history), interval=200)
        ani.save(output_path, writer='ffmpeg', dpi=300)
        plt.close()
        
        return ani
        
    except ImportError:
        print("matplotlib.animation or ffmpeg not available. Skipping animation.")
        return None

# Utility function to test the implementations
def test_visualization_functions():
    """Test all visualization functions with mock data"""
    print("Testing visualization functions...")
    
    # Create mock data for testing
    class MockState:
        def __init__(self, weights_dict):
            self.weights = weights_dict
    
    # Test 1: Memory trajectories
    print("Testing memory trajectories...")
    mock_states = []
    for i in range(10):
        weights = {
            'w1': torch.randn(1, 5, 10, 10),
            'w2': torch.randn(1, 5, 10, 10)
        }
        mock_states.append(MockState(weights))
    
    fig1 = visualize_memory_trajectories(mock_states, sample_idx=0)
    print("✓ Memory trajectories test completed")
    
    # Test 2: Surprises and learning rates
    print("Testing surprises and learning rates...")
    surprises = torch.randn(1, 1, 20)
    adaptive_lrs = torch.randn(1, 1, 20).abs()
    fig2 = visualize_surprises_and_lrs(surprises, adaptive_lrs)
    print("✓ Surprises and learning rates test completed")
    
    # Test 3: Memory recall quality
    print("Testing memory recall quality...")
    query_sequences = torch.randn(5, 1, 10, 64)
    target_values = torch.randn(5, 10, 64)
    
    # Create a mock memory module
    class MockMemory(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(64, 64)
        
        def forward(self, x):
            return self.linear(x), MockState({'w': torch.randn(1, 1, 64, 64)})
    
    mock_model = MockMemory()
    fig3 = visualize_memory_recall_quality(mock_model, query_sequences, target_values, 'cpu')
    print("✓ Memory recall quality test completed")
    
    print("All visualization function tests completed successfully!")

if __name__ == '__main__':
    test_visualization_functions()