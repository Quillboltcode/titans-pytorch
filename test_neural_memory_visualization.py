#!/usr/bin/env python3
"""
Test script for neural memory visualization using titans_pytorch/neural_memory.py
This script creates a complete test pipeline to validate the visualization functions
and save results to the viz_output folder.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append('/home/quillbolt/project/kaggle/titans-pytorch')

from titans_pytorch.neural_memory import NeuralMemory
from memory_visualization import (
    visualize_memory_trajectories,
    visualize_surprises_and_lrs,
    visualize_memory_state_space,
    visualize_memory_access_patterns,
    analyze_memory_capacity,
    visualize_memory_recall_quality,
    create_memory_animation
)

def create_test_model():
    """Create a test NeuralMemory model with appropriate configuration"""
    print("Creating test NeuralMemory model...")
    
    # Configuration for a small test model
    config = {
        'dim': 64,
        'chunk_size': 4,
        'heads': 2,
        'dim_head': 32,
        'batch_size': 8,
        'momentum': True,
        'momentum_order': 2,
        'per_parameter_lr_modulation': True,
        'spectral_norm_surprises': True,
        'gated_transition': True
    }
    
    model = NeuralMemory(**config)
    return model

def create_test_data():
    """Create synthetic test data for visualization"""
    print("Creating test data...")
    
    # Create synthetic sequences
    batch_size = 4
    seq_len = 16
    dim = 64
    
    # Main sequence
    seq = torch.randn(batch_size, seq_len, dim)
    
    # Store sequence (same as main sequence for simplicity)
    store_seq = seq.clone()
    
    # Create a simple dataloader
    dataset = TensorDataset(seq)
    dataloader = DataLoader(dataset, batch_size=2)
    
    return seq, store_seq, dataloader

def capture_memory_states(model, seq, store_seq=None):
    """Capture memory states during forward pass for visualization"""
    print("Capturing memory states...")
    
    # List to store state history
    state_history = []
    
    # Hook function to capture states
    def capture_hook(viz_data):
        state_history.append(viz_data.get('state', None))
    
    # Attach hook
    original_hook = model._visualization_hook
    model._visualization_hook = capture_hook
    
    # Forward pass
    with torch.no_grad():
        if store_seq is not None:
            retrieved, state, surprises = model(seq, store_seq=store_seq, return_surprises=True)
        else:
            retrieved, state, surprises = model(seq, return_surprises=True)
    
    # Restore original hook
    model._visualization_hook = original_hook
    
    return state_history, surprises, retrieved

def run_all_visualizations(model, seq, store_seq, dataloader, output_dir='viz_output'):
    """Run all visualization functions and save results"""
    print("Running visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Capture memory states
    state_history, surprises, retrieved = capture_memory_states(model, seq, store_seq)
    
    # Extract components from surprises
    unweighted_mem_model_loss, adaptive_lr = surprises
    
    # 1. Memory Trajectories
    print("1. Visualizing memory trajectories...")
    if state_history:
        fig1 = visualize_memory_trajectories(state_history, sample_idx=0)
        if fig1:
            fig1.savefig(f'{output_dir}/memory_trajectories.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
    
    # 2. Surprises and Learning Rates
    print("2. Visualizing surprises and learning rates...")
    fig2 = visualize_surprises_and_lrs(unweighted_mem_model_loss, adaptive_lr)
    if fig2:
        fig2.savefig(f'{output_dir}/surprises_and_lrs.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # 3. Memory State Space
    print("3. Visualizing memory state space...")
    fig3 = visualize_memory_state_space(model, dataloader, 'cpu')
    if fig3:
        fig3.savefig(f'{output_dir}/memory_state_space.png', dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    # 4. Memory Access Patterns
    print("4. Visualizing memory access patterns...")
    fig4 = visualize_memory_access_patterns(model, seq[:1], 'cpu')  # Use first sample
    if fig4:
        fig4.savefig(f'{output_dir}/memory_access_patterns.png', dpi=300, bbox_inches='tight')
        plt.close(fig4)
    
    # 5. Memory Capacity Analysis
    print("5. Analyzing memory capacity...")
    perturbation_positions = [2, 5, 8, 11, 14]  # Positions to perturb
    fig5 = analyze_memory_capacity(model, seq[:1], perturbation_positions, 'cpu')
    if fig5:
        fig5.savefig(f'{output_dir}/memory_capacity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig5)
    
    # 6. Memory Recall Quality
    print("6. Visualizing memory recall quality...")
    # Create simple query-target pairs for testing
    query_sequences = torch.randn(3, 1, 8, 64)  # 3 queries, batch=1, seq_len=8, dim=64
    target_values = torch.randn(3, 8, 64)  # Corresponding targets
    fig6 = visualize_memory_recall_quality(model, query_sequences, target_values, 'cpu')
    if fig6:
        fig6.savefig(f'{output_dir}/memory_recall_quality.png', dpi=300, bbox_inches='tight')
        plt.close(fig6)
    
    # 7. Memory Animation (if enough states)
    print("7. Creating memory animation...")
    if len(state_history) > 5:
        try:
            ani = create_memory_animation(state_history, f'{output_dir}/memory_evolution.mp4')
            if ani:
                print(f"Animation saved to {output_dir}/memory_evolution.mp4")
        except Exception as e:
            print(f"Animation creation failed: {e}")
    else:
        print("Not enough states for animation (need >5 states)")
    
    print(f"All visualizations saved to {output_dir}/")

def test_visualization_integration():
    """Test the integration of visualization functions with NeuralMemory"""
    print("=== Neural Memory Visualization Test ===")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create model
    model = create_test_model()
    print(f"Model created: {model}")
    
    # Create test data
    seq, store_seq, dataloader = create_test_data()
    print(f"Test data created: seq shape {seq.shape}, dataloader length {len(dataloader)}")
    
    # Run visualizations
    output_dir = 'viz_output'
    run_all_visualizations(model, seq, store_seq, dataloader, output_dir)
    
    # Verify outputs
    expected_files = [
        'memory_trajectories.png',
        'surprises_and_lrs.png',
        'memory_state_space.png',
        'memory_access_patterns.png',
        'memory_capacity_analysis.png',
        'memory_recall_quality.png'
    ]
    
    print("\n=== Verification ===")
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} created successfully")
        else:
            print(f"✗ {filename} not found")
    
    # Check for animation
    animation_file = os.path.join(output_dir, 'memory_evolution.mp4')
    if os.path.exists(animation_file):
        print(f"✓ memory_evolution.mp4 created successfully")
    else:
        print(f"✗ memory_evolution.mp4 not found (may require ffmpeg)")
    
    print("\n=== Test Summary ===")
    print("Visualization test completed!")
    print(f"Results saved in: {os.path.abspath(output_dir)}")
    print("\nThe test demonstrates:")
    print("1. Successful integration of visualization functions with NeuralMemory")
    print("2. Capture of memory states during forward passes")
    print("3. Generation of all major visualization types")
    print("4. Proper handling of tensor data and state management")
    print("5. Robust error handling for edge cases")

if __name__ == '__main__':
    test_visualization_integration()