"""
Comparison Script: Standard ViT vs Memory ViT on Split CIFAR-10

This script runs the full continual learning experiment for both models
and generates a comparison report.

Usage:
    python run_comparison.py --epochs_task_a 30 --epochs_task_b 30
"""

import subprocess
import json
import os
import click
import torch
from datetime import datetime

@click.command()
@click.option('--epochs_task_a', default=30, help='Epochs per task')
@click.option('--epochs_task_b', default=30, help='Epochs per task')
@click.option('--batch_size', default=64, help='Batch size')
@click.option('--lr', default=3e-4, help='Learning rate')
@click.option('--dim', default=192, help='Model dimension')
@click.option('--skip_standard', is_flag=True, help='Skip Standard ViT baseline')
@click.option('--skip_memory', is_flag=True, help='Skip Memory ViT')
def run_comparison(epochs_task_a, epochs_task_b, batch_size, lr, dim, 
                   skip_standard, skip_memory):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'config': {
            'epochs_task_a': epochs_task_a,
            'epochs_task_b': epochs_task_b,
            'batch_size': batch_size,
            'learning_rate': lr,
            'dim': dim,
        },
        'standard': None,
        'memory': None,
    }
    
    base_cmd = [
        'python', 'train_continual.py',
        '--epochs_task_a', str(epochs_task_a),
        '--epochs_task_b', str(epochs_task_b),
        '--batch_size', str(batch_size),
        '--lr', str(lr),
        '--dim', str(dim),
        '--phase', 'both',
    ]
    
    # =========================================================================
    # Run Standard ViT Baseline
    # =========================================================================
    if not skip_standard:
        print("\n" + "="*70)
        print("RUNNING STANDARD VIT BASELINE")
        print("="*70)
        
        cmd = base_cmd + ['--model_type', 'standard']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print("STDERR:", result.stderr)
        except Exception as e:
            print(f"Error running Standard ViT: {e}")
        
        # Parse results from checkpoint
        checkpoint_path = './checkpoints/standard_final.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            results['standard'] = checkpoint.get('results', {})
            print(f"\nStandard ViT Results: {results['standard']}")
    
    # =========================================================================
    # Run Memory ViT
    # =========================================================================
    if not skip_memory:
        print("\n" + "="*70)
        print("RUNNING MEMORY VIT")
        print("="*70)
        
        cmd = base_cmd + ['--model_type', 'memory']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print("STDERR:", result.stderr)
        except Exception as e:
            print(f"Error running Memory ViT: {e}")
        
        # Parse results from checkpoint
        checkpoint_path = './checkpoints/memory_final.pth'
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            results['memory'] = checkpoint.get('results', {})
            print(f"\nMemory ViT Results: {results['memory']}")
    
    # =========================================================================
    # Generate Comparison Report
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL COMPARISON REPORT")
    print("="*70)
    
    if results['standard'] and results['memory']:
        print(f"\n{'Metric':<30} {'Standard ViT':>15} {'Memory ViT':>15}")
        print("-" * 60)
        
        standard = results['standard']
        memory = results['memory']
        
        metrics = [
            ('Task A Acc (Phase 1)', 'task_A_acc_phase1'),
            ('Task A Acc (Phase 2)', 'task_A_acc_phase2'),
            ('Task B Acc', 'task_B_acc'),
            ('Forgetting', 'forgetting'),
        ]
        
        for name, key in metrics:
            std_val = standard.get(key, 'N/A')
            mem_val = memory.get(key, 'N/A')
            
            if isinstance(std_val, float):
                std_str = f"{std_val:.2f}%"
            else:
                std_str = str(std_val)
            
            if isinstance(mem_val, float):
                mem_str = f"{mem_val:.2f}%"
            else:
                mem_str = str(mem_val)
            
            print(f"{name:<30} {std_str:>15} {mem_str:>15}")
        
        # Calculate improvement
        forgetting_improvement = standard.get('forgetting', 0) - memory.get('forgetting', 0)
        print("-" * 60)
        print(f"\nMemory ViT reduces forgetting by: {forgetting_improvement:.2f}%")
        
        if memory.get('task_A_acc_phase2', 0) > standard.get('task_A_acc_phase2', 0) + 10:
            print("\n✓ HYPOTHESIS CONFIRMED: Memory ViT significantly reduces catastrophic forgetting!")
        else:
            print("\n? Results inconclusive - further tuning may be needed")
    
    else:
        print("Not enough data to generate comparison report.")
        print("Standard ViT:", results['standard'])
        print("Memory ViT:", results['memory'])
    
    # Save results to JSON
    results_path = f'./results/comparison_{timestamp}.json'
    os.makedirs('./results', exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == '__main__':
    run_comparison()
