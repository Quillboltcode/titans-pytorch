#!/usr/bin/env python3
"""
Visualization script for analyzing final test evaluation using the best model.
This script provides comprehensive analysis and visualization of model performance
on the test dataset, including accuracy metrics, confusion matrices, and
per-class performance analysis.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from tqdm import tqdm

# Add project root to path
# sys.path.append('/home/quillbolt/project/kaggle/titans-pytorch')

from VitMemoryv3 import SimpleViT
from train_vit_memory_cifar_accelerator import count_flops

def load_best_model(checkpoint_path, num_classes=10, dim=192, use_memory=True):
    """Load the best model from checkpoint"""
    try:
        # Create model
        model = SimpleViT(
            image_size=32,
            patch_size=4,
            num_classes=num_classes,
            dim=dim,
            depth=6,
            heads=3,
            mlp_dim=dim*4,
            use_memory=use_memory,
            memory_chunk_size=64
        )
        
        # Load checkpoint
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            checkpoint = load_file(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Strip 'module.' prefix if present (from DataParallel/DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_test_loader(dataset='cifar10', batch_size=100):
    """Create test data loader"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if dataset == 'cifar10':
        testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    else:
        testset = datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluate model on test set and return predictions"""
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_imgs = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            preds = logits.argmax(dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_imgs.append(imgs.cpu().numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_imgs)

def plot_confusion_matrix(labels, preds, class_names, title="Confusion Matrix"):
    """Plot confusion matrix"""
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    return plt.gcf()

def plot_class_accuracy(labels, preds, class_names, title="Per-Class Accuracy"):
    """Plot per-class accuracy"""
    accuracy = []
    for i in range(len(class_names)):
        mask = labels == i
        if np.any(mask):
            acc = np.mean(preds[mask] == i)
        else:
            acc = 0
        accuracy.append(acc)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), accuracy)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0, 1)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return plt.gcf()

def plot_incorrect_predictions(images, labels, preds, class_names, num_samples=10):
    """Plot examples of incorrect predictions"""
    incorrect_mask = labels != preds
    incorrect_images = images[incorrect_mask][:num_samples]
    incorrect_labels = labels[incorrect_mask][:num_samples]
    incorrect_preds = preds[incorrect_mask][:num_samples]
    
    if len(incorrect_labels) == 0:
        return plt.figure()

    rows = int(np.ceil(len(incorrect_labels) / 5))
    plt.figure(figsize=(15, 3 * rows))
    
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    for i in range(len(incorrect_labels)):
        img = incorrect_images[i].transpose(1, 2, 0) # CHW -> HWC
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        plt.subplot(rows, 5, i+1)
        plt.imshow(img)
        plt.title(f'True: {class_names[incorrect_labels[i]]}\nPred: {class_names[incorrect_preds[i]]}',
                 color='red', fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    return plt.gcf()

def analyze_performance(labels, preds, class_names):
    """Generate comprehensive performance analysis"""
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    
    # Calculate overall metrics
    accuracy = np.mean(labels == preds)
    
    # Find best and worst classes
    class_accuracies = [report[name]['f1-score'] for name in class_names]
    best_classes = np.argsort(class_accuracies)[-3:][::-1]
    worst_classes = np.argsort(class_accuracies)[:3]
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Best Classes (by F1-score): {[class_names[i] for i in best_classes]}")
    print(f"Worst Classes (by F1-score): {[class_names[i] for i in worst_classes]}")
    
    return report, accuracy, best_classes, worst_classes

def main():
    """Main visualization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize test evaluation results')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], 
                       help='Dataset to use')
    parser.add_argument('--output_dir', type=str, default='visualizations', 
                       help='Directory to save visualizations')
    parser.add_argument('--baseline', action='store_true', help='Use Baseline ViT (no memory)')
    parser.add_argument('--dim', type=int, default=192, help='Model dimension')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = load_best_model(args.checkpoint, num_classes=num_classes, dim=args.dim, use_memory=not args.baseline)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Calculate Parameters and FLOPs
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {params:,}")
    
    # Calculate FLOPs
    print("Calculating FLOPs...")
    std_flops, mem_flops = count_flops(model)
    print(f"Standard FLOPs: {std_flops/1e9:.4f} GFLOPs")
    print(f"Memory-aware FLOPs: {mem_flops/1e9:.4f} GFLOPs")
    
    # Load test data
    print("Loading test data...")
    test_loader = get_test_loader(args.dataset)
    
    # Get class names
    if args.dataset == 'cifar10':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        # For CIFAR-100, use generic class names
        class_names = [f'Class {i}' for i in range(100)]
    
    # Evaluate model
    print("Evaluating model...")
    preds, labels, images = evaluate_model(model, test_loader)
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # 1. Confusion Matrix
    cm_fig = plot_confusion_matrix(labels, preds, class_names, 
                                  f"{args.dataset.upper()} Confusion Matrix")
    cm_fig.savefig(f"{args.output_dir}/{args.dataset}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    
    # 2. Per-class accuracy
    acc_fig = plot_class_accuracy(labels, preds, class_names, 
                                f"{args.dataset.upper()} Per-Class Accuracy")
    acc_fig.savefig(f"{args.output_dir}/{args.dataset}_class_accuracy.png", dpi=300, bbox_inches='tight')
    
    # 3. Incorrect predictions examples
    incorr_fig = plot_incorrect_predictions(images, labels, preds, class_names)
    incorr_fig.savefig(f"{args.output_dir}/{args.dataset}_incorrect_predictions.png", dpi=300, bbox_inches='tight')
    
    # 4. Performance analysis
    print("\nPerformance Analysis:")
    report, accuracy, best_classes, worst_classes = analyze_performance(labels, preds, class_names)
    
    # Save performance summary and classification report
    summary_path = f"{args.output_dir}/{args.dataset}_performance_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*40 + "\n")
        f.write("      Performance and Model Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Model Checkpoint: {args.checkpoint}\n")
        f.write(f"Dataset: {args.dataset.upper()}\n\n")
        
        f.write("--- Model Stats ---\n")
        f.write(f"Total Parameters: {params:,}\n")
        f.write(f"Standard FLOPs: {std_flops/1e9:.4f} GFLOPs\n")
        f.write(f"Memory-aware FLOPs: {mem_flops/1e9:.4f} GFLOPs\n\n")
        
        f.write("--- Evaluation Metrics ---\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        
        f.write("--- Per-Class Performance (by F1-score) ---\n")
        f.write(f"Top 3 Best Performing Classes:\n")
        f.write('\n'.join([f"  - {class_names[i]}: {report[class_names[i]]['f1-score']:.4f}" for i in best_classes]))
        f.write(f"\n\nTop 3 Worst Performing Classes:\n")
        f.write('\n'.join([f"  - {class_names[i]}: {report[class_names[i]]['f1-score']:.4f}" for i in worst_classes]))
        f.write("\n\n" + "-"*40 + "\n\n")
        f.write("--- Full Classification Report ---\n")
        f.write(classification_report(labels, preds, target_names=class_names))
    
    print(f"\nPerformance summary saved to {summary_path}")
    print(f"Visualizations saved to {args.output_dir}/")
    print("Analysis complete!")

if __name__ == '__main__':
    main()