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

def load_best_model(checkpoint_path, num_classes=10, dim=192):
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
            use_memory=True,
            memory_chunk_size=64
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
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
    
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            logits = model(imgs)
            preds = logits.argmax(dim=-1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    return np.concatenate(all_preds), np.concatenate(all_labels)

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

def plot_incorrect_predictions(labels, preds, class_names, num_samples=10):
    """Plot examples of incorrect predictions"""
    incorrect_mask = labels != preds
    incorrect_labels = labels[incorrect_mask][:num_samples]
    incorrect_preds = preds[incorrect_mask][:num_samples]
    
    plt.figure(figsize=(15, 8))
    for i, (true_label, pred_label) in enumerate(zip(incorrect_labels, incorrect_preds)):
        plt.text(0.1, 0.9 - i*0.1, 
                f'True: {class_names[true_label]} | Pred: {class_names[pred_label]}',
                fontsize=10)
    
    plt.title(f'Examples of Incorrect Predictions ({num_samples} samples)')
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()

def analyze_performance(labels, preds, class_names):
    """Generate comprehensive performance analysis"""
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    
    # Calculate overall metrics
    accuracy = np.mean(labels == preds)
    
    # Find best and worst classes
    class_accuracies = [report[str(i)]['precision'] for i in range(len(class_names))]
    best_classes = np.argsort(class_accuracies)[-3:][::-1]
    worst_classes = np.argsort(class_accuracies)[:3]
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Best Classes: {[class_names[i] for i in best_classes]}")
    print(f"Worst Classes: {[class_names[i] for i in worst_classes]}")
    
    return report

def main():
    """Main visualization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize test evaluation results')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], 
                       help='Dataset to use')
    parser.add_argument('--output_dir', type=str, default='visualizations', 
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    num_classes = 10 if args.dataset == 'cifar10' else 100
    model = load_best_model(args.checkpoint, num_classes=num_classes)
    
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
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
    preds, labels = evaluate_model(model, test_loader)
    
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
    incorr_fig = plot_incorrect_predictions(labels, preds, class_names)
    incorr_fig.savefig(f"{args.output_dir}/{args.dataset}_incorrect_predictions.png", dpi=300, bbox_inches='tight')
    
    # 4. Performance analysis
    print("\nPerformance Analysis:")
    report = analyze_performance(labels, preds, class_names)
    
    # Save classification report
    with open(f"{args.output_dir}/{args.dataset}_classification_report.txt", 'w') as f:
        f.write(classification_report(labels, preds, target_names=class_names))
    
    print(f"\nVisualizations saved to {args.output_dir}/")
    print("Analysis complete!")

if __name__ == '__main__':
    main()