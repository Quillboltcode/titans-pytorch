import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from accelerate import Accelerator
import click
import numpy as np
from collections import namedtuple

from VitMemoryv3 import SimpleViT

# NeuralMemState structure (from titans_pytorch)
NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates',
])

def detach_mem_state(state):
    """Properly detach NeuralMemState for memory management"""
    if state is None:
        return None
    if isinstance(state, (list, tuple)):
        return [detach_mem_state(s) for s in state]

    # Check if it's a NeuralMemState namedtuple
    if hasattr(state, '_fields') and 'seq_index' in state._fields:
        # Handle NeuralMemState namedtuple
        detached_state = {}
        for field in state._fields:
            value = getattr(state, field)
            if field == 'weights' and value is not None:
                # Weights are TensorDict
                detached_state[field] = value.apply(lambda t: t.detach() if torch.is_tensor(t) else t)
            elif field == 'states' and value is not None:
                # States are tuples of TensorDicts
                detached_state[field] = tuple(
                    td.apply(lambda t: t.detach() if torch.is_tensor(t) else t)
                    for td in value
                )
            elif torch.is_tensor(value):
                detached_state[field] = value.detach()
            else:
                # For non-tensor fields like seq_index (int), just copy
                detached_state[field] = value

        return NeuralMemState(**detached_state)
    else:
        # If not NeuralMemState, just detach if tensor, else copy
        if torch.is_tensor(state):
            return state.detach()
        else:
            return state

def check_state_change(old_states, new_states, step_name=""):
    """Check if NeuralMemState has changed"""
    if old_states is None or new_states is None:
        print(f"{step_name} States are None")
        return False

    changed = False
    for i, (old, new) in enumerate(zip(old_states, new_states)):
        if old is None or new is None:
            continue
        # Check if any tensor in the state has changed
        for field in old._fields:
            old_val = getattr(old, field)
            new_val = getattr(new, field)
            if torch.is_tensor(old_val) and torch.is_tensor(new_val):
                if not torch.equal(old_val, new_val):
                    print(f"{step_name} Layer {i} {field} changed")
                    changed = True
            elif hasattr(old_val, 'apply') and hasattr(new_val, 'apply'):  # TensorDict
                # Check if any tensor in TensorDict changed
                for key in old_val.keys():
                    if torch.is_tensor(old_val[key]) and torch.is_tensor(new_val[key]):
                        if not torch.equal(old_val[key], new_val[key]):
                            print(f"{step_name} Layer {i} {field}.{key} changed")
                            changed = True
    return changed

def evaluate_with_ttt(model, test_loader, accelerator, ttt_steps=1):
    """
    Evaluate with Test-Time Training for NeuralMemory models
    """
    # Set model to eval mode (batchnorm/dropout)
    model.eval()

    # Configure trainable parameters for TTT
    model.set_trainable_parameters(memory_training=True, freeze_attention=True)

    # Initialize memory states for each layer
    # NeuralMemory handles its own updates internally - no external optimizer needed
    states = [None] * len(model.transformer.layers) if hasattr(model.transformer, 'layers') else None

    # Metrics tracking
    total_loss = 0
    correct = 0
    total = 0

    # TTT Evaluation loop - NO torch.no_grad() needed
    progress_bar = tqdm(test_loader, desc="TTT Evaluation")

    for batch_idx, (imgs, labels) in enumerate(progress_bar):
        imgs = imgs.to(accelerator.device)
        labels = labels.to(accelerator.device)

        # TTT Phase: Update memory using current batch
        for ttt_step in range(ttt_steps):
            initial_states = [detach_mem_state(s) for s in states] if states else None

            # Forward pass with memory updates enabled
            # NeuralMemory computes gradients internally for its updates
            logits, new_states = model(
                imgs,
                state=states,
                memory_training=True,  # Enables memory updates
                freeze_attention=True   # Keeps attention frozen
            )

            # Check if states changed
            if batch_idx == 0 and ttt_step == 0:  # Only check first batch/step for logging
                changed = check_state_change(initial_states, new_states, f"TTT Step {ttt_step}")
                if not changed:
                    print("Warning: NeuralMemState did not change during TTT update!")

            # No external backward needed - NeuralMemory handles internal updates
            # But we need to compute loss to provide supervision signal
            _ = F.cross_entropy(logits, labels)  # Loss is used internally by NeuralMemory

            # Update states for next iteration
            states = [detach_mem_state(s) for s in new_states] if new_states else None

        # Inference Phase: Final prediction with adapted memory
        with torch.no_grad():  # Now we can disable gradients for final prediction
            logits, _ = model(
                imgs,
                state=states,
                memory_training=False,  # No more updates
                freeze_attention=True
            )

        # Standard evaluation metrics
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()

        predictions = logits.argmax(dim=-1)
        predictions, labels = accelerator.gather_for_metrics((predictions, labels))

        total += labels.size(0)
        correct += predictions.eq(labels).sum().item()

        # Update progress bar
        current_acc = 100. * correct / total
        progress_bar.set_postfix({'acc': f'{current_acc:.2f}%'})

        # Detach states between batches to prevent infinite gradient graphs
        if states is not None:
            states = [detach_mem_state(s) for s in states]

        # Only process first few batches for testing
        if batch_idx >= 2:
            break

    # Final metrics
    test_acc = 100. * correct / total
    avg_loss = total_loss / (batch_idx + 1)

    return test_acc, avg_loss

@click.command()
@click.option('--checkpoint', default=None, help='Path to model checkpoint (optional)')
@click.option('--ttt_steps', default=1, help='Number of TTT update steps per batch')
@click.option('--batch_size', default=100, help='Batch size for evaluation')
def main(checkpoint, ttt_steps, batch_size):
    accelerator = Accelerator()
    
    # Load Data
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Load Model
    model = SimpleViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=192,
        depth=6,
        heads=3,
        mlp_dim=192*4,
        use_memory=True,
        memory_chunk_size=64
    )
    
    # Handle different checkpoint formats or random initialization
    if checkpoint and os.path.exists(checkpoint):
        checkpoint_data = torch.load(checkpoint, map_location='cpu')
        if 'model_state_dict' in checkpoint_data:
            model.load_state_dict(checkpoint_data['model_state_dict'])
        else:
            model.load_state_dict(checkpoint_data)  # Assume direct state_dict
        print(f"Loaded checkpoint from {checkpoint}")
    else:
        print("No checkpoint provided or not found. Using random initialization.")
    
    model = accelerator.prepare(model)
    
    # Run TTT evaluation
    acc, loss = evaluate_with_ttt(model, test_loader, accelerator, ttt_steps=ttt_steps)
    
    print(f"\nTTT Evaluation Results (Steps={ttt_steps}):")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Loss: {loss:.4f}")

if __name__ == '__main__':
    main()