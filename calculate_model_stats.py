import torch
import torch.nn as nn
from thop import profile, clever_format
from titan_text_model import SmallTitanModel

def calculate_model_statistics():
    """Calculate and print parameter count and FLOP count for SmallTitanModel"""
    print("=" * 80)
    print("SmallTitanModel Statistics")
    print("=" * 80)
    
    # Initialize model (gated=True, segment_size=256 as default)
    model = SmallTitanModel(gated=True, segment_size=256)
    model.eval()
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"\nParameter Count:")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Non-trainable:        {non_trainable_params:,}")
    print(f"  Size (FP32):          {total_params * 4 / 1e6:.2f} MB")
    print(f"  Size (BF16):          {total_params * 2 / 1e6:.2f} MB")
    
    # Calculate FLOPs (using a typical input for language models)
    batch_size = 1
    seq_length = 256
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
    
    try:
        # Calculate FLOPs using thop
        flops, params = profile(model, inputs=(input_ids,), verbose=False)
        flops, params = clever_format([flops, params], "%.2f")
        
        print(f"\nFLOPs:")
        print(f"  Forward Pass (BS=1, Seq=256): {flops}")
        
        # Calculate FLOPs for larger batch sizes
        print(f"  Estimated per Token: {float(flops.split()[0]) / seq_length:.2f} {flops.split()[1]}/token")
        
    except Exception as e:
        print(f"\nWarning: Could not calculate FLOPs using thop: {e}")
        print("FLOP calculation requires compatible model structure")
    
    print("\n" + "=" * 80)
    print("Model Structure Summary:")
    print("-" * 80)
    print(model)

if __name__ == "__main__":
    calculate_model_statistics()
