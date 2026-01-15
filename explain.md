The NeuralMemory class in `titans_pytorch/neural_memory.py` implements a dynamic memory mechanism for neural networks, particularly transformers, inspired by concepts from "Test-Time Training" (TTT) and associative memory models. It allows the model to store and retrieve key-value associations in the parameters of a small neural network (the "memory model"), enabling efficient handling of long sequences or adaptive learning. Below, I'll explain its architecture and how it works, based on the code.

### Key Architectural Components

The NeuralMemory is a model-within-a-model.

Outer Model: The main Transformer/Network. It learns how to update the memory.
Inner Model (Memory): A small MLP (Multi-Layer Perceptron) whose weights are the memory.

1. **Memory Model (Core Storage)**:
   - The memory is represented by a small MLP (Multi-Layer Perceptron), defaulting to `MemoryMLP` from `memory_models.py`.
   - `MemoryMLP` is a simple feedforward network with configurable depth (default 2 layers), expansion factor (default 4x hidden dimension), and Xavier-initialized weights. It takes inputs (keys) and produces outputs (values) via matrix multiplications and GeLU activations.
   - The MLP's parameters (weights) are the "memory" — they encode learned associations between keys and values. This is analogous to fast weights in linear attention or Schmidhuber's work, where the network acts as a differentiable associative memory.
   - Optionally wrapped in `ResidualNorm` for normalization and residual connections, as in the TTT paper.


2. **Multi-Head Support**:
   - Supports multiple heads (default 1) for parallel processing, similar to multi-head attention.
   - Heads split the input dimension and process independently, then merge/combine outputs.
   - Includes optional RMSNorm for queries, keys, and multi-head outputs.

3. **Chunking and Pooling**:
   - Sequences are processed in chunks (configurable size, default 1) for efficiency.
   - Chunk representations are pooled using average pooling or attention pooling (if `attn_pool_chunks=True`).

4. **Adaptive Optimization**:
   - **Adaptive Learning Rate**: Learned per-token/chunk, transformed via sigmoid (default max 1e-2).
   - **Momentum**: Supports Nth-order momentum (default 1st order) using associative scans for cumulative updates.
   - **Weight Decay**: Learned decay factor applied via associative scans to prevent overfitting.
   - **Per-Parameter Modulation**: Allows the network to modulate learning rates per memory model parameter.
   - **Spectral Normalization**: Optional Newton-Schulz iteration to normalize updates (inspired by Muon optimizer).

5. **State Management**:
   - Uses `NeuralMemState` (a namedtuple) to track sequence index, weights, cached sequences, past states (updates/momentum), and updates.
   - Supports detaching states for gradient flow control.

6. **Other Features**:
   - **Gated Transitions**: Smoothly transitions from initial weights to updated ones.
   - **Gradient Clamping**: Soft-clamps gradient norms for stability.
   - **Hyper-Connections**: Allows queries/keys/values to receive different input views for expressivity.
   - **Batch Processing**: Handles mini-batch updates at configurable batch sizes.

### How It Works

NeuralMemory operates in two main phases: **storing memories** (updating the memory model) and **retrieving memories** (querying the memory). It's designed for sequential processing, often in transformers, where it augments attention with dynamic memory.

#### 1. Initialization
- The memory model (MLP) is initialized with random weights.
- States (weights, momentum, etc.) are set up per batch/head.

#### 2. Storing Memories (`store_memories` method)
- **Input Processing**:
  - Take a sequence (e.g., embeddings from a transformer layer).
  - Normalize with RMSNorm.
  - Derive keys and values via linear projections (with optional activation).
  - Chunk the sequence and pool to representations.
- **Compute "Surprises" (Updates)**:
  - Treat keys as inputs to the memory model (MLP) and compute the loss (MSE) against values.
  - Use `torch.func.grad` and `vmap` to compute per-sample gradients of the MLP's weights w.r.t. this loss.
  - These gradients are the "surprises" — updates to the memory.
  - Apply adaptive learning rates as loss weights during gradient computation.
- **Refine Updates**:
  - Apply momentum via associative scans (cumulative integration of surprises).
  - Apply weight decay (forgetting) via associative scans.
  - Optional: Spectral normalize updates, per-layer modulation, gradient clamping.
- **Update Weights**:
  - Use associative scans to accumulate updates over chunks/sequences.
  - Update the MLP's weights incrementally, respecting batch boundaries.
- **Output**: Updated weights, next state, and optional "surprises" (losses/learning rates) for analysis.

This phase dynamically learns associations: the MLP's weights adapt to map keys to values, with stability from momentum/decay.

#### 3. Retrieving Memories (`retrieve_memories` method)
- **Input Processing**:
  - Take queries (from the sequence).
  - Normalize and project to query embeddings.
  - Split into heads if multi-head.
- **Query the Memory**:
  - Pass queries through the updated memory model (MLP) using `torch.func.functional_call`.
  - The MLP outputs retrieved values based on learned associations.
- **Post-Processing**:
  - Apply multi-head RMSNorm, gating, and head merging.
  - Restore original sequence dimensions (handle padding for chunking).
- **Output**: Retrieved values (memory-augmented outputs).

Retrieval is a simple forward pass: queries fetch stored values via the MLP.

#### 4. Full Forward Pass (`forward` method)
- Handles both storing and retrieving in sequence.
- Splits input into store/retrieve sequences (if using different views).
- Accumulates updates across chunks/batches.
- Updates weights at batch boundaries.
- Returns retrieved outputs and next state.
- Supports single-token decoding, caching, and gradient detachment.

### Key Insights and Usage
- **Efficiency**: Chunking and associative scans enable O(sequence length) updates without full recomputation.
- **Expressivity**: The MLP can learn complex mappings; multi-head and hyper-connections add flexibility.
- **Stability**: Momentum, decay, and clamping prevent divergence.
- **Integration**: Used in transformers (e.g., in `mac_transformer.py` or `implicit_mlp_attention.py`) for memory-augmented attention.
- **Limitations**: Requires careful tuning of chunk sizes, learning rates, and batch sizes; can be computationally intensive for large memories.

This architecture enables transformers to "remember" across long contexts dynamically, improving performance on tasks like language modeling or reasoning.

### Detailed Layer-by-Layer Execution

To understand the implementation in `titans_pytorch/neural_memory.py`, here is a breakdown of the specific layers and the flow of data during the forward and backward passes.

#### 1. Layer Definitions (`__init__`)

*   **The Memory Core (`self.memory_model`)**:
    *   **Code**: `self.memory_model = MemoryMLP(...)`
    *   **Role**: This is the storage container. Unlike a standard KV-cache which stores raw tensors, this stores information in the **weights** of this MLP.
    *   **Structure**: Typically a 2-layer MLP with GELU activation.

*   **The Projections (The "Learner")**:
    *   `self.to_keys` & `self.to_values`: Projects the input sequence into training data ($K, V$) for the Memory MLP.
    *   `self.to_queries`: Projects the input sequence into queries ($Q$) to ask the Memory MLP.
    *   `self.to_adaptive_step`: Predicts a learning rate ($\eta$) for every token. Determines "how much to learn from this token".
    *   `self.to_decay_factor`: Predicts a decay rate ($\alpha$). Determines "how much to forget previous memories".
    *   `self.to_momentum`: Predicts momentum to smooth out updates.

*   **The Optimizer (`self.per_sample_grad_fn`)**:
    *   **Code**: `vmap(grad(forward_and_loss))`
    *   **Role**: A differentiable optimizer. It calculates the gradient of the Memory MLP's loss with respect to its weights. This gradient is the "surprise" signal used to update the memory.

#### 2. The Forward Pass (Execution)

When data enters `NeuralMemory.forward`, two distinct processes occur: **Storing** (Training the inner model) and **Retrieving** (Inference on the inner model).

**Step A: Storing Memories (`store_memories`)**
This phase updates the weights of the Memory MLP based on the input sequence.
1.  **Generate Training Data**: Input `seq` is projected to Keys ($K$) and Values ($V$). Adaptive learning rates and decay factors are computed.
2.  **Calculate "Surprise" (Gradient Descent)**:
    *   The model treats $K$ as input and $V$ as the target.
    *   It runs the *current* Memory MLP on $K$.
    *   It calculates the MSE Loss against $V$.
    *   It computes the **Gradient** of this loss w.r.t. the MLP weights.
    *   *Concept*: High gradient = high "surprise" (memory didn't know this data).
3.  **Update Weights (Associative Scan)**:
    *   Instead of a loop, it uses an **Associative Scan** to apply thousands of sequential updates (Momentum + Decay + Gradient) in parallel.
    *   **Result**: A sequence of updated weights ($W_{t}$) for every timestep.

**Step B: Retrieving Memories (`retrieve_memories`)**
This phase uses the updated weights to generate output features.
1.  **Generate Queries**: Input `seq` is projected to Queries ($Q$).
2.  **Functional Call**:
    *   The code performs a "stateless" forward pass of the Memory MLP.
    *   It uses $Q$ as input and the **New Weights** ($W_{t}$) calculated in Step A.
    *   `Output = MemoryMLP(Q; Weights=W_t)`

#### 3. The Backward Pass (Meta-Learning)

This is the "Outer Loop" training. When you run `loss.backward()` on the main model (e.g., the Transformer):

1.  **Differentiation through Optimization**:
    *   Because the "Surprise" (Step A) was calculated using differentiable PyTorch functions (`torch.func.grad`), the gradient flows *through* the weight update steps.
2.  **What is learned?**:
    *   The model learns **how to generate Keys and Values** that are easy for the Memory MLP to store.
    *   It learns **how to predict Learning Rates** to focus on important info and ignore noise.
    *   It learns **how to Decay** old info to manage the limited capacity of the Memory MLP.


    # Note
    graph of neural memory
    b,h,w of memory
    cell sample->size of mem