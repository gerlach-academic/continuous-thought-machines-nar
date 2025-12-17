# CLRS Algorithmic Reasoning with CTM

This module enables training the Continuous Thought Machine (CTM) on CLRS algorithmic reasoning benchmarks. It provides adapters for both [SALSA-CLRS](https://github.com/jkminder/salsa-clrs) and the original [dm-clrs](https://github.com/deepmind/clrs) benchmarks.

## Installation

1. Install the base CTM requirements:
```bash
pip install -r requirements.txt
```

2. Install SALSA-CLRS from GitHub:
```bash
pip install git+https://github.com/jkminder/salsa-clrs.git
```

Note: SALSA-CLRS depends on `dm-clrs` and `jax`, which will be installed automatically.

## Available Algorithms

| Algorithm | Task Type | Description |
|-----------|-----------|-------------|
| `bfs` | Node Classification | Breadth-First Search reachability |
| `dfs` | Node Classification | Depth-First Search reachability |
| `dijkstra` | Node Regression | Shortest path distances |
| `mst_prim` | Edge Classification | Minimum Spanning Tree |
| `fast_mis` | Node Classification | Maximum Independent Set |
| `eccentricity` | Node Regression | Graph Eccentricity |

## Quick Start

### Training

Train CTM on BFS:
```bash
python -m tasks.clrs.train --algorithm bfs --num_nodes 16 32
```

Train on Dijkstra with custom settings:
```bash
python -m tasks.clrs.train \
    --algorithm dijkstra \
    --graph_generator er \
    --num_nodes 4 8 16 \
    --d_model 512 \
    --iterations 50 \
    --lr 1e-4 \
    --training_iterations 50000
```

### Using the Dataset Adapter

```python
from tasks.clrs.dataset import CLRSDatasetAdapter, collate_clrs_batch
from torch.utils.data import DataLoader

# Create dataset
dataset = CLRSDatasetAdapter(
    algorithm="bfs",
    split="train",
    num_samples=10000,
    graph_generator="er",
    graph_generator_kwargs={"n": [16, 32], "p_range": (0.1, 0.3)},
    max_nodes=64
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_clrs_batch
)

# Iterate
for inputs, targets, metadata in dataloader:
    # inputs: [B, max_nodes, input_dim]
    # targets: [B, max_nodes, output_dim]
    # metadata: dict with attention_mask, adjacency_mask, etc.
    pass
```

### Using the Model

```python
from tasks.clrs.model import ContinuousThoughtMachineGraph

model = ContinuousThoughtMachineGraph(
    input_dim=64,
    d_model=512,
    d_input=128,
    heads=8,
    iterations=50,
    max_nodes=64,
    out_dims_per_node=2  # Binary classification
)

# Forward pass
predictions, certainties, synch_out = model(
    inputs,  # [B, N, D_in]
    attention_mask=mask,  # [B, N]
    adjacency_mask=adj_mask  # [B, N, N] optional
)
# predictions: [B, N * out_dims, T]
# certainties: [B, 2, T]
```

## How It Works

### Graph → CTM Mapping

The CTM is designed for sequential/image inputs, so we adapt graph data:

1. **Node Features**: Each node becomes a "token" with features encoding:
   - Positional information (node ID via sinusoidal encoding)
   - Structural information (degree, connectivity)
   - Algorithm-specific inputs (source node, edge weights)

2. **Graph Structure**: Can be injected via:
   - Attention masking (nodes only attend to neighbors)
   - Adjacency-aware positional encodings

3. **Output**: Per-node predictions are made at each reasoning step, with the most "certain" step used for final prediction.

### CTM Reasoning on Graphs

The CTM's internal recurrence allows it to:
1. Iteratively propagate information through the graph via attention
2. Build up node representations through synchronization
3. Make predictions when confident (self-determined stopping)

This is particularly suited for algorithms like BFS/DFS that require multi-step reasoning.

## Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 512 | Core hidden dimension |
| `d_input` | 128 | Attention projection dimension |
| `heads` | 8 | Attention heads |
| `iterations` | 50 | Internal reasoning steps |
| `memory_length` | 25 | NLM history length |
| `n_synch_out` | 64 | Output synchronization neurons |
| `synapse_depth` | 2 | Synapse U-Net depth |

### Training Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `lr` | 1e-4 | Learning rate |
| `training_iterations` | 50000 | Number of training steps |
| `warmup_steps` | 1000 | LR warmup steps |

## Extending to New Algorithms

To add support for additional CLRS algorithms:

1. Add the algorithm config to `config.py`:
```python
ALGORITHM_CONFIGS["new_algo"] = {
    "task_type": "node_classification",
    "output_type": "mask",
    "description": "Description of the algorithm"
}
```

2. Update `_build_target` in `dataset.py` if needed for custom output parsing.

## References

- [SALSA-CLRS Paper](https://arxiv.org/abs/2309.12253)
- [Original CLRS Benchmark](https://arxiv.org/abs/2205.15659)
- [CTM Paper](https://arxiv.org/abs/2505.05522)
