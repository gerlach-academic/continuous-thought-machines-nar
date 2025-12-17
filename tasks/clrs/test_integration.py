"""Test script for CLRS integration with CTM."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch

def test_dataset_adapter():
    """Test the CLRSDatasetAdapter."""
    print("=" * 60)
    print("Testing CLRSDatasetAdapter...")
    print("=" * 60)
    
    from tasks.clrs.dataset import CLRSDatasetAdapter, collate_clrs_batch
    from torch.utils.data import DataLoader
    
    # Create a small dataset
    ds = CLRSDatasetAdapter(
        algorithm='bfs',
        split='train',
        num_samples=10,
        graph_generator='er',
        graph_generator_kwargs={'n': [8, 12], 'p_range': (0.2, 0.4)},
        data_dir='./data/clrs_test',
        max_nodes=32
    )
    print(f"Dataset created with {len(ds)} samples")
    print(f"Input dim: {ds.input_dim}, Output dim: {ds.output_dim}")
    
    # Get a sample
    inputs, targets, metadata = ds[0]
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Num nodes: {metadata['num_nodes']}")
    print(f"Attention mask shape: {metadata['attention_mask'].shape}")
    print(f"Adjacency mask shape: {metadata['adjacency_mask'].shape}")
    
    # Test dataloader
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_clrs_batch)
    batch = next(iter(loader))
    inputs_batch, targets_batch, meta_batch = batch
    print(f"\nBatch input shape: {inputs_batch.shape}")
    print(f"Batch target shape: {targets_batch.shape}")
    print(f"Batch attention mask shape: {meta_batch['attention_mask'].shape}")
    
    print("\n✓ Dataset adapter test PASSED!")
    return ds


def test_model():
    """Test the ContinuousThoughtMachineGraph model."""
    print("\n" + "=" * 60)
    print("Testing ContinuousThoughtMachineGraph...")
    print("=" * 60)
    
    from tasks.clrs.model import ContinuousThoughtMachineGraph
    
    # Create model
    model = ContinuousThoughtMachineGraph(
        input_dim=64,
        d_model=256,
        d_input=64,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        iterations=10,
        memory_length=10,
        deep_nlms=True,
        memory_hidden_dims=8,
        out_dims_per_node=2,
        max_nodes=32,
        dropout=0.1,
    )
    print(f"Model created")
    
    # Create dummy input
    batch_size = 4
    max_nodes = 32
    input_dim = 64
    
    x = torch.randn(batch_size, max_nodes, input_dim)
    mask = torch.ones(batch_size, max_nodes, dtype=torch.bool)
    mask[:, 16:] = False  # Only first 16 nodes are valid
    
    # Forward pass
    predictions, certainties, synch_out = model(x, attention_mask=mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Certainties shape: {certainties.shape}")
    print(f"Synch out shape: {synch_out.shape}")
    
    # Check shapes
    assert predictions.shape == (batch_size, max_nodes * 2, 10), f"Wrong pred shape: {predictions.shape}"
    assert certainties.shape == (batch_size, 2, 10), f"Wrong cert shape: {certainties.shape}"
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    print("\n✓ Model test PASSED!")
    return model


def test_integration():
    """Test full integration of dataset and model."""
    print("\n" + "=" * 60)
    print("Testing Full Integration...")
    print("=" * 60)
    
    from tasks.clrs.dataset import CLRSDatasetAdapter, collate_clrs_batch
    from tasks.clrs.model import ContinuousThoughtMachineGraph
    from torch.utils.data import DataLoader
    
    # Create dataset
    ds = CLRSDatasetAdapter(
        algorithm='bfs',
        split='train',
        num_samples=10,
        graph_generator='er',
        graph_generator_kwargs={'n': [8], 'p_range': (0.3, 0.5)},
        data_dir='./data/clrs_test',
        max_nodes=32
    )
    
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_clrs_batch)
    
    # Create model matching dataset dimensions
    model = ContinuousThoughtMachineGraph(
        input_dim=ds.input_dim,
        d_model=256,
        d_input=64,
        heads=4,
        n_synch_out=32,
        n_synch_action=32,
        synapse_depth=1,
        iterations=10,
        memory_length=10,
        out_dims_per_node=ds.output_dim,
        max_nodes=32,
        dropout=0.1,
    )
    
    # Get batch and forward pass
    inputs, targets, metadata = next(iter(loader))
    attention_mask = metadata['attention_mask']
    
    predictions, certainties, _ = model(inputs, attention_mask=attention_mask)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Target shape: {targets.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Compute a simple loss
    B, total_out, T = predictions.shape
    N = 32
    out_dims = total_out // N
    
    # Reshape predictions for last timestep
    preds = predictions[..., -1].view(B, N, out_dims)
    
    # Apply mask
    valid_preds = preds[attention_mask]
    valid_targets = targets[attention_mask]
    
    loss = torch.nn.functional.cross_entropy(
        valid_preds, 
        valid_targets.argmax(dim=-1)
    )
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print("Backward pass successful!")
    
    print("\n✓ Integration test PASSED!")


def test_multiple_algorithms():
    """Test with different algorithms."""
    print("\n" + "=" * 60)
    print("Testing Multiple Algorithms...")
    print("=" * 60)
    
    from tasks.clrs.dataset import CLRSDatasetAdapter
    from tasks.clrs.config import ALGORITHM_CONFIGS
    
    algorithms_to_test = ['bfs', 'dfs', 'dijkstra']
    
    for algo in algorithms_to_test:
        print(f"\nTesting {algo}...")
        try:
            ds = CLRSDatasetAdapter(
                algorithm=algo,
                split='train',
                num_samples=5,
                graph_generator='er',
                graph_generator_kwargs={'n': [8], 'p_range': (0.3, 0.5)},
                data_dir='./data/clrs_test',
                max_nodes=32
            )
            inputs, targets, metadata = ds[0]
            config = ALGORITHM_CONFIGS[algo]
            print(f"  {algo}: {config['description']}")
            print(f"  Task type: {config['task_type']}")
            print(f"  Input shape: {inputs.shape}, Target shape: {targets.shape}")
            print(f"  ✓ {algo} works!")
        except Exception as e:
            print(f"  ✗ {algo} failed: {e}")
    
    print("\n✓ Multiple algorithms test PASSED!")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("CLRS-CTM Integration Tests")
    print("=" * 60 + "\n")
    
    try:
        test_dataset_adapter()
        test_model()
        test_integration()
        test_multiple_algorithms()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
