"""
Test script for teacher forcing modes in CLRS.

Tests:
1. Dataset hint extraction
2. Hard teacher forcing schedule computation
3. Soft teacher forcing with certainty-based emission
4. Full training step with hint loss
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import numpy as np


def test_hint_extraction():
    """Test that hints are correctly extracted from CLRS data."""
    print("\n=== Testing Hint Extraction ===")
    
    from tasks.clrs.dataset import CLRSDatasetAdapter
    
    # Create dataset with hints
    dataset = CLRSDatasetAdapter(
        algorithm="bfs",
        split="train",
        num_samples=10,
        graph_generator="er",
        graph_generator_kwargs={"n": [8], "p_range": (0.2, 0.4)},
        data_dir="./data/clrs_hint_test",
        use_hints=True,
        max_nodes=16,
    )
    
    # Get a sample
    inputs, targets, metadata = dataset[0]
    
    print(f"Inputs shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Attention mask: {metadata['attention_mask'].sum()} valid nodes")
    
    if 'hints' in metadata:
        print(f"\nHints found: {list(metadata['hints'].keys())}")
        for name, hint in metadata['hints'].items():
            print(f"  {name}: shape {hint.shape}")
        print(f"Number of hint steps: {metadata['num_hint_steps']}")
    else:
        print("WARNING: No hints extracted!")
        return False
    
    print("✓ Hint extraction test passed")
    return True


def test_hard_teacher_forcing():
    """Test hard teacher forcing schedule computation."""
    print("\n=== Testing Hard Teacher Forcing ===")
    
    from tasks.clrs.hints import HardTeacherForcing, HintScheduleConfig, TeacherForcingMode
    
    config = HintScheduleConfig(
        mode=TeacherForcingMode.HARD,
        iterations_per_hint=5,
    )
    
    hard_tf = HardTeacherForcing(config)
    
    # Test schedule: 50 iterations, 8 hint steps
    schedule = hard_tf.compute_schedule(50, 8)
    
    print(f"Schedule for 50 iterations, 8 hint steps:")
    supervised_iters = (schedule >= 0).nonzero(as_tuple=True)[0].tolist()
    print(f"Supervised iterations: {supervised_iters}")
    print(f"Hint step at each: {schedule[supervised_iters].tolist()}")
    
    # Check expected behavior
    expected = [4, 9, 14, 19, 24, 29, 34, 39]  # (i+1)*5 - 1 for i in 0..7
    assert supervised_iters == expected, f"Expected {expected}, got {supervised_iters}"
    
    print("✓ Hard teacher forcing test passed")
    return True


def test_soft_teacher_forcing():
    """Test soft teacher forcing with certainty-based emission."""
    print("\n=== Testing Soft Teacher Forcing ===")
    
    from tasks.clrs.hints import SoftTeacherForcing, HintScheduleConfig, TeacherForcingMode
    
    config = HintScheduleConfig(
        mode=TeacherForcingMode.SOFT,
        certainty_threshold=0.7,
        min_iterations_between=2,
        max_iterations_per_hint=10,
    )
    
    soft_tf = SoftTeacherForcing(config)
    
    # Create fake certainties: starts low, spikes at certain points
    B, T = 2, 30
    certainties = torch.zeros(B, 2, T)
    certainties[:, 1, :] = 0.3  # Low certainty default
    
    # Sample 0: high certainty at steps 5, 12, 20
    certainties[0, 1, 5] = 0.8
    certainties[0, 1, 12] = 0.9
    certainties[0, 1, 20] = 0.85
    
    # Sample 1: never high certainty (should force at max_iterations)
    certainties[1, 1, :] = 0.4
    
    num_hint_steps = 5
    emission_mask, hint_indices = soft_tf.compute_emission_points(certainties, num_hint_steps)
    
    print(f"Sample 0 emissions: {emission_mask[0].nonzero(as_tuple=True)[0].tolist()}")
    print(f"Sample 1 emissions: {emission_mask[1].nonzero(as_tuple=True)[0].tolist()}")
    
    # Sample 0 should emit at high certainty points
    sample0_emissions = emission_mask[0].nonzero(as_tuple=True)[0].tolist()
    assert 5 in sample0_emissions, "Should emit at step 5 (high certainty)"
    
    # Sample 1 should emit at max_iterations intervals
    sample1_emissions = emission_mask[1].nonzero(as_tuple=True)[0].tolist()
    assert 9 in sample1_emissions, "Should force emit at step 10 (max_iterations)"
    
    print("✓ Soft teacher forcing test passed")
    return True


def test_adaptive_scheduler():
    """Test the combined adaptive hint scheduler."""
    print("\n=== Testing Adaptive Hint Scheduler ===")
    
    from tasks.clrs.hints import AdaptiveHintScheduler, HintScheduleConfig, TeacherForcingMode
    
    # Test with soft mode
    config = HintScheduleConfig(
        mode=TeacherForcingMode.SOFT,
        certainty_threshold=0.6,
        min_iterations_between=2,
        max_iterations_per_hint=8,
        hint_loss_weight=1.0,
        progressive_hints=True,
    )
    
    scheduler = AdaptiveHintScheduler(config)
    
    # Test progressive hint weight decay
    scheduler.update_progress(0.0)
    weight_start = scheduler.get_effective_hint_weight()
    
    scheduler.update_progress(0.5)
    weight_mid = scheduler.get_effective_hint_weight()
    
    scheduler.update_progress(1.0)
    weight_end = scheduler.get_effective_hint_weight()
    
    print(f"Hint weights: start={weight_start:.3f}, mid={weight_mid:.3f}, end={weight_end:.3f}")
    
    assert weight_start > weight_mid > weight_end, "Weight should decrease over training"
    assert abs(weight_end - 0.1) < 0.01, "Final weight should be ~0.1"
    
    print("✓ Adaptive scheduler test passed")
    return True


def test_training_step_with_hints():
    """Test a full training step with hint loss."""
    print("\n=== Testing Training Step with Hints ===")
    
    from tasks.clrs.dataset import CLRSDatasetAdapter, collate_clrs_batch
    from tasks.clrs.model import ContinuousThoughtMachineGraph
    from tasks.clrs.hints import AdaptiveHintScheduler, HintScheduleConfig, TeacherForcingMode
    from torch.utils.data import DataLoader
    
    device = torch.device('cpu')
    
    # Create small dataset with hints
    dataset = CLRSDatasetAdapter(
        algorithm="bfs",
        split="train",
        num_samples=4,
        graph_generator="er",
        graph_generator_kwargs={"n": [6], "p_range": (0.3, 0.5)},
        data_dir="./data/clrs_hint_test",
        use_hints=True,
        max_nodes=8,
    )
    
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False,
        collate_fn=collate_clrs_batch
    )
    
    # Create model
    model = ContinuousThoughtMachineGraph(
        input_dim=dataset.input_dim,
        d_model=64,
        d_input=32,
        heads=2,
        n_synch_out=16,
        n_synch_action=16,
        synapse_depth=1,
        iterations=10,  # Short for testing
        memory_length=5,
        out_dims_per_node=2,
        max_nodes=8,
        dropout=0.0,
    ).to(device)
    
    # Initialize lazy modules
    sample = dataset[0]
    with torch.no_grad():
        _ = model(sample[0].unsqueeze(0), attention_mask=sample[2]['attention_mask'].unsqueeze(0))
    
    # Create hint scheduler
    config = HintScheduleConfig(
        mode=TeacherForcingMode.SOFT,
        certainty_threshold=0.5,
        min_iterations_between=1,
        max_iterations_per_hint=5,
        hint_loss_weight=1.0,
        progressive_hints=False,
    )
    scheduler = AdaptiveHintScheduler(config).to(device)
    
    # Training step
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    batch = next(iter(loader))
    inputs, targets, metadata = batch
    
    optimizer.zero_grad()
    predictions, certainties, _ = model(
        inputs, attention_mask=metadata['attention_mask']
    )
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Certainties shape: {certainties.shape}")
    
    # Compute hint loss
    if 'hints' in metadata:
        hints = metadata['hints']
        print(f"Hints in batch: {list(hints.keys())}")
        
        hint_loss, hint_metrics = scheduler(
            predictions, certainties, hints, metadata['attention_mask']
        )
        
        print(f"Hint loss: {hint_loss.item():.4f}")
        print(f"Hint metrics: {hint_metrics}")
        
        # Backward pass
        hint_loss.backward()
        optimizer.step()
        
        print("✓ Training step with hints passed")
        return True
    else:
        print("WARNING: No hints in batch!")
        return False


def test_all():
    """Run all tests."""
    print("=" * 60)
    print("CLRS Teacher Forcing Tests")
    print("=" * 60)
    
    results = {}
    
    results['hint_extraction'] = test_hint_extraction()
    results['hard_tf'] = test_hard_teacher_forcing()
    results['soft_tf'] = test_soft_teacher_forcing()
    results['adaptive_scheduler'] = test_adaptive_scheduler()
    results['training_step'] = test_training_step_with_hints()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n❌ Some tests failed!")
    
    return all_passed


if __name__ == "__main__":
    test_all()
