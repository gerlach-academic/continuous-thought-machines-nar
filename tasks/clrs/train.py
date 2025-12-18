"""
Training script for CLRS algorithmic reasoning with CTM.

This script trains the Continuous Thought Machine on CLRS/SALSA-CLRS
graph algorithm tasks like BFS, DFS, Dijkstra, etc.

Supports three training modes:
- none: Only supervise final output (default)
- hard: Teacher forcing at fixed intervals (1 CTM iter per algorithm step)
- soft: Certainty-based hint emission (CTM decides when to output)

Usage:
    python -m tasks.clrs.train --algorithm bfs --num_nodes 16 32
    python -m tasks.clrs.train --algorithm dijkstra --teacher_forcing hard --use_hints
    python -m tasks.clrs.train --algorithm bfs --teacher_forcing soft --use_hints
"""
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

# Add parent dirs to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tasks.clrs.dataset import CLRSDatasetAdapter, collate_clrs_batch
from tasks.clrs.model import ContinuousThoughtMachineGraph
from tasks.clrs.config import CLRSConfig, ALGORITHM_CONFIGS, TeacherForcingMode
from tasks.clrs.hints import (
    HintScheduleConfig, 
    AdaptiveHintScheduler,
    TeacherForcingMode as TFMode,
)
from utils.housekeeping import set_seed

try:
    from autoclip.torch import QuantileClip
    AUTOCLIP_AVAILABLE = True
except ImportError:
    AUTOCLIP_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CTM on CLRS algorithmic reasoning tasks"
    )
    
    # Task configuration
    parser.add_argument('--algorithm', type=str, default='bfs',
                        choices=['bfs', 'dfs', 'dijkstra', 'mst_prim', 'fast_mis', 'eccentricity'],
                        help='CLRS algorithm to train on')
    parser.add_argument('--graph_generator', type=str, default='er',
                        choices=['er', 'ws', 'delaunay', 'grid', 'path', 'tree'],
                        help='Graph generator type')
    parser.add_argument('--num_nodes', type=int, nargs='+', default=[4, 7, 11, 13, 16],
                        help='Number of nodes for training graphs')
    parser.add_argument('--max_nodes', type=int, default=64,
                        help='Maximum nodes (for padding)')
    parser.add_argument('--use_hints', action='store_true',
                        help='Use algorithm hints for training')
    
    # Dataset configuration
    parser.add_argument('--num_train_samples', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--num_val_samples', type=int, default=1000,
                        help='Number of validation samples')
    parser.add_argument('--data_dir', type=str, default='./data/clrs',
                        help='Directory for data storage')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model hidden dimension')
    parser.add_argument('--d_input', type=int, default=128,
                        help='Attention dimension')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_synch_out', type=int, default=64,
                        help='Neurons for output synchronization')
    parser.add_argument('--n_synch_action', type=int, default=64,
                        help='Neurons for action synchronization')
    parser.add_argument('--synapse_depth', type=int, default=2,
                        help='Synapse U-Net depth')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Internal reasoning iterations')
    parser.add_argument('--memory_length', type=int, default=25,
                        help='NLM history length')
    parser.add_argument('--deep_nlms', action=argparse.BooleanOptionalAction, 
                        default=True, help='Use deep NLMs')
    parser.add_argument('--memory_hidden_dims', type=int, default=16,
                        help='NLM hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--neuron_select_type', type=str, default='random-pairing',
                        choices=['first-last', 'random', 'random-pairing'],
                        help='Neuron selection strategy')
    parser.add_argument('--use_adjacency_mask', action='store_true',
                        help='Use graph structure in attention')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--batch_size_test', type=int, default=64,
                        help='Test batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--training_iterations', type=int, default=50000,
                        help='Number of training iterations')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Learning rate warmup steps')
    parser.add_argument('--gradient_clipping', type=float, default=-1,
                        help='Gradient clipping quantile (-1 to disable)')
    parser.add_argument('--use_most_certain', action=argparse.BooleanOptionalAction,
                        default=True, help='Use most certain prediction for loss')
    
    # Teacher forcing settings
    parser.add_argument('--teacher_forcing', type=str, default='none',
                        choices=['none', 'hard', 'soft'],
                        help='Teacher forcing mode for hints')
    parser.add_argument('--iterations_per_hint', type=int, default=5,
                        help='CTM iterations per algorithm step (hard mode)')
    parser.add_argument('--certainty_threshold', type=float, default=0.7,
                        help='Certainty threshold for hint emission (soft mode)')
    parser.add_argument('--min_iterations_between', type=int, default=2,
                        help='Minimum iterations between hints (soft mode)')
    parser.add_argument('--max_iterations_per_hint', type=int, default=10,
                        help='Maximum iterations before forcing hint (soft mode, only used if force_at_final_only=False)')
    parser.add_argument('--force_final_hints', action=argparse.BooleanOptionalAction,
                        default=True, help='Force remaining hints at final iteration (soft mode)')
    parser.add_argument('--hint_loss_weight', type=float, default=1.0,
                        help='Weight for hint loss')
    parser.add_argument('--output_loss_weight', type=float, default=1.0,
                        help='Weight for output loss')
    parser.add_argument('--progressive_hints', action=argparse.BooleanOptionalAction,
                        default=True, help='Decay hint weight over training')
    
    # Logging and checkpointing
    parser.add_argument('--log_dir', type=str, default='logs/clrs',
                        help='Logging directory')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Checkpoint save frequency')
    parser.add_argument('--eval_every', type=int, default=500,
                        help='Evaluation frequency')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Logging frequency')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--reload', action='store_true',
                        help='Reload from checkpoint')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda, cpu, mps)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers')
    
    return parser.parse_args()


def compute_node_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    attention_mask: torch.Tensor,
    certainties: Optional[torch.Tensor] = None,
    use_most_certain: bool = True,
    task_type: str = "node_classification"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute loss for node-level predictions.
    
    Args:
        predictions: [B, N * out_dims, T]
        targets: [B, N, out_dims]
        attention_mask: [B, N] valid nodes
        certainties: [B, 2, T]
        use_most_certain: Use most certain iteration
        task_type: Type of task (classification/regression)
        
    Returns:
        loss: Scalar loss
        metrics: Dict of metrics
    """
    B, total_out, T = predictions.shape
    N = attention_mask.shape[1]
    out_dims = total_out // N
    
    # Reshape predictions
    predictions = predictions.view(B, N, out_dims, T)
    
    if use_most_certain and certainties is not None:
        # Select most certain iteration
        cert = certainties[:, 1, :]  # [B, T]
        best_t = cert.argmax(dim=-1)  # [B]
        
        # Gather predictions at best iteration
        best_t_expanded = best_t.view(B, 1, 1, 1).expand(-1, N, out_dims, -1)
        preds = predictions.gather(3, best_t_expanded).squeeze(-1)  # [B, N, out_dims]
    else:
        # Use last iteration
        preds = predictions[..., -1]
    
    # Compute loss based on task type
    if task_type == "node_classification":
        # Cross-entropy loss
        preds_flat = preds[attention_mask].view(-1, out_dims)
        targets_flat = targets[attention_mask].view(-1, out_dims)
        
        # Convert to class labels if one-hot
        if out_dims > 1:
            target_labels = targets_flat.argmax(dim=-1)
            loss = F.cross_entropy(preds_flat, target_labels)
            
            # Compute accuracy
            pred_labels = preds_flat.argmax(dim=-1)
            accuracy = (pred_labels == target_labels).float().mean().item()
        else:
            loss = F.binary_cross_entropy_with_logits(preds_flat.squeeze(), targets_flat.squeeze())
            accuracy = ((preds_flat.squeeze() > 0) == (targets_flat.squeeze() > 0.5)).float().mean().item()
    
    elif task_type == "node_regression":
        preds_flat = preds[attention_mask]
        targets_flat = targets[attention_mask]
        loss = F.mse_loss(preds_flat, targets_flat)
        accuracy = 0.0  # Not applicable
    
    else:
        # Default: MSE
        loss = F.mse_loss(preds, targets)
        accuracy = 0.0
    
    metrics = {
        'loss': loss.item(),
        'accuracy': accuracy,
    }
    
    return loss, metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task_type: str = "node_classification",
    use_most_certain: bool = True,
    max_batches: int = -1
) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    
    all_metrics = {'loss': [], 'accuracy': []}
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches > 0 and i >= max_batches:
                break
            
            inputs, targets, metadata = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            attention_mask = metadata['attention_mask'].to(device)
            adjacency_mask = metadata['adjacency_mask'].to(device)
            
            predictions, certainties, _ = model(
                inputs, 
                attention_mask=attention_mask,
                adjacency_mask=adjacency_mask
            )
            
            _, metrics = compute_node_loss(
                predictions, targets, attention_mask,
                certainties, use_most_certain, task_type
            )
            
            for k, v in metrics.items():
                all_metrics[k].append(v)
    
    # Average metrics
    return {k: np.mean(v) for k, v in all_metrics.items()}


def train(args):
    """Main training function."""
    # Setup
    set_seed(args.seed)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, args.algorithm, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    
    # Device setup
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Get algorithm config
    algo_config = ALGORITHM_CONFIGS[args.algorithm]
    task_type = algo_config['task_type']
    print(f"Training on {args.algorithm}: {algo_config['description']}")
    
    # Create datasets
    print("Creating datasets...")
    train_kwargs = {
        "n": args.num_nodes,
        "p_range": (0.1, 0.3)
    }
    if args.graph_generator == "ws":
        train_kwargs["k"] = [4, 6, 8]
    
    train_dataset = CLRSDatasetAdapter(
        algorithm=args.algorithm,
        split="train",
        num_samples=args.num_train_samples,
        graph_generator=args.graph_generator,
        graph_generator_kwargs=train_kwargs,
        data_dir=args.data_dir,
        use_hints=args.use_hints,
        max_nodes=args.max_nodes,
    )
    
    val_dataset = CLRSDatasetAdapter(
        algorithm=args.algorithm,
        split="val",
        num_samples=args.num_val_samples,
        graph_generator=args.graph_generator,
        graph_generator_kwargs={"n": [max(args.num_nodes)], "p_range": (0.1, 0.3)},
        data_dir=args.data_dir,
        use_hints=args.use_hints,
        max_nodes=args.max_nodes,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_clrs_batch,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        collate_fn=collate_clrs_batch,
        num_workers=args.num_workers
    )
    
    # Create model
    print("Creating model...")
    model = ContinuousThoughtMachineGraph(
        input_dim=train_dataset.input_dim,
        d_model=args.d_model,
        d_input=args.d_input,
        heads=args.heads,
        n_synch_out=args.n_synch_out,
        n_synch_action=args.n_synch_action,
        synapse_depth=args.synapse_depth,
        iterations=args.iterations,
        memory_length=args.memory_length,
        deep_nlms=args.deep_nlms,
        memory_hidden_dims=args.memory_hidden_dims,
        out_dims_per_node=train_dataset.output_dim,
        max_nodes=args.max_nodes,
        dropout=args.dropout,
        neuron_select_type=args.neuron_select_type,
        use_adjacency_mask=args.use_adjacency_mask,
    ).to(device)
    
    # Initialize lazy modules
    sample_inputs, _, sample_meta = train_dataset[0]
    sample_inputs = sample_inputs.unsqueeze(0).to(device)
    sample_mask = sample_meta['attention_mask'].unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(sample_inputs, attention_mask=sample_mask)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create hint scheduler if using teacher forcing
    hint_scheduler = None
    use_hints = args.use_hints and args.teacher_forcing != 'none'
    if use_hints:
        tf_mode = TFMode(args.teacher_forcing)
        hint_config = HintScheduleConfig(
            mode=tf_mode,
            iterations_per_hint=args.iterations_per_hint,
            certainty_threshold=args.certainty_threshold,
            min_iterations_between=args.min_iterations_between,
            max_iterations_per_hint=args.max_iterations_per_hint,
            force_final_hints=args.force_final_hints,
            hint_loss_weight=args.hint_loss_weight,
            output_loss_weight=args.output_loss_weight,
            progressive_hints=args.progressive_hints,
        )
        hint_scheduler = AdaptiveHintScheduler(hint_config).to(device)
        print(f"Using {args.teacher_forcing} teacher forcing with hints")
        if args.teacher_forcing == 'soft':
            print(f"  force_final_hints={args.force_final_hints}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    if args.gradient_clipping > 0 and AUTOCLIP_AVAILABLE:
        optimizer = QuantileClip.as_optimizer(
            optimizer=optimizer,
            quantile=args.gradient_clipping,
            history_length=1000
        )
    
    # Warmup + cosine scheduler
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.training_iterations - args.warmup_steps)
        return max(0.01, 0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print("Starting training...")
    model.train()
    
    train_iter = iter(train_loader)
    best_val_acc = 0.0
    running_loss = 0.0
    running_acc = 0.0
    running_hint_loss = 0.0

    
    pbar = tqdm(range(args.training_iterations), desc="Training")
    
    for step in pbar:
        # Get batch (with cycling)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        inputs, targets, metadata = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        attention_mask = metadata['attention_mask'].to(device)
        adjacency_mask = metadata['adjacency_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions, certainties, _ = model(
            inputs,
            attention_mask=attention_mask,
            adjacency_mask=adjacency_mask
        )
        
        # Compute output loss
        output_loss, metrics = compute_node_loss(
            predictions, targets, attention_mask,
            certainties, args.use_most_certain, task_type
        )
        
        # Compute hint loss if using teacher forcing
        hint_loss = torch.tensor(0.0, device=device)
        hint_metrics = {}
        if hint_scheduler is not None and 'hints' in metadata:
            # Update training progress for progressive hints
            progress = step / args.training_iterations
            hint_scheduler.update_progress(progress)
            
            # Move hints to device
            hints = {k: v.to(device) for k, v in metadata['hints'].items()}
            
            # Compute hint loss
            hint_loss, hint_metrics = hint_scheduler(
                predictions, certainties, hints, attention_mask
            )
        
        # Combined loss
        total_loss = args.output_loss_weight * output_loss + hint_loss
        
        # Backward pass
        total_loss.backward()
        
        if args.gradient_clipping > 0 and not AUTOCLIP_AVAILABLE:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Update running stats
        running_loss = 0.95 * running_loss + 0.05 * metrics['loss']
        running_acc = 0.95 * running_acc + 0.05 * metrics['accuracy']
        if hint_loss.item() > 0:
            running_hint_loss = 0.95 * running_hint_loss + 0.05 * hint_loss.item()
        
        # Logging
        if step % args.log_every == 0:
            log_dict = {
                'loss': f"{running_loss:.4f}",
                'acc': f"{running_acc:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            }
            if use_hints:
                log_dict['h_loss'] = f"{running_hint_loss:.4f}"
                if 'natural_emissions' in hint_metrics:
                    log_dict['nat'] = f"{hint_metrics['natural_emissions']:.1f}"
                    log_dict['frc'] = f"{hint_metrics['forced_emissions']:.1f}"
            pbar.set_postfix(log_dict)
        
        # Evaluation
        if step % args.eval_every == 0 and step > 0:
            val_metrics = evaluate(
                model, val_loader, device, task_type,
                args.use_most_certain, max_batches=10
            )
            
            print(f"\n[Step {step}] Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'args': vars(args)
                }, os.path.join(log_dir, 'best_model.pt'))
            
            model.train()
        
        # Checkpointing
        if step % args.save_every == 0 and step > 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'running_loss': running_loss,
                'running_acc': running_acc,
                'best_val_acc': best_val_acc,
                'args': vars(args)
            }, os.path.join(log_dir, f'checkpoint_{step}.pt'))
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_metrics = evaluate(
        model, val_loader, device, task_type,
        args.use_most_certain, max_batches=-1
    )
    print(f"Final Val Loss: {final_metrics['loss']:.4f}, "
          f"Final Val Acc: {final_metrics['accuracy']:.4f}")
    
    # Save final model
    torch.save({
        'step': args.training_iterations,
        'model_state_dict': model.state_dict(),
        'final_metrics': final_metrics,
        'best_val_acc': best_val_acc,
        'args': vars(args)
    }, os.path.join(log_dir, 'final_model.pt'))
    
    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")
    print(f"Logs saved to: {log_dir}")
    
    return model, best_val_acc


if __name__ == '__main__':
    args = parse_args()
    train(args)
