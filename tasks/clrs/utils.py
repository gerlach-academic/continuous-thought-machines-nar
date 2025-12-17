"""
Utility functions for CLRS tasks.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def compute_node_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """
    Compute node-level classification accuracy.
    
    Args:
        predictions: [B, N, C] logits or probabilities
        targets: [B, N, C] one-hot or [B, N] class labels
        mask: [B, N] valid nodes
        threshold: Threshold for binary classification
        
    Returns:
        Accuracy as float
    """
    if predictions.dim() == 3 and predictions.shape[-1] > 1:
        pred_labels = predictions.argmax(dim=-1)
        if targets.dim() == 3:
            true_labels = targets.argmax(dim=-1)
        else:
            true_labels = targets
    else:
        pred_labels = (predictions.squeeze(-1) > threshold).long()
        true_labels = (targets.squeeze(-1) > threshold).long()
    
    # Apply mask
    valid_preds = pred_labels[mask]
    valid_targets = true_labels[mask]
    
    if valid_preds.numel() == 0:
        return 0.0
    
    return (valid_preds == valid_targets).float().mean().item()


def compute_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    positive_class: int = 1
) -> Dict[str, float]:
    """
    Compute F1 score for binary node classification.
    
    Args:
        predictions: [B, N, 2] logits
        targets: [B, N, 2] one-hot targets
        mask: [B, N] valid nodes
        positive_class: Index of positive class
        
    Returns:
        Dict with precision, recall, f1
    """
    pred_labels = predictions.argmax(dim=-1)
    true_labels = targets.argmax(dim=-1) if targets.dim() == 3 else targets
    
    # Get valid predictions
    valid_preds = pred_labels[mask]
    valid_targets = true_labels[mask]
    
    # Compute TP, FP, FN
    tp = ((valid_preds == positive_class) & (valid_targets == positive_class)).sum().float()
    fp = ((valid_preds == positive_class) & (valid_targets != positive_class)).sum().float()
    fn = ((valid_preds != positive_class) & (valid_targets == positive_class)).sum().float()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor
) -> Dict[str, float]:
    """
    Compute regression metrics for node-level predictions.
    
    Args:
        predictions: [B, N, 1] predicted values
        targets: [B, N, 1] target values
        mask: [B, N] valid nodes
        
    Returns:
        Dict with mse, mae, rmse
    """
    valid_preds = predictions[mask].squeeze()
    valid_targets = targets[mask].squeeze()
    
    mse = F.mse_loss(valid_preds, valid_targets).item()
    mae = F.l1_loss(valid_preds, valid_targets).item()
    rmse = np.sqrt(mse)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }


def decode_predictions_over_time(
    predictions: torch.Tensor,
    certainties: torch.Tensor,
    strategy: str = "most_certain"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode predictions from temporal dimension.
    
    Args:
        predictions: [B, out_dims, T]
        certainties: [B, 2, T]
        strategy: 'most_certain', 'last', 'average', 'weighted_average'
        
    Returns:
        decoded_predictions: [B, out_dims]
        selected_timesteps: [B] (for most_certain)
    """
    if strategy == "most_certain":
        cert = certainties[:, 1, :]  # [B, T]
        best_t = cert.argmax(dim=-1)  # [B]
        B, D, T = predictions.shape
        
        # Gather predictions at best timestep
        idx = best_t.view(B, 1, 1).expand(-1, D, 1)
        decoded = predictions.gather(2, idx).squeeze(-1)
        
        return decoded, best_t
    
    elif strategy == "last":
        return predictions[..., -1], torch.full(
            (predictions.shape[0],), predictions.shape[-1] - 1,
            device=predictions.device
        )
    
    elif strategy == "average":
        return predictions.mean(dim=-1), None
    
    elif strategy == "weighted_average":
        cert = certainties[:, 1, :]  # [B, T]
        weights = F.softmax(cert, dim=-1)  # [B, T]
        weights = weights.unsqueeze(1)  # [B, 1, T]
        
        decoded = (predictions * weights).sum(dim=-1)
        return decoded, None
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_graph_attention_mask(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_hops: int = 2,
    include_self: bool = True
) -> torch.Tensor:
    """
    Create multi-hop attention mask from graph structure.
    
    Args:
        edge_index: [2, E] edge indices
        num_nodes: Number of nodes
        num_hops: Number of hops for neighborhood
        include_self: Include self-loops
        
    Returns:
        mask: [N, N] boolean mask (True = can attend)
    """
    # Build adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        adj[src, dst] = True
        adj[dst, src] = True  # Assume undirected
    
    if include_self:
        adj.fill_diagonal_(True)
    
    # Multi-hop expansion
    mask = adj.clone()
    current = adj.float()
    
    for _ in range(num_hops - 1):
        current = current @ adj.float()
        mask = mask | (current > 0)
    
    return mask


def visualize_reasoning_steps(
    predictions: torch.Tensor,
    certainties: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    max_nodes_to_show: int = 10
) -> Dict[str, np.ndarray]:
    """
    Extract data for visualizing reasoning over time.
    
    Args:
        predictions: [B, N*C, T]
        certainties: [B, 2, T]
        targets: [B, N, C]
        mask: [B, N]
        max_nodes_to_show: Maximum nodes to include
        
    Returns:
        Dict with visualization data
    """
    B, total_out, T = predictions.shape
    N = mask.shape[1]
    C = total_out // N
    
    # Take first batch item
    preds = predictions[0].view(N, C, T).cpu().numpy()
    certs = certainties[0].cpu().numpy()
    targs = targets[0].cpu().numpy()
    node_mask = mask[0].cpu().numpy()
    
    # Get valid nodes
    valid_idx = np.where(node_mask)[0][:max_nodes_to_show]
    
    return {
        'predictions': preds[valid_idx],  # [n_nodes, C, T]
        'certainties': certs,  # [2, T]
        'targets': targs[valid_idx],  # [n_nodes, C]
        'timesteps': np.arange(T),
        'node_indices': valid_idx
    }


class EarlyStopping:
    """Early stopping helper."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def get_algorithm_metrics(algorithm: str) -> List[str]:
    """Get relevant metrics for an algorithm."""
    classification_algos = ['bfs', 'dfs', 'fast_mis']
    regression_algos = ['dijkstra', 'eccentricity']
    edge_algos = ['mst_prim']
    
    base_metrics = ['loss']
    
    if algorithm in classification_algos:
        return base_metrics + ['accuracy', 'f1', 'precision', 'recall']
    elif algorithm in regression_algos:
        return base_metrics + ['mse', 'mae', 'rmse']
    elif algorithm in edge_algos:
        return base_metrics + ['accuracy', 'edge_f1']
    else:
        return base_metrics + ['accuracy']
