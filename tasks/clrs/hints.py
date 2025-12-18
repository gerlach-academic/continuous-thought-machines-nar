"""
Hint-based teacher forcing for CLRS algorithms with CTM.

This module provides both hard and soft teacher forcing mechanisms:

Hard Teacher Forcing:
    - Maps CTM iterations directly to algorithm hint steps
    - Forces the model to output specific hints at specific iterations
    - Good for initial training but may limit CTM's reasoning potential
    
Soft Teacher Forcing:
    - Allows the CTM to reason freely between hint outputs
    - Uses certainty-based gating: when certainty exceeds threshold, 
      the model should output the next hint
    - Better matches CTM's philosophy of adaptive computation
    - Can also set a minimum iterations between hints and max iterations budget
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TeacherForcingMode(Enum):
    """Teacher forcing modes for hint supervision."""
    NONE = "none"           # No hint supervision, only final output
    HARD = "hard"           # Strict step-to-iteration mapping
    SOFT = "soft"           # Certainty-based hint emission


@dataclass
class HintScheduleConfig:
    """Configuration for hint scheduling."""
    mode: TeacherForcingMode = TeacherForcingMode.NONE
    
    # For HARD mode
    iterations_per_hint: int = 1  # How many CTM iterations per algorithm step
    
    # For SOFT mode
    certainty_threshold: float = 0.7   # Certainty level to trigger hint emission
    min_iterations_between: int = 1    # Minimum thinking time (prevent degenerate solutions)
    force_final_hints: bool = True     # Force remaining hints at end of iteration budget
    
    # Legacy (no longer forces during iteration, only at end if force_final_hints=True)
    max_iterations_per_hint: int = 10  # Not used in soft mode anymore
    
    # General settings
    hint_loss_weight: float = 1.0      # Weight for hint reconstruction loss
    output_loss_weight: float = 1.0    # Weight for final output loss
    progressive_hints: bool = True     # Start with more hints, decrease over training
    

def extract_hint_sequence(data, specs: Dict) -> Dict[str, torch.Tensor]:
    """
    Extract hint sequence from CLRS data.
    
    Hints have shape [N, T, ...] where N is nodes and T is algorithm steps.
    We need to reorganize them into a sequence of states.
    
    Args:
        data: PyG CLRSData object
        specs: Algorithm specifications
        
    Returns:
        Dictionary mapping hint names to [T, N, ...] tensors
    """
    hints = {}
    
    # Get hint attribute names from data
    if hasattr(data, 'hints') and data.hints:
        hint_names = data.hints
    else:
        # Infer from specs
        hint_names = [k for k, v in specs.items() if v[0].name == 'HINT']
    
    for hint_name in hint_names:
        if hasattr(data, hint_name):
            hint_data = getattr(data, hint_name)
            if hint_data is not None:
                # Hints from SALSA-CLRS have shape [N, T, ...] 
                # We transpose to [T, N, ...]
                if hint_data.dim() >= 2:
                    hint_tensor = hint_data.transpose(0, 1)
                else:
                    hint_tensor = hint_data.unsqueeze(0)
                hints[hint_name] = hint_tensor
    
    return hints


def get_num_hint_steps(hints: Dict[str, torch.Tensor]) -> int:
    """Get the number of algorithm steps from hint sequence."""
    if not hints:
        return 0
    # All hints should have same number of steps
    first_hint = next(iter(hints.values()))
    return first_hint.shape[0]


class HardTeacherForcing(nn.Module):
    """
    Hard teacher forcing: strict iteration-to-hint mapping.
    
    Maps CTM iterations to algorithm steps linearly:
    - iterations_per_hint=1: step 0 -> iter 0, step 1 -> iter 1, ...
    - iterations_per_hint=5: step 0 -> iter 5, step 1 -> iter 10, ...
    
    If CTM has more iterations than needed, remaining iterations
    should all predict the final output.
    """
    
    def __init__(self, config: HintScheduleConfig):
        super().__init__()
        self.config = config
        self.iterations_per_hint = config.iterations_per_hint
    
    def compute_schedule(
        self,
        num_iterations: int,
        num_hint_steps: int
    ) -> torch.Tensor:
        """
        Compute which hint step each CTM iteration should predict.
        
        Args:
            num_iterations: Total CTM iterations
            num_hint_steps: Number of algorithm hint steps
            
        Returns:
            target_step: [T] tensor mapping iteration -> hint step index
                        -1 means no hint supervision for that iteration
        """
        target_steps = torch.full((num_iterations,), -1, dtype=torch.long)
        
        for hint_idx in range(num_hint_steps):
            # Which iteration should output this hint?
            iter_idx = (hint_idx + 1) * self.iterations_per_hint - 1
            if iter_idx < num_iterations:
                target_steps[iter_idx] = hint_idx
        
        return target_steps
    
    def compute_loss(
        self,
        predictions: torch.Tensor,  # [B, N * out_dims, T]
        hints: Dict[str, torch.Tensor],  # Each [B, T_hint, N, ...]
        attention_mask: torch.Tensor,  # [B, N]
        hint_heads: nn.ModuleDict,  # Hint prediction heads (optional)
        num_iterations: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hard teacher forcing loss.
        
        For each hint step, we supervise the corresponding CTM iteration.
        Uses output predictions directly to supervise hints (without hint-specific heads).
        """
        if not hints:
            return torch.tensor(0.0, device=predictions.device), {}
        
        B, total_out, T = predictions.shape
        N = attention_mask.shape[1]
        out_dims = total_out // N
        device = predictions.device
        
        # Reshape predictions to [B, N, out_dims, T]
        predictions_reshaped = predictions.view(B, N, out_dims, T)
        
        # Get number of hint steps from first hint tensor
        num_hint_steps = next(iter(hints.values())).shape[1]
        
        schedule = self.compute_schedule(num_iterations, num_hint_steps)
        schedule = schedule.to(device)
        
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        metrics = {}
        
        # For each hint type
        for hint_name, hint_tensor in hints.items():
            hint_loss = torch.tensor(0.0, device=device)
            num_supervised = 0
            
            # For each iteration that should predict a hint
            for t in range(num_iterations):
                hint_idx = schedule[t].item()
                if hint_idx < 0:
                    continue  # No supervision this iteration
                
                if hint_idx >= hint_tensor.shape[1]:
                    continue  # Out of bounds
                
                # Get predictions at this iteration: [B, N, out_dims]
                pred = predictions_reshaped[..., t]
                
                # Get target hint: [B, N] or [B, N, D]
                target = hint_tensor[:, hint_idx]  
                
                # Compute loss based on hint shape
                if target.dim() == 2:  # [B, N] - node-level scalar or mask
                    pred_vals = pred[..., 0]  # Use first output dimension
                    
                    # Binary mask hint
                    if target.max() <= 1 and target.min() >= 0:
                        loss = F.binary_cross_entropy_with_logits(
                            pred_vals[attention_mask],
                            target[attention_mask].float()
                        )
                    else:
                        # Scalar hint
                        loss = F.mse_loss(
                            pred_vals[attention_mask],
                            target[attention_mask].float()
                        )
                elif target.dim() == 3:  # [B, N, D]
                    pred_used = pred[..., :target.shape[-1]]
                    loss = F.mse_loss(
                        pred_used[attention_mask],
                        target[attention_mask].float()
                    )
                else:
                    continue  # Skip unsupported shapes
                
                hint_loss = hint_loss + loss
                num_supervised += 1
            
            if num_supervised > 0:
                hint_loss = hint_loss / num_supervised
                total_loss = total_loss + hint_loss
                metrics[f'loss_{hint_name}'] = hint_loss.item()
        
        return total_loss, metrics


class SoftTeacherForcing(nn.Module):
    """
    Soft teacher forcing: certainty-based hint emission.
    
    The CTM can "think" freely, and when it becomes sufficiently certain
    (certainty > threshold), it emits the next hint. This allows variable
    computation time between hints.
    
    Key insight: The model should naturally learn efficient hint timing because:
    1. Fixed iteration budget creates implicit pressure to not waste iterations
    2. Emitting when uncertain → higher hint loss
    3. Not emitting all hints → missing supervision signal
    
    The model learns its own "rhythm" without explicit forcing.
    
    Parameters:
    - certainty_threshold: When certainty exceeds this, model is "ready" to emit
    - min_iterations_between: Small minimum to prevent degenerate solutions
    - force_final_hints: If True, force remaining hints at final iterations
    """
    
    def __init__(self, config: HintScheduleConfig):
        super().__init__()
        self.config = config
        self.certainty_threshold = config.certainty_threshold
        self.min_between = config.min_iterations_between
        # Note: max_between is now optional - only used if force_final is True
        self.max_between = config.max_iterations_per_hint
        self.force_final = getattr(config, 'force_final_hints', True)
    
    def compute_emission_points(
        self,
        certainties: torch.Tensor,  # [B, 2, T]
        num_hint_steps: int,
        force_remaining: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute when hints should be emitted based on certainty.
        
        The model decides when to emit based on its own certainty.
        No forcing except optionally at the very end if hints remain.
        
        Args:
            certainties: [B, 2, T] certainty values (index 1 is certainty)
            num_hint_steps: Number of algorithm hint steps
            force_remaining: If True, force any remaining hints at final iterations
            
        Returns:
            emission_mask: [B, T] bool tensor - True where hint should be emitted
            hint_indices: [B, T] int tensor - which hint step at each position
        """
        B, _, T = certainties.shape
        device = certainties.device
        
        cert = certainties[:, 1, :]  # [B, T] - certainty values
        
        emission_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        hint_indices = torch.full((B, T), -1, dtype=torch.long, device=device)
        
        # Process each sample independently (hints may be emitted at different times)
        for b in range(B):
            current_hint = 0
            iters_since_last = 0
            
            for t in range(T):
                if current_hint >= num_hint_steps:
                    # All hints emitted - done with hints for this sample
                    continue
                
                iters_since_last += 1
                should_emit = False
                
                # Pure certainty-based emission (after minimum iterations)
                if iters_since_last >= self.min_between:
                    if cert[b, t] >= self.certainty_threshold:
                        should_emit = True
                
                if should_emit:
                    emission_mask[b, t] = True
                    hint_indices[b, t] = current_hint
                    current_hint += 1
                    iters_since_last = 0
            
            # Handle remaining hints at end of iteration budget
            if force_remaining and current_hint < num_hint_steps:
                # Force remaining hints in final iterations
                remaining = num_hint_steps - current_hint
                for i in range(remaining):
                    t_idx = T - remaining + i
                    if t_idx >= 0:
                        emission_mask[b, t_idx] = True
                        hint_indices[b, t_idx] = current_hint + i
        
        return emission_mask, hint_indices
    
    def compute_loss(
        self,
        predictions: torch.Tensor,  # [B, N * out_dims, T]
        certainties: torch.Tensor,  # [B, 2, T]
        hints: Dict[str, torch.Tensor],  # Each [B, T_hint, N, ...]
        attention_mask: torch.Tensor,  # [B, N]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute soft teacher forcing loss.
        
        Only supervises at emission points determined by certainty.
        Tracks natural emissions (certainty-based) vs forced (at end).
        """
        if not hints:
            return torch.tensor(0.0), {}
        
        B, total_out, T = predictions.shape
        N = attention_mask.shape[1]
        device = predictions.device
        num_hint_steps = get_num_hint_steps(hints)
        
        # Get certainty values
        cert = certainties[:, 1, :]  # [B, T]
        
        emission_mask, hint_indices = self.compute_emission_points(
            certainties, num_hint_steps, force_remaining=self.force_final
        )
        
        # Count natural vs forced emissions
        natural_emissions = 0
        forced_emissions = 0
        
        for b in range(B):
            current_hint = 0
            iters_since_last = 0
            for t in range(T):
                if emission_mask[b, t]:
                    if iters_since_last >= self.min_between and cert[b, t] >= self.certainty_threshold:
                        natural_emissions += 1
                    else:
                        forced_emissions += 1
                    current_hint += 1
                    iters_since_last = 0
                else:
                    iters_since_last += 1
        
        total_loss = torch.tensor(0.0, device=device)
        metrics = {
            'num_emissions': emission_mask.float().sum().item() / B,
            'natural_emissions': natural_emissions / B,
            'forced_emissions': forced_emissions / B,
        }
        
        # For each hint type, compute loss at emission points
        for hint_name, hint_tensor in hints.items():
            hint_loss = torch.tensor(0.0, device=device)
            num_supervised = 0
            
            # Get predictions at emission points
            for b in range(B):
                for t in range(T):
                    if not emission_mask[b, t]:
                        continue
                    
                    hint_idx = hint_indices[b, t].item()
                    if hint_idx < 0 or hint_idx >= hint_tensor.shape[1]:
                        continue
                    
                    # Get prediction and target
                    pred = predictions[b, :, t].view(N, -1)  # [N, out_dims]
                    target = hint_tensor[b, hint_idx]  # [N, ...]
                    mask = attention_mask[b]  # [N]
                    
                    # Compute loss based on hint type
                    if target.dim() == 1:
                        if target.max() <= 1:
                            # Binary mask
                            loss = F.binary_cross_entropy_with_logits(
                                pred[mask, 0], target[mask].float()
                            )
                        else:
                            # Scalar
                            loss = F.mse_loss(pred[mask, 0], target[mask].float())
                    else:
                        # Multi-dimensional hint
                        loss = F.mse_loss(pred[mask], target[mask].float())
                    
                    hint_loss = hint_loss + loss
                    num_supervised += 1
            
            if num_supervised > 0:
                hint_loss = hint_loss / num_supervised
                total_loss = total_loss + hint_loss
                metrics[f'loss_{hint_name}'] = hint_loss.item()
        
        return total_loss, metrics


class AdaptiveHintScheduler(nn.Module):
    """
    Combined hint scheduler supporting both hard and soft modes.
    
    Also supports progressive training: start with more hint supervision
    and gradually reduce it as training progresses.
    """
    
    def __init__(self, config: HintScheduleConfig):
        super().__init__()
        self.config = config
        self.mode = config.mode
        
        if self.mode == TeacherForcingMode.HARD:
            self.scheduler = HardTeacherForcing(config)
        elif self.mode == TeacherForcingMode.SOFT:
            self.scheduler = SoftTeacherForcing(config)
        else:
            self.scheduler = None
        
        # For progressive training
        self.register_buffer('training_progress', torch.tensor(0.0))
    
    def update_progress(self, progress: float):
        """Update training progress (0 to 1) for progressive hints."""
        self.training_progress.fill_(progress)
    
    def get_effective_hint_weight(self) -> float:
        """Get current hint loss weight based on training progress."""
        if not self.config.progressive_hints:
            return self.config.hint_loss_weight
        
        # Linear decay from full weight to 0.1x over training
        progress = self.training_progress.item()
        decay = 1.0 - 0.9 * progress
        return self.config.hint_loss_weight * decay
    
    def forward(
        self,
        predictions: torch.Tensor,
        certainties: torch.Tensor,
        hints: Dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        hint_heads: Optional[nn.ModuleDict] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hint supervision loss.
        
        Args:
            predictions: [B, N * out_dims, T] model predictions
            certainties: [B, 2, T] certainty values
            hints: Dictionary of hint tensors
            attention_mask: [B, N] valid node mask
            hint_heads: Optional module dict for hint-specific predictions
            
        Returns:
            hint_loss: Weighted hint reconstruction loss
            metrics: Dictionary of per-hint metrics
        """
        if self.mode == TeacherForcingMode.NONE or self.scheduler is None:
            return torch.tensor(0.0, device=predictions.device), {}
        
        if self.mode == TeacherForcingMode.HARD:
            loss, metrics = self.scheduler.compute_loss(
                predictions, hints, attention_mask, 
                hint_heads or nn.ModuleDict(), predictions.shape[-1]
            )
        else:  # SOFT
            loss, metrics = self.scheduler.compute_loss(
                predictions, certainties, hints, attention_mask
            )
        
        # Apply weight
        weight = self.get_effective_hint_weight()
        weighted_loss = loss * weight
        metrics['hint_weight'] = weight
        
        return weighted_loss, metrics


# ============================================================================
# Hint-aware prediction heads
# ============================================================================

class HintPredictionHead(nn.Module):
    """
    Prediction head for a specific hint type.
    
    Different hint types need different output structures:
    - MASK: Binary per-node prediction
    - SCALAR: Continuous per-node prediction
    - POINTER: Edge prediction (node predicts its predecessor)
    - GRAPH: Single graph-level prediction
    """
    
    def __init__(
        self,
        input_dim: int,
        hint_type: str,  # 'mask', 'scalar', 'pointer', 'graph'
        max_nodes: int = 64,
    ):
        super().__init__()
        self.hint_type = hint_type
        self.max_nodes = max_nodes
        
        if hint_type == 'mask':
            self.head = nn.Linear(input_dim, 1)
        elif hint_type == 'scalar':
            self.head = nn.Linear(input_dim, 1)
        elif hint_type == 'pointer':
            self.head = nn.Linear(input_dim, max_nodes)  # Pointer to any node
        elif hint_type == 'graph':
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, 1)
            )
        else:
            raise ValueError(f"Unknown hint type: {hint_type}")
    
    def forward(
        self,
        hidden: torch.Tensor,  # [B, N, D] or [B, D] for graph
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict hint values.
        
        Args:
            hidden: Hidden state
            attention_mask: Valid node mask
            
        Returns:
            Hint predictions
        """
        if self.hint_type == 'graph':
            # Pool over nodes then predict
            if hidden.dim() == 3:
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1)
                    hidden = (hidden * mask).sum(1) / mask.sum(1)
                else:
                    hidden = hidden.mean(1)
            return self.head(hidden)
        else:
            return self.head(hidden)


def create_hint_heads(
    specs: Dict,
    input_dim: int,
    max_nodes: int = 64
) -> nn.ModuleDict:
    """
    Create prediction heads for all hints in an algorithm.
    
    Args:
        specs: Algorithm specifications with hint types
        input_dim: Input dimension from model
        max_nodes: Maximum number of nodes
        
    Returns:
        ModuleDict mapping hint names to prediction heads
    """
    from clrs._src.specs import Type, Location
    
    heads = nn.ModuleDict()
    
    for name, (stage, location, dtype, *rest) in specs.items():
        if stage.name != 'HINT':
            continue
        
        # Map CLRS types to our hint types
        if dtype == Type.MASK or dtype == Type.MASK_ONE:
            hint_type = 'mask'
        elif dtype == Type.SCALAR:
            hint_type = 'scalar'
        elif dtype == Type.POINTER:
            hint_type = 'pointer'
        else:
            # Default to scalar
            hint_type = 'scalar'
        
        if location == Location.GRAPH:
            hint_type = 'graph'
        
        heads[name] = HintPredictionHead(input_dim, hint_type, max_nodes)
    
    return heads
