"""
Continuous Thought Machine for Graph Reasoning (CTM-Graph).

This module provides a CTM variant designed for graph-structured inputs,
suitable for CLRS algorithmic reasoning tasks. The key adaptations are:

1. Graph-aware input processing (nodes as tokens, edges as attention structure)
2. Node-level output predictions
3. Optional graph structure injection into attention
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# Import core CTM modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from models.modules import SynapseUNET, Squeeze, SuperLinear
from models.utils import compute_normalized_entropy


class GraphBackbone(nn.Module):
    """
    Backbone for processing graph-structured inputs.
    
    Projects node features and optionally incorporates edge information.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process node features.
        
        Args:
            x: Node features [B, N, D_in]
            
        Returns:
            Processed features [B, N, D_hidden]
        """
        h = self.input_proj(x)
        
        for layer in self.layers:
            h = h + layer(h)  # Residual connections
        
        return h


class ContinuousThoughtMachineGraph(nn.Module):
    """
    Continuous Thought Machine adapted for graph reasoning tasks.
    
    Key differences from base CTM:
    1. Input is a sequence of node features instead of images
    2. Attention can be masked based on graph structure
    3. Output is per-node predictions
    4. Supports graph-level aggregation for global predictions
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        d_input: int = 128,
        heads: int = 8,
        n_synch_out: int = 64,
        n_synch_action: int = 64,
        synapse_depth: int = 2,
        iterations: int = 50,
        memory_length: int = 25,
        deep_nlms: bool = True,
        memory_hidden_dims: int = 16,
        do_layernorm_nlm: bool = False,
        out_dims_per_node: int = 2,
        max_nodes: int = 64,
        dropout: float = 0.1,
        dropout_nlm: Optional[float] = None,
        neuron_select_type: str = 'random-pairing',
        n_random_pairing_self: int = 0,
        use_adjacency_mask: bool = False,
        backbone_layers: int = 2,
    ):
        """
        Initialize the Graph CTM.
        
        Args:
            input_dim: Dimension of input node features
            d_model: Core hidden dimension
            d_input: Dimension for attention projections
            heads: Number of attention heads
            n_synch_out: Neurons for output synchronization
            n_synch_action: Neurons for action synchronization
            synapse_depth: Depth of synapse U-Net
            iterations: Number of internal reasoning steps
            memory_length: History length for NLMs
            deep_nlms: Whether to use deep NLMs
            memory_hidden_dims: Hidden dim for NLMs
            do_layernorm_nlm: LayerNorm in NLMs
            out_dims_per_node: Output dimension per node
            max_nodes: Maximum nodes for padding
            dropout: Dropout rate
            dropout_nlm: Separate dropout for NLMs
            neuron_select_type: Neuron selection strategy
            n_random_pairing_self: Self-pairing neurons
            use_adjacency_mask: Use graph structure in attention
            backbone_layers: Number of backbone layers
        """
        super().__init__()
        
        # Store config
        self.input_dim = input_dim
        self.d_model = d_model
        self.d_input = d_input
        self.iterations = iterations
        self.memory_length = memory_length
        self.max_nodes = max_nodes
        self.out_dims_per_node = out_dims_per_node
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.neuron_select_type = neuron_select_type
        self.use_adjacency_mask = use_adjacency_mask
        
        dropout_nlm = dropout if dropout_nlm is None else dropout_nlm
        
        # Input processing (backbone)
        self.backbone = GraphBackbone(
            input_dim=input_dim,
            hidden_dim=d_input,
            num_layers=backbone_layers,
            dropout=dropout
        )
        
        # Attention projections
        self.kv_proj = nn.Sequential(
            nn.Linear(d_input, d_input),
            nn.LayerNorm(d_input)
        )
        self.q_proj = nn.Linear(self.calculate_synch_representation_size(n_synch_action), d_input)
        self.attention = nn.MultiheadAttention(
            d_input, heads, dropout, batch_first=True
        )
        
        # Core CTM modules
        self.synapses = self._build_synapses(synapse_depth, d_model, dropout)
        self.trace_processor = self._build_nlms(
            deep_nlms, do_layernorm_nlm, memory_length, 
            memory_hidden_dims, d_model, dropout_nlm
        )
        
        # Start states
        self.register_parameter(
            'start_activated_state',
            nn.Parameter(torch.zeros(d_model).uniform_(
                -math.sqrt(1/d_model), math.sqrt(1/d_model)
            ))
        )
        self.register_parameter(
            'start_trace',
            nn.Parameter(torch.zeros(d_model, memory_length).uniform_(
                -math.sqrt(1/(d_model+memory_length)), 
                math.sqrt(1/(d_model+memory_length))
            ))
        )
        
        # Synchronization setup
        self.synch_representation_size_action = self.calculate_synch_representation_size(n_synch_action)
        self.synch_representation_size_out = self.calculate_synch_representation_size(n_synch_out)
        
        self._setup_synchronization_params('action', n_synch_action, n_random_pairing_self)
        self._setup_synchronization_params('out', n_synch_out, n_random_pairing_self)
        
        # Output projection (node-level)
        self.output_projector = nn.Sequential(
            nn.Linear(self.synch_representation_size_out, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_nodes * out_dims_per_node)
        )
        
        # Prediction reshaper for certainty computation
        self.prediction_reshaper = [max_nodes, out_dims_per_node]
    
    def calculate_synch_representation_size(self, n_synch: int) -> int:
        """Calculate synchronization representation size."""
        if self.neuron_select_type == 'random-pairing':
            return n_synch
        else:
            return (n_synch * (n_synch + 1)) // 2
    
    def _build_synapses(self, depth: int, d_model: int, dropout: float) -> nn.Module:
        """Build synapse module."""
        if depth == 1:
            return nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        else:
            return SynapseUNET(d_model, depth, 16, dropout)
    
    def _build_nlms(
        self, deep: bool, do_norm: bool, memory_length: int,
        hidden_dims: int, d_model: int, dropout: float
    ) -> nn.Module:
        """Build Neuron-Level Models."""
        if deep:
            return nn.Sequential(
                SuperLinear(
                    in_dims=memory_length, out_dims=2 * hidden_dims,
                    N=d_model, do_norm=do_norm, dropout=dropout
                ),
                nn.GLU(),
                SuperLinear(
                    in_dims=hidden_dims, out_dims=2,
                    N=d_model, do_norm=do_norm, dropout=dropout
                ),
                nn.GLU(),
                Squeeze(-1)
            )
        else:
            return nn.Sequential(
                SuperLinear(
                    in_dims=memory_length, out_dims=2,
                    N=d_model, do_norm=do_norm, dropout=dropout
                ),
                nn.GLU(),
                Squeeze(-1)
            )
    
    def _setup_synchronization_params(
        self, synch_type: str, n_synch: int, n_random_pairing_self: int
    ):
        """Set up synchronization parameters."""
        left, right = self._initialize_neuron_indices(n_synch, n_random_pairing_self)
        synch_size = (
            self.synch_representation_size_action 
            if synch_type == 'action' 
            else self.synch_representation_size_out
        )
        
        self.register_buffer(f'{synch_type}_neuron_indices_left', left)
        self.register_buffer(f'{synch_type}_neuron_indices_right', right)
        self.register_parameter(
            f'decay_params_{synch_type}',
            nn.Parameter(torch.zeros(synch_size), requires_grad=True)
        )
    
    def _initialize_neuron_indices(
        self, n_synch: int, n_random_pairing_self: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize neuron indices for synchronization."""
        if self.neuron_select_type == 'random-pairing':
            left = torch.from_numpy(
                np.random.choice(np.arange(self.d_model), size=n_synch)
            )
            right = torch.cat([
                left[:n_random_pairing_self],
                torch.from_numpy(
                    np.random.choice(
                        np.arange(self.d_model), 
                        size=n_synch - n_random_pairing_self
                    )
                )
            ])
        else:
            left = torch.from_numpy(
                np.random.choice(np.arange(self.d_model), size=n_synch)
            )
            right = torch.from_numpy(
                np.random.choice(np.arange(self.d_model), size=n_synch)
            )
        
        return left, right
    
    def compute_synchronization(
        self,
        activated_state: torch.Tensor,
        decay_alpha: Optional[torch.Tensor],
        decay_beta: Optional[torch.Tensor],
        r: torch.Tensor,
        synch_type: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute synchronization representation."""
        if synch_type == 'action':
            n_synch = self.n_synch_action
            left_idx = self.action_neuron_indices_left
            right_idx = self.action_neuron_indices_right
        else:
            n_synch = self.n_synch_out
            left_idx = self.out_neuron_indices_left
            right_idx = self.out_neuron_indices_right
        
        if self.neuron_select_type == 'random-pairing':
            left = activated_state[:, left_idx]
            right = activated_state[:, right_idx]
            pairwise_product = left * right
        else:
            selected_left = activated_state[:, left_idx]
            selected_right = activated_state[:, right_idx]
            outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
            i, j = torch.triu_indices(n_synch, n_synch)
            pairwise_product = outer[:, i, j]
        
        # Recurrent synchronization update
        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1
        
        synchronization = decay_alpha / torch.sqrt(decay_beta)
        return synchronization, decay_alpha, decay_beta
    
    def compute_certainty(self, predictions: torch.Tensor) -> torch.Tensor:
        """Compute prediction certainty."""
        B = predictions.size(0)
        reshaped = predictions.reshape([B] + self.prediction_reshaper)
        ne = compute_normalized_entropy(reshaped)
        return torch.stack((ne, 1 - ne), -1)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        adjacency_mask: Optional[torch.Tensor] = None,
        track: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input node features [B, N, D_in]
            attention_mask: Valid node mask [B, N]
            adjacency_mask: Graph structure mask [B, N, N]
            track: Whether to track internal states
            
        Returns:
            predictions: [B, N * out_dims, T]
            certainties: [B, 2, T]
            synchronization_out: Final output synchronization
        """
        B = x.size(0)
        device = x.device
        
        # Process input through backbone
        kv = self.backbone(x)  # [B, N, d_input]
        kv = self.kv_proj(kv)
        
        # Initialize recurrent state
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1)
        
        # Prepare output storage
        out_size = self.max_nodes * self.out_dims_per_node
        predictions = torch.empty(
            B, out_size, self.iterations, 
            device=device, dtype=torch.float32
        )
        certainties = torch.empty(
            B, 2, self.iterations,
            device=device, dtype=torch.float32
        )
        
        # Initialize synchronization
        decay_alpha_action, decay_beta_action = None, None
        decay_alpha_out, decay_beta_out = None, None
        
        # Clamp decay parameters
        self.decay_params_action.data = torch.clamp(self.decay_params_action, 0, 15)
        self.decay_params_out.data = torch.clamp(self.decay_params_out, 0, 15)
        
        r_action = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1)
        r_out = torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)
        
        # Prepare attention mask
        attn_mask = None
        if self.use_adjacency_mask and adjacency_mask is not None:
            # Convert adjacency to attention mask (True = masked out)
            attn_mask = ~adjacency_mask
        
        # Tracking storage
        if track:
            tracking_data = {
                'pre_activations': [],
                'post_activations': [],
                'synch_out': [],
                'synch_action': [],
                'attention': []
            }
        
        # Recurrent reasoning loop
        for step in range(self.iterations):
            # Compute action synchronization
            synch_action, decay_alpha_action, decay_beta_action = \
                self.compute_synchronization(
                    activated_state, decay_alpha_action, 
                    decay_beta_action, r_action, 'action'
                )
            
            # Attend to input data
            q = self.q_proj(synch_action).unsqueeze(1)  # [B, 1, d_input]
            attn_out, attn_weights = self.attention(
                q, kv, kv,
                key_padding_mask=~attention_mask if attention_mask is not None else None,
                average_attn_weights=False,
                need_weights=track
            )
            attn_out = attn_out.squeeze(1)  # [B, d_input]
            
            # Apply synapses
            pre_synapse = torch.cat([attn_out, activated_state], dim=-1)
            state = self.synapses(pre_synapse)
            
            # Update state trace
            state_trace = torch.cat(
                [state_trace[:, :, 1:], state.unsqueeze(-1)], 
                dim=-1
            )
            
            # Apply NLMs
            activated_state = self.trace_processor(state_trace)
            
            # Compute output synchronization
            synch_out, decay_alpha_out, decay_beta_out = \
                self.compute_synchronization(
                    activated_state, decay_alpha_out,
                    decay_beta_out, r_out, 'out'
                )
            
            # Get predictions
            current_pred = self.output_projector(synch_out)
            current_certainty = self.compute_certainty(current_pred)
            
            predictions[..., step] = current_pred
            certainties[..., step] = current_certainty
            
            # Track internal states
            if track:
                tracking_data['pre_activations'].append(state.detach().cpu())
                tracking_data['post_activations'].append(activated_state.detach().cpu())
                tracking_data['synch_out'].append(synch_out.detach().cpu())
                tracking_data['synch_action'].append(synch_action.detach().cpu())
                if attn_weights is not None:
                    tracking_data['attention'].append(attn_weights.detach().cpu())
        
        if track:
            return predictions, certainties, synch_out, tracking_data
        
        return predictions, certainties, synch_out
    
    def get_most_certain_predictions(
        self,
        predictions: torch.Tensor,
        certainties: torch.Tensor
    ) -> torch.Tensor:
        """
        Get predictions from the most certain iteration.
        
        Args:
            predictions: [B, out_dims, T]
            certainties: [B, 2, T]
            
        Returns:
            Most certain predictions [B, out_dims]
        """
        # Use certainty (index 1) to select best iteration
        # Higher certainty = lower entropy = more confident
        cert_values = certainties[:, 1, :]  # [B, T]
        best_idx = cert_values.argmax(dim=-1)  # [B]
        
        B = predictions.size(0)
        return predictions[torch.arange(B), :, best_idx]


def build_ctm_graph(
    config: Dict[str, Any],
    input_dim: int = 64,
    max_nodes: int = 64
) -> ContinuousThoughtMachineGraph:
    """
    Build a CTM-Graph model from configuration.
    
    Args:
        config: Configuration dictionary
        input_dim: Input feature dimension
        max_nodes: Maximum number of nodes
        
    Returns:
        Configured CTM-Graph model
    """
    return ContinuousThoughtMachineGraph(
        input_dim=input_dim,
        d_model=config.get('d_model', 512),
        d_input=config.get('d_input', 128),
        heads=config.get('heads', 8),
        n_synch_out=config.get('n_synch_out', 64),
        n_synch_action=config.get('n_synch_action', 64),
        synapse_depth=config.get('synapse_depth', 2),
        iterations=config.get('iterations', 50),
        memory_length=config.get('memory_length', 25),
        deep_nlms=config.get('deep_nlms', True),
        memory_hidden_dims=config.get('memory_hidden_dims', 16),
        out_dims_per_node=config.get('out_dims_per_node', 2),
        max_nodes=max_nodes,
        dropout=config.get('dropout', 0.1),
        neuron_select_type=config.get('neuron_select_type', 'random-pairing'),
        n_random_pairing_self=config.get('n_random_pairing_self', 0),
        use_adjacency_mask=config.get('use_adjacency_mask', False),
    )
