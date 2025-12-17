"""
Dataset adapters for CLRS/SALSA-CLRS benchmarks.

This module wraps SALSA-CLRS PyG datasets into a format suitable for the CTM.
The CTM expects flat tensor inputs, so we convert graph-structured data into
a format that can be processed by attention-based mechanisms.
"""
import math
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
from torch.utils.data import Dataset
import numpy as np

try:
    from salsaclrs import (
        SALSACLRSDataset,
        SALSACLRSDataLoader, 
        load_dataset as salsa_load_dataset,
        ALGORITHMS as SALSA_ALGORITHMS,
    )
    from salsaclrs.data import CLRSData
    # Register required classes as safe globals for PyTorch 2.6+
    import torch
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([
        CLRSData, 
        DataEdgeAttr, 
        DataTensorAttr,
        GlobalStorage,
    ])
    SALSA_AVAILABLE = True
except ImportError:
    SALSA_AVAILABLE = False
    SALSA_ALGORITHMS = []

from .config import CLRSConfig, ALGORITHM_CONFIGS


class CLRSDatasetAdapter(Dataset):
    """
    Adapter that wraps SALSA-CLRS datasets for CTM consumption.
    
    The CTM processes inputs via attention mechanisms. For graphs, we represent:
    1. Node features as a sequence of tokens
    2. Edge information encoded into node features or as positional information
    3. Graph structure as attention masks
    
    This allows the CTM to reason over graph structure implicitly through
    its synchronization-based representation learning.
    """
    
    def __init__(
        self,
        algorithm: str,
        split: str = "train",
        num_samples: int = 10000,
        graph_generator: str = "er",
        graph_generator_kwargs: Optional[Dict] = None,
        data_dir: str = "./data/clrs",
        use_hints: bool = False,
        max_nodes: int = 64,
        node_feature_dim: int = 32,
        include_edge_features: bool = True,
        use_prebuilt: bool = False,
        **kwargs
    ):
        """
        Initialize the CLRS dataset adapter.
        
        Args:
            algorithm: Algorithm name ('bfs', 'dfs', 'dijkstra', etc.)
            split: Dataset split ('train', 'val', 'test')
            num_samples: Number of samples to generate
            graph_generator: Graph generator type ('er', 'ws', 'delaunay', etc.)
            graph_generator_kwargs: Arguments for graph generator
            data_dir: Directory to store/load data
            use_hints: Whether to include intermediate algorithm hints
            max_nodes: Maximum number of nodes (for padding)
            node_feature_dim: Dimension of node feature embeddings
            include_edge_features: Whether to include edge features in input
            use_prebuilt: Whether to use pre-built SALSA-CLRS datasets
        """
        if not SALSA_AVAILABLE:
            raise ImportError(
                "SALSA-CLRS is required. Install with:\n"
                "pip install git+https://github.com/jkminder/salsa-clrs.git"
            )
        
        self.algorithm = algorithm
        self.split = split
        self.max_nodes = max_nodes
        self.node_feature_dim = node_feature_dim
        self.include_edge_features = include_edge_features
        self.use_hints = use_hints
        
        # Get algorithm config
        if algorithm not in ALGORITHM_CONFIGS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                f"Available: {list(ALGORITHM_CONFIGS.keys())}"
            )
        self.algo_config = ALGORITHM_CONFIGS[algorithm]
        
        # Set up graph generator kwargs with defaults
        if graph_generator_kwargs is None:
            graph_generator_kwargs = self._default_graph_kwargs(graph_generator, split)
        
        # Load or create the dataset
        if use_prebuilt and split in ["train", "val"]:
            # Use prebuilt SALSA-CLRS datasets
            self._dataset = salsa_load_dataset(
                algorithm=algorithm,
                split=split,
                local_dir=data_dir
            )
            if isinstance(self._dataset, dict):
                # Test split returns dict, pick first one
                self._dataset = list(self._dataset.values())[0]
        else:
            # Generate custom dataset
            self._dataset = SALSACLRSDataset(
                root=data_dir,
                split=split,
                algorithm=algorithm,
                num_samples=num_samples,
                graph_generator=graph_generator,
                graph_generator_kwargs=graph_generator_kwargs,
                hints=use_hints,
                ignore_all_hints=not use_hints,
                max_cores=-1,  # Serial processing to avoid multiprocessing issues
                **kwargs
            )
        
        # Store specs for output processing
        self.specs = self._dataset.specs
        
        # Precompute feature dimensions
        self._setup_dimensions()
    
    def _default_graph_kwargs(self, generator: str, split: str) -> Dict:
        """Get default graph generator kwargs."""
        if split == "train":
            n = [4, 7, 11, 13, 16]
        elif split == "val":
            n = [16]
        else:
            n = [16, 32]
        
        kwargs = {"n": n}
        if generator in ["er", "ws"]:
            # Use log(n)/n scaling for ER graphs
            base_p = math.log(max(n)) / max(n)
            kwargs["p_range"] = (base_p, base_p * 3)
        if generator == "ws":
            kwargs["k"] = [4, 6, 8]
        
        return kwargs
    
    def _setup_dimensions(self):
        """Set up input/output dimensions based on algorithm."""
        # Input dimension: node features + optional edge features
        # We encode: node ID (one-hot or positional), degree, and any input features
        self.input_dim = self.node_feature_dim
        if self.include_edge_features:
            self.input_dim += self.node_feature_dim  # Edge encoding
        
        # Output dimension depends on task type
        task_type = self.algo_config["task_type"]
        if task_type == "node_classification":
            self.output_dim = 2  # Binary classification per node
        elif task_type == "node_regression":
            self.output_dim = 1  # Scalar per node
        elif task_type == "edge_classification":
            self.output_dim = 2  # Binary per edge
        else:
            self.output_dim = 2
    
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Get a single sample.
        
        Returns:
            inputs: Tensor of shape [max_nodes, input_dim] - node representations
            targets: Tensor of shape [max_nodes, output_dim] or scalar
            metadata: Dict with additional information (num_nodes, edge_index, etc.)
        """
        data = self._dataset[idx]
        
        # Get graph info
        num_nodes = self._get_num_nodes(data)
        edge_index = data.edge_index
        
        # Build node features
        node_features = self._build_node_features(data, num_nodes)
        
        # Build target
        target = self._build_target(data, num_nodes)
        
        # Pad to max_nodes
        inputs = self._pad_features(node_features, num_nodes)
        targets = self._pad_target(target, num_nodes)
        
        # Create attention mask for valid nodes
        attention_mask = torch.zeros(self.max_nodes, dtype=torch.bool)
        attention_mask[:num_nodes] = True
        
        # Build adjacency-based attention mask (nodes can attend to neighbors)
        adj_mask = self._build_adjacency_mask(edge_index, num_nodes)
        
        metadata = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "attention_mask": attention_mask,
            "adjacency_mask": adj_mask,
            "algorithm": self.algorithm,
        }
        
        return inputs, targets, metadata
    
    def _get_num_nodes(self, data) -> int:
        """Get number of nodes from PyG data object."""
        if hasattr(data, 'num_nodes'):
            return data.num_nodes
        # Infer from edge_index
        return int(data.edge_index.max().item()) + 1
    
    def _build_node_features(self, data, num_nodes: int) -> torch.Tensor:
        """
        Build node feature matrix from CLRS data.
        
        We encode:
        1. Positional encoding (node ID)
        2. Degree information
        3. Algorithm-specific input features
        """
        features = []
        
        # 1. Positional encoding using sinusoidal embeddings
        pos_enc = self._positional_encoding(num_nodes, self.node_feature_dim // 2)
        features.append(pos_enc)
        
        # 2. Degree encoding
        edge_index = data.edge_index
        in_degree = torch.zeros(num_nodes)
        out_degree = torch.zeros(num_nodes)
        
        # Count degrees
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            out_degree[src] += 1
            in_degree[dst] += 1
        
        degree_enc = self._encode_scalar(
            (in_degree + out_degree) / (2 * num_nodes), 
            self.node_feature_dim // 4
        )
        features.append(degree_enc)
        
        # 3. Algorithm-specific features
        if hasattr(data, 's') and data.s is not None:
            # Source node indicator (for BFS, DFS, Dijkstra)
            source_enc = self._encode_scalar(data.s.float(), self.node_feature_dim // 4)
            features.append(source_enc)
        else:
            # Pad with zeros
            features.append(torch.zeros(num_nodes, self.node_feature_dim // 4))
        
        # 4. Edge features (aggregated to nodes)
        if self.include_edge_features:
            edge_feat = self._aggregate_edge_features(data, num_nodes)
            features.append(edge_feat)
        
        # Concatenate and ensure correct dimension
        node_features = torch.cat(features, dim=-1)
        
        # Pad or truncate to input_dim
        if node_features.shape[-1] < self.input_dim:
            padding = torch.zeros(num_nodes, self.input_dim - node_features.shape[-1])
            node_features = torch.cat([node_features, padding], dim=-1)
        elif node_features.shape[-1] > self.input_dim:
            node_features = node_features[:, :self.input_dim]
        
        return node_features.float()
    
    def _positional_encoding(self, length: int, dim: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(length, dim)
        pe[:, 0::2] = torch.sin(position * div_term[:dim//2 + dim%2])
        pe[:, 1::2] = torch.cos(position * div_term[:dim//2])
        
        return pe
    
    def _encode_scalar(self, values: torch.Tensor, dim: int) -> torch.Tensor:
        """Encode scalar values using learned-style encoding."""
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        
        # Use sinusoidal encoding of the scalar value
        freqs = torch.exp(torch.linspace(0, math.log(100), dim // 2))
        
        encoded = torch.zeros(values.shape[0], dim)
        encoded[:, 0::2] = torch.sin(values * freqs[:dim//2 + dim%2])
        encoded[:, 1::2] = torch.cos(values * freqs[:dim//2])
        
        return encoded
    
    def _aggregate_edge_features(self, data, num_nodes: int) -> torch.Tensor:
        """Aggregate edge features to nodes."""
        edge_feat = torch.zeros(num_nodes, self.node_feature_dim)
        edge_index = data.edge_index
        
        # Check for edge weights
        if hasattr(data, 'weights') and data.weights is not None:
            weights = data.weights.float()
        elif hasattr(data, 'A') and data.A is not None:
            # Adjacency weights
            weights = data.A.float()
        else:
            weights = torch.ones(edge_index.shape[1])
        
        # Aggregate: sum of weighted edge encodings for each node
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            w = weights[i].item() if i < len(weights) else 1.0
            
            # Encode edge weight
            enc = self._encode_scalar(torch.tensor([w]), self.node_feature_dim // 2)
            edge_feat[dst, :self.node_feature_dim//2] += enc.squeeze(0)
            edge_feat[src, self.node_feature_dim//2:] += enc.squeeze(0)
        
        # Normalize by degree
        degree = torch.zeros(num_nodes)
        for i in range(edge_index.shape[1]):
            degree[edge_index[0, i]] += 1
            degree[edge_index[1, i]] += 1
        degree = degree.clamp(min=1)
        
        edge_feat = edge_feat / degree.unsqueeze(-1)
        
        return edge_feat
    
    def _build_target(self, data, num_nodes: int) -> torch.Tensor:
        """Build target tensor from CLRS data."""
        task_type = self.algo_config["task_type"]
        
        # Find the output attribute
        output_key = None
        for key in data.keys():
            if key in ['pi', 'reach', 'd', 'in_mst', 'in_mis', 'ecc']:
                output_key = key
                break
        
        if output_key is None:
            # Try to find output from specs
            for key, spec in self.specs.items():
                if spec[0].name == 'OUTPUT':
                    output_key = key
                    break
        
        if output_key is None:
            raise ValueError(f"Could not find output attribute in data: {list(data.keys())}")
        
        output_data = getattr(data, output_key)
        
        # Ensure output_data is a tensor
        if not isinstance(output_data, torch.Tensor):
            output_data = torch.tensor(output_data)
        
        # Get the actual number of outputs from the data
        actual_num = output_data.shape[0] if output_data.dim() > 0 else 1
        
        if task_type == "node_classification":
            # Binary classification per node
            target = torch.zeros(num_nodes, 2)
            if output_data.dim() == 1:
                # Binary mask - treat as class labels
                n = min(len(output_data), num_nodes)
                target[:n, 1] = output_data[:n].float()
                target[:n, 0] = 1 - target[:n, 1]
            elif output_data.dim() == 2:
                n = min(output_data.shape[0], num_nodes)
                target[:n] = output_data[:n].float()
            else:
                # Handle other cases
                target[:actual_num, 1] = output_data.flatten()[:num_nodes].float()
                target[:, 0] = 1 - target[:, 1]
        elif task_type == "node_regression":
            # Scalar per node
            target = torch.zeros(num_nodes, 1)
            if output_data.dim() == 0:
                target[:] = output_data.float()
            else:
                n = min(len(output_data.flatten()), num_nodes)
                target[:n, 0] = output_data.flatten()[:n].float()
        elif task_type == "edge_classification":
            # Need to convert edge output to node-level
            # For now, aggregate edge predictions to nodes
            target = torch.zeros(num_nodes, 2)
            edge_index = data.edge_index
            for i in range(min(edge_index.shape[1], len(output_data))):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if src < num_nodes and dst < num_nodes:
                    val = output_data[i].float().item()
                    target[src, 1] += val
                    target[dst, 1] += val
            target[:, 0] = 1 - (target[:, 1] > 0).float()
            target[:, 1] = (target[:, 1] > 0).float()
        else:
            target = torch.zeros(num_nodes, 1)
            if output_data.numel() > 0:
                n = min(output_data.numel(), num_nodes)
                target[:n, 0] = output_data.flatten()[:n].float()
        
        return target
    
    def _pad_features(self, features: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Pad features to max_nodes."""
        if num_nodes >= self.max_nodes:
            return features[:self.max_nodes]
        
        padding = torch.zeros(self.max_nodes - num_nodes, features.shape[-1])
        return torch.cat([features, padding], dim=0)
    
    def _pad_target(self, target: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Pad target to max_nodes."""
        if num_nodes >= self.max_nodes:
            return target[:self.max_nodes]
        
        padding = torch.zeros(self.max_nodes - num_nodes, target.shape[-1])
        return torch.cat([target, padding], dim=0)
    
    def _build_adjacency_mask(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Build adjacency mask for attention."""
        adj_mask = torch.zeros(self.max_nodes, self.max_nodes, dtype=torch.bool)
        
        # Set diagonal (self-attention)
        for i in range(min(num_nodes, self.max_nodes)):
            adj_mask[i, i] = True
        
        # Set edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src < self.max_nodes and dst < self.max_nodes:
                adj_mask[src, dst] = True
                adj_mask[dst, src] = True  # Symmetric for undirected
        
        return adj_mask


def create_clrs_datasets(
    algorithm: str,
    config: Optional[CLRSConfig] = None,
    data_dir: str = "./data/clrs",
    **kwargs
) -> Tuple[CLRSDatasetAdapter, CLRSDatasetAdapter, CLRSDatasetAdapter]:
    """
    Create train, validation, and test datasets for a CLRS algorithm.
    
    Args:
        algorithm: Algorithm name
        config: Optional CLRSConfig with settings
        data_dir: Directory for data storage
        **kwargs: Additional arguments passed to dataset
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    if config is None:
        config = CLRSConfig(algorithm=algorithm)
    
    common_kwargs = {
        "algorithm": algorithm,
        "data_dir": data_dir,
        "use_hints": config.use_hints,
        "graph_generator": config.graph_generator,
        **kwargs
    }
    
    train_ds = CLRSDatasetAdapter(
        split="train",
        num_samples=config.num_train_samples,
        graph_generator_kwargs={"n": config.num_nodes, "p_range": config.p_range},
        **common_kwargs
    )
    
    val_ds = CLRSDatasetAdapter(
        split="val", 
        num_samples=config.num_val_samples,
        graph_generator_kwargs={"n": [max(config.num_nodes)], "p_range": config.p_range},
        **common_kwargs
    )
    
    test_ds = CLRSDatasetAdapter(
        split="test",
        num_samples=config.num_test_samples,
        graph_generator_kwargs={"n": [max(config.num_nodes) * 2], "p_range": config.p_range},
        **common_kwargs
    )
    
    return train_ds, val_ds, test_ds


def collate_clrs_batch(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Collate function for CLRS batches.
    
    Args:
        batch: List of (inputs, targets, metadata) tuples
    
    Returns:
        Batched inputs, targets, and combined metadata
    """
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    
    # Combine metadata
    metadata = {
        "num_nodes": torch.tensor([b[2]["num_nodes"] for b in batch]),
        "attention_mask": torch.stack([b[2]["attention_mask"] for b in batch]),
        "adjacency_mask": torch.stack([b[2]["adjacency_mask"] for b in batch]),
        "algorithm": batch[0][2]["algorithm"],
    }
    
    return inputs, targets, metadata
