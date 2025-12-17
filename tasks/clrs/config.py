"""
Configuration for CLRS algorithms and task settings.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class CLRSConfig:
    """Configuration for a CLRS task."""
    
    # Algorithm settings
    algorithm: str = "bfs"
    
    # Graph generation settings
    graph_generator: str = "er"  # 'er', 'ws', 'delaunay', 'grid', 'path', 'tree'
    num_nodes: List[int] = field(default_factory=lambda: [16, 32])
    p_range: Tuple[float, float] = (0.1, 0.3)  # For ER/WS graphs
    
    # Dataset settings
    num_train_samples: int = 10000
    num_val_samples: int = 1000
    num_test_samples: int = 1000
    use_hints: bool = False  # Whether to use intermediate hints
    
    # Model settings
    d_model: int = 512
    d_input: int = 128
    heads: int = 8
    iterations: int = 50
    memory_length: int = 25
    n_synch_out: int = 64
    n_synch_action: int = 64
    synapse_depth: int = 2
    
    # Training settings
    batch_size: int = 32
    batch_size_test: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-5
    training_iterations: int = 50000
    warmup_steps: int = 1000
    
    # Task-specific output dimension (will be set based on algorithm)
    out_dims: Optional[int] = None
    
    def get_graph_generator_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for the graph generator."""
        kwargs = {"n": self.num_nodes}
        if self.graph_generator in ["er", "ws"]:
            kwargs["p_range"] = self.p_range
        return kwargs


# Algorithm-specific configurations
ALGORITHM_CONFIGS = {
    # Node classification tasks (output per node)
    "bfs": {
        "task_type": "node_classification",
        "output_type": "mask",  # Binary mask of reachable nodes
        "description": "Breadth-First Search reachability",
    },
    "dfs": {
        "task_type": "node_classification", 
        "output_type": "mask",
        "description": "Depth-First Search reachability",
    },
    "dijkstra": {
        "task_type": "node_regression",
        "output_type": "distance",  # Shortest path distances
        "description": "Dijkstra's shortest path distances",
    },
    "mst_prim": {
        "task_type": "edge_classification",
        "output_type": "mask",  # Which edges are in MST
        "description": "Prim's Minimum Spanning Tree",
    },
    "fast_mis": {
        "task_type": "node_classification",
        "output_type": "mask",  # Which nodes are in MIS
        "description": "Fast Maximum Independent Set",
    },
    "eccentricity": {
        "task_type": "node_regression",
        "output_type": "scalar",  # Eccentricity per node
        "description": "Graph Eccentricity",
    },
}


# Default graph sizes for different splits
DEFAULT_GRAPH_SIZES = {
    "train": [4, 7, 11, 13, 16],
    "val": [16],
    "test_small": [16, 32],
    "test_medium": [64, 80],
    "test_large": [128, 160],
}
