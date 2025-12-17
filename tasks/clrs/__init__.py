"""
CLRS/SALSA-CLRS Task for the Continuous Thought Machine (CTM).

This module provides adapters to run CLRS algorithmic reasoning benchmarks
with the CTM architecture. It supports both:
- SALSA-CLRS: Sparse and scalable PyTorch Geometric datasets
- dm-clrs: Original DeepMind CLRS benchmark (via SALSA-CLRS dependency)

Available algorithms (SALSA-CLRS):
- bfs: Breadth-First Search
- dfs: Depth-First Search  
- dijkstra: Dijkstra's shortest path
- mst_prim: Prim's Minimum Spanning Tree
- fast_mis: Fast Maximum Independent Set
- eccentricity: Graph Eccentricity

Usage:
    from tasks.clrs.dataset import CLRSDatasetAdapter
    from tasks.clrs.train import train_clrs
"""

from .dataset import CLRSDatasetAdapter, create_clrs_datasets
from .config import CLRSConfig, ALGORITHM_CONFIGS

__all__ = [
    'CLRSDatasetAdapter',
    'create_clrs_datasets', 
    'CLRSConfig',
    'ALGORITHM_CONFIGS',
]
