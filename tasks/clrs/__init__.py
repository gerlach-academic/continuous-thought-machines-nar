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

Teacher Forcing Modes:
- none: Only supervise final output (default)
- hard: Force hint output at fixed CTM iteration intervals
- soft: Certainty-based hint emission (CTM decides when ready)

Usage:
    from tasks.clrs.dataset import CLRSDatasetAdapter
    from tasks.clrs.train import train_clrs
    
    # With soft teacher forcing:
    python -m tasks.clrs.train --algorithm bfs --use_hints --teacher_forcing soft
"""

from .dataset import CLRSDatasetAdapter, create_clrs_datasets
from .config import CLRSConfig, ALGORITHM_CONFIGS, TeacherForcingMode
from .hints import (
    HintScheduleConfig,
    AdaptiveHintScheduler,
    HardTeacherForcing,
    SoftTeacherForcing,
)

__all__ = [
    'CLRSDatasetAdapter',
    'create_clrs_datasets', 
    'CLRSConfig',
    'ALGORITHM_CONFIGS',
    'TeacherForcingMode',
    'HintScheduleConfig',
    'AdaptiveHintScheduler',
    'HardTeacherForcing',
    'SoftTeacherForcing',
]
