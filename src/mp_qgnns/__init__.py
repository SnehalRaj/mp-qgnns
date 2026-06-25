"""Message-passing quantum graph neural networks on the Weisfeiler-Leman hierarchy.

Subpackages:
    core      Johnson combinatorics, exact set-j-WL, the compound-pyramid layer,
              and the PennyLane circuits.
    models    EquivariantQGNN, JohnsonGIN, GIN (CFI); TSPQGNN; QM9QGNN.
    data_io   graph pairs, TSP instances, and molecular datasets.
    training  task-specific training and evaluation loops.
"""

from .core.compound import CompoundPyramidLayer, compound_matrix
from .core.subsets import (
    enumerate_subsets,
    isotype_features,
    isotype_features_batched,
    johnson_adjacency,
)
from .core.wl import distinguishes, first_separating_j, set_j_wl_histograms
from .data_io.cfi import load_cfi, prism_vs_k33, relabel, relabelled_dataset
from .models.cfi import GIN, EquivariantQGNN, JohnsonGIN
from .models.qm9 import QM9QGNN
from .models.tsp import TSPQGNN
from .training.cfi import FisherReadout, evaluate
from .trainability import gradient_variance, variance_sweep

__all__ = [
    "enumerate_subsets",
    "johnson_adjacency",
    "isotype_features",
    "isotype_features_batched",
    "compound_matrix",
    "CompoundPyramidLayer",
    "set_j_wl_histograms",
    "distinguishes",
    "first_separating_j",
    "load_cfi",
    "prism_vs_k33",
    "relabel",
    "relabelled_dataset",
    "EquivariantQGNN",
    "JohnsonGIN",
    "GIN",
    "TSPQGNN",
    "QM9QGNN",
    "FisherReadout",
    "evaluate",
    "gradient_variance",
    "variance_sweep",
]
