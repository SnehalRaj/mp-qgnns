"""PennyLane circuits for the two quantum operations used by the models.

The first is the equivariant Johnson mixing on the node register: the node
register is the Hamming-weight-j subspace of n qubits, where the basis state of a
subset S has its qubits in S set to one, and the mixing is the evolution under
the hopping Hamiltonian

    H = sum_{a<b} (X_a X_b + Y_a Y_b) / 2 ,

which on the weight-j subspace is the symmetric Johnson adjacency A_J. Since H is
invariant under qubit permutations, exp(i*alpha*H) is S_N-equivariant and matches
the reduced-basis mixing of models.cfi.EquivariantQGNN
(tests/test_backend_equiv.py). The second is the RBS pyramid on the embedding
register (below), the circuit form of compound.CompoundPyramidLayer.
"""
from __future__ import annotations

from math import comb

import numpy as np
import pennylane as qml

from .subsets import enumerate_subsets


def johnson_hamiltonian(n: int) -> qml.Hamiltonian:
    """Hopping Hamiltonian whose weight-j block is the Johnson adjacency."""
    coeffs, ops = [], []
    for a in range(n):
        for b in range(a + 1, n):
            coeffs += [0.5, 0.5]
            ops += [qml.PauliX(a) @ qml.PauliX(b), qml.PauliY(a) @ qml.PauliY(b)]
    return qml.Hamiltonian(coeffs, ops)


def subset_basis_indices(n: int, j: int) -> list[int]:
    """Computational-basis index of each weight-j subset (PennyLane wire order)."""
    return [sum(1 << (n - 1 - v) for v in s) for s in enumerate_subsets(n, j)]


def mixing_unitary(n: int, j: int, alpha: float) -> np.ndarray:
    """Reduced-basis unitary of exp(i*alpha*H) on the weight-j subspace.

    Built from the PennyLane Hamiltonian and restricted to the weight-j
    computational-basis states, in the subset order of ``enumerate_subsets``.
    """
    full = qml.matrix(qml.evolve(johnson_hamiltonian(n), -alpha), wire_order=range(n))
    idx = subset_basis_indices(n, j)
    return np.asarray(full)[np.ix_(idx, idx)]


def apply_mixing(amplitudes: np.ndarray, n: int, j: int, alpha: float) -> np.ndarray:
    """Evolve a weight-j amplitude vector (subset order) by exp(i*alpha*H)."""
    if amplitudes.shape[0] != comb(n, j):
        raise ValueError("amplitude vector must have length C(n, j)")
    return mixing_unitary(n, j, alpha) @ amplitudes


# ---------------------------------------------------------------------------
# RBS pyramid on the embedding register. A reconfigurable beam splitter is a
# two-qubit Givens rotation (qml.SingleExcitation); a pyramid of them acts as an
# orthogonal matrix on the single-particle sector and, on the weight-k sector,
# as that matrix's k-th compound. This is the circuit counterpart of
# compound.CompoundPyramidLayer (checked in tests/test_compound_backend.py).
# ---------------------------------------------------------------------------


def pyramid_pairs(D: int) -> list[tuple[int, int]]:
    """Nearest-neighbour RBS layout with D(D-1)/2 gates."""
    pairs = []
    for offset in range(D - 1):
        for q in range(D - 1 - offset):
            pairs.append((q, q + 1))
    return pairs


def rbs_pyramid(angles, wires) -> None:
    """Apply a pyramid of SingleExcitation gates (use inside a QNode)."""
    pairs = pyramid_pairs(len(wires))
    for theta, (a, b) in zip(angles, pairs):
        qml.SingleExcitation(theta, wires=[wires[a], wires[b]])


def pyramid_unitary(angles, D: int) -> np.ndarray:
    """Full 2**D unitary of the RBS pyramid."""
    return np.asarray(qml.matrix(rbs_pyramid, wire_order=range(D))(angles, range(D)))


def hw_block(matrix: np.ndarray, D: int, k: int) -> np.ndarray:
    """Weight-k block of a 2**D operator, in subset order."""
    idx = subset_basis_indices(D, k)
    return matrix[np.ix_(idx, idx)]
