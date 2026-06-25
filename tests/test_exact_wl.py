from mp_qgnns import load_cfi, prism_vs_k33
from mp_qgnns.core.wl import distinguishes, first_separating_j


def test_prism_vs_k33_separates_at_three():
    a0, a1, n, sep = prism_vs_k33()
    assert sep == 3
    assert not distinguishes(a0, a1, n, 1)
    assert not distinguishes(a0, a1, n, 2)
    assert distinguishes(a0, a1, n, 3)


def test_cfi_k3_first_separates_at_three():
    a0, a1, n, sep = load_cfi("k3")
    assert sep == 3
    assert first_separating_j(a0, a1, n, 4) == 3


def test_cfi_k4_first_separates_at_four():
    a0, a1, n, sep = load_cfi("k4")
    assert sep == 4
    assert first_separating_j(a0, a1, n, 4) == 4
