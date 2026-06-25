from mp_qgnns.trainability import variance_sweep


def test_gradient_variance_decays_polynomially():
    rows = variance_sweep([4, 8, 12], k=2, num_samples=150)
    var = [r["grad_var"] for r in rows]
    dim = [r["dim"] for r in rows]
    assert var[-1] > 0
    # sub-exponential: the variance falls no faster than a polynomial in the
    # subspace dimension, which at fixed k is polynomial in the qubit count.
    assert var[0] / var[-1] < (dim[-1] / dim[0]) ** 3
