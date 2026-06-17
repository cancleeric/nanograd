"""Correctness tests: analytic gradients vs numerical finite differences."""

import math

from nanograd import MLP, Value


def numerical_grad(f, x, eps=1e-6):
    """Central-difference estimate of df/dx at scalar x."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def test_add_mul_pow():
    a = Value(-4.0)
    b = Value(2.0)
    c = a * b + b ** 3
    c.backward()
    assert math.isclose(c.data, -8.0 + 8.0)
    # dc/da = b = 2 ; dc/db = a + 3b^2 = -4 + 12 = 8
    assert math.isclose(a.grad, 2.0)
    assert math.isclose(b.grad, 8.0)


def test_relu_gradient():
    x = Value(3.0)
    y = x.relu() * 2
    y.backward()
    assert y.data == 6.0
    assert x.grad == 2.0  # active branch

    xn = Value(-1.0)
    (xn.relu() * 2).backward()
    assert xn.grad == 0.0  # dead branch


def test_matches_numerical():
    # f(x) = tanh(2x + 1) * x^2 ; autograd grad must match finite difference.
    def f_scalar(xv: float) -> float:
        return math.tanh(2 * xv + 1) * xv * xv

    x = Value(0.7)
    out = (x * 2 + 1).tanh() * (x * x)
    out.backward()

    assert math.isclose(x.grad, numerical_grad(f_scalar, 0.7), rel_tol=1e-4)


def test_mlp_trains():
    import random

    random.seed(0)
    model = MLP(2, [8, 1])
    for _ in range(50):
        pred = model([Value(1.0), Value(-1.0)])
        loss = (pred - 1.0) ** 2
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= 0.05 * p.grad
    assert abs(model([Value(1.0), Value(-1.0)]).data - 1.0) < 0.1
