"""A tiny reverse-mode automatic differentiation engine.

The whole idea fits in one class. A ``Value`` wraps a single scalar and
remembers how it was produced, so that calling ``.backward()`` on the final
node walks the computation graph in reverse and fills in ``.grad`` for every
value that contributed to it -- the same chain rule that powers PyTorch /
TensorFlow, written out in a way you can read in five minutes.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    """A scalar value and its gradient in an autodiff graph."""

    def __init__(self, data: float, _children: Iterable["Value"] = (), _op: str = ""):
        self.data: float = float(data)
        self.grad: float = 0.0
        # Internal autograd bookkeeping.
        self._backward: Callable[[], None] = lambda: None
        self._prev: set["Value"] = set(_children)
        self._op: str = _op  # for debugging / graph visualisation

    # -- core ops -------------------------------------------------------
    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(out)/d(self) = 1, d(out)/d(other) = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> "Value":
        assert isinstance(other, (int, float)), "only supports number powers"
        out = Value(self.data ** other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # -- activations ----------------------------------------------------
    def relu(self) -> "Value":
        out = Value(0.0 if self.data < 0 else self.data, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            self.grad += e * out.grad

        out._backward = _backward
        return out

    # -- the chain rule -------------------------------------------------
    def backward(self) -> None:
        """Populate ``.grad`` for every node feeding into ``self``."""
        topo: list[Value] = []
        visited: set[Value] = set()

        def build(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        # Seed: d(self)/d(self) = 1, then propagate backwards.
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # -- sugar ----------------------------------------------------------
    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: float) -> "Value":
        return self + other

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other: float) -> "Value":
        return Value(other) + (-self)

    def __rmul__(self, other: float) -> "Value":
        return self * other

    def __truediv__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        return self * other ** -1

    def __rtruediv__(self, other: float) -> "Value":
        return Value(other) * self ** -1

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
