"""Train the MLP to separate two interleaving half-moons.

Run:  python -m examples.train_moons

The dataset is generated with the standard library so the example is fully
reproducible and runs in a second -- no third-party dependencies.
"""

from __future__ import annotations

import math
import random

from nanograd import MLP, Value

random.seed(1337)


def make_moons(n: int = 100, noise: float = 0.1):
    """Two interleaving half circles -- the classic non-linear toy set."""
    xs, ys = [], []
    for i in range(n):
        label = i % 2  # 0 -> upper moon, 1 -> lower moon
        t = math.pi * (i / n)
        if label == 0:
            x, y = math.cos(t), math.sin(t)
        else:
            x, y = 1 - math.cos(t), 0.5 - math.sin(t)
        x += random.gauss(0, noise)
        y += random.gauss(0, noise)
        xs.append([x, y])
        ys.append(label)
    return xs, ys


def loss(model: MLP, xs, ys):
    # Margin (SVM-style) loss + L2 regularisation, all through autograd.
    preds = [model(x) for x in xs]
    targets = [2 * y - 1 for y in ys]  # {0,1} -> {-1,+1}
    losses = [(1 + -ti * pi).relu() for ti, pi in zip(targets, preds)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg = 1e-4 * sum((p * p for p in model.parameters()), Value(0.0))
    total = data_loss + reg
    acc = sum((pi.data > 0) == (ti > 0) for ti, pi in zip(targets, preds)) / len(ys)
    return total, acc


def main() -> None:
    xs, ys = make_moons(100)
    model = MLP(2, [16, 16, 1])  # 2 inputs -> two 16-wide hidden -> 1 output
    print(f"parameters: {len(model.parameters())}")

    for step in range(100):
        total, acc = loss(model, xs, ys)

        model.zero_grad()
        total.backward()

        lr = 1.0 - 0.9 * step / 100  # simple decaying-LR SGD
        for p in model.parameters():
            p.data -= lr * p.grad

        if step % 10 == 0:
            print(f"step {step:3d}  loss {total.data:.4f}  acc {acc*100:.1f}%")

    print(f"final accuracy: {acc*100:.1f}%")
    return model, xs, ys


if __name__ == "__main__":
    main()
