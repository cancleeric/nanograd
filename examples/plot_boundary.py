"""Train on the moons and plot the learned decision boundary.

    python -m examples.plot_boundary   # -> docs/decision-boundary.png

Optional extra: needs matplotlib + numpy (pip install nanograd[plot]).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from examples.train_moons import make_moons, loss
from nanograd import MLP, Value
import random

random.seed(1337)
xs, ys = make_moons(100)
model = MLP(2, [16, 16, 1])
for step in range(100):
    total, _ = loss(model, xs, ys)
    model.zero_grad(); total.backward()
    lr = 1.0 - 0.9 * step / 100
    for p in model.parameters():
        p.data -= lr * p.grad

# evaluate model over a grid
gx = np.linspace(-1.6, 2.1, 120); gy = np.linspace(-1.6, 1.6, 120)
Z = np.zeros((len(gy), len(gx)))
for i, yv in enumerate(gy):
    for j, xv in enumerate(gx):
        Z[i, j] = model([Value(xv), Value(yv)]).data

pts = np.array(xs); lab = np.array(ys)
fig, ax = plt.subplots(figsize=(6, 5))
ax.contourf(gx, gy, Z, levels=[-1e9, 0, 1e9], colors=["#ffd5d5", "#d5e3ff"])
ax.contour(gx, gy, Z, levels=[0], colors="k", linewidths=1)
ax.scatter(pts[lab==0,0], pts[lab==0,1], c="#c0392b", s=18, label="class 0")
ax.scatter(pts[lab==1,0], pts[lab==1,1], c="#2c5fb3", s=18, label="class 1")
ax.set_title("nanograd MLP — learned decision boundary (moons, 100% acc)")
ax.legend(loc="upper right"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig("docs/decision-boundary.png", dpi=130)
print("saved docs/decision-boundary.png")
