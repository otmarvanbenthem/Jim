"""Compare the diagnostics of one or more Jim runs.

Each argument is a run directory containing diagnostics.npz (and diagnostics.json).
Prints every array and scalar, then plots each array against its index.

    python plot_diagnostics.py DATA/run1
    python plot_diagnostics.py DATA/run1 DATA/run2 --legend flowMC NS-AW --out diag.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("dirs", nargs="+", help="run directories")
parser.add_argument("--legend", nargs="+", help="one label per directory")
parser.add_argument("--out", default="diagnostics.png")
args = parser.parse_args()

dirs = [Path(d).expanduser() for d in args.dirs]
labels = args.legend or [d.name for d in dirs]

# Load the arrays: {name: array} for each run, plus the scalars from the json
runs, scalars = [], []
for d, label in zip(dirs, labels):
    with np.load(d / "diagnostics.npz") as f:
        runs.append({k: np.asarray(f[k]) for k in f.files})
    js = d / "diagnostics.json"
    scalars.append(json.load(open(js)) if js.is_file() else {})

    print(f"\n=== {label} ({d}) ===")
    for k, v in scalars[-1].items():
        print(f"  json  {k}: {v}")
    for k, v in runs[-1].items():
        info = f"shape={v.shape} dtype={v.dtype}"
        if v.size and np.issubdtype(v.dtype, np.number):
            info += f" min={np.nanmin(v):.4g} max={np.nanmax(v):.4g}"
        print(f"  npz   {k}: {info}")

# Plot every numeric array with at least 2 points that any run has
names = sorted({k for r in runs for k, v in r.items()
                if v.ndim in (1, 2) and v.shape[0] > 1 and np.issubdtype(v.dtype, np.number)})
if not names:
    raise SystemExit("No 1D/2D numeric arrays to plot.")

fig, axes = plt.subplots(len(names), 1, figsize=(12, 3.5 * len(names)), squeeze=False)
for ax, k in zip(axes.ravel(), names):
    for i, run in enumerate(runs):
        if k not in run:
            continue
        v = run[k]
        if v.ndim == 1:
            ax.plot(v, color=f"C{i}", label=labels[i])
        else:
            # 2D: assume rows = iterations, columns = live points/chains
            lo, med, hi = np.nanpercentile(v, [16, 50, 84], axis=1)
            ax.plot(med, color=f"C{i}", label=labels[i])
            ax.fill_between(np.arange(len(med)), lo, hi, color=f"C{i}", alpha=0.2)
    ax.set_title(k)
    ax.set_xlabel("index")
for ax in axes.ravel()[len(names):]:
    ax.axis("off")

axes[0, 0].legend()
fig.tight_layout()
fig.savefig(args.out, dpi=100)
print("Saved", args.out)