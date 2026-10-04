"""Corner plot of one or more Jim runs, with the injected values marked.

Each argument is a run directory containing samples.npz (and config.final.toml).

    python plot_corner.py DATA/run1
    python plot_corner.py DATA/run1 DATA/run2 --legend flowMC NS-AW --out compare.png
"""
import argparse
import tomllib
from pathlib import Path

import corner
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.lines import Line2D

LABELS = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$", "q": r"$q$", "s1_z": r"$\chi_1$",
    "s2_z": r"$\chi_2$", "iota": r"$\iota$", "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$", "t_det": r"$t_\mathrm{det}\,[\mathrm{s}]$",
    "phase_c": r"$\phi_c$", "psi": r"$\psi$", "ra": r"$\alpha$",
    "dec": r"$\delta$", "lambda_1": r"$\Lambda_1$", "lambda_2": r"$\Lambda_2$",
}

parser = argparse.ArgumentParser()
parser.add_argument("dirs", nargs="+", help="run directories")
parser.add_argument("--legend", nargs="+", help="one label per directory")
parser.add_argument("--out", default="compare.png")
args = parser.parse_args()

dirs = [Path(d).expanduser() for d in args.dirs]
labels = args.legend or [d.name for d in dirs]

# Load samples: {parameter: array} for each run
runs = []
for d in dirs:
    with np.load(d / "samples.npz") as f:
        runs.append({k: np.asarray(f[k], dtype=float) for k in f.files})

# Plot the parameters that every run has
names = [k for k in runs[0] if all(k in r for r in runs)]
print("Plotting:", names)

# Injected values from the first run's config (q is computed from eta)
truths = None
config = dirs[0] / "config.final.toml"
if config.is_file():
    with open(config, "rb") as f:
        inj = tomllib.load(f)["data"]["injection_parameters"]
    eta = inj["eta"]
    inj["q"] = (1 - 2 * eta - np.sqrt(1 - 4 * eta)) / (2 * eta)
    truths = [inj.get(k) for k in names]  # None = no line (e.g. t_det)

fig = None
for i, run in enumerate(runs):
    samples = np.column_stack([run[k] for k in names])
    fig = corner.corner(
        samples, fig=fig, color=f"C{i}", labels=[LABELS.get(k, k) for k in names],
        truths=truths if i == 0 else None, truth_color="k",
        hist_kwargs={"density": True}, plot_datapoints=False, plot_density=False,
    )

fig.legend(handles=[Line2D([], [], color=f"C{i}", label=l) for i, l in enumerate(labels)],
           loc="upper right", fontsize=14) 
fig.savefig(args.out, dpi=200)
print("Saved", args.out)