import json
from pathlib import Path

import corner
import numpy as np
# Define directories and base filename
data_dir = Path("./DATA")
base_name = "sample_data_GW170817_FlowMC_Multiband.npz"
# Build file paths

npz_path = data_dir / base_name

# Load .npz archive safely into a dictionary
with np.load(npz_path) as data:
    chains = {key: data[key] for key in data.files}

print(chains)

parameter_labels = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$",
    "q": r"$q$",
    "s1_z": r"$\chi_1$",
    "s2_z": r"$\chi_2$",
    "iota": r"$\iota$",
    "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$",
    "psi": r"$\psi$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
}
#%%
print("Plotting")
# Replaces jim.prior.parameter_names: use the keys in the file, in file order
parameter_names = list(chains.keys())

samples = np.column_stack([np.asarray(chains[k], dtype=float) for k in parameter_names])
print(samples.shape)  # (n_samples, n_params)

fig = corner.corner(
    samples,
    labels=[parameter_labels.get(k, k) for k in parameter_names],
)
print("Plotting done")
fig.savefig(Path(__file__).parent/ "Figures" / "GW170817_FlowMC_Multiband_ET_OTMAR.png")