import json
from pathlib import Path

import corner
import numpy as np

file_path = Path("/DATA")
file_name = "sample_data_GW170817_FlowMC_Multiband.json"  # no leading slash

with open(file_path / file_name, "r", encoding="utf-8") as f:
    chain = json.load(f)


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
print("plotting")
# Replaces jim.prior.parameter_names: use the keys in the file, in file order
parameter_names = list(chain.keys())

samples = np.column_stack([np.asarray(chain[k], dtype=float) for k in parameter_names])
print(samples.shape)  # (n_samples, n_params)

fig = corner.corner(
    samples,
    labels=[parameter_labels.get(k, k) for k in parameter_names],
)

fig.savefig(Path(__file__).parent/ "Figures" / "GW170817_FlowMC_Multiband_ET_OTMAR.png")