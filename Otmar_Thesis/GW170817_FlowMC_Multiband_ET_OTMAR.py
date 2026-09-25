#To determine correct use of GPU, since I will be working on my home pc and Snellius
import os
import shutil
import sys
# 1. Detect if running on local AMD/ROCm machine
has_rocm = shutil.which("rocminfo") is not None or os.path.exists("/opt/rocm")

# If on local ROCm machine and SDMA isn't disabled yet, set vars and relaunch
if has_rocm:
  # os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false" 
  # os.environ["HSA_ENABLE_SDMA"] = "0"
  #os.environ["HIP_VISIBLE_DEVICES"] = "0"
 #  os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" #not sure what this does but helped
   print('Using AMD GPU')


#If on Snellius
else:
    print('Using Nvidia GPU')

#Rest of code
#To start the project, we will use NS AW as our sampler. Most if not all, will be copied from GW150914_NS_AW.py
#Also using the guides

import time
from pathlib import Path

import corner
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jimgw.core.jim import Jim
from jimgw.core.prior import (
    CombinePrior,
    UniformPrior,
    CosinePrior,
    SinePrior,
    PowerLawPrior,
    SinePrior,
)
from jimgw.core.single_event.detector import get_ET
from jimgw.core.single_event.likelihood import MultibandedTransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
)
from jimgw.core.transforms import (
    BoundToBound,
    CosineTransform,
    PowerLawTransform,
    SineTransform,
    reverse_bijective_transform,
)
from jimgw.core.single_event.data import PowerSpectrum

from jimgw.samplers.config import FlowMCConfig


# --- Fetch data ---

# --- Waveform model ---
waveform = RippleIMRPhenomD_NRTidalv2(f_ref=5.0)

# --- Injection simulated signal ---

gps =  1766797218
t_det_min, t_det_max = -0.1, 0.1
injection_parameters = {
    # Mass & Spin parameters
    "M_c": 1.186,          # Chirp mass (solar masses)
    "eta": 0.248,          # Symmetric mass ratio
    "s1_z": 0.005,         # Primary dimensionless spin
    "s2_z": 0.005,         # Secondary dimensionless spin

    # Extrinsic parameters
    "ra": 3.446,           # Right ascension (rad)
    "dec": -0.408,         # Declination (rad)
    "psi": 1.75,           # Polarization angle (rad)
    "d_L": 40.0,           # Luminosity distance (Mpc)
    "iota": 2.53,          # Inclination angle (rad)
    "phase_c": 0.0,        # Coalescence phase (rad)
    "t_c": 0.0,            # Geocentric trigger time shift (s)

    # Tidal deformation parameters (Option A: Component-wise)
    "lambda_1": 300.0,     # Primary dimensionless tidal deformability (m1 ~ 1.46 M_sun)
    "lambda_2": 450.0,     # Secondary dimensionless tidal deformability (m2 ~ 1.27 M_sun)

}


fmin = 5.0
fmax = 2048
duration = 64
sampling_frequency = 2 * fmax

ifos = get_ET()
for ifo in ifos:
    ifo.set_psd(PowerSpectrum.from_file("curves_Jan_2020/et_d.txt", is_asd=True)) #load psd from asd txt file
    ifo.inject_signal(
        duration,
        sampling_frequency,
        trigger_time=gps,
        waveform_model=waveform,
        parameters=injection_parameters,
        f_min=fmin,
        f_max=fmax,
        zero_noise=False,
    )
print("injection done")
# --- Prior ---



prior = CombinePrior(
    [
        UniformPrior(1.170, 1.200, parameter_names=["M_c"]),
        UniformPrior(0.125, 1.0, parameter_names=["q"]),
        UniformPrior(-0.99, 0.99, parameter_names=["s1_z"]),
        UniformPrior(-0.99, 0.99
                     , parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(1.0, 100.0, 2.0, parameter_names=["d_L"]),
        UniformPrior(-0.1, 0.1, parameter_names=["t_c"]), #was t_c
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
        UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
    ]
)
print("prior done")
# --- Transforms ---

sample_transforms = [
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(trigger_time=gps, ifo=ifos[0]),
 
]

likelihood_transforms = [
    MassRatioToSymmetricMassRatioTransform,

]


# --- Likelihood ---

likelihood = MultibandedTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    f_min=fmin,
    f_max=fmax,
    trigger_time=gps,
    prior=prior,
)

print("multiband done")
# --- Sample ---

jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    periodic={
        "psi": (0.0, float(jnp.pi)),
        "phase_c": (0.0, 2 * float(jnp.pi)),
        "azimuth": (0.0, 2 * float(jnp.pi)),
    },
    sampler_config=FlowMCConfig(
        n_chains=1000,
        n_local_steps=100,
        n_global_steps=1000,
        n_training_loops=50,
        n_production_loops=10,
        n_NFproposal_batch_size=64,
        global_thinning=100,
    ),
    verbose=True,
)

start_time = time.time()
jim.sample()
end_time = time.time()
print(f"Sampling took {(end_time - start_time) / 60:.2f} mins")

# --- Results ---

diagnostics = jim.get_diagnostics()
print(f"Likelihood evaluations: {diagnostics['n_likelihood_evaluations']:,}")

chains = jim.get_samples()

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
fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T,
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
)
fig.savefig(Path(__file__).parent / "GW170817_FlowMC_Multiband_ET_OTMAR.png")

# %%
