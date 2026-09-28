#To determine correct use of GPU, since I will be working on my home pc and Snellius
import os
import shutil
import sys
# 1. Detect if running on local AMD/ROCm machine
has_rocm = shutil.which("rocminfo") is not None or os.path.exists("/opt/rocm")

# If on local ROCm machine and SDMA isn't disabled yet, set vars and relaunch
if has_rocm:
   os.environ["XLA_FLAGS"] = "--xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=false" 
   os.environ["HSA_ENABLE_SDMA"] = "0"
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
)
from jimgw.core.single_event.detector import get_H1, get_L1, get_V1
from jimgw.core.single_event.likelihood import MultibandedTransientLikelihoodFD
from jimgw.core.single_event.data import Data
from jimgw.core.single_event.waveform import RippleIMRPhenomXAS
from jimgw.core.single_event.transforms import (
    SkyFrameToDetectorFrameSkyPositionTransform,
    MassRatioToSymmetricMassRatioTransform,
    GeocentricArrivalTimeToDetectorArrivalTimeTransform,
)
from jimgw.core.transforms import (
    BoundToBound,
    CosineTransform,
    PowerLawTransform,
    reverse_bijective_transform,
)
from jimgw.samplers.config import FlowMCConfig


# --- Fetch data ---

# --- Waveform model ---
waveform = RippleIMRPhenomXAS(f_ref=20)

# --- Injection simulated signal ---

gps =  time.time() - 10000
q = 0.85
injection_parameters = {
"M_c"     : 28.3,
"q"       : 0.85,
"eta"     : q / (1 + q) ** 2,
"s1_z"    : 0.3,
"s2_z"   : -0.2,
"iota"    : 0.4,
"d_L"     : 440.0,
"t_c"     : 0.03,
"phase_c" : 0.5,
"psi"     : 0.1,
"ra"      : 1.375,
"dec"     : -1.21,
}


fmin = 20.0
fmax = 512
duration = 4.0
sampling_frequency = 2 * fmax

ifos = [get_H1(), get_L1(), get_V1()]
for ifo in ifos:
    ifo.load_and_set_psd()
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
        UniformPrior(20.0, 40.0, parameter_names=["M_c"]),
        UniformPrior(0.125, 1.0, parameter_names=["q"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s1_z"]),
        UniformPrior(-0.05, 0.05, parameter_names=["s2_z"]),
        SinePrior(parameter_names=["iota"]),
        PowerLawPrior(1.0, 2000.0, 2.0, parameter_names=["d_L"]),
        UniformPrior(-0.1, 0.1, parameter_names=["t_c"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"]),
        UniformPrior(0.0, jnp.pi, parameter_names=["psi"]),
        UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"]),
        CosinePrior(parameter_names=["dec"]),
       # UniformPrior(0.0, 5000.0, parameter_names=["lambda_1"]),
       # UniformPrior(0.0, 5000.0, parameter_names=["lambda_2"]),
    ]
)

print("prior done")
# --- Transforms ---

sample_transforms = [
    GeocentricArrivalTimeToDetectorArrivalTimeTransform(trigger_time=gps, ifo=ifos[0]),
    SkyFrameToDetectorFrameSkyPositionTransform(trigger_time=gps, ifos=ifos),
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
    time_offset=0.1, #not sure what this does but did help with memory
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
        n_chains=100,
        n_local_steps=10,
        n_global_steps=100,
        n_training_loops=5,
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
   # "lambda_1": r"$\Lambda_1$",
   # "lambda_2": r"$\Lambda_2$",
}

fig = corner.corner(
    np.stack([chains[key] for key in jim.prior.parameter_names]).T[::10],
    labels=[parameter_labels.get(k, k) for k in jim.prior.parameter_names],
     truths = [injection_parameters[k] for k in jim.prior.parameter_names],
)
fig.savefig(Path(__file__).parent / "GW150914_multiband_Otmar.png")
