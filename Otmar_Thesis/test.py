import os
import shutil
import sys
# 1. Detect if running on local AMD/ROCm machine
has_rocm = shutil.which("rocminfo") is not None or os.path.exists("/opt/rocm")

# If on local ROCm machine and SDMA isn't disabled yet, set vars and relaunch
if has_rocm:
 #   os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0"
   os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
   
    #os.environ["HSA_ENABLE_SDMA"] = "0"
  # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"   
   # print("Using AMD home GPU (rx 9060xt)")
   print('Running JimGW on cpu')

else:
    print('Using Nvidia GPU')


import logging
import warnings
import jax






jax.config.update("jax_enable_x64", True) 
logging.getLogger("flowMC").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

from jimgw.core.single_event.detector import get_H1, get_L1
from jimgw.core.single_event.waveform import RippleTaylorF2
from jimgw.core.single_event.likelihood import TransientLikelihoodFD
from jimgw.core.single_event.transforms import MassRatioToSymmetricMassRatioTransform
from jimgw.core.prior import CombinePrior, UniformPrior
from jimgw.core.jim import Jim
from jimgw.samplers.config import FlowMCConfig

#%%
gps_time = 1126259462.0

injection_parameters = {
    "M_c": 30.0,
    "eta": 0.24,
    "s1_z": 0.0,
    "s2_z": 0.0,
    "lambda_1": 0.0,
    "lambda_2": 0.0,
    "d_L": 1000.0,
    "t_c": 0.0,
    "phase_c": 0.0,
    "iota": 0.4,
    "psi": 0.3,
    "ra": 1.5,
    "dec": 0.5,
}

#%%
waveform = RippleTaylorF2(f_ref=20.0)

f_min = 20.0
f_max = 1024.0
duration = 4.0
sampling_frequency = 2 * f_max

ifos = [get_H1(), get_L1()]
for ifo in ifos:
    ifo.load_and_set_psd()
    ifo.inject_signal(
        duration,
        sampling_frequency,
        trigger_time=gps_time,
        waveform_model=waveform,
        parameters=injection_parameters,
        f_min=f_min,
        f_max=f_max,
    )

#%%
prior = CombinePrior(
    [
        UniformPrior(29.0, 31.0, ["M_c"]),
        UniformPrior(0.125, 1.0, ["q"]),
    ]
)
#%%
likelihood_transforms = [MassRatioToSymmetricMassRatioTransform]

sample_transforms = []  # sampler operates directly in prior space (M_c, q)
#%%

fixed_parameters = {
    k: v for k, v in injection_parameters.items() if k not in ["M_c", "eta"]
}

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    trigger_time=gps_time,
    f_min=f_min,
    f_max=f_max,
    fixed_parameters=fixed_parameters,
)

#%%
jim = Jim(
    likelihood,
    prior,
    sampler_config=FlowMCConfig(
        n_chains=50,
        n_local_steps=50,
        n_global_steps=100,
        n_training_loops=5,
        n_production_loops=3,
        n_epochs=5,
        rq_spline_hidden_units=[32, 32],
        rq_spline_n_layers=4,
        rq_spline_n_bins=8,
        n_max_examples=3000,
    ),
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
)

jim.sample()

#%%

from corner import corner
import matplotlib.pyplot as plt
import numpy as np

samples = jim.get_samples()

fig = corner(
    np.column_stack([samples["M_c"], samples["q"]]),
    labels=[r"$\mathcal{M}_c\ [M_\odot]$", r"$q$"],
    truths=[
        injection_parameters["M_c"],
        MassRatioToSymmetricMassRatioTransform.inverse(
            {"eta": injection_parameters["eta"]}
        )[0]["q"],
    ],
)
plt.show()