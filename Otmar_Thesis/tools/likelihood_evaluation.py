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

from jimgw.samplers.config import BlackJAXNSAWConfig


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
duration = 6800
sampling_frequency = 2 * fmax


ifos = get_ET()
for ifo in ifos:
    ifo.set_psd() #load psd from asd txt file
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

# --- Likelihood ---

likelihood = MultibandedTransientLikelihoodFD(
    ifos,
    waveform=waveform,
    f_min=fmin,
    f_max=fmax,
    trigger_time=gps,
    prior=prior,
)
