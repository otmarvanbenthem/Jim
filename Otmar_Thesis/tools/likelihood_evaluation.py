import time
from pathlib import Path

import corner
from matplotlib import pyplot as plt
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
from jimgw.core.single_event.detector import get_ET, get_ET_2L
from jimgw.core.single_event.likelihood import TransientLikelihoodFD, MultibandedTransientLikelihoodFD
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
    "d_L": 880.0,           # Luminosity distance (Mpc)
    "iota": 2.53,          # Inclination angle (rad)
    "phase_c": 0.0,        # Coalescence phase (rad)
    "t_c": 0.0,            # Geocentric trigger time shift (s)

    # Tidal deformation parameters (Option A: Component-wise)
    "lambda_1": 300.0,     # Primary dimensionless tidal deformability (m1 ~ 1.46 M_sun)
    "lambda_2": 450.0,     # Secondary dimensionless tidal deformability (m2 ~ 1.27 M_sun)

}

fixed_parameters = {
    # Mass & Spin parameters
    "M_c": 1.186,          # Chirp mass (solar masses)
    "eta": 0.248,          # Symmetric mass ratio
    "s1_z": 0.005,         # Primary dimensionless spin
    "s2_z": 0.005,         # Secondary dimensionless spin

    # Extrinsic parameters
   # "ra": 3.446,           # Right ascension (rad)
   # "dec": -0.408,         # Declination (rad)
    "psi": 1.75,           # Polarization angle (rad)
    "d_L": 880.0,           # Luminosity distance (Mpc)
    "iota": 2.53,          # Inclination angle (rad)
    "phase_c": 0.0,        # Coalescence phase (rad)
    "t_c": 0.0,            # Geocentric trigger time shift (s)

    # Tidal deformation parameters (Option A: Component-wise)
    "lambda_1": 300.0,     # Primary dimensionless tidal deformability (m1 ~ 1.46 M_sun)
    "lambda_2": 450.0,     # Secondary dimensionless tidal deformability (m2 ~ 1.27 M_sun)

}

detector = "ET"
f_min = 5.0
f_max = 2048
duration = 6800
sampling_frequency = 2 * f_max

if (detector == "ET"):
    ifos = get_ET()

elif (detector == "ET_2L"):
    ifos = get_ET_2L()

for ifo in ifos:
    ifo.load_and_set_psd() #load psd from asd txt file
    # simulate the noise and inject the signal
    ifo.simulate_and_set_noise(
        duration=duration,
        sampling_frequency=sampling_frequency,
        trigger_time=gps,
        f_min=f_min,
        f_max=f_max,
        rng_key=jax.random.key(0),
    )
    ifo.inject_signal(waveform, injection_parameters)
print("injection done")

# --- Likelihood ---

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=waveform,
    f_min=f_min,
    f_max=f_max,
    fixed_parameters=fixed_parameters
)



print("Evaluating logL in chunks...")

# Define a vmapped function for a fixed chunk size
chunk_size = 50
f_chunk = jax.jit(jax.vmap(likelihood.evaluate))

RA, DEC = jnp.meshgrid(
    jnp.linspace(0, 2 * jnp.pi, 50), jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 50)
)
ra_flat = RA.ravel()
dec_flat = DEC.ravel()
n_total = ra_flat.shape[0]

logL_list = []
for i in range(0, n_total, chunk_size):
  chunk_ra = ra_flat[i : i + chunk_size]
  chunk_dec = dec_flat[i : i + chunk_size]

  params = {"ra": chunk_ra, "dec": chunk_dec}
  val = f_chunk(params)
  logL_list.append(val)

# Concatenate chunks and reshape
logL2 = jnp.concatenate(logL_list).reshape(RA.shape)

#plot

ra = np.linspace(0, 2*np.pi, 50)
dec = np.linspace(-np.pi, np.pi, 50)
L = np.asarray(logL2)
plt.pcolormesh(ra, dec, L, shading="auto", cmap="viridis")
plt.colorbar(label=r"$\ln L$")
#plt.contour(ra, dec, L, levels=L.max() - np.array([0.5, 2.0, 4.5]), colors="w")
plt.savefig("grid.png", dpi=150, bbox_inches="tight")
# %%
