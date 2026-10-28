import time
from pathlib import Path
import tomllib 
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
from jimgw.core.single_event.likelihood import TransientLikelihoodFD, MultibandedTransientLikelihoodFD, HeterodynedTransientLikelihoodFD
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

LABELS = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$", "q": r"$q$", "s1_z": r"$\chi_1$",
    "s2_z": r"$\chi_2$", "iota": r"$\iota$", "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$", "t_det": r"$t_\mathrm{det}\,[\mathrm{s}]$",
    "phase_c": r"$\phi_c$", "psi": r"$\psi$", "ra": r"$\alpha$",
    "dec": r"$\delta$", "lambda_1": r"$\Lambda_1$", "lambda_2": r"$\Lambda_2$",
}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dir", nargs="+", help="run directory")
    p.add_argument("--params", nargs="+", default=None,
                   help="parameters to evaluate likelihood, in this order "
                        "(maximum of two)")
    p.add_argument("--out", default="likelihood_evaluation.png")
    p.add_argument("--ref-ifo", type=int, default=0,
                   help="detector index used as t_det reference in the config "
                        "fallback (default 0)")
    p.add_argument("--likelihood", nargs="+", default="TransientLikelihoodFD", help="Which type of (fast) likelihood to use ")
    args = p.parse_args()
    #if args.legend and len(args.legend) != len(args.dirs):
       # p.error(f"--legend needs {len(args.dirs)} labels, got {len(args.legend)}")
    return p, args

def resolve_detectors(path, override=None):
    """Return (name, list of detectors) for the geometry the run used.

    `override` ("ET_2L" or "ET") wins. Otherwise the [data] table of the config
    is scanned for detector names: "ET1"/"ET2"/"ET3" or "triangle" -> ET
    triangle, "2L" or "ET_L..." -> ET_2L. If nothing matches, fall back to
    ET_2L with a warning.
    """
    with open(path, "rb") as f:        # tomllib needs binary mode
        cfg = tomllib.load(f)

    getters = {"ET_2L": get_ET_2L, "ET": get_ET}

    name = override
    if name is None:
        dets = cfg.get("data", {}).get("detectors", [])
        if isinstance(dets, str):
            dets = [dets]
        dets = [str(d).lower() for d in dets]
        text = " ".join(dets)
        if "2l" in text or "et_l" in text or "etl" in text:
            name = "ET_2L"
        elif "et" in dets or "triangle" in text or any(f"et{i}" in dets for i in (1, 2, 3)):
            name = "ET"
        else:
            print("WARNING: could not infer the detector geometry from the config; "
                  "assuming ET_2L. Pass --detectors ET or --detectors ET_2L to set it.")
            name = "ET_2L"
        print(f"Detector geometry: {name} (inferred from config)")
    else:
        print(f"Detector geometry: {name} (from --detectors)")
    return name, getters[name]()



def truths_from_config(dir):
    """Return the injection parameters from <dir>/config.final.toml."""
    path = Path(dir).expanduser() / "config.final.toml"
    with open(path, "rb") as f:        # tomllib needs binary mode
        cfg = tomllib.load(f)

    inj = dict(cfg["data"]["injection_parameters"])
    if "eta" in inj:
        eta = inj["eta"]
        inj["q"] = (1 - 2 * eta - np.sqrt(1 - 4 * eta)) / (2 * eta)
    return inj

def get_info(path):
    
    with open(path, "rb") as f:        # tomllib needs binary mode
        cfg = tomllib.load(f)
    freq = dict(cfg["likelihood"])
    f_min = freq["f_min"]
    f_max = freq["f_max"]
    sampling_frequency = 2*f_max
    duration = cfg["data"]["duration"]
    gps = cfg["data"]["trigger_time"]
    
    return f_min, f_max, sampling_frequency, duration, gps


def main():
    path_config = Path(args.dir).expanduser() / "config.final.toml"
    parser, args = parse_args()
    params = [str(d)  for d in args.params]
    fixed_params = truths_from_config(args.dir)
  
   
    name_det, getter = resolve_detectors(path_config, None)

    f_min, f_max, sampling_frequency, duration, gps = get_freq(path_config)
   
    waveform = RippleIMRPhenomD_NRTidalv2(f_ref=5.0) 

    for ifo in getter:
        ifo.load_and_set_psd() #load psd from asd txt file
   
        ifo.simulate_and_set_noise(
        duration=duration,
        sampling_frequency=sampling_frequency,
        trigger_time=gps,
        f_min=f_min,
        f_max=f_max,
        rng_key=jax.random.key(0),
         )
        ifo.inject_signal(waveform, fixed_params)
    print("injection done")

    keys_to_remove = params
    fixed_params = {k: v for k, v in fixed_params.items() if k not in params}

    #From here: Go to evaluating the likelihood. REMEMBER: want option to do 1 or 2 params.



if __name__ == "__main__":
    main()