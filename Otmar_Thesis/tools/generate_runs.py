
import time
from pathlib import Path
import tomllib 
from matplotlib import pyplot as plt
import numpy as np
import argparse
from tools import get_network_optimal_snr, get_d_L_for_target_snr

from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2

def parse_args():
    
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="path to template config file")
    p.add_argument("--outdir_data", help="path to directory to put generated data")
    p.add_argument("--outdir_config", help="path to directory for output config file")
    p.add_argument("--detectors", nargs="+", help="list of detectors to use")
    p.add_argument("--snr", nargs="+", type=float, default=[30], help="list of SNRs")
    p.add_argument("--submit", action="store_true",
                   help="submit the jobs after running")
    args = p.parse_args()
    return p, args



    

def get_info(path):
    """Read a template config and return everything needed by
    get_d_L_for_target_snr / get_network_optimal_snr, except the detector
    and the target SNR (both come from the command-line args).

    Returns
    -------
    waveform : waveform model
    parameters : dict
        Injection parameters from [data.injection_parameters]
        (get_d_L_for_target_snr ignores the d_L in here).
    kwargs : dict
        duration, sampling_frequency, trigger_time, f_min, f_max.
    """
    with open(path, "rb") as f:  # tomllib needs binary mode
        cfg = tomllib.load(f)

    # waveform, change later to include more waveforms
    approximant = cfg["waveform"]["approximant"]
    if approximant != "IMRPhenomD_NRTidalv2":
        raise ValueError(f"Unsupported approximant {approximant!r}.")
    waveform = RippleIMRPhenomD_NRTidalv2(f_ref=cfg["waveform"]["f_ref"])

    # source parameters
    parameters = dict(cfg["data"]["injection_parameters"])

    # data settings
    f_min = cfg["likelihood"]["f_min"]
    f_max = cfg["likelihood"]["f_max"]
    kwargs = dict(
        duration=cfg["data"]["duration"],
        sampling_frequency=cfg["data"]["sampling_frequency"],
        trigger_time=cfg["data"]["trigger_time"],
        f_min=f_min,
        f_max=f_max,
    )

    return waveform, parameters, kwargs

def write_config_file(path_reference_config, detector, snr, output_path_config):

def write_job_file(config_file, output_path_job,submit): #Needs to be output_path_config, compared to the reference config being the complete


