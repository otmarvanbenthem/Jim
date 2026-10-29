
import time
from pathlib import Path
import tomllib 
from matplotlib import pyplot as plt
import numpy as np
import argparse

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="path to template config file")
    p.add_argument("--outdir_data", help = "path to directory put generated data")
    p.add_argument("--outdir_config", help = "path to directory output configure file  ")
    p.add_argument("--detectors", help = "list of detectors to use") \
    p.add_argument("--snr", help = "list of snr's")
    args = p.parse_args()
    #if args.legend and len(args.legend) != len(args.dirs):
       # p.error(f"--legend needs {len(args.dirs)} labels, got {len(args.legend)}")
    return p, args
