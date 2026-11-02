"""Generate Jim config (.toml) and SLURM job (.sh) files for a set of detectors and SNRs.

For every (detector, SNR) pair the script
  1. computes the luminosity distance giving that network SNR,
  2. writes a config file based on the template config,
  3. writes a SLURM job file that runs it (and optionally submits it).

Naming (config, job, output directory and slurm log all share one stem):
  {detector}_{fast likelihood}_{fast likelihood parameters}_{sampler}_{sampler parameters}_{f_min}_{f_max}_{network snr}
  config : <stem>.toml
  job    : <stem>.sh
  outdir : <stem>/
  slurm  : slurm-<stem>.out
"""

import argparse
import json
import re
import subprocess
import tomllib
from pathlib import Path

from tools import get_d_L_for_target_snr

from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2

# ---------------------------------------------------------------------------
# Where things live in the template config. Adjust these to match your .toml.
# Each key is a path of (section, ..., key).
# ---------------------------------------------------------------------------
KEY_D_L = ("data", "injection_parameters", "d_L")
KEY_DETECTOR = ("data", "detector")
KEY_OUTDIR = ("data", "outdir")

# Name tag of the fast likelihood and the sampler: (config path, abbreviation map).
KEY_FAST_LIKELIHOOD = ("likelihood", "type")
KEY_SAMPLER = ("sampler", "type")
LIKELIHOOD_ABBREVIATIONS = {
    "HeterodynedTransientLikelihoodFD": "T_rel",
    "RelativeBinning": "T_rel",
}
SAMPLER_ABBREVIATIONS = {
    "NS-AW": "NS_AW",
    "nested_sampling_acceptance_walk": "NS_AW",
}

# Parameters that go into the name, read from the config: {config key path: short label}.
# They are read from the config file, so the name always matches what is actually run.
FAST_LIKELIHOOD_PARAMS = {("likelihood", "eps"): "e"}
SAMPLER_PARAMS = {("sampler", "n_live"): "n_live"}

# SLURM settings
SLURM_HEADER = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --mem=30G
#SBATCH -t 60:00:00
#SBATCH --output=slurm-{stem}.out
"""
VENV = "$HOME/Jim/.venv/bin/activate"
WORKDIR = "$HOME/Jim/Otmar_Thesis/"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="path to template config file")
    p.add_argument("--outdir_data", required=True, help="path to directory to put generated data")
    p.add_argument("--outdir_config", required=True,
                   help="path to directory for output config and job files")
    p.add_argument("--detectors", required=True, nargs="+", help="list of detectors to use")
    p.add_argument("--snr", nargs="+", type=float, default=[30.0], help="list of SNRs")
    p.add_argument("--submit", action="store_true", help="submit the jobs after running")
    return p, p.parse_args()


# ---------------------------------------------------------------------------
# Reading the template
# ---------------------------------------------------------------------------
def _get(cfg, path):
    for k in path:
        cfg = cfg[k]
    return cfg


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
    cfg : dict
        The full parsed template config.
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
    kwargs = dict(
        duration=cfg["data"]["duration"],
        sampling_frequency=cfg["data"]["sampling_frequency"],
        trigger_time=cfg["data"]["trigger_time"],
        f_min=cfg["likelihood"]["f_min"],
        f_max=cfg["likelihood"]["f_max"],
    )

    return waveform, parameters, kwargs, cfg


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def fmt_num(x):
    """5.0 -> '5', 0.005 -> '0_005', 2000 -> '2000'."""
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return str(x).replace(".", "_").replace("-", "m")

def _tag(cfg, key_path, abbreviations):
    """Generates tag from given values"""
    value = str(_get(cfg, key_path))
    return abbreviations.get(value, value.replace(" ", "_").replace("-", "_"))


def _params_tag(cfg, params):
    """From parameters, generates parameter tags."""
    return "_".join(f"{label}_{fmt_num(_get(cfg, path))}" for path, label in params.items())


def make_stem(cfg, detector, snr):
    """Shared base name of config, job, output directory and slurm log."""
    f_min, f_max = cfg["likelihood"]["f_min"], cfg["likelihood"]["f_max"]
    parts = [
        detector,
        _tag(cfg, KEY_FAST_LIKELIHOOD, LIKELIHOOD_ABBREVIATIONS),
        _params_tag(cfg, FAST_LIKELIHOOD_PARAMS),
        _tag(cfg, KEY_SAMPLER, SAMPLER_ABBREVIATIONS),
        _params_tag(cfg, SAMPLER_PARAMS),
        f"{fmt_num(f_min)}_Hz",
        f"{fmt_num(f_max)}_Hz",
        f"snr_{fmt_num(snr)}",
    ]
    return "_".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Writing files
# ---------------------------------------------------------------------------
def _set_value(text, key_path, value):
    """Replace `key = ...` inside its [section] of a TOML text, keeping all other
    formatting and comments of the template."""
    *section, key = key_path
    header = re.compile(r"^\s*\[\s*" + r"\s*\.\s*".join(map(re.escape, section)) + r"\s*\]\s*(#.*)?$")
    any_header = re.compile(r"^\s*\[")
    lines = text.splitlines()

    in_section = False
    for i, line in enumerate(lines):
        if header.match(line):
            in_section = True
            continue
        if in_section and any_header.match(line):
            break
        if in_section and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            comment = re.search(r"\s#.*$", line)
            lines[i] = f"{key} = {json.dumps(value)}" + (comment.group(0) if comment else "")
            return "\n".join(lines) + "\n"
    raise KeyError(f"Could not find {'.'.join(key_path)} in the template config.")


def write_config_file(path_reference_config, detector, d_L, output_data_dir, output_path_config):
    """Copy the reference config, changing only detector, d_L and output directory."""
    text = Path(path_reference_config).read_text()
    text = _set_value(text, KEY_D_L, round(float(d_L), 4))
    text = _set_value(text, KEY_DETECTOR, detector)
    text = _set_value(text, KEY_OUTDIR, str(output_data_dir))
    output_path_config.write_text(text)


def write_job_file(config_file, output_path_job, submit):
    """Write the SLURM job file for `config_file` (the generated config, not the template)."""
    config_file = Path(config_file)
    stem = config_file.stem
    body = SLURM_HEADER.format(stem=stem) + f"""
# Silence the JAX cache warning
export PYTHONWARNINGS="ignore:Error writing persistent compilation cache entry:UserWarning"

source {VENV}
cd {WORKDIR}

start=$SECONDS
jim-run {config_file}
echo "Python step took $((SECONDS - start)) s"
"""
    output_path_job.write_text(body)
    if submit:
        subprocess.run(["sbatch", str(output_path_job)], check=True)


def main():
    p, args = parse_args()
    path_config = Path(args.input).expanduser()
    outdir_data = Path(args.outdir_data).expanduser().resolve()
    outdir_config = Path(args.outdir_config).expanduser().resolve()
    outdir_config.mkdir(parents=True, exist_ok=True)

    waveform, parameters, kwargs, cfg = get_info(path_config)

    for detector in args.detectors:  # e.g. ["ET", "ET_2L"]
        for snr in args.snr:         # e.g. [30.0, 40.0, 50.0]
            stem = make_stem(cfg, detector, snr)

            d_L = get_d_L_for_target_snr(
                detector, waveform, parameters, target_snr=snr, **kwargs
            )

            config_path = outdir_config / f"{stem}.toml"
            job_path = outdir_config / f"{stem}.sh"

            write_config_file(path_config, detector, d_L, outdir_data / stem, config_path)
            write_job_file(config_path, job_path, args.submit)


if __name__ == "__main__":
    main()