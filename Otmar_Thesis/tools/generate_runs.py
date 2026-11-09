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
KEY_DETECTOR = ("data", "detectors")
KEY_OUTDIR = ("output", "dir")
 
# Name tag of the fast likelihood and the sampler: (config path, abbreviation map).
# The fast likelihood is the first sub-table of [likelihood], e.g. [likelihood.multiband].
KEY_SAMPLER = ("sampler", "type")
LIKELIHOOD_ABBREVIATIONS = {
    "multiband": "mult",
    "heterodyne": "rel",  # relative binning / heterodyned likelihood
}
SAMPLER_ABBREVIATIONS = {
    "blackjax-ns-aw": "NS_AW",
}
 
# Parameters that go into the name, read from the config: {config key path: short label}.
# They are read from the config file, so the name always matches what is actually run.
# Fast likelihood parameters are all numeric entries of its sub-table; this maps key -> short label.
FAST_LIKELIHOOD_PARAM_LABELS = {"epsilon": "e"}
# Sampler parameters are all numeric entries of [sampler]; this maps key -> short label (optional).
SAMPLER_PARAM_LABELS = {}
 
# SLURM settings
SLURM_HEADER = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --mem=30G
#SBATCH -t 40:00:00
#SBATCH --output={slurm_output}
"""
# Name of the slurm log. {stem} is the full run name; %j (optional) is the SLURM job id.
# Examples: "slurm-{stem}.out"  or  "slurm-{stem}_%j.out"
SLURM_OUTPUT_PATTERN = "slurm-{stem}.out"
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
    p.add_argument("--detectors", required=True, nargs="+", help="list of detectors to use, given as for example: ET ET_2L")
    p.add_argument("--snr", nargs="+", type=float, default=[30.0], help="list of SNRs, given as for example: 30.0 40.0 50.0")
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
    if isinstance(x, str):  # text values: keep letters and digits only
        return re.sub(r"[^A-Za-z0-9]+", "_", x).strip("_")
    return str(x).replace(".", "_").replace("-", "m")
 
 
def _tag(cfg, key_path, abbreviations):
    value = str(_get(cfg, key_path))
    return abbreviations.get(value, value.replace(" ", "_").replace("-", "_"))
 
 
def _params_tag(cfg, params):
    return "_".join(f"{label}_{fmt_num(_get(cfg, path))}" for path, label in params.items())
 
 
def _fast_likelihood(cfg):
    """(name tag, parameter tag) of the fast likelihood: the first sub-table of [likelihood]."""
    subs = {k: v for k, v in cfg["likelihood"].items() if isinstance(v, dict)}
    if not subs:
        return "", ""
    name, params = next(iter(subs.items()))
    ptag = "_".join(
        f"{FAST_LIKELIHOOD_PARAM_LABELS.get(k, k)}_{fmt_num(v)}"
        for k, v in params.items()
        if not isinstance(v, (dict, list))
    )
    return LIKELIHOOD_ABBREVIATIONS.get(name, name), ptag
 
 
def _sampler_params(cfg):
    """Parameter tag with every numeric entry of [sampler], in config order."""
    return "_".join(
        f"{SAMPLER_PARAM_LABELS.get(k, k)}_{fmt_num(v)}"
        for k, v in cfg["sampler"].items()
        if not isinstance(v, (dict, list))
    )
 
 
def make_stem(cfg, detector, snr):
    """Shared base name of config, job, output directory and slurm log."""
    f_min, f_max = cfg["likelihood"]["f_min"], cfg["likelihood"]["f_max"]
    fast_name, fast_params = _fast_likelihood(cfg)
    parts = [
        detector,
        fast_name,
        fast_params,
        _tag(cfg, KEY_SAMPLER, SAMPLER_ABBREVIATIONS),
        _sampler_params(cfg),
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
    text = _set_value(text, KEY_DETECTOR, [detector])
    text = _set_value(text, KEY_OUTDIR, str(output_data_dir))
    output_path_config.write_text(text)
 
 
def _cluster_path(path):
    """Rewrite a local path that lies inside the Otmar_Thesis folder so that it starts with
    WORKDIR ($HOME/Jim/Otmar_Thesis/), e.g.
    /home/me/Documents/Jim/Otmar_Thesis/Job_scripts/x/x.toml -> $HOME/Jim/Otmar_Thesis/Job_scripts/x/x.toml
    """
    parts = Path(path).parts
    root = Path(WORKDIR).name  # "Otmar_Thesis"
    if root not in parts:
        raise ValueError(f"{path} is not inside a '{root}' directory, cannot build the cluster path.")
    rel = Path(*parts[parts.index(root) + 1:])
    return f"{WORKDIR.rstrip('/')}/{rel}"
 
 
def write_job_file(config_file, output_path_job, submit, output_dir):
    """Write the SLURM job file for `config_file` (the generated config, not the template)."""
    config_file = Path(config_file)
    stem = config_file.stem
    config_file = _cluster_path(config_file)
    body = SLURM_HEADER.format(slurm_output=SLURM_OUTPUT_PATTERN.format(stem=stem)) + f"""
# Silence the JAX cache warning
export PYTHONWARNINGS="ignore:Error writing persistent compilation cache entry:UserWarning"
 
source {VENV}
cd {WORKDIR}
 
start=$SECONDS
jim-run {config_file}
runtime=$((SECONDS - start))
echo "Python step took $runtime s"
mkdir -p {output_dir}
echo "$runtime" > {output_dir}/runtime_seconds.txt
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
 
    print("Starting loop over detectors and snrs")
    for detector in args.detectors:  # e.g. ["ET", "ET_2L"]
        for snr in args.snr:         # e.g. [30.0, 40.0, 50.0]
            print(f"Generating files for detector: {detector}, with snr: {snr}")
            stem = make_stem(cfg, detector, snr)
 
            d_L = get_d_L_for_target_snr(
                detector, waveform, parameters, target_snr=snr, **kwargs
            )
 
            # one subdirectory per run, named after the stem, holding config and job file
            run_dir = outdir_config / stem
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / f"{stem}.toml"
            job_path = run_dir / f"{stem}.sh"
 
            write_config_file(path_config, detector, d_L, outdir_data / stem, config_path)
            write_job_file(config_path, job_path, args.submit, outdir_data / stem)
 
 
if __name__ == "__main__":
    main()
