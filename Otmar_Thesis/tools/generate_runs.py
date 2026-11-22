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

Parameter sweeps (all combinations are generated, parameters not given keep the template value):
  --sampler_param KEY VALUE [VALUE ...]     a key of the [sampler] table
  --likelihood_param KEY VALUE [VALUE ...]  a key of the fast likelihood table, e.g. [likelihood.heterodyne]
"""
 
import argparse
import copy
import itertools
import json
import re
import subprocess
import time
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
# Fast likelihood parameters that may be swept with --likelihood_param even when they are not (or
# commented out) in the template: {key: example value, used to pick the type}. Such a key is added
# to the [likelihood.<fast likelihood>] table of the generated config.
FAST_LIKELIHOOD_OPTIONAL_PARAMS = {"n_bins": 1}
# Groups of fast likelihood parameters that exclude each other (the template says: specify at most
# one of epsilon / n_bins). Sweeping one key removes the other keys of its group from the generated
# config, and two keys of the same group cannot be swept in the same run of this script.
FAST_LIKELIHOOD_EXCLUSIVE = [("epsilon", "n_bins")]
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
    p.add_argument("--sampler_param", action="append", nargs="+", metavar=("KEY", "VALUE"), default=[],
                   help="sampler parameter to sweep over: a key of the [sampler] table followed by one or "
                        "more values. Repeat for several parameters; all combinations are generated. "
                        "Example: --sampler_param n_live 1000 2000 --sampler_param n_target 500 1000. "
                        "Parameters not given keep the value of the template config.")
    p.add_argument("--likelihood_param", action="append", nargs="+", metavar=("KEY", "VALUE"), default=[],
                   help="fast likelihood parameter to sweep over: a key of the fast likelihood table "
                        "(e.g. [likelihood.heterodyne]) followed by one or more values. Repeat for several "
                        "parameters; all combinations are generated (also combined with --sampler_param). "
                        "Example: --likelihood_param epsilon 0.01 0.005. "
                        "epsilon and n_bins exclude each other: sweep one or the other, and the other "
                        "is removed from the generated config. Parameters not given keep the template value.")
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
# Sampler parameter sweep
# ---------------------------------------------------------------------------
def _parse_value(raw, template_value):
    """Convert a command-line string to the type of the value in the template config."""
    if isinstance(template_value, bool):
        if raw.lower() not in ("true", "false"):
            raise ValueError(f"Expected true/false, got {raw!r}.")
        return raw.lower() == "true"
    if isinstance(template_value, int):
        return int(float(raw)) if float(raw).is_integer() else float(raw)
    if isinstance(template_value, float):
        return float(raw)
    return raw


def _build_grid(table, param_args, flag, label, optional=None):
    """Turn repeated `KEY VALUE [VALUE ...]` arguments into a list of {key: value} dicts, one
    per combination (cartesian product). Returns [{}] if nothing was given, so the template
    values are used. Keys must be scalar entries of `table`, or keys of `optional`
    ({key: example value}) that may be added to it. `flag` and `label` are for error messages."""
    if not param_args:
        return [{}]
    optional = optional or {}
    scalar = {k: v for k, v in table.items() if not isinstance(v, (dict, list))}
    keys, value_lists = [], []
    for key, *raw_values in param_args:
        if key in scalar:
            template_value = scalar[key]
        elif key in optional:
            template_value = optional[key]
        else:
            valid = list(scalar) + [k for k in optional if k not in scalar]
            raise KeyError(f"'{key}' is not a scalar {label} parameter in the template. Available: {valid}")
        if key in keys:
            raise ValueError(f"{flag} {key} given more than once.")
        if not raw_values:
            raise ValueError(f"{flag} {key} needs at least one value.")
        keys.append(key)
        value_lists.append([_parse_value(v, template_value) for v in raw_values])
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def get_sampler_grid(cfg, sampler_param_args):
    """Grid of --sampler_param combinations. Works for any sampler: keys are just entries
    of the [sampler] table."""
    return _build_grid(cfg["sampler"], sampler_param_args, "--sampler_param", "[sampler]")


def _fast_likelihood_name(cfg):
    """Name of the fast likelihood: the first sub-table of [likelihood], e.g. 'heterodyne'."""
    subs = [k for k, v in cfg["likelihood"].items() if isinstance(v, dict)]
    if not subs:
        raise KeyError("The template has no fast likelihood sub-table under [likelihood].")
    return subs[0]


def _excluded_keys(keys):
    """Keys that must be dropped from the config when `keys` are set (FAST_LIKELIHOOD_EXCLUSIVE)."""
    keys = set(keys)
    return {other for group in FAST_LIKELIHOOD_EXCLUSIVE if keys & set(group)
            for other in group if other not in keys}


def get_likelihood_grid(cfg, likelihood_param_args):
    """Grid of --likelihood_param combinations: keys are entries of the fast likelihood
    table, e.g. [likelihood.heterodyne]."""
    if not likelihood_param_args:
        return [{}]
    name = _fast_likelihood_name(cfg)
    swept = {key for key, *_ in likelihood_param_args}
    for group in FAST_LIKELIHOOD_EXCLUSIVE:
        clash = sorted(swept & set(group))
        if len(clash) > 1:
            raise ValueError(f"--likelihood_param {' and '.join(clash)} exclude each other, "
                             f"generate them in separate runs of this script.")
    return _build_grid(cfg["likelihood"][name], likelihood_param_args, "--likelihood_param",
                       f"[likelihood.{name}]", FAST_LIKELIHOOD_OPTIONAL_PARAMS)


def with_sampler_params(cfg, sampler_params):
    """Copy of cfg with the given [sampler] values replaced (key order is kept)."""
    new_cfg = copy.deepcopy(cfg)
    new_cfg["sampler"].update(sampler_params)
    return new_cfg


def with_likelihood_params(cfg, likelihood_params):
    """Copy of cfg with the given fast likelihood values set. Parameters that exclude
    the given ones (e.g. epsilon when n_bins is set) are removed."""
    new_cfg = copy.deepcopy(cfg)
    if likelihood_params:
        table = new_cfg["likelihood"][_fast_likelihood_name(cfg)]
        for key in _excluded_keys(likelihood_params):
            table.pop(key, None)
        table.update(likelihood_params)
    return new_cfg


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


def _section_bounds(lines, section):
    """(index of the [section] header, index of the next header or end of file)."""
    header = re.compile(r"^\s*\[\s*" + r"\s*\.\s*".join(map(re.escape, section)) + r"\s*\]\s*(#.*)?$")
    any_header = re.compile(r"^\s*\[")
    for start, line in enumerate(lines):
        if header.match(line):
            end = next((j for j in range(start + 1, len(lines)) if any_header.match(lines[j])), len(lines))
            return start, end
    raise KeyError(f"Could not find section [{'.'.join(section)}] in the template config.")


def _add_value(text, key_path, value):
    """Add `key = value` at the end of its [section], after the last non-empty line
    (so before the next section)."""
    *section, key = key_path
    lines = text.splitlines()
    start, end = _section_bounds(lines, section)
    last = end - 1
    while last > start and not lines[last].strip():
        last -= 1
    lines.insert(last + 1, f"{key} = {json.dumps(value)}")
    return "\n".join(lines) + "\n"


def _remove_key(text, key_path):
    """Delete the line `key = ...` from its [section]. Does nothing if the key is not there
    (a commented-out `#key = ...` is left alone)."""
    *section, key = key_path
    lines = text.splitlines()
    start, end = _section_bounds(lines, section)
    for i in range(start + 1, end):
        if re.match(rf"^\s*{re.escape(key)}\s*=", lines[i]):
            del lines[i]
            break
    return "\n".join(lines) + "\n"


def _set_or_add_value(text, key_path, value):
    """Like _set_value, but adds the key to its section if it is not there yet."""
    try:
        return _set_value(text, key_path, value)
    except KeyError:
        return _add_value(text, key_path, value)
 
 
def write_config_file(path_reference_config, detector, d_L, output_data_dir, output_path_config,
                      sampler_params=None, likelihood_params=None):
    """Copy the reference config, changing only detector, d_L, output directory and the
    given [sampler] and fast likelihood parameters."""
    text = Path(path_reference_config).read_text()
    text = _set_value(text, KEY_D_L, round(float(d_L), 4))
    text = _set_value(text, KEY_DETECTOR, [detector])
    text = _set_value(text, KEY_OUTDIR, str(output_data_dir))
    for key, value in (sampler_params or {}).items():
        text = _set_value(text, ("sampler", key), value)
    if likelihood_params:
        fast_section = ("likelihood", _fast_likelihood_name(tomllib.loads(text)))
        for key in sorted(_excluded_keys(likelihood_params)):
            text = _remove_key(text, (*fast_section, key))
        for key, value in likelihood_params.items():
            text = _set_or_add_value(text, (*fast_section, key), value)
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
 
 
FAILED = []  # job files that could not be submitted, reported at the end of main()


def submit_job(job_path, retries=5, wait=10):
    """Submit with sbatch, retrying on transient errors. Returns True on success."""
    for attempt in range(1, retries + 1):
        res = subprocess.run(
            ["sbatch", str(job_path)],
            capture_output=True, text=True,
            cwd=job_path.parent,  # slurm log lands next to the job file, not in the cwd
        )
        if res.returncode == 0:
            print(f"  submitted {job_path.name}: {res.stdout.strip()}")
            return True
        print(f"  sbatch failed (attempt {attempt}/{retries}): {res.stderr.strip()}")
        time.sleep(wait * attempt)
    FAILED.append(job_path)
    return False


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
        submit_job(output_path_job)
        time.sleep(1)  # don't hammer the scheduler
 
 
def main():
    p, args = parse_args()
    path_config = Path(args.input).expanduser()
    outdir_data = Path(args.outdir_data).expanduser().resolve()
    outdir_config = Path(args.outdir_config).expanduser().resolve()
    outdir_config.mkdir(parents=True, exist_ok=True)
 
    waveform, parameters, kwargs, cfg = get_info(path_config)
    sampler_grid = get_sampler_grid(cfg, args.sampler_param)  # e.g. [{"n_live": 1000}, {"n_live": 2000}]
    likelihood_grid = get_likelihood_grid(cfg, args.likelihood_param)  # e.g. [{"epsilon": 0.01}, {"epsilon": 0.005}]

    print("Starting loop over detectors, snrs, sampler parameters and fast likelihood parameters")
    for detector in args.detectors:  # e.g. ["ET", "ET_2L"]
        for snr in args.snr:         # e.g. [30.0, 40.0, 50.0]
            # d_L only depends on detector and snr, so compute it once for all sampler settings
            d_L = get_d_L_for_target_snr(
                detector, waveform, parameters, target_snr=snr, **kwargs
            )
            for sampler_params, likelihood_params in itertools.product(sampler_grid, likelihood_grid):
                print(f"Generating files for detector: {detector}, with snr: {snr}, "
                      f"sampler parameters: {sampler_params or 'template'}, "
                      f"likelihood parameters: {likelihood_params or 'template'}")
                run_cfg = with_likelihood_params(with_sampler_params(cfg, sampler_params), likelihood_params)
                stem = make_stem(run_cfg, detector, snr)

                # one subdirectory per run, named after the stem, holding config and job file
                run_dir = outdir_config / stem
                run_dir.mkdir(parents=True, exist_ok=True)
                config_path = run_dir / f"{stem}.toml"
                job_path = run_dir / f"{stem}.sh"

                write_config_file(path_config, detector, d_L, outdir_data / stem, config_path,
                                  sampler_params, likelihood_params)
                write_job_file(config_path, job_path, args.submit, outdir_data / stem)

    if FAILED:
        print("\nThese jobs were NOT submitted:")
        for f in FAILED:
            print(f"  sbatch {f}")


if __name__ == "__main__":
    main()