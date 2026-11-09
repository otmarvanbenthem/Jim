"""Corner plot of one or more Jim runs, with the injected values marked.

Each argument is a run directory containing samples.npz. Truth lines are read
from truths.json in the first run directory if it exists (written by the run
in the same space as the samples); otherwise they are reconstructed from
config.final.toml (approximate, see truths_from_config).

    python plot_corner.py DATA/run1
    python plot_corner.py DATA/run1 DATA/run2 --legend flowMC NS-AW --out compare.png
    python plot_corner.py DATA/run1 DATA/run2 --params ra dec d_L iota --out sky.png
"""
import jax

jax.config.update("jax_enable_x64", True)  # must come before importing jimgw

import argparse
import json
import tomllib
from pathlib import Path

import corner
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.lines import Line2D

LABELS = {
    "M_c": r"$\mathcal{M}_c\,[M_\odot]$", "q": r"$q$", "s1_z": r"$\chi_1$",
    "s2_z": r"$\chi_2$", "iota": r"$\iota$", "d_L": r"$d_L\,[\mathrm{Mpc}]$",
    "t_c": r"$t_c\,[\mathrm{s}]$", "t_det": r"$t_\mathrm{det}\,[\mathrm{s}]$",
    "phase_c": r"$\phi_c$", "psi": r"$\psi$", "ra": r"$\alpha$",
    "dec": r"$\delta$", "lambda_1": r"$\Lambda_1$", "lambda_2": r"$\Lambda_2$",
}

# Saved in samples.npz but not physical parameters: excluded from the default
# selection (can still be requested explicitly with --params).
NOT_PARAMS = {"log_likelihood", "log_prior", "log_posterior", "weights"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+", help="run directories")
    p.add_argument("--legend", nargs="+", help="one label per directory")
    p.add_argument("--params", nargs="+", default=None,
                   help="parameters to plot, in this order "
                        "(default: all shared by every run, excluding log_likelihood etc.)")
    p.add_argument("--out", default="compare.png")
    p.add_argument("--no-truths", action="store_true", help="do not draw truth lines")
    p.add_argument("--ref-ifo", type=int, default=0,
                   help="detector index used as t_det reference in the config "
                        "fallback (default 0)")
    p.add_argument("--detectors", choices=["ET_2L", "ET"], default=None,
                   help="detector geometry for the config fallback: ET_2L (two L-shaped) "
                        "or ET (triangle). Default: infer from config.final.toml")
    args = p.parse_args()
    if args.legend and len(args.legend) != len(args.dirs):
        p.error(f"--legend needs {len(args.dirs)} labels, got {len(args.legend)}")
    return p, args


def load_runs(parser, dirs):
    runs = []
    for d in dirs:
        path = d / "samples.npz"
        if not path.is_file():
            found = sorted(x.name for x in d.iterdir()) if d.is_dir() else "directory does not exist"
            parser.error(f"{path} not found. Contents of {d}: {found}")
        with np.load(path) as f:
            runs.append({k: np.asarray(f[k], dtype=float) for k in f.files})
    return runs


def select_names(parser, runs, requested):
    shared = [k for k in runs[0] if all(k in r for r in runs)]
    if requested is None:
        return [k for k in shared if k not in NOT_PARAMS]
    missing = [p for p in requested if p not in shared]
    if missing:
        parser.error(f"parameter(s) not found in all runs: {missing}. Available: {shared}")
    return list(dict.fromkeys(requested))  # keep order, drop duplicates


def resolve_detectors(cfg, override=None):
    """Return (name, list of detectors) for the geometry the run used.

    `override` ("ET_2L" or "ET") wins. Otherwise the [data] table of the config
    is scanned for detector names: "ET1"/"ET2"/"ET3" or "triangle" -> ET
    triangle, "2L" or "ET_L..." -> ET_2L. If nothing matches, fall back to
    ET_2L with a warning.
    """
    from jimgw.core.single_event.detector import get_ET, get_ET_2L
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


def truths_from_config(cfg, names, runs, ref_ifo, detectors=None):
    """Approximate injected values from config.final.toml.

    Used only when no truths.json exists. t_det is computed for detector
    `ref_ifo` of the run's detector set (ET_2L or the ET triangle, see
    resolve_detectors); ra/dec are the geocentric injection values, which are
    wrong if the samples are in a detector-frame sky parametrisation.
    """
    inj = dict(cfg["data"]["injection_parameters"])
    if "eta" in inj:
        eta = inj["eta"]
        inj["q"] = (1 - 2 * eta - np.sqrt(1 - 4 * eta)) / (2 * eta)

    if cfg.get("sampling", {}).get("detector_frame_sky", False) and ("ra" in names or "dec" in names):
        print("WARNING: detector_frame_sky = true; the ra/dec truth lines from the config "
              "may be in the wrong frame. Prefer truths.json.")

    if "t_det" in names and "t_det" not in inj:
        import jax.numpy as jnp
        from jimgw.core.single_event.time_utils import cached_greenwich_mean_sidereal_time

        _, ifos = resolve_detectors(cfg, detectors)
        if not 0 <= ref_ifo < len(ifos):
            raise SystemExit(f"--ref-ifo {ref_ifo} out of range: this geometry has "
                             f"{len(ifos)} detectors (0..{len(ifos) - 1})")

        gmst = cached_greenwich_mean_sidereal_time(cfg["data"]["trigger_time"])
        ra, dec = jnp.float64(inj["ra"]), jnp.float64(inj["dec"])
        cands = [float(inj["t_c"] + ifo.delay_from_geocenter(ra, dec, gmst))
                 for ifo in ifos]
        lo, hi = np.percentile(runs[0]["t_det"], [1, 99])
        print(f"t_det candidates per detector: {cands}; posterior 1-99%: [{lo:.5f}, {hi:.5f}]")
        inj["t_det"] = cands[ref_ifo]
    return inj


def get_truths(args, dirs, names, runs):
    if args.no_truths:
        return None
    truths_file = dirs[0] / "truths.json"
    if truths_file.is_file():
        with open(truths_file) as f:
            truth = json.load(f)
        print("Truths from", truths_file)
    else:
        config = dirs[0] / "config.final.toml"
        if not config.is_file():
            print("No truths.json or config.final.toml; no truth lines")
            return None
        with open(config, "rb") as f:
            cfg = tomllib.load(f)
        if cfg["data"].get("type") != "injection":
            return None
        print("No truths.json; reconstructing truths from config (approximate)")
        truth = truths_from_config(cfg, names, runs, args.ref_ifo, args.detectors)
    return [truth.get(k) for k in names]  # None = no line


def main():
    parser, args = parse_args()
    dirs = [Path(d).expanduser() for d in args.dirs]
    labels = args.legend or [d.name for d in dirs]

    runs = load_runs(parser, dirs)
    for d, r in zip(dirs, runs):
        k0 = next(iter(r))
        print(f"{d.name}: {len(r[k0])} samples, {len(np.unique(r[k0]))} unique ({k0})")

    names = select_names(parser, runs, args.params)
    print("Plotting:", names)
    truths = get_truths(args, dirs, names, runs)

    # Font sizes scale with the number of panels
    n = len(names)
    label_fs = max(18, min(35, 3 * n))
    tick_fs = max(12, min(20, int(1.5 * n)))
    legend_fs = max(16, min(70, 5 * n))

    fig = None
    for i, run in enumerate(runs):
        samples = np.column_stack([run[k] for k in names])
        fig = corner.corner(
            samples, fig=fig, color=f"C{i}", labels=[LABELS.get(k, k) for k in names],
            label_kwargs={"fontsize": label_fs}, labelpad=0.1,
            truths=truths if i == 0 else None, truth_color="k",
            hist_kwargs={"density": True}, plot_datapoints=True, plot_density=False, plot_contours=False
        )

    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=tick_fs)

    fig.legend(handles=[Line2D([], [], color=f"C{i}", label=l) for i, l in enumerate(labels)],
               loc="upper right", fontsize=legend_fs)
    fig.savefig(args.out, dpi=100, bbox_inches="tight")
    print("Saved", args.out)


if __name__ == "__main__":
    main()