import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jimgw.core.single_event.detector import get_ET, get_ET_2
from jimgw.core.single_event.detector import get_ET, get_ET_2L

G, c, M_SUN = 6.67430e-11, 299792458.0, 1.98847e30

def duration(f_min, f_max, Mc):
    """Mc in solar masses, f in Hz. Returns duration in seconds."""
    Mc = Mc * M_SUN * G / c**3  # convert to seconds
    tau = lambda f: (5 / 256) * Mc ** (-5 / 3) * (np.pi * f) ** (-8 / 3)
    return tau(f_min) - tau(f_max)

def new_dl(old_dl,old_snr,new_snr):   
    '''Calculate d_l for a target network snr, given a reference snr and a reference d_l''' 
    dl = (old_snr * old_dl)/new_snr
    return dl

"""Helper tools for injection studies with ET and ET 2L."""


L

_DETECTOR_GETTERS = {
    "ET": get_ET,
    "ET_2L": get_ET_2L,
}


def _build_ifos(detector: str) -> list:
    """Build the detectors of ET or ET 2L, each with its PSD loaded."""
    if detector not in _DETECTOR_GETTERS:
        raise ValueError(f"detector must be one of {list(_DETECTOR_GETTERS)}, got {detector!r}.")
    ifos = _DETECTOR_GETTERS[detector]()
    if not isinstance(ifos, (list, tuple)):
        ifos = [ifos]
    ifos = list(ifos)
    for ifo in ifos:
        ifo.load_and_set_psd()
    return ifos


def get_network_optimal_snr(
    detector: str,
    waveform,
    parameters: dict[str, float],
    duration: float,
    sampling_frequency: float,
    trigger_time: float,
    f_min: float,
    f_max: float,
    allow_wraparound: bool = False,
) -> float:
    """Network optimal SNR of a signal in ET or ET 2L, from zero-noise injections."""
    snr_sq = 0.0
    for ifo in _build_ifos(detector):
        ifo.set_zero_noise(
            duration=duration,
            sampling_frequency=sampling_frequency,
            trigger_time=trigger_time,
            f_min=f_min,
            f_max=f_max,
        )
        # inject_signal returns (optimal SNR, matched-filter SNR)
        optimal_snr, _ = ifo.inject_signal(
            waveform, parameters, allow_wraparound=allow_wraparound
        )
        snr_sq += float(jnp.abs(optimal_snr)) ** 2
    return float(jnp.sqrt(snr_sq))


def get_d_L_for_target_snr(
    detector: str,
    waveform,
    parameters: dict[str, float],
    target_snr: float,
    duration: float,
    sampling_frequency: float,
    trigger_time: float,
    f_min: float,
    f_max: float,
    reference_d_L: float = 500.0,
    allow_wraparound: bool = False,
) -> float:
    """Luminosity distance (Mpc) at which the network optimal SNR equals ``target_snr``.

    The SNR scales as 1 / d_L, so it is computed once at ``reference_d_L``
    (zero noise) and rescaled:

        d_L = reference_d_L * SNR(reference_d_L) / target_snr

    Parameters
    ----------
    detector : "ET" or "ET_2L"
        Which detector configuration to use (get_ET() or get_ET_2L()).
    waveform : waveform model
        Frequency-domain waveform model, e.g. ``RippleIMRPhenomD_NRTidalv2(f_ref=5)``.
    parameters : dict
        Likelihood-space source parameters (waveform parameters plus ra, dec,
        psi, t_c). Any ``d_L`` in it is ignored in favour of ``reference_d_L``.
    target_snr : float
        Desired network optimal SNR.
    duration, sampling_frequency, trigger_time, f_min, f_max
        Data settings, as used for the real injection.
    reference_d_L : float
        Distance used for the scaling run. Any positive value works.
    allow_wraparound : bool
        Passed on to ``inject_signal``.
    """
    if target_snr <= 0:
        raise ValueError("target_snr must be positive.")

    params = dict(parameters)
    params["d_L"] = reference_d_L

    snr_ref = get_network_optimal_snr(
        detector,
        waveform,
        params,
        duration=duration,
        sampling_frequency=sampling_frequency,
        trigger_time=trigger_time,
        f_min=f_min,
        f_max=f_max,
        allow_wraparound=allow_wraparound,
    )
    if snr_ref <= 0:
        raise ValueError("Reference SNR is zero; cannot rescale to a target SNR.")

    return reference_d_L * snr_ref / target_snr


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)

    from jimgw.core.single_event.waveform import RippleIMRPhenomD_NRTidalv2

    waveform = RippleIMRPhenomD_NRTidalv2(f_ref=5.0)

    injection_parameters = {
        "M_c": 1.186,
        "eta": 0.248,
        "s1_z": 0.005,
        "s2_z": 0.005,
        "lambda_1": 300.0,
        "lambda_2": 450.0,
        "iota": 2.53,
        "phase_c": 0.0,
        "psi": 1.75,
        "ra": 3.446,
        "dec": -0.408,
        "t_c": 0.0,
    }

    kwargs = dict(
        duration=6800.0,
        sampling_frequency=4096.0,
        trigger_time=1766797218,
        f_min=5.0,
        f_max=2048.0,
    )

    for det in ["ET", "ET_2L"]:
        d_L = get_d_L_for_target_snr(
            det, waveform, injection_parameters, target_snr=30.0, **kwargs
        )
        check = get_network_optimal_snr(
            det, waveform, {**injection_parameters, "d_L": d_L}, **kwargs
        )
        print(f"{det}: d_L = {d_L:.2f} Mpc for SNR 30 (check: {check:.3f})")



