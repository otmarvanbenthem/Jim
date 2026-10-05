import numpy as np

G, c, M_SUN = 6.67430e-11, 299792458.0, 1.98847e30

def duration(f_min, f_max, Mc):
    """Mc in solar masses, f in Hz. Returns duration in seconds."""
    Mc = Mc * M_SUN * G / c**3  # convert to seconds
    tau = lambda f: (5 / 256) * Mc ** (-5 / 3) * (np.pi * f) ** (-8 / 3)
    return tau(f_min) - tau(f_max)

print(duration(f_min=20, f_max=2048, Mc=1.186))