import numpy as np
from refl1d.sample.reflectivity import reflectivity_amplitude as reflamp, convolve

def compute_nr_from_sld(
    sld_arr,
    Q=None,                         # pass measured Q if you have it
    qmin=0.00843, qmax=0.09275, n_q=400,
    dq_over_q=0.025,                # instrument resolution (fractional)
    sigma=3.0                       # interfacial roughness in Å (scalar or array o
):
    """
    sld_arr: (z, rho[, irho])
      z:    depth grid in Å, strictly increasing (fronting -> backing)
      rho:  SLD in units of 1e-6 Å^-2
      irho: optional absorption in 1e-6 Å^-2
    Returns: (Q, R) reflectivity at Q
    """
    z   = np.asarray(sld_arr[0], dtype=float)
    rho = np.asarray(sld_arr[1], dtype=float)
    irho = None if len(sld_arr) < 3 or sld_arr[2] is None else np.asarray(sld_arr[2])
    # sanity
    if z.ndim!=1 or rho.ndim!=1 or z.size!=rho.size:
        raise ValueError("z and rho must be 1D and the same length.")
    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing.")
    # Build Q
    if Q is None:
        Q = np.linspace(qmin, qmax, n_q)
    else:
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 1:
            raise ValueError("Q must be 1D")
    dQ = np.clip(Q * dq_over_q, 1e-9, None)
    # Convert continuous profile to slabs expected by reflamp:
    # thickness w_j for each slab j is the delta in z; the last layer is semi-infin
    w = np.diff(z)
    w = np.r_[w, 0.0]                         # last (substrate) thickness 0 => sem
    rrho = rho
    iirho = np.zeros_like(rrho) if irho is None else irho
    # Sigma: per-interface roughness; length must be number_of_interfaces = len(w)
    if np.isscalar(sigma):
        sigma_arr = np.full(len(w)-1, float(sigma))
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.size != len(w)-1:
            raise ValueError("sigma must be scalar or length len(z)-1")