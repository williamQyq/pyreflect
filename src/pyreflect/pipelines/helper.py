import numpy as np
from refnx.reflect import SLD, ReflectModel

from refl1d.sample.reflectivity import reflectivity_amplitude as reflamp, convolve


def compute_nr_from_sld(
    sld_arr,
    Q=None,
    qmin=0.008,
    qmax=0.20,
    n_q=400,
    dq_over_q=0.01,
    sigma=3.0,
    rho_fronting=0.0,     # air
    rho_backing=2.075     # Si substrate
):
    """
    Compute neutron reflectivity from a continuous SLD profile using refl1d.

    Parameters
    ----------
    sld_arr : tuple
        (z, rho) or (z, rho, irho)
        z, rho are ordered from substrate -> air
    """

    # -----------------------------
    # Parse input
    # -----------------------------
    z = np.asarray(sld_arr[0], dtype=float)
    rho = np.asarray(sld_arr[1], dtype=float)
    irho = None

    if len(sld_arr) >= 3 and sld_arr[2] is not None:
        irho = np.asarray(sld_arr[2], dtype=float)

    if z.ndim != 1 or rho.ndim != 1 or z.size != rho.size:
        raise ValueError("z and rho must be 1D arrays of equal length")

    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing (substrate → air)")

    # -----------------------------
    # Reverse SLD: substrate → air  →  air → substrate
    # -----------------------------
    z = z[::-1]
    rho = rho[::-1]
    if irho is not None:
        irho = irho[::-1]

    # shift so fronting interface is at z = 0
    z = z - z[0]

    # -----------------------------
    # Build Q
    # -----------------------------
    if Q is None:
        Q = np.linspace(qmin, qmax, n_q)
    else:
        Q = np.asarray(Q, dtype=float)

    dQ = np.clip(Q * dq_over_q, 1e-9, None)

    # -----------------------------
    # Convert continuous profile → slabs
    # -----------------------------
    w_film = np.diff(z)

    # slab thicknesses: fronting | film | substrate
    w = np.r_[0.0, w_film, 0.0]

    # SLDs: fronting | film | substrate
    rrho = np.r_[rho_fronting, rho, rho_backing]

    if irho is None:
        iirho = np.zeros_like(rrho)
    else:
        iirho = np.r_[0.0, irho, 0.0]

    # -----------------------------
    # Roughness per interface
    # -----------------------------
    n_interfaces = len(rrho) - 1

    if np.isscalar(sigma):
        sigma_arr = np.full(n_interfaces, float(sigma))
    else:
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.size != n_interfaces:
            raise ValueError("sigma must be scalar or length len(rrho)-1")

    # -----------------------------
    # Reflectivity calculation
    # -----------------------------
    r = reflamp(Q, w, rrho, iirho, sigma_arr)
    R = np.abs(r) ** 2

    # Instrument resolution convolution
    R = convolve(Q, R, Q, dQ)

    return Q, R

def compute_nr_from_sld_refnx(
    sld_arr,
    qmin=0.00843,
    qmax=0.09275,
    n_q=400,
    dq_over_q=0.025,
    sigma=3.0,
):
    """
    Compute neutron reflectivity using refnx from a continuous SLD profile.

    Parameters
    ----------
    sld_arr : tuple
        (z, rho) or (z, rho, irho)
        z, rho ordered from substrate -> air
    qmin, qmax : float
        Q range (Å^-1)
    n_q : int
        Number of Q points
    dq_over_q : float
        Fractional resolution ΔQ / Q
    sigma : float
        Interfacial roughness in Å

    Returns
    -------
    Q : ndarray
        Momentum transfer (Å^-1)
    R : ndarray
        Reflectivity
    """

    # -----------------------------
    # Parse input
    # -----------------------------
    z = np.asarray(sld_arr[0], dtype=float)
    rho = np.asarray(sld_arr[1], dtype=float)

    if z.ndim != 1 or rho.ndim != 1 or z.size != rho.size:
        raise ValueError("z and rho must be 1D arrays of equal length")

    if not np.all(np.diff(z) > 0):
        raise ValueError("z must be strictly increasing (substrate → air)")

    # -----------------------------
    # Reverse: substrate → air  →  air → substrate
    # -----------------------------
    z = z[::-1]
    rho = rho[::-1]

    # shift so fronting interface starts at z = 0
    z = z - z[0]

    # -----------------------------
    # Define fronting & backing
    # -----------------------------
    air = SLD(0.0, name="air")
    substrate = SLD(rho[-1], name="substrate")

    # -----------------------------
    # Build slab layers
    # -----------------------------
    layers = []
    for i in range(len(z) - 1):
        thickness = z[i + 1] - z[i]
        mid_sld = 0.5 * (rho[i] + rho[i + 1])

        layer = SLD(mid_sld, name=f"layer{i}")(thickness, sigma)
        layers.append(layer)

    # -----------------------------
    # Build structure: air | film | substrate
    # -----------------------------
    structure = air(0, sigma)
    for layer in layers:
        structure |= layer
    structure |= substrate(0, sigma)

    # -----------------------------
    # Reflectivity model
    # -----------------------------
    model = ReflectModel(structure, bkg=0.0, scale=1.0)
    model.dq = dq_over_q

    # -----------------------------
    # Compute NR
    # -----------------------------
    Q = np.linspace(qmin, qmax, n_q)
    R = model(Q)

    return Q, R


# for experimental data
def reverse_y_order(sld_array):
    """
    Flip SLD y from left to right, substrate to air direction
    """
    A_flipped = sld_array.copy()
    A_flipped[1] = A_flipped[1, ::-1]
    return A_flipped


def find_substrate_critical_idx(arr, target=2.075):
    """
    the idx of critical point where substrate transit to material film
    """
    y = arr[1]
    for i in range(len(y) - 1):
        if y[i] <= target and y[i + 1] > target:
            return i


def align_points(A, B):
    """
    A is expt sld, B is pred sld

    Align B to A using shift.
    Both A and B are shape (2, N).
    """
    # Step 1: find index in A where y ≈ 2.07 (left edge)
    target_y = 2.07

    # Search from left to right, the closest idx x which y close to target y
    idx_A = find_substrate_critical_idx(A)

    idx_B = find_substrate_critical_idx(B)

    # Step 2: compute the translation vector
    shift = A[:, idx_A] - B[:, idx_B]

    # Step 3: apply the shift
    B_aligned = B + shift[:, np.newaxis]

    return B_aligned