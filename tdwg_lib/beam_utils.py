import torch
import math 

def gaussian_beam(
    x: torch.Tensor,
    x_center: float,
    w0: float,
    z0: float = 0.0,
    wavelength: float = 1.55,
    include_phase: bool = True,
    dtype=torch.complex128,
):
    """
    [ChatGPT-created]
    Fundamental Gaussian beam sampled at longitudinal offset z0 from the waist (focus).

    Parameters
    ----------
    x : (Nx,) torch.Tensor
        Transverse coordinate grid (real). Can be nonuniform; we normalize with trapz over x.
    x_center : float
        Beam center (transverse).
    w0 : float
        Beam waist at focus (1/e field radius at z=0).
    z0 : float, default 0.0
        Longitudinal distance from the waist where the beam is sampled (same units as x).
    wavelength : float, default 1.55
        Wavelength (same length units as x and w0). Used for Rayleigh range and curvature.
    include_phase : bool, default True
        If True, include quadratic phase from wavefront curvature at z0.
        (Global phase is omitted since it doesn’t affect normalization.)
    dtype : torch dtype for complex output (default complex128)

    Returns
    -------
    mode : (Nx,) torch.Tensor, complex
        Discretely L2-normalized Gaussian beam field at plane z0.
    """
    # Work in float64 for geometry; cast to complex at the end
    xr = x.to(torch.float64)
    xc = torch.tensor(x_center, dtype=torch.float64, device=x.device)

    # Beam parameters
    k = 2.0 * math.pi / wavelength                       # wavenumber
    zR = math.pi * (w0 ** 2) / wavelength                # Rayleigh range
    wz = w0 * math.sqrt(1.0 + (z0 / zR) ** 2)            # spot size at z0
    # radius of curvature R(z) (∞ at focus)
    if z0 == 0.0:
        R = math.inf
    else:
        R = z0 * (1.0 + (zR / z0) ** 2)

    # Field envelope at plane z0 (no huge numbers; always well scaled)
    X = xr - xc
    env = (w0 / wz) * torch.exp(-(X ** 2) / (wz ** 2))   # real-valued envelope

    # Optional phase: quadratic curvature term; omit global phase
    if include_phase and math.isfinite(R):
        phase = torch.exp(-1j * torch.tensor(k / (2.0 * R), dtype=torch.complex128, device=x.device) * (X.to(torch.complex128) ** 2))
        field = (env.to(torch.complex64)) * phase
    else:
        field = env.to(torch.complex64)

    # Discrete L2 normalization using trapz over x (robust for nonuniform grids)
    power = torch.trapz(torch.abs(field) ** 2, xr)
    mode = field / torch.sqrt(power + torch.finfo(torch.float32).eps)
    return mode.to(dtype)


def make_HG_modes(x_axis: torch.Tensor,
                  x_center: float,
                  w0: float,
                  n: int,
                  *,
                  dtype=torch.float64) -> torch.Tensor:
    """
    [ChatGPT-created]
    Return (n, Nx) tensor of 1D Hermite–Gauss modes centered at x_center with waist w0,
    orthonormal on the *continuous* line (up to discretization error),
    evaluated on x_axis. Output dtype is complex128 (real values cast to complex).

    Uses the stable normalized Hermite-function recurrence:
        ψ0(u) = π^(-1/4) exp(-u^2/2)
        ψ1(u) = √2 * u * ψ0(u)
        ψ_{k+1}(u) = √(2/(k+1)) * u * ψ_k(u) - √(k/(k+1)) * ψ_{k-1}(u)
    and maps to HG with waist via
        φ_n(x; w0) = (1/√w0) ψ_n( √2 (x - x0) / w0 )

    Parameters
    ----------
    x_axis : torch.Tensor, shape (Nx,)
        Grid points (need not be uniform; normalization uses trapz).
    x_center : float
        Mode center.
    w0 : float
        Waist parameter.
    n : int
        Number of modes (from 0 to n-1).
    dtype : torch dtype for internal real computation (default float64).

    Returns
    -------
    modes : torch.Tensor, shape (n, Nx), dtype=complex128
        Each row i is the (discretely) L2-normalized HG_i on x_axis.
    """
    if n <= 0:
        return torch.empty((0, x_axis.numel()), dtype=torch.complex128, device=x_axis.device)

    x = x_axis.to(dtype)
    device = x.device

    # dimensionless scaled coordinate u = √2 (x - x0)/w0
    u = math.sqrt(2.0) * (x - x_center) / w0

    # ψ0(u) and scaling to HG with waist: φ0(x; w0) = (1/√w0) ψ0(u)
    pref0 = (math.pi ** (-0.25)) / math.sqrt(w0)
    phi0 = pref0 * torch.exp(-0.5 * u * u)

    modes = []
    # helper to discretely L2-normalize using trapz over x (handles non-uniform grids)
    def _normalize(vec):
        # real-valued here, but keep general |·|^2 in case of later complex extensions
        power = torch.trapz(vec.abs() ** 2, x)
        # avoid zero division on pathological grids
        return vec / torch.sqrt(power + torch.finfo(vec.dtype).eps)

    # φ0
    modes.append(_normalize(phi0))

    if n == 1:
        return torch.stack(modes, dim=0).to(torch.complex128)

    # φ1 = (1/√w0) ψ1(u) = (1/√w0) * √2 u ψ0 = √2 u * φ0
    sqrt2 = math.sqrt(2.0)
    phi1 = sqrt2 * u * phi0
    modes.append(_normalize(phi1))

    # Stable three-term recurrence for normalized modes
    phi_nm2 = phi0  # φ_{k-1}
    phi_nm1 = phi1  # φ_{k}
    for k in range(1, n - 1):
        # φ_{k+1} = √(2/(k+1)) u φ_k - √(k/(k+1)) φ_{k-1}
        a = math.sqrt(2.0 / (k + 1.0))
        b = math.sqrt(k / (k + 1.0))
        phi_np1 = a * u * phi_nm1 - b * phi_nm2
        modes.append(_normalize(phi_np1))
        phi_nm2, phi_nm1 = phi_nm1, phi_np1

    modes = torch.stack(modes, dim=0).to(torch.complex64)
    return modes

def make_boxed_modes(x_axis, N, xmode_out_lim, separation = 0):
    x2ind = lambda x: torch.argmin(torch.abs(x_axis-x)) 
    out_xsep_list = torch.linspace(-xmode_out_lim, xmode_out_lim, N+1)
    output_modes = []
    for i in range(N):
        output_mode = torch.zeros_like(x_axis)
        output_mode[x2ind(out_xsep_list[i] + separation/2):x2ind(out_xsep_list[i+1] - separation/2)] = 1
        output_modes.append(output_mode)
    output_modes = torch.vstack(output_modes)
    return output_modes