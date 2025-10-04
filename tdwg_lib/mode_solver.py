"""
Eigenmode solver for 2D waveguides.

This module solves for propagating modes in graded-index waveguides using:
- FFT-based differential operators for efficient computation
- Sparse eigenvalue solver for mode extraction
- Support for both paraxial (Fresnel) and non-paraxial formulations

Uses NumPy and SciPy for numerical computations.
"""

from scipy.linalg import circulant
from scipy.sparse.linalg import eigs, eigsh, LinearOperator
import tdwg_lib.ftutils_np as ftutils
import numpy as np
import scipy

def solve_modes(x_axis, n_ref, k0, dn, Nmodes, verbose = True, fresnel = False):
    """
    Solve for propagating modes in a graded-index waveguide using eigenmode decomposition.

    Uses FFT-based differential operators and sparse eigenvalue solver to find the
    highest-effective-index modes. Supports both paraxial (Fresnel) and non-paraxial formulations.

    Inputs:
    -------
    x_axis : numpy.ndarray, shape (Nx,)
        Transverse coordinate grid (must be uniformly spaced).
    n_ref : float
        Reference refractive index (typically effective index of slab mode).
    k0 : float
        Free-space wavenumber (2π/λ₀) in inverse length units.
    dn : numpy.ndarray, shape (Nx,)
        Refractive index modulation profile (complex array; imaginary part represents loss/gain).
    Nmodes : int
        Number of modes to solve for (highest effective index modes).
    verbose : bool, default True
        If True, print warning when unbound/leaky modes are found.
    fresnel : bool, default False
        If True, use paraxial (Fresnel) approximation; if False, use full scalar wave equation.

    Returns:
    --------
    eigenval : numpy.ndarray, shape (Nmodes,)
        Real-valued propagation constants (beta values) of the modes in ascending order.
    eigenvec : numpy.ndarray, shape (Nmodes, Nx)
        Real-valued mode profiles normalized to discrete L2 norm, each row is one mode.
    """
    Nx = len(x_axis)
    dx = x_axis[1] - x_axis[0]
    kx_axis = 2*np.pi * ftutils.ft_f_axis(Nx, dx).astype(complex)  # frequency axis

    def wave_eq_operator_action(u):
        # u is a vector of shape (Nx,)
        u_hat = ftutils.fft_centered(u)
        # Compute second derivative via FFT (in Fourier space)

        if fresnel: 
            u_xx_hat = -kx_axis**2 * u_hat
            u_xx = ftutils.ifft_centered(u_xx_hat)
            return 1 / (2 * n_ref * k0) * u_xx + k0 * (dn + n_ref) * u
        if not fresnel: 
            u_xx_hat = np.sqrt((k0 * n_ref)**2 - kx_axis**2) * u_hat
            u_xx = ftutils.ifft_centered(u_xx_hat)
            return u_xx + k0 * dn * u
        
        # Apply the full operator: i/(2*n_ref*k0)*d^2u/dx^2 + i*k0*n(x)*u
        
    wave_eq_operator = LinearOperator(
        shape=(Nx, Nx),
        matvec=wave_eq_operator_action,
        dtype=complex
    )
    
    eigenval, eigenvec = eigs(wave_eq_operator, k=Nmodes, which="LR")
    eigenvec = eigenvec.T/np.sqrt(dx)
    # rotate eigenvecs to be real
    eigenvec = eigenvec * np.expand_dims(np.exp(-1j*np.angle(eigenvec[:,int(Nx/2)+1])), -1)
    # standardize sign of modes such that just left of the center, modes will be positive
    eigenvec = eigenvec / np.sign(eigenvec[:, int(Nx/2)+1])[:,np.newaxis]
    
    # check for cladding/leaky modes
    if verbose: 
        if np.real(eigenval-n_ref*k0).min()<0: 
            print('Found unbound mode')
    
    return np.real(eigenval), np.real(eigenvec)