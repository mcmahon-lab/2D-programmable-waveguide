"""
Waveguide configuration and utilities.

This module provides:
- Waveguide class as a container for storing simulation geometry and parameters
- Absorbing boundary conditions (PML-like) for suppressing reflections
"""

import numpy as np
import torch
import tdwg_lib.ftutils_torch as ftutils

class Waveguide():
    """
    Container class for waveguide simulation geometry and parameters.

    Stores coordinate axes, refractive index profiles, and derived quantities
    (Fourier axes, spacing, etc.) for 2D waveguide simulations.
    """
    def __init__(self, neff, x_axis, z_axis, background_delta_n = None, device = 'cpu'):
        """
        Initialize waveguide with simulation geometry.

        All length units are in microns (μm).

        Inputs:
        -------
        neff : float
            Effective refractive index of the slab waveguide (reference index).
        x_axis : torch.Tensor, shape (Nx,)
            Transverse coordinate grid (must be uniformly spaced).
        z_axis : torch.Tensor, shape (Nz,)
            Propagation coordinate grid (must be uniformly spaced).
        background_delta_n : torch.Tensor, shape (Nx,), optional
            Background refractive index modulation profile. If None, uses uniform profile (all zeros).
        device : str, default 'cpu'
            PyTorch device for tensors ('cpu' or 'cuda').

        Attributes:
        -----------
        n : float - effective refractive index
        x_axis, z_axis : torch.Tensor - coordinate grids
        Nx, Nz : int - grid sizes
        dx, dz : float - grid spacings
        Lz : float - total propagation length
        lam0 : float - wavelength (1.55 μm)
        k0 : float - free-space wavenumber
        fx_axis, kx_axis : torch.Tensor - frequency/wavenumber axes
        x2ind, z2ind : callable - coordinate-to-index converters
        background_delta_n : torch.Tensor - background index profile
        """
        #all the units for length are in microns!
        self.n =  neff # effective refractive index for the slab waveguide

        self.z_axis = z_axis
        self.Nz = len(z_axis)
        self.dz = z_axis[1] - z_axis[0]
        self.Lz = z_axis.max()

        self.x_axis = x_axis
        self.Nx = len(x_axis)
        self.dx = x_axis[1] - x_axis[0]

        self.lam0 = 1.55 #wavelength of the fundamental
        self.k0 = 2*np.pi/self.lam0 # k-number in free space

        self.fx_axis = ftutils.ft_f_axis(self.Nx, self.dx)
        self.kx_axis = 2*np.pi*self.fx_axis.to(torch.complex64)

        self.x2ind = lambda x: int(np.argmin(np.abs(self.x_axis-x)))
        self.z2ind = lambda z: int(np.argmin(np.abs(self.z_axis-z)))
        self.zlist2ind = lambda z: int(np.argmin(np.abs(self.z_list-z)))

        if background_delta_n is None:
            background_delta_n = torch.zeros_like(x_axis)
        if background_delta_n.shape != (self.Nx,):
            raise ValueError('spatial_map has wrong shape, should be [self.Nx,]')
        self.background_delta_n = background_delta_n.to(device)
        
def add_absorbing_boundary(x_axis, dn_slice, k0, abs_width=10, sigma_max=0.05, power=2):
    """
    Add PML-like absorbing boundaries to suppress reflections at domain edges.

    Adds imaginary refractive index component (absorption) that increases smoothly
    near the boundaries, mimicking a Perfectly Matched Layer (PML).

    Inputs:
    -------
    x_axis : torch.Tensor, shape (Nx,)
        Transverse coordinate grid.
    dn_slice : torch.Tensor, shape (Nx,)
        Refractive index modulation profile (real or complex).
    k0 : float
        Free-space wavenumber (2π/λ₀), included for consistency but not currently used.
    abs_width : float, default 10
        Width of absorbing region at each boundary (same units as x_axis).
    sigma_max : float, default 0.05
        Maximum imaginary index (absorption coefficient) at boundaries.
    power : float, default 2
        Polynomial order for absorption profile (higher = sharper transition).

    Returns:
    --------
    n_complex : torch.Tensor, shape (Nx,), complex
        Modified index profile with absorbing boundaries: dn_slice + i*sigma(x).
    """
    Nx = len(x_axis)
    sigma = torch.zeros_like(x_axis)

    x_min = x_axis[0]
    x_max = x_axis[-1]

    # Left boundary
    left_mask = x_axis <= x_min + abs_width
    x_left = x_axis[left_mask]
    sigma[left_mask] = sigma_max * ((x_min + abs_width - x_left) / abs_width) ** power

    # Right boundary
    right_mask = x_axis >= x_max - abs_width
    x_right = x_axis[right_mask]
    sigma[right_mask] = sigma_max * ((x_right - (x_max - abs_width)) / abs_width) ** power

    # Final complex index
    n_complex = dn_slice + 1j * sigma
    return n_complex
