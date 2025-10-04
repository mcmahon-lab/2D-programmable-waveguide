"""
Split-step Fourier method solver for wave propagation.

This module provides split-step solvers for simulating wave propagation in 2D waveguides:
- Generic split-step solver with user-defined Fourier and real-space operators
- Beam Propagation Method (BPM) solver inherited from the generic split-step solver
- Can store or not store internal fields for propagation visualization or memory-efficiency, respectively

Uses PyTorch for GPU-accelerated computations.
"""

import torch
import torch.fft as fft
import numpy as np
import tdwg_lib.ftutils_torch as ftutils

class SplitStepSolver():
    """
    Generic split-step Fourier method solver for wave propagation.

    Implements the split-step algorithm with user-defined dispersive (D) and nonlinear (N) operators.
    The algorithm alternates between applying D in Fourier space and N in real space.
    """
    def __init__(self, D_step, D_half_step = None, Ncom = 1):
        """
        Initialize split-step solver with dispersive operators.

        Inputs:
        -------
        D_step : torch.Tensor, shape (Nx,)
            Full-step dispersive operator in Fourier space (applied at each propagation step).
        D_half_step : torch.Tensor, shape (Nx,), optional
            Half-step dispersive operator for symmetric splitting. If None, uses sqrt(D_step).
        Ncom : int, default 1
            Compression factor: fields are stored every Ncom steps when monitor=True.
        """
        self.D_step = D_step
        if D_half_step is None:
            D_half_step = torch.sqrt(D_step)
        self.D_half_step = D_half_step
        self.Ncom = Ncom

    def run_simulation(self, a, N, monitor = False):
        """
        Run split-step propagation simulation.

        Propagates initial field 'a' through Nz steps using symmetric split-step algorithm:
        D_half → (N → D)^Nz → D_half†, where D is dispersive and N is nonlinear operator.

        Inputs:
        -------
        a : torch.Tensor, shape (Nx,) or (Nmodes, Nx)
            Initial field(s) in real space (complex-valued).
        N : torch.Tensor, shape (Nz, Nx) or (Nz, 1) or (Nz,)
            Nonlinear operator at each z step (typically exp(i*phase)).
        monitor : bool, default False
            If True, store intermediate fields and compute output statistics.

        Returns:
        --------
        a : torch.Tensor, shape matching input
            Output field(s) after propagation.

        Side effects (if monitor=True):
        --------------------------------
        Sets the following attributes:
        - Emat_x : torch.Tensor, shape (Nz//Ncom, ...) - fields in real space
        - Emat_f : torch.Tensor, shape (Nz//Ncom, ...) - fields in Fourier space
        - Eout_x, Ein_x : numpy arrays - output/input fields in real space
        - Eout_f, Ein_f : numpy arrays - output/input fields in Fourier space
        - Iout_x, Iin_x : numpy arrays - output/input intensities in real space
        - Iout_f, Iin_f : numpy arrays - output/input intensities in Fourier space
        """
        if monitor:
            self.a_list = []
            self.ak_list = []

        D_step = self.D_step.to(a.device)
        D_half_step = self.D_half_step.to(a.device)
        N = N.to(a.device)

        ak = fft.fft(a)
        ak = D_half_step * ak
        for (z_ind, N_step) in enumerate(N):
            a = fft.ifft(ak)
            a = N_step * a
            ak = fft.fft(a)
            ak = D_step * ak

            if monitor:
                if (z_ind + 1) % self.Ncom == 0:
                    self.a_list.append(a)
                    self.ak_list.append(torch.fft.ifftshift(ak))
        ak = D_half_step.conj() * ak
        a = fft.ifft(ak)

        if monitor:
            assert len(self.a_list) > 0, "No fields were stored—check Ncom or N length."
            self.Emat_x = torch.stack(self.a_list)
            self.Emat_f = torch.stack(self.ak_list)

            self.Eout_x = self.Emat_x[-1].detach().cpu().numpy()
            self.Eout_f = self.Emat_f[-1].detach().cpu().numpy()
            self.Iout_x = abs(self.Emat_x[-1].detach().cpu().numpy())**2
            self.Iout_f = abs(self.Emat_f[-1].detach().cpu().numpy())**2

            self.Ein_x = self.Emat_x[0].detach().cpu().numpy()
            self.Ein_f = self.Emat_f[0].detach().cpu().numpy()
            self.Iin_x = abs(self.Emat_x[0].detach().cpu().numpy())**2
            self.Iin_f = abs(self.Emat_f[0].detach().cpu().numpy())**2
        return a    


class BPMSplitStepSolver(SplitStepSolver):
    """
    Beam Propagation Method (BPM) solver using split-step Fourier method.

    Specialized split-step solver for paraxial wave propagation in waveguides with
    refractive index modulation. Inherits from SplitStepSolver and sets up
    dispersive operator for Fresnel diffraction.
    """
    def __init__(self, x_axis, z_axis, n_ref, Ncom=1, k0=2*np.pi/1.55):
        """
        Initialize BPM solver with simulation geometry.

        Inputs:
        -------
        x_axis : torch.Tensor, shape (Nx,)
            Transverse coordinate grid (must be uniformly spaced).
        z_axis : torch.Tensor, shape (Nz,)
            Propagation coordinate grid (must be uniformly spaced).
        n_ref : float
            Reference refractive index (effective index of slab mode).
        Ncom : int, default 1
            Compression factor: fields are stored every Ncom steps when monitor=True.
        k0 : float, default 2π/1.55
            Free-space wavenumber (2π/λ₀) in μm⁻¹.
        """
        self.k0 = k0  # Wave number (2π/λ)
        self.n_ref = n_ref

        self.x_axis = x_axis
        self.Nx = len(x_axis)
        self.dx = x_axis[1] - x_axis[0]

        self.z_axis = z_axis
        self.dz = z_axis[1] - z_axis[0]

        self.Ncom = Ncom # wavefront is saved every Ncom integration steps

        self.fx_axis = ftutils.ft_f_axis(self.Nx, self.dx)
        self.kx_axis = 2*np.pi*self.fx_axis.to(torch.complex64)

        self.phase_shift = self.kx_axis**2/(2*self.n_ref*self.k0)*self.dz # Fresnel
        D_step = torch.fft.fftshift(torch.exp((-1j*self.phase_shift)))
        D_half_step = torch.fft.fftshift(torch.exp((-1j*self.phase_shift/2)))

        super().__init__(D_step, D_half_step, Ncom)

    def run_simulation(self, a, delta_n, monitor=False):
        """
        Run BPM simulation through waveguide with refractive index modulation.

        Propagates initial field through 2D refractive index profile using paraxial
        (Fresnel) approximation. The nonlinear operator is N = exp(i*k0*delta_n*dz).

        Inputs:
        -------
        a : torch.Tensor, shape (Nx,) or (Nmodes, Nx)
            Initial field(s) in real space (complex-valued).
        delta_n : torch.Tensor, shape (Nz, Nx) or (Nz, 1) or (Nz,)
            Refractive index modulation profile (complex; imaginary part = absorption/gain).
        monitor : bool, default False
            If True, store intermediate fields (see SplitStepSolver.run_simulation).

        Returns:
        --------
        a : torch.Tensor, shape matching input
            Output field(s) after propagation through waveguide.
        """
        device = a.device
        delta_n = delta_n.to(device)
        k0 = self.k0
        dz = torch.tensor(self.dz, device=device, dtype=delta_n.dtype)

        # Compute nonlinear operator: N[z, x] = exp(i k0 delta_n dz)
        phase = 1j * k0 * delta_n * dz
        N = torch.exp(phase)

        return super().run_simulation(a, N, monitor)