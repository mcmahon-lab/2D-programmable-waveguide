"""
Fourier transform utilities using PyTorch.

This module provides centered and normalized FFT variants for 1D and 2D transforms:
- Centered transforms: DC component at array center (using fftshift)
- Isometric transforms: preserve integrals (physical units)
- Orthogonal transforms: preserve L2 norm (unitary)
- Frequency axes: asymmetric with DC at zero (not symmetric about zero)

All functions use PyTorch's FFT implementation with GPU acceleration support.
"""

import torch.fft as fft

def ft_t_axis(N, dt, device = 'cpu'):
    """
    Generate centered time/space coordinate axis for FFT.

    Inputs:
    -------
    N : int - Number of grid points (must be even)
    dt : float - Grid spacing
    device : str - PyTorch device ('cpu' or 'cuda')

    Returns:
    --------
    t_axis : torch.Tensor, shape (N,) - Centered coordinate axis from -N*dt/2 to N*dt/2
    """
    if N % 2:
        raise Exception("N must be an even integer")
    return fft.fftshift(fft.fftfreq(N).to(device))*N*dt

def ft_f_axis(N, dt, device = 'cpu'):
    """
    Generate centered frequency/wavenumber axis for FFT.

    Creates asymmetric frequency axis with DC at zero (not symmetric about zero).

    Inputs:
    -------
    N : int - Number of grid points (must be even)
    dt : float - Grid spacing in time/space domain
    device : str - PyTorch device ('cpu' or 'cuda')

    Returns:
    --------
    f_axis : torch.Tensor, shape (N,) - Centered frequency axis with DC component at center
    """
    if N % 2:
        raise Exception("N must be an even integer")
    return fft.fftshift(fft.fftfreq(N).to(device))/dt

### 1D fourier transform functions ###

def fft_centered(y):
    """
    Compute centered FFT (DC component at array center).

    Inputs: y : torch.Tensor
    Returns: Y : torch.Tensor - FFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.fft(y)
    y = fft.fftshift(y)
    return y

def ifft_centered(Y):
    """
    Compute centered inverse FFT (DC component at array center).

    Inputs: Y : torch.Tensor
    Returns: y : torch.Tensor - IFFT with DC at center
    """
    Y = fft.ifftshift(Y)
    Y = fft.ifft(Y)
    Y = fft.fftshift(Y)
    return Y

def fft_iso(y, dt):
    """
    Compute isometric FFT (preserves integrals).

    Inputs: y : torch.Tensor, dt : float (grid spacing)
    Returns: Y : torch.Tensor - FFT scaled to preserve integral
    """
    return fft.fft(y)*dt

def ifft_iso(Y, dt):
    """
    Compute isometric inverse FFT (preserves integrals).

    Inputs: Y : torch.Tensor, dt : float (grid spacing)
    Returns: y : torch.Tensor - IFFT scaled to preserve integral
    """
    return fft.ifft(Y)/dt

def fft_centered_iso(y, dt):
    """
    Compute centered isometric FFT (DC at center, preserves integrals).

    Inputs: y : torch.Tensor, dt : float
    Returns: Y : torch.Tensor
    """
    y = fft.ifftshift(y)
    y = fft_iso(y, dt)
    y = fft.fftshift(y)
    return y

def ifft_centered_iso(Y, dt):
    """
    Compute centered isometric inverse FFT (DC at center, preserves integrals).

    Inputs: Y : torch.Tensor, dt : float
    Returns: y : torch.Tensor
    """
    Y = fft.ifftshift(Y)
    Y = ifft_iso(Y, dt)
    Y = fft.fftshift(Y)
    return Y

def fft_centered_ortho(y):
    """
    Compute centered orthogonal FFT (DC at center, preserves L2 norm).

    Inputs: y : torch.Tensor
    Returns: Y : torch.Tensor - Unitary FFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.fft(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y

def ifft_centered_ortho(y):
    """
    Compute centered orthogonal inverse FFT (DC at center, preserves L2 norm).

    Inputs: Y : torch.Tensor
    Returns: y : torch.Tensor - Unitary IFFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.ifft(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y

### 2D fourier transform functions ###

def fft2_centered(y):
    """
    Compute 2D centered FFT (DC component at array center).

    Inputs: y : torch.Tensor, shape (Ny, Nx)
    Returns: Y : torch.Tensor - 2D FFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.fft2(y)
    y = fft.fftshift(y)
    return y

def ifft2_centered(Y):
    """
    Compute 2D centered inverse FFT (DC component at array center).

    Inputs: Y : torch.Tensor, shape (Ny, Nx)
    Returns: y : torch.Tensor - 2D IFFT with DC at center
    """
    Y = fft.ifftshift(Y)
    Y = fft.ifft2(Y)
    Y = fft.fftshift(Y)
    return Y

def fft2_iso(y, dx, dy):
    """
    Compute 2D isometric FFT (preserves integrals).

    Inputs: y : torch.Tensor, dx : float, dy : float (grid spacings)
    Returns: Y : torch.Tensor - 2D FFT scaled to preserve integral
    """
    return fft.fft2(y)*dx*dy

def ifft2_iso(Y, dx, dy):
    """
    Compute 2D isometric inverse FFT (preserves integrals).

    Inputs: Y : torch.Tensor, dx : float, dy : float (grid spacings)
    Returns: y : torch.Tensor - 2D IFFT scaled to preserve integral
    """
    return fft.ifft2(Y)/(dx*dy)

def fft2_centered_iso(y, dx, dy):
    """
    Compute 2D centered isometric FFT (DC at center, preserves integrals).

    Inputs: y : torch.Tensor, dx : float, dy : float
    Returns: Y : torch.Tensor
    """
    y = fft.ifftshift(y)
    y = fft2_iso(y, dx, dy)
    y = fft.fftshift(y)
    return y

def ifft2_centered_iso(Y, dx, dy):
    """
    Compute 2D centered isometric inverse FFT (DC at center, preserves integrals).

    Inputs: Y : torch.Tensor, dx : float, dy : float
    Returns: y : torch.Tensor
    """
    Y = fft.ifftshift(Y)
    Y = ifft2_iso(Y, dx, dy)
    Y = fft.fftshift(Y)
    return Y