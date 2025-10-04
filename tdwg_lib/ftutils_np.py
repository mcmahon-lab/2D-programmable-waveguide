"""
Fourier transform utilities using NumPy.

This module provides centered and normalized FFT variants for 1D and 2D transforms:
- Centered transforms: DC component at array center (using fftshift)
- Isometric transforms: preserve integrals (physical units)
- Orthogonal transforms: preserve L2 norm (unitary)
- Frequency axes: asymmetric with DC at zero (not symmetric about zero)

All functions use NumPy's FFT implementation.
"""

import numpy.fft as fft

def ft_t_axis(N, dt):
    """
    Generate centered time/space coordinate axis for FFT.

    Inputs:
    -------
    N : int - Number of grid points
    dt : float - Grid spacing

    Returns:
    --------
    t_axis : numpy.ndarray, shape (N,) - Centered coordinate axis from -N*dt/2 to N*dt/2
    """
    return fft.fftshift(fft.fftfreq(N))*N*dt

def ft_f_axis(N, dt):
    """
    Generate centered frequency/wavenumber axis for FFT.

    Creates asymmetric frequency axis with DC at zero (not symmetric about zero).

    Inputs:
    -------
    N : int - Number of grid points
    dt : float - Grid spacing in time/space domain

    Returns:
    --------
    f_axis : numpy.ndarray, shape (N,) - Centered frequency axis with DC component at center
    """
    return fft.fftshift(fft.fftfreq(N))/dt

### 1D fourier transform functions ###

def fft_centered(y):
    """
    Compute centered FFT (DC component at array center).

    Inputs: y : numpy.ndarray
    Returns: Y : numpy.ndarray - FFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.fft(y)
    y = fft.fftshift(y)
    return y

def ifft_centered(Y):
    """
    Compute centered inverse FFT (DC component at array center).

    Inputs: Y : numpy.ndarray
    Returns: y : numpy.ndarray - IFFT with DC at center
    """
    Y = fft.ifftshift(Y)
    Y = fft.ifft(Y)
    Y = fft.fftshift(Y)
    return Y

def fft_iso(y, dt):
    """
    Compute isometric FFT (preserves integrals).

    Inputs: y : numpy.ndarray, dt : float (grid spacing)
    Returns: Y : numpy.ndarray - FFT scaled to preserve integral
    """
    return fft.fft(y)*dt

def ifft_iso(Y, dt):
    """
    Compute isometric inverse FFT (preserves integrals).

    Inputs: Y : numpy.ndarray, dt : float (grid spacing)
    Returns: y : numpy.ndarray - IFFT scaled to preserve integral
    """
    return fft.ifft(Y)/dt

def fft_centered_iso(y, dt):
    """
    Compute centered isometric FFT (DC at center, preserves integrals).

    Inputs: y : numpy.ndarray, dt : float
    Returns: Y : numpy.ndarray
    """
    y = fft.ifftshift(y)
    y = fft_iso(y, dt)
    y = fft.fftshift(y)
    return y

def ifft_centered_iso(Y, dt):
    """
    Compute centered isometric inverse FFT (DC at center, preserves integrals).

    Inputs: Y : numpy.ndarray, dt : float
    Returns: y : numpy.ndarray
    """
    Y = fft.ifftshift(Y)
    Y = ifft_iso(Y, dt)
    Y = fft.fftshift(Y)
    return Y

def fft_centered_ortho(y):
    """
    Compute centered orthogonal FFT (DC at center, preserves L2 norm).

    Inputs: y : numpy.ndarray
    Returns: Y : numpy.ndarray - Unitary FFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.fft(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y

def ifft_centered_ortho(y):
    """
    Compute centered orthogonal inverse FFT (DC at center, preserves L2 norm).

    Inputs: Y : numpy.ndarray
    Returns: y : numpy.ndarray - Unitary IFFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.ifft(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y

### 2D fourier transform functions ###

def fft2_centered(y):
    """
    Compute 2D centered FFT (DC component at array center).

    Inputs: y : numpy.ndarray, shape (Ny, Nx)
    Returns: Y : numpy.ndarray - 2D FFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.fft2(y)
    y = fft.fftshift(y)
    return y

def ifft2_centered(Y):
    """
    Compute 2D centered inverse FFT (DC component at array center).

    Inputs: Y : numpy.ndarray, shape (Ny, Nx)
    Returns: y : numpy.ndarray - 2D IFFT with DC at center
    """
    Y = fft.ifftshift(Y)
    Y = fft.ifft2(Y)
    Y = fft.fftshift(Y)
    return Y

def fft2_iso(y, dx, dy):
    """
    Compute 2D isometric FFT (preserves integrals).

    Inputs: y : numpy.ndarray, dx : float, dy : float (grid spacings)
    Returns: Y : numpy.ndarray - 2D FFT scaled to preserve integral
    """
    return fft.fft2(y)*dx*dy

def ifft2_iso(Y, dx, dy):
    """
    Compute 2D isometric inverse FFT (preserves integrals).

    Inputs: Y : numpy.ndarray, dx : float, dy : float (grid spacings)
    Returns: y : numpy.ndarray - 2D IFFT scaled to preserve integral
    """
    return fft.ifft2(Y)/(dx*dy)

def fft2_centered_iso(y, dx, dy):
    """
    Compute 2D centered isometric FFT (DC at center, preserves integrals).

    Inputs: y : numpy.ndarray, dx : float, dy : float
    Returns: Y : numpy.ndarray
    """
    y = fft.ifftshift(y)
    y = fft2_iso(y, dx, dy)
    y = fft.fftshift(y)
    return y

def ifft2_centered_iso(Y, dx, dy):
    """
    Compute 2D centered isometric inverse FFT (DC at center, preserves integrals).

    Inputs: Y : numpy.ndarray, dx : float, dy : float
    Returns: y : numpy.ndarray
    """
    Y = fft.ifftshift(Y)
    Y = ifft2_iso(Y, dx, dy)
    Y = fft.fftshift(Y)
    return Y

def fft2_centered_ortho(y):
    """
    Compute 2D centered orthogonal FFT (DC at center, preserves L2 norm).

    Inputs: y : numpy.ndarray, shape (Ny, Nx)
    Returns: Y : numpy.ndarray - Unitary 2D FFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.fft2(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y

def ifft2_centered_ortho(y):
    """
    Compute 2D centered orthogonal inverse FFT (DC at center, preserves L2 norm).

    Inputs: Y : numpy.ndarray, shape (Ny, Nx)
    Returns: y : numpy.ndarray - Unitary 2D IFFT with DC at center
    """
    y = fft.ifftshift(y)
    y = fft.ifft2(y, norm = 'ortho')
    y = fft.fftshift(y)
    return y