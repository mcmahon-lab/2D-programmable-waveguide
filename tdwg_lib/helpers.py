"""
Helper utilities for 2D waveguide simulations.

This module provides:
- Gaussian smoothing filters for 1D and 2D tensors in Fourier space
- Timestamp generation utilities
"""

import torch
import time
import numpy as np
import math

def timestring():
    """
    Generate a timestamp string for file naming and logging.

    Returns:
    --------
    str
        Timestamp in format "YYYY-MM-DD--HH-MM-SS".
    """
    return time.strftime("%Y-%m-%d--%H-%M-%S")

def smoothen1d(x_axis, tensor, scale):
    """
    Apply Gaussian smoothing filter to a 1D tensor in Fourier space.

    Convolves the input with a Gaussian kernel of characteristic width 'scale' using
    efficient FFT-based filtering. Useful for emulating finite resolution or numerical stability.

    Inputs:
    -------
    x_axis : torch.Tensor, shape (Nx,)
        Spatial coordinate grid (must be uniformly spaced).
    tensor : torch.Tensor, shape (Nx,)
        Input tensor to be smoothed (real or complex).
    scale : float
        Characteristic width of Gaussian filter in same units as x_axis.

    Returns:
    --------
    tensor_filtered : torch.Tensor, shape (Nx,)
        Smoothed tensor with same dtype as input.
    """
    Nx = len(x_axis)
    dx = x_axis[1] - x_axis[0]

    # Frequency axes
    fx = torch.fft.fftfreq(Nx, d=dx).to(tensor.device)

    # Construct Gaussian filter: exp(- (k^2 * scale^2))
    gaussian_filter = torch.exp(- fx**2 * scale**2 )

    # FFT of the tensor
    tensor_fft = torch.fft.fft(tensor)

    # Apply the filter
    tensor_fft_filtered = tensor_fft * gaussian_filter

    # Inverse FFT to get back to spatial domain
    tensor_filtered = torch.fft.ifft(tensor_fft_filtered)#.real

    return tensor_filtered

def smoothen2d(x_axis, z_axis, tensor, scale_x, scale_z):
    """
    Apply 2D Gaussian smoothing filter to a 2D tensor in Fourier space.

    Convolves the input with a separable 2D Gaussian kernel using efficient FFT-based
    filtering. Applies different smoothing scales in x and z directions.

    Inputs:
    -------
    x_axis : torch.Tensor, shape (Nx,)
        Transverse coordinate grid (must be uniformly spaced).
    z_axis : torch.Tensor, shape (Nz,)
        Propagation coordinate grid (must be uniformly spaced).
    tensor : torch.Tensor, shape (Nz, Nx)
        Input 2D tensor to be smoothed (real or complex).
    scale_x : float
        Characteristic width of Gaussian filter in x direction (same units as x_axis).
    scale_z : float
        Characteristic width of Gaussian filter in z direction (same units as z_axis).

    Returns:
    --------
    tensor_filtered : torch.Tensor, shape (Nz, Nx)
        Smoothed 2D tensor with same dtype as input.
    """
    Nx = len(x_axis)
    Nz = len(z_axis)
    dx = x_axis[1] - x_axis[0]
    dz = z_axis[1] - z_axis[0]

    # Frequency axes
    fx = torch.fft.fftfreq(Nx, d=dx).to(tensor.device)
    fz = torch.fft.fftfreq(Nz, d=dz).to(tensor.device)

    # Construct Gaussian filter: exp(- (k^2 * scale^2))
    gaussian_filter_x = torch.exp(- fx**2 * scale_x**2 )
    gaussian_filter_z = torch.exp(- fz**2 * scale_z**2 )

    # FFT of the tensor
    tensor_fft2 = torch.fft.fft2(tensor)

    # Apply the filter
    tensor_fft2_filtered = tensor_fft2 * gaussian_filter_x
    tensor_fft2_filtered = (tensor_fft2_filtered.T * gaussian_filter_z).T

    # Inverse FFT to get back to spatial domain
    tensor_filtered = torch.fft.ifft2(tensor_fft2_filtered)#.real

    return tensor_filtered