import torch
import time
import numpy as np
import math

def timestring():
    return time.strftime("%Y-%m-%d--%H-%M-%S")

def smoothen1d(x_axis, tensor, scale):
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